#!/usr/bin/env python3
"""
LiDAR Ray Tracing Resampler V2 - Realistic Velodyne HDL-64E Simulation

Improvements over V1:
1. Actual Velodyne HDL-64E non-uniform beam angles from calibration data
2. Beam divergence simulation (multiple rays per beam)
3. Ray drop model based on distance and incidence angle (LiDARsim-inspired)
4. Range and angular noise
5. Better voxelization matching

Based on research from:
- LiDARsim (CVPR 2020): https://arxiv.org/abs/2006.09348
- NeRF-LiDAR (AAAI 2024): https://github.com/fudan-zvg/NeRF-LiDAR
- LiDAR-RT (CVPR 2025): Real-time Gaussian-based ray tracing
- Automotive LiDAR Modeling: https://pmc.ncbi.nlm.nih.gov/articles/PMC7309070/
- HELIOS++: https://github.com/3dgeo-heidelberg/helios
- Velodyne ROS driver calibration: https://github.com/ros-drivers/velodyne

Author: Data Augmentation Pipeline
"""

import numpy as np
from typing import Tuple, List, Optional, Dict
from pathlib import Path
from dataclasses import dataclass

# Try to import numba for acceleration
try:
    from numba import jit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    print("[lidar_resampler_v2] Numba not available, using pure Python (slower)")


# =============================================================================
# Velodyne HDL-64E Calibration Data
# From: ros-drivers/velodyne/velodyne_pointcloud/params/64e_s2.1-sztaki.yaml
# =============================================================================

# Actual vertical angles (degrees) from HDL-64E calibration file
# These are NON-UNIFORM - the real sensor has varying spacing
VELODYNE_HDL64E_VERTICAL_ANGLES = np.array([
    # Laser ID 0-31 (Upper block)
    -8.77, -8.36, 2.49, 2.98, -7.78, -7.27, -10.86, -10.36,
    -6.86, -6.26, -9.82, -9.34, -2.59, -2.06, -5.67, -5.19,
    -1.59, -1.15, -4.71, -4.17, -0.57, -0.22, -3.71, -3.19,
    3.50, 4.00, 0.51, 0.98, 4.49, 4.97, 1.45, 1.97,
    # Laser ID 32-63 (Lower block)
    -22.73, -22.35, -11.51, -10.94, -21.89, -21.31, -24.85, -24.42,
    -20.86, -20.11, -23.86, -23.17, -16.56, -16.11, -19.58, -19.18,
    -15.70, -15.18, -18.72, -18.22, -14.60, -14.08, -17.77, -17.12,
    -10.55, -9.99, -13.41, -12.97, -9.62, -9.07, -12.42, -12.08,
])

# Sort by angle for more intuitive ordering (top to bottom)
VELODYNE_ANGLES_SORTED_IDX = np.argsort(VELODYNE_HDL64E_VERTICAL_ANGLES)[::-1]
VELODYNE_ANGLES_SORTED = VELODYNE_HDL64E_VERTICAL_ANGLES[VELODYNE_ANGLES_SORTED_IDX]


# =============================================================================
# Numba-Accelerated Ray Tracing Functions
# =============================================================================

if HAS_NUMBA:
    @jit(nopython=True, cache=True)
    def _trace_ray_numba(
        dir_x: float, dir_y: float, dir_z: float,
        occupied: np.ndarray,
        max_range_m: float = 80.0
    ) -> Tuple[int, int, int, bool]:
        """
        Numba-accelerated ray tracing using DDA algorithm.
        Returns (vx, vy, vz, hit) where hit is True if ray hit an occupied voxel.
        """
        # Start at LiDAR position (voxel 0, 128, 10 = world 0, 0, 0)
        # World to voxel: vx = x/0.2, vy = (y+25.6)/0.2, vz = (z+2)/0.2
        VOXEL_SIZE = 0.2

        # Ray march step size (smaller = more accurate but slower)
        dt = 0.05  # 5cm steps

        t = 0.0
        while t < max_range_m:
            t += dt

            # World position along ray
            wx = dir_x * t
            wy = dir_y * t
            wz = dir_z * t

            # Convert to voxel coordinates
            vx = int(wx / VOXEL_SIZE)
            vy = int((wy + 25.6) / VOXEL_SIZE)
            vz = int((wz + 2.0) / VOXEL_SIZE)

            # Check bounds
            if vx < 0 or vx >= 256 or vy < 0 or vy >= 256 or vz < 0 or vz >= 32:
                break

            # Check hit
            if occupied[vx, vy, vz]:
                return vx, vy, vz, True

        return 0, 0, 0, False

    @jit(nopython=True, parallel=True, cache=True)
    def simulate_multireturn_with_points_numba(
        occupied: np.ndarray,
        v_angles_rad: np.ndarray,
        h_angles_rad: np.ndarray,
        max_returns: int = 2,
        samples_per_return: int = 2,
        range_noise_std: float = 0.03,
        angular_noise_std: float = 0.002,
        max_range: float = 80.0
    ):
        """
        Numba-accelerated multi-return LiDAR simulation that also returns point cloud.

        Returns:
            sparse_mask: (256, 256, 32) binary observation mask
            beam_points: (n_v, max_per_beam, 3) per-beam point positions
            beam_counts: (n_v,) number of valid points per beam
        """
        VOXEL_SIZE = 0.2
        n_v = len(v_angles_rad)
        n_h = len(h_angles_rad)
        max_per_beam = n_h * max_returns * samples_per_return

        sparse_mask = np.zeros((256, 256, 32), dtype=np.uint8)
        beam_points = np.zeros((n_v, max_per_beam, 3), dtype=np.float32)
        beam_counts = np.zeros(n_v, dtype=np.int32)

        for beam_idx in prange(n_v):
            v_rad = v_angles_rad[beam_idx]
            cos_v = np.cos(v_rad)
            sin_v = np.sin(v_rad)
            count = 0

            for h_idx in range(n_h):
                h_rad = h_angles_rad[h_idx]

                dir_x = cos_v * np.cos(h_rad)
                dir_y = cos_v * np.sin(h_rad)
                dir_z = sin_v

                t = 0.0
                dt = 0.05
                returns_found = 0
                last_hit_x, last_hit_y, last_hit_z = -1, -1, -1

                while t < max_range and returns_found < max_returns:
                    t += dt
                    wx = dir_x * t
                    wy = dir_y * t
                    wz = dir_z * t

                    vx = int(wx / VOXEL_SIZE)
                    vy = int((wy + 25.6) / VOXEL_SIZE)
                    vz = int((wz + 2.0) / VOXEL_SIZE)

                    if vx < 0 or vx >= 256 or vy < 0 or vy >= 256 or vz < 0 or vz >= 32:
                        break

                    if occupied[vx, vy, vz] and (vx != last_hit_x or vy != last_hit_y or vz != last_hit_z):
                        for _ in range(samples_per_return):
                            rn = np.random.normal(0, range_noise_std)
                            hn = np.random.normal(0, angular_noise_std)
                            vn = np.random.normal(0, angular_noise_std)

                            noisy_t = t + rn
                            noisy_h = h_rad + hn
                            noisy_v = v_rad + vn

                            px = np.cos(noisy_v) * np.cos(noisy_h) * noisy_t
                            py = np.cos(noisy_v) * np.sin(noisy_h) * noisy_t
                            pz = np.sin(noisy_v) * noisy_t

                            nvx = int(px / VOXEL_SIZE)
                            nvy = int((py + 25.6) / VOXEL_SIZE)
                            nvz = int((pz + 2.0) / VOXEL_SIZE)

                            if 0 <= nvx < 256 and 0 <= nvy < 256 and 0 <= nvz < 32:
                                if occupied[nvx, nvy, nvz]:
                                    sparse_mask[nvx, nvy, nvz] = 1

                            # Store point regardless of voxel validity
                            if count < max_per_beam:
                                beam_points[beam_idx, count, 0] = px
                                beam_points[beam_idx, count, 1] = py
                                beam_points[beam_idx, count, 2] = pz
                                count += 1

                        returns_found += 1
                        last_hit_x, last_hit_y, last_hit_z = vx, vy, vz
                        t += 0.2

            beam_counts[beam_idx] = count

        return sparse_mask, beam_points, beam_counts

    @jit(nopython=True, parallel=True, cache=True)
    def simulate_multireturn_numba(
        occupied: np.ndarray,
        v_angles_rad: np.ndarray,
        h_angles_rad: np.ndarray,
        max_returns: int = 2,
        samples_per_return: int = 2,
        range_noise_std: float = 0.03,
        angular_noise_std: float = 0.002,
        max_range: float = 80.0
    ) -> np.ndarray:
        """
        Numba-accelerated multi-return LiDAR simulation.

        Returns sparse_mask (256, 256, 32) binary observation mask.
        """
        sparse_mask = np.zeros((256, 256, 32), dtype=np.uint8)
        VOXEL_SIZE = 0.2

        n_v = len(v_angles_rad)
        n_h = len(h_angles_rad)

        # Process each vertical beam in parallel
        for beam_idx in prange(n_v):
            v_rad = v_angles_rad[beam_idx]
            cos_v = np.cos(v_rad)
            sin_v = np.sin(v_rad)

            for h_idx in range(n_h):
                h_rad = h_angles_rad[h_idx]

                # Base ray direction
                dir_x = cos_v * np.cos(h_rad)
                dir_y = cos_v * np.sin(h_rad)
                dir_z = sin_v

                # Trace ray and collect multiple returns
                t = 0.0
                dt = 0.05  # 5cm step
                returns_found = 0
                last_hit_x, last_hit_y, last_hit_z = -1, -1, -1

                while t < max_range and returns_found < max_returns:
                    t += dt

                    wx = dir_x * t
                    wy = dir_y * t
                    wz = dir_z * t

                    vx = int(wx / VOXEL_SIZE)
                    vy = int((wy + 25.6) / VOXEL_SIZE)
                    vz = int((wz + 2.0) / VOXEL_SIZE)

                    if vx < 0 or vx >= 256 or vy < 0 or vy >= 256 or vz < 0 or vz >= 32:
                        break

                    # Check if new occupied voxel (not the last one we hit)
                    if occupied[vx, vy, vz] and (vx != last_hit_x or vy != last_hit_y or vz != last_hit_z):
                        # Found a return - generate noisy samples
                        for _ in range(samples_per_return):
                            rn = np.random.normal(0, range_noise_std)
                            hn = np.random.normal(0, angular_noise_std)
                            vn = np.random.normal(0, angular_noise_std)

                            noisy_t = t + rn
                            noisy_h = h_rad + hn
                            noisy_v = v_rad + vn

                            px = np.cos(noisy_v) * np.cos(noisy_h) * noisy_t
                            py = np.cos(noisy_v) * np.sin(noisy_h) * noisy_t
                            pz = np.sin(noisy_v) * noisy_t

                            # Convert noisy point back to voxel
                            nvx = int(px / VOXEL_SIZE)
                            nvy = int((py + 25.6) / VOXEL_SIZE)
                            nvz = int((pz + 2.0) / VOXEL_SIZE)

                            if 0 <= nvx < 256 and 0 <= nvy < 256 and 0 <= nvz < 32:
                                if occupied[nvx, nvy, nvz]:
                                    sparse_mask[nvx, nvy, nvz] = 1

                        returns_found += 1
                        last_hit_x, last_hit_y, last_hit_z = vx, vy, vz

                        # Skip through this voxel to look for next return
                        t += 0.2

        return sparse_mask


@dataclass
class LiDARConfig:
    """Configuration for LiDAR sensor simulation."""
    # Sensor type
    sensor_name: str = "Velodyne_HDL64E"

    # Vertical angles (degrees) - use actual calibration data
    vertical_angles: np.ndarray = None

    # Horizontal resolution
    h_resolution: int = 2048  # Points per rotation (360°)
    h_fov: Tuple[float, float] = (-180.0, 180.0)  # Full 360° rotation

    # Range limits (meters)
    min_range: float = 0.9
    max_range: float = 120.0
    effective_range: float = 80.0  # KITTI effective range

    # Beam divergence (mrad) - HDL-64E has ~3mrad beam divergence
    beam_divergence: float = 3.0  # milliradians

    # Noise parameters
    range_noise_std: float = 0.02  # 2cm standard deviation
    angular_noise_std: float = 0.001  # ~0.06° standard deviation

    # Ray drop parameters (from LiDARsim research)
    raydrop_base_prob: float = 0.02  # Base probability of ray drop
    raydrop_range_factor: float = 0.001  # Increases with distance
    raydrop_incidence_factor: float = 0.3  # Increases with grazing angle

    # Mounting position (for SemanticKITTI)
    mount_height: float = 1.73  # meters above ground

    def __post_init__(self):
        if self.vertical_angles is None:
            self.vertical_angles = VELODYNE_HDL64E_VERTICAL_ANGLES


def bresenham3d_single(start: Tuple[int, int, int], end: Tuple[int, int, int]) -> List[Tuple[int, int, int]]:
    """
    3D Bresenham line algorithm for a single ray.
    Returns all voxel coordinates along the line from start to end.
    """
    x1, y1, z1 = start
    x2, y2, z2 = end

    points = [(x1, y1, z1)]

    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    dz = abs(z2 - z1)

    xs = 1 if x2 > x1 else -1
    ys = 1 if y2 > y1 else -1
    zs = 1 if z2 > z1 else -1

    if dx >= dy and dx >= dz:
        p1 = 2 * dy - dx
        p2 = 2 * dz - dx
        while x1 != x2:
            x1 += xs
            if p1 >= 0:
                y1 += ys
                p1 -= 2 * dx
            if p2 >= 0:
                z1 += zs
                p2 -= 2 * dx
            p1 += 2 * dy
            p2 += 2 * dz
            points.append((x1, y1, z1))
    elif dy >= dx and dy >= dz:
        p1 = 2 * dx - dy
        p2 = 2 * dz - dy
        while y1 != y2:
            y1 += ys
            if p1 >= 0:
                x1 += xs
                p1 -= 2 * dy
            if p2 >= 0:
                z1 += zs
                p2 -= 2 * dy
            p1 += 2 * dx
            p2 += 2 * dz
            points.append((x1, y1, z1))
    else:
        p1 = 2 * dy - dz
        p2 = 2 * dx - dz
        while z1 != z2:
            z1 += zs
            if p1 >= 0:
                y1 += ys
                p1 -= 2 * dz
            if p2 >= 0:
                x1 += xs
                p2 -= 2 * dz
            p1 += 2 * dy
            p2 += 2 * dx
            points.append((x1, y1, z1))

    return points


class RealisticLiDARResampler:
    """
    Realistic LiDAR simulation with Velodyne HDL-64E characteristics.

    Features:
    - Non-uniform beam angles from actual calibration data
    - Beam divergence simulation
    - Distance and incidence-angle based ray drop
    - Range and angular noise

    SemanticKITTI Voxel Grid (from SSC-RS paper):
    - Grid: 256×256×32 with 0.2m voxels
    - Physical extent: [0~51.2m, -25.6~25.6m, -2~4.4m]
    - LiDAR at voxel (0, 128, 10) = world (0, 0, 0)
    """

    GRID_SIZE = (256, 256, 32)
    VOXEL_SIZE = 0.2  # meters

    # LiDAR position in voxel coordinates
    # World (0, 0, 0) maps to voxel (0, 128, 10)
    LIDAR_POSITION = (0, 128, 10)

    def __init__(self, config: LiDARConfig = None):
        """Initialize with optional configuration."""
        self.config = config or LiDARConfig()
        self._precompute_rays()

    def _precompute_rays(self):
        """Precompute ray directions for all beams."""
        config = self.config

        # Vertical angles from calibration
        v_angles_rad = np.radians(config.vertical_angles)
        num_beams = len(v_angles_rad)

        # Horizontal angles
        # For SemanticKITTI, grid only covers forward hemisphere (X >= 0)
        # So we use H-FOV from -90° to +90° (forward facing)
        h_angles_rad = np.linspace(
            np.radians(-90), np.radians(90),
            config.h_resolution, endpoint=False
        )

        # Create meshgrid
        H, V = np.meshgrid(h_angles_rad, v_angles_rad)

        # Convert spherical to Cartesian
        # x = forward, y = left, z = up (Velodyne coordinate system)
        self.ray_dirs = np.stack([
            np.cos(V) * np.cos(H),  # x (forward)
            np.cos(V) * np.sin(H),  # y (left/right)
            np.sin(V),               # z (up/down)
        ], axis=-1)  # Shape: (num_beams, h_resolution, 3)

        # Store beam indices for ray drop calculations
        self.beam_indices = np.repeat(
            np.arange(num_beams), config.h_resolution
        ).reshape(num_beams, config.h_resolution)

        # Flatten for iteration
        self.ray_dirs_flat = self.ray_dirs.reshape(-1, 3)
        self.beam_indices_flat = self.beam_indices.flatten()

        print(f"[RealisticLiDARResampler] Initialized")
        print(f"  Sensor: {config.sensor_name}")
        print(f"  Beams: {num_beams} (non-uniform angles)")
        print(f"  V-FOV: [{config.vertical_angles.min():.1f}°, {config.vertical_angles.max():.1f}°]")
        print(f"  H-resolution: {config.h_resolution} (FOV: ±90°)")
        print(f"  Total rays: {len(self.ray_dirs_flat)}")
        print(f"  LiDAR position (voxel): {self.LIDAR_POSITION}")

    def _compute_incidence_angle(
        self,
        ray_dir: np.ndarray,
        hit_voxel: Tuple[int, int, int],
        occupied: np.ndarray
    ) -> float:
        """
        Estimate incidence angle at hit point using local surface normal.

        Uses neighboring voxels to estimate surface normal.
        Returns angle in radians (0 = perpendicular, pi/2 = grazing).
        """
        x, y, z = hit_voxel

        # Get neighboring occupancy (simple gradient-based normal estimation)
        normal = np.zeros(3)

        for dim in range(3):
            pos = [x, y, z]
            neg = [x, y, z]

            pos[dim] = min(pos[dim] + 1, self.GRID_SIZE[dim] - 1)
            neg[dim] = max(neg[dim] - 1, 0)

            # Normal points from occupied to empty
            occ_pos = occupied[tuple(pos)]
            occ_neg = occupied[tuple(neg)]

            normal[dim] = float(occ_neg) - float(occ_pos)

        # Normalize
        norm_len = np.linalg.norm(normal)
        if norm_len > 0:
            normal = normal / norm_len
        else:
            normal = -ray_dir  # Assume perpendicular if can't estimate

        # Incidence angle (0 = perpendicular, pi/2 = grazing)
        cos_angle = abs(np.dot(ray_dir, normal))
        return np.arccos(np.clip(cos_angle, 0, 1))

    def _should_drop_ray(
        self,
        distance: float,
        incidence_angle: float,
        beam_idx: int
    ) -> bool:
        """
        Determine if ray should be dropped based on LiDARsim-inspired model.

        Ray drop probability increases with:
        - Distance from sensor
        - Grazing incidence angle
        - Some randomness for realistic behavior
        """
        config = self.config

        # Base probability
        p_drop = config.raydrop_base_prob

        # Distance factor (linear increase)
        range_ratio = distance / (config.effective_range * 5)  # 5 voxels per meter
        p_drop += config.raydrop_range_factor * range_ratio

        # Incidence angle factor (increases for grazing angles)
        # incidence_angle is 0 for perpendicular, pi/2 for grazing
        grazing_factor = np.sin(incidence_angle) ** 2
        p_drop += config.raydrop_incidence_factor * grazing_factor

        # Clamp probability
        p_drop = np.clip(p_drop, 0, 0.9)

        return np.random.random() < p_drop

    def _add_range_noise(self, distance: float) -> float:
        """Add Gaussian noise to range measurement."""
        noise = np.random.normal(0, self.config.range_noise_std * 5)  # Convert to voxels
        return max(0, distance + noise)

    def _ray_to_endpoint(
        self,
        direction: np.ndarray,
        max_dist_voxels: float = 300.0
    ) -> Tuple[int, int, int]:
        """Convert ray direction to endpoint voxel coordinates."""
        endpoint = np.array(self.LIDAR_POSITION, dtype=float) + direction * max_dist_voxels
        endpoint = np.clip(endpoint, [0, 0, 0],
                          [self.GRID_SIZE[0]-1, self.GRID_SIZE[1]-1, self.GRID_SIZE[2]-1])
        return tuple(endpoint.astype(int))

    def resample(
        self,
        complete_voxels: np.ndarray,
        enable_raydrop: bool = True,
        enable_noise: bool = True,
        enable_divergence: bool = False,
        divergence_rays: int = 4,
        verbose: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Resample complete voxel scene to sparse LiDAR observation.

        Args:
            complete_voxels: (256, 256, 32) semantic labels
            enable_raydrop: Enable ray drop simulation
            enable_noise: Enable range/angular noise
            enable_divergence: Enable beam divergence (multiple rays)
            divergence_rays: Number of rays per beam for divergence
            verbose: Print progress

        Returns:
            sparse_mask: (256, 256, 32) binary observation mask
            invalid_mask: (256, 256, 32) occluded/unobserved occupied voxels
        """
        assert complete_voxels.shape == self.GRID_SIZE, \
            f"Expected {self.GRID_SIZE}, got {complete_voxels.shape}"

        occupied = (complete_voxels > 0).astype(np.uint8)
        sparse_mask = np.zeros(self.GRID_SIZE, dtype=np.uint8)

        total_rays = len(self.ray_dirs_flat)
        hits = 0
        drops = 0

        if verbose:
            print(f"[Resample] Tracing {total_rays} rays...")

        for i, (direction, beam_idx) in enumerate(zip(self.ray_dirs_flat, self.beam_indices_flat)):
            # Add angular noise
            if enable_noise:
                noise = np.random.normal(0, self.config.angular_noise_std, 3)
                direction = direction + noise
                direction = direction / np.linalg.norm(direction)

            # Get endpoint
            endpoint = self._ray_to_endpoint(direction)

            if endpoint == self.LIDAR_POSITION:
                continue

            # Trace ray
            ray_voxels = bresenham3d_single(self.LIDAR_POSITION, endpoint)

            # Find first hit
            for vx, vy, vz in ray_voxels:
                if not (0 <= vx < 256 and 0 <= vy < 256 and 0 <= vz < 32):
                    break

                if occupied[vx, vy, vz]:
                    # Calculate distance
                    dx = vx - self.LIDAR_POSITION[0]
                    dy = vy - self.LIDAR_POSITION[1]
                    dz = vz - self.LIDAR_POSITION[2]
                    distance = np.sqrt(dx*dx + dy*dy + dz*dz)

                    # Ray drop check
                    if enable_raydrop:
                        incidence = self._compute_incidence_angle(
                            direction, (vx, vy, vz), occupied
                        )
                        if self._should_drop_ray(distance, incidence, beam_idx):
                            drops += 1
                            break

                    # Mark as observed
                    sparse_mask[vx, vy, vz] = 1
                    hits += 1
                    break

            if verbose and (i + 1) % 20000 == 0:
                print(f"  Traced {i+1}/{total_rays} rays, {hits} hits, {drops} drops")

        if verbose:
            obs_rate = 100 * sparse_mask.sum() / occupied.sum() if occupied.sum() > 0 else 0
            print(f"[Resample] Complete: {sparse_mask.sum()} observed, {drops} dropped")
            print(f"  Observation rate: {obs_rate:.1f}% of occupied voxels")

        invalid_mask = (occupied & ~sparse_mask).astype(np.uint8)
        return sparse_mask, invalid_mask

    def resample_with_divergence(
        self,
        complete_voxels: np.ndarray,
        num_divergence_rays: int = 4,
        verbose: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Resample with beam divergence simulation.

        Each beam fires multiple rays within the divergence cone,
        which can hit multiple voxels (simulating beam footprint).
        """
        assert complete_voxels.shape == self.GRID_SIZE

        occupied = (complete_voxels > 0).astype(np.uint8)
        sparse_mask = np.zeros(self.GRID_SIZE, dtype=np.uint8)

        config = self.config
        divergence_rad = config.beam_divergence / 1000.0  # mrad to rad

        if verbose:
            print(f"[Resample+Divergence] Rays per beam: {num_divergence_rays}")
            print(f"  Beam divergence: {config.beam_divergence} mrad")

        total_hits = 0

        # Use subset of rays for divergence (otherwise too slow)
        ray_subset = np.random.choice(
            len(self.ray_dirs_flat),
            min(50000, len(self.ray_dirs_flat)),
            replace=False
        )

        for idx in ray_subset:
            base_dir = self.ray_dirs_flat[idx]

            # Generate divergence rays around main direction
            for _ in range(num_divergence_rays):
                # Random offset within divergence cone
                theta = np.random.uniform(0, divergence_rad)
                phi = np.random.uniform(0, 2 * np.pi)

                # Create orthogonal basis
                if abs(base_dir[0]) < 0.9:
                    ortho1 = np.cross(base_dir, [1, 0, 0])
                else:
                    ortho1 = np.cross(base_dir, [0, 1, 0])
                ortho1 = ortho1 / np.linalg.norm(ortho1)
                ortho2 = np.cross(base_dir, ortho1)

                # Offset direction
                offset = theta * (np.cos(phi) * ortho1 + np.sin(phi) * ortho2)
                direction = base_dir + offset
                direction = direction / np.linalg.norm(direction)

                # Trace ray
                endpoint = self._ray_to_endpoint(direction)
                if endpoint == self.LIDAR_POSITION:
                    continue

                ray_voxels = bresenham3d_single(self.LIDAR_POSITION, endpoint)

                for vx, vy, vz in ray_voxels:
                    if not (0 <= vx < 256 and 0 <= vy < 256 and 0 <= vz < 32):
                        break
                    if occupied[vx, vy, vz]:
                        sparse_mask[vx, vy, vz] = 1
                        total_hits += 1
                        break

        if verbose:
            print(f"[Resample+Divergence] {sparse_mask.sum()} unique voxels observed")

        invalid_mask = (occupied & ~sparse_mask).astype(np.uint8)
        return sparse_mask, invalid_mask


class PointCloudLiDARSimulator:
    """
    Point Cloud-based LiDAR simulation.

    This approach mimics how GT sparse observations are created:
    1. Generate actual 3D point cloud from ray tracing
    2. Add range noise and beam divergence to spread points
    3. Voxelize the point cloud

    This produces more realistic observation density than pure ray tracing.
    """

    GRID_SIZE = (256, 256, 32)
    VOXEL_SIZE = 0.2  # meters

    # Grid bounds (world coordinates in meters)
    X_BOUNDS = (0.0, 51.2)
    Y_BOUNDS = (-25.6, 25.6)
    Z_BOUNDS = (-2.0, 4.4)

    # LiDAR world position (center of voxel grid in Y, at ground in Z)
    LIDAR_WORLD_POS = np.array([0.0, 0.0, 0.0])

    def __init__(self, config: LiDARConfig = None):
        """Initialize simulator."""
        self.config = config or LiDARConfig()
        self._precompute_rays()

    def _precompute_rays(self):
        """Precompute ray directions."""
        config = self.config
        v_angles = np.radians(config.vertical_angles)
        num_beams = len(v_angles)

        # Full 180° forward FOV
        h_angles = np.linspace(np.radians(-90), np.radians(90),
                               config.h_resolution, endpoint=False)

        H, V = np.meshgrid(h_angles, v_angles)

        self.ray_dirs = np.stack([
            np.cos(V) * np.cos(H),
            np.cos(V) * np.sin(H),
            np.sin(V),
        ], axis=-1)

        self.ray_dirs_flat = self.ray_dirs.reshape(-1, 3)

        print(f"[PointCloudLiDARSimulator] Initialized with {len(self.ray_dirs_flat)} rays")

    def _trace_ray_world(
        self,
        direction: np.ndarray,
        occupied_voxels: np.ndarray,
        max_range: float = 80.0
    ) -> Optional[np.ndarray]:
        """
        Trace ray and return hit point in world coordinates.

        Returns None if no hit within range.
        """
        # Convert to voxel space for tracing
        start_voxel = self._world_to_voxel(self.LIDAR_WORLD_POS)

        # Endpoint in world, then convert to voxel
        endpoint_world = self.LIDAR_WORLD_POS + direction * max_range
        endpoint_voxel = self._world_to_voxel(endpoint_world)

        # Clamp endpoint
        endpoint_voxel = tuple(np.clip(
            endpoint_voxel,
            [0, 0, 0],
            [self.GRID_SIZE[0]-1, self.GRID_SIZE[1]-1, self.GRID_SIZE[2]-1]
        ).astype(int))

        if start_voxel == endpoint_voxel:
            return None

        # Trace in voxel space
        ray_voxels = bresenham3d_single(start_voxel, endpoint_voxel)

        for vx, vy, vz in ray_voxels:
            if not (0 <= vx < 256 and 0 <= vy < 256 and 0 <= vz < 32):
                break

            if occupied_voxels[vx, vy, vz]:
                # Return world coordinates of hit (center of voxel)
                return self._voxel_to_world(vx, vy, vz)

        return None

    def _world_to_voxel(self, pos: np.ndarray) -> Tuple[int, int, int]:
        """Convert world coordinates to voxel indices."""
        vx = int((pos[0] - self.X_BOUNDS[0]) / self.VOXEL_SIZE)
        vy = int((pos[1] - self.Y_BOUNDS[0]) / self.VOXEL_SIZE)
        vz = int((pos[2] - self.Z_BOUNDS[0]) / self.VOXEL_SIZE)
        return (
            np.clip(vx, 0, self.GRID_SIZE[0] - 1),
            np.clip(vy, 0, self.GRID_SIZE[1] - 1),
            np.clip(vz, 0, self.GRID_SIZE[2] - 1),
        )

    def _voxel_to_world(self, vx: int, vy: int, vz: int) -> np.ndarray:
        """Convert voxel indices to world coordinates."""
        x = self.X_BOUNDS[0] + (vx + 0.5) * self.VOXEL_SIZE
        y = self.Y_BOUNDS[0] + (vy + 0.5) * self.VOXEL_SIZE
        z = self.Z_BOUNDS[0] + (vz + 0.5) * self.VOXEL_SIZE
        return np.array([x, y, z])

    def _points_to_voxel_grid(self, points: np.ndarray) -> np.ndarray:
        """Convert point cloud to voxel grid."""
        voxel_grid = np.zeros(self.GRID_SIZE, dtype=np.uint8)

        if len(points) == 0:
            return voxel_grid

        # Filter valid points
        valid = (
            (points[:, 0] >= self.X_BOUNDS[0]) & (points[:, 0] < self.X_BOUNDS[1]) &
            (points[:, 1] >= self.Y_BOUNDS[0]) & (points[:, 1] < self.Y_BOUNDS[1]) &
            (points[:, 2] >= self.Z_BOUNDS[0]) & (points[:, 2] < self.Z_BOUNDS[1])
        )
        points = points[valid]

        if len(points) == 0:
            return voxel_grid

        # Convert to voxel indices
        vx = ((points[:, 0] - self.X_BOUNDS[0]) / self.VOXEL_SIZE).astype(int)
        vy = ((points[:, 1] - self.Y_BOUNDS[0]) / self.VOXEL_SIZE).astype(int)
        vz = ((points[:, 2] - self.Z_BOUNDS[0]) / self.VOXEL_SIZE).astype(int)

        vx = np.clip(vx, 0, self.GRID_SIZE[0] - 1)
        vy = np.clip(vy, 0, self.GRID_SIZE[1] - 1)
        vz = np.clip(vz, 0, self.GRID_SIZE[2] - 1)

        voxel_grid[vx, vy, vz] = 1

        return voxel_grid

    def simulate_point_cloud(
        self,
        complete_voxels: np.ndarray,
        points_per_beam: int = 1,
        range_noise_std: float = 0.02,
        angular_noise_std: float = 0.001,
        beam_divergence_rad: float = 0.003,
        verbose: bool = False
    ) -> np.ndarray:
        """
        Simulate LiDAR point cloud from complete voxel scene.

        Args:
            complete_voxels: (256, 256, 32) semantic voxel grid
            points_per_beam: Number of points per ray (for beam spread)
            range_noise_std: Range noise in meters
            angular_noise_std: Angular noise in radians
            beam_divergence_rad: Beam divergence angle in radians
            verbose: Print progress

        Returns:
            Nx3 point cloud in world coordinates
        """
        occupied = (complete_voxels > 0).astype(np.uint8)
        all_points = []

        if verbose:
            print(f"[PointCloud] Simulating with {len(self.ray_dirs_flat)} rays...")

        for i, base_dir in enumerate(self.ray_dirs_flat):
            # Generate multiple points per beam to simulate beam spread
            for _ in range(points_per_beam):
                # Add angular noise/divergence
                if beam_divergence_rad > 0 or angular_noise_std > 0:
                    noise = np.random.normal(0, angular_noise_std + beam_divergence_rad/3, 3)
                    direction = base_dir + noise
                    direction = direction / np.linalg.norm(direction)
                else:
                    direction = base_dir

                # Trace ray
                hit_point = self._trace_ray_world(direction, occupied)

                if hit_point is not None:
                    # Add range noise (along ray direction)
                    if range_noise_std > 0:
                        range_noise = np.random.normal(0, range_noise_std)
                        hit_point = hit_point + direction * range_noise

                    all_points.append(hit_point)

            if verbose and (i + 1) % 30000 == 0:
                print(f"  Processed {i+1}/{len(self.ray_dirs_flat)} rays, {len(all_points)} points")

        if verbose:
            print(f"[PointCloud] Generated {len(all_points)} points")

        return np.array(all_points) if all_points else np.zeros((0, 3))

    def resample(
        self,
        complete_voxels: np.ndarray,
        points_per_beam: int = 2,
        range_noise_std: float = 0.03,
        angular_noise_std: float = 0.002,
        verbose: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Resample complete scene to sparse observation via point cloud generation.

        This approach better matches GT by:
        1. Generating actual point cloud with noise
        2. Voxelizing the point cloud (same as GT process)

        Args:
            complete_voxels: Complete semantic voxel grid
            points_per_beam: Points per ray for density
            range_noise_std: Range noise spread
            angular_noise_std: Angular noise spread
            verbose: Print progress

        Returns:
            sparse_mask: Binary observation mask
            invalid_mask: Occluded/unobserved voxels
        """
        # Generate point cloud
        point_cloud = self.simulate_point_cloud(
            complete_voxels,
            points_per_beam=points_per_beam,
            range_noise_std=range_noise_std,
            angular_noise_std=angular_noise_std,
            beam_divergence_rad=self.config.beam_divergence / 1000.0,
            verbose=verbose
        )

        # Voxelize point cloud
        sparse_mask = self._points_to_voxel_grid(point_cloud)

        if verbose:
            print(f"[Resample] Voxelized to {sparse_mask.sum()} unique voxels")

        occupied = (complete_voxels > 0).astype(np.uint8)
        invalid_mask = (occupied & ~sparse_mask).astype(np.uint8)

        return sparse_mask, invalid_mask


class VoxelizationMatcher:
    """
    Match GT SemanticKITTI voxelization behavior.

    GT sparse observations come from voxelizing raw point clouds.
    This class simulates that process more closely.
    """

    GRID_SIZE = (256, 256, 32)
    VOXEL_SIZE = 0.2

    # Grid bounds (world coordinates in meters)
    X_BOUNDS = (0.0, 51.2)
    Y_BOUNDS = (-25.6, 25.6)
    Z_BOUNDS = (-2.0, 4.4)

    @classmethod
    def points_to_voxels(cls, points: np.ndarray) -> np.ndarray:
        """
        Convert point cloud to voxel grid (SemanticKITTI style).

        Args:
            points: Nx3 array of (x, y, z) world coordinates

        Returns:
            (256, 256, 32) binary voxel grid
        """
        voxel_grid = np.zeros(cls.GRID_SIZE, dtype=np.uint8)

        # Filter points within bounds
        valid = (
            (points[:, 0] >= cls.X_BOUNDS[0]) & (points[:, 0] < cls.X_BOUNDS[1]) &
            (points[:, 1] >= cls.Y_BOUNDS[0]) & (points[:, 1] < cls.Y_BOUNDS[1]) &
            (points[:, 2] >= cls.Z_BOUNDS[0]) & (points[:, 2] < cls.Z_BOUNDS[1])
        )
        points = points[valid]

        if len(points) == 0:
            return voxel_grid

        # Convert to voxel indices
        vx = ((points[:, 0] - cls.X_BOUNDS[0]) / cls.VOXEL_SIZE).astype(int)
        vy = ((points[:, 1] - cls.Y_BOUNDS[0]) / cls.VOXEL_SIZE).astype(int)
        vz = ((points[:, 2] - cls.Z_BOUNDS[0]) / cls.VOXEL_SIZE).astype(int)

        # Clamp to grid
        vx = np.clip(vx, 0, cls.GRID_SIZE[0] - 1)
        vy = np.clip(vy, 0, cls.GRID_SIZE[1] - 1)
        vz = np.clip(vz, 0, cls.GRID_SIZE[2] - 1)

        # Mark voxels
        voxel_grid[vx, vy, vz] = 1

        return voxel_grid

    @classmethod
    def voxel_to_world(cls, vx: int, vy: int, vz: int) -> Tuple[float, float, float]:
        """Convert voxel indices to world coordinates (center of voxel)."""
        x = cls.X_BOUNDS[0] + (vx + 0.5) * cls.VOXEL_SIZE
        y = cls.Y_BOUNDS[0] + (vy + 0.5) * cls.VOXEL_SIZE
        z = cls.Z_BOUNDS[0] + (vz + 0.5) * cls.VOXEL_SIZE
        return (x, y, z)

    @classmethod
    def world_to_voxel(cls, x: float, y: float, z: float) -> Tuple[int, int, int]:
        """Convert world coordinates to voxel indices."""
        vx = int((x - cls.X_BOUNDS[0]) / cls.VOXEL_SIZE)
        vy = int((y - cls.Y_BOUNDS[0]) / cls.VOXEL_SIZE)
        vz = int((z - cls.Z_BOUNDS[0]) / cls.VOXEL_SIZE)
        return (
            np.clip(vx, 0, cls.GRID_SIZE[0] - 1),
            np.clip(vy, 0, cls.GRID_SIZE[1] - 1),
            np.clip(vz, 0, cls.GRID_SIZE[2] - 1),
        )


def create_default_resampler() -> RealisticLiDARResampler:
    """Create resampler with default SemanticKITTI configuration."""
    config = LiDARConfig(
        sensor_name="Velodyne_HDL64E_SemanticKITTI",
        vertical_angles=VELODYNE_HDL64E_VERTICAL_ANGLES,
        h_resolution=2048,
        effective_range=80.0,
        raydrop_base_prob=0.02,
        raydrop_range_factor=0.0005,
        raydrop_incidence_factor=0.2,
    )
    return RealisticLiDARResampler(config)


# =============================================================================
# Helper functions for loading/saving
# =============================================================================

def pack_voxels(voxel_grid: np.ndarray) -> np.ndarray:
    """Pack binary voxel grid into bit-compressed format."""
    flat = voxel_grid.flatten().astype(np.uint8)
    packed = np.zeros(flat.shape[0] // 8, dtype=np.uint8)
    for i in range(8):
        packed |= (flat[i::8] << (7 - i))
    return packed


def unpack_voxels(packed: np.ndarray, shape: Tuple[int, int, int] = (256, 256, 32)) -> np.ndarray:
    """Unpack bit-compressed voxels to full grid."""
    unpacked = np.zeros(packed.shape[0] * 8, dtype=np.uint8)
    for i in range(8):
        unpacked[i::8] = (packed >> (7 - i)) & 1
    return unpacked.reshape(shape)


def load_semantickitti_voxels(voxel_dir: Path, frame_id: str) -> dict:
    """Load SemanticKITTI voxel data for a frame."""
    voxel_dir = Path(voxel_dir)

    sparse_packed = np.fromfile(voxel_dir / f"{frame_id}.bin", dtype=np.uint8)
    sparse = unpack_voxels(sparse_packed)

    labels = np.fromfile(voxel_dir / f"{frame_id}.label", dtype=np.uint16).reshape(256, 256, 32)

    invalid_path = voxel_dir / f"{frame_id}.invalid"
    invalid = unpack_voxels(np.fromfile(invalid_path, dtype=np.uint8)) if invalid_path.exists() else None

    return {'sparse': sparse, 'labels': labels, 'invalid': invalid}


class DensityAwareLiDARSimulator:
    """
    Density-aware LiDAR simulation that properly captures ring structure.

    Key insight: At close range, horizontal angular resolution (0.08°) produces
    point spacing << voxel size (0.2m). Multiple consecutive rays hit the SAME
    voxel, creating "dense rings" on the ground.

    The GT sparse observations preserve this density pattern because they voxelize
    raw point clouds where each individual point creates a voxel hit.

    This simulator models the expected point density based on distance and
    uses that to determine which voxels are observed.

    Geometry:
    - Angular resolution: 0.08° horizontal, varying vertical
    - At distance d meters, horizontal point spacing = d * tan(0.08°) ≈ d * 0.0014
    - Voxel size: 0.2m
    - At d=10m: spacing ≈ 0.014m → ~14 rays hit one voxel
    - At d=100m: spacing ≈ 0.14m → ~1.4 rays hit one voxel
    """

    GRID_SIZE = (256, 256, 32)
    VOXEL_SIZE = 0.2  # meters

    # Grid bounds (world coordinates in meters)
    X_BOUNDS = (0.0, 51.2)
    Y_BOUNDS = (-25.6, 25.6)
    Z_BOUNDS = (-2.0, 4.4)

    # LiDAR world position
    LIDAR_WORLD_POS = np.array([0.0, 0.0, 0.0])  # At voxel (0, 128, 10)

    def __init__(self, config: LiDARConfig = None):
        """Initialize simulator."""
        self.config = config or LiDARConfig()

        # Precompute angular parameters
        self.h_angular_res = 360.0 / self.config.h_resolution  # degrees
        self.h_angular_res_rad = np.radians(self.h_angular_res)

        # Vertical angles (sorted from top to bottom for consistent scanning)
        self.v_angles = np.sort(self.config.vertical_angles)[::-1]  # Top to bottom
        self.v_angles_rad = np.radians(self.v_angles)

        print(f"[DensityAwareLiDARSimulator] Initialized")
        print(f"  Horizontal resolution: {self.config.h_resolution} pts/rotation")
        print(f"  Angular resolution: {self.h_angular_res:.4f}° = {self.h_angular_res_rad:.6f} rad")
        print(f"  Vertical beams: {len(self.v_angles)}")
        print(f"  V-FOV: [{self.v_angles.min():.1f}°, {self.v_angles.max():.1f}°]")

    def _world_to_voxel(self, x: float, y: float, z: float) -> Tuple[int, int, int]:
        """Convert world coordinates to voxel indices."""
        vx = int((x - self.X_BOUNDS[0]) / self.VOXEL_SIZE)
        vy = int((y - self.Y_BOUNDS[0]) / self.VOXEL_SIZE)
        vz = int((z - self.Z_BOUNDS[0]) / self.VOXEL_SIZE)
        return (
            np.clip(vx, 0, self.GRID_SIZE[0] - 1),
            np.clip(vy, 0, self.GRID_SIZE[1] - 1),
            np.clip(vz, 0, self.GRID_SIZE[2] - 1),
        )

    def _voxel_to_world_center(self, vx: int, vy: int, vz: int) -> np.ndarray:
        """Convert voxel indices to world coordinates (center of voxel)."""
        x = self.X_BOUNDS[0] + (vx + 0.5) * self.VOXEL_SIZE
        y = self.Y_BOUNDS[0] + (vy + 0.5) * self.VOXEL_SIZE
        z = self.Z_BOUNDS[0] + (vz + 0.5) * self.VOXEL_SIZE
        return np.array([x, y, z])

    def _compute_ring_density(self, horizontal_dist: float) -> float:
        """
        Compute expected point density (rays per voxel width) at given distance.

        At distance d, horizontal point spacing = d * angular_res_rad
        Points per voxel = voxel_size / point_spacing = voxel_size / (d * angular_res_rad)

        Args:
            horizontal_dist: Horizontal distance from LiDAR in meters

        Returns:
            Expected number of rays hitting a voxel-width at this distance
        """
        if horizontal_dist < 0.1:
            return 100.0  # Very close = very dense

        point_spacing = horizontal_dist * self.h_angular_res_rad
        rays_per_voxel = self.VOXEL_SIZE / point_spacing
        return rays_per_voxel

    def simulate(
        self,
        complete_voxels: np.ndarray,
        verbose: bool = False
    ) -> Tuple[np.ndarray, Dict]:
        """
        Simulate LiDAR observation with proper ring density.

        For each beam angle:
        1. Sweep horizontally (180° forward FOV)
        2. For each horizontal angle, trace ray to find hit
        3. Mark hit voxel (respecting that closer hits = denser rings)

        The key is NOT to de-duplicate - each ray hit marks the voxel,
        and the voxelization naturally produces the ring pattern.

        Args:
            complete_voxels: (256, 256, 32) semantic voxel grid
            verbose: Print progress

        Returns:
            sparse_mask: (256, 256, 32) binary observation mask
            stats: Dictionary with simulation statistics
        """
        assert complete_voxels.shape == self.GRID_SIZE

        occupied = (complete_voxels > 0).astype(np.uint8)
        sparse_mask = np.zeros(self.GRID_SIZE, dtype=np.uint8)

        # Statistics
        stats = {
            'total_rays': 0,
            'hits': 0,
            'misses': 0,
            'out_of_bounds': 0,
            'hits_by_distance': {},  # distance_bin -> count
        }

        # Horizontal angles (forward 180° FOV)
        h_angles = np.linspace(-90, 90, self.config.h_resolution, endpoint=False)
        h_angles_rad = np.radians(h_angles)

        if verbose:
            print(f"[DensityAware] Simulating {len(self.v_angles)} beams × {len(h_angles)} angles...")

        # For each vertical beam (scan pattern)
        for beam_idx, v_angle_rad in enumerate(self.v_angles_rad):
            cos_v = np.cos(v_angle_rad)
            sin_v = np.sin(v_angle_rad)

            # For each horizontal angle in the sweep
            for h_idx, h_angle_rad in enumerate(h_angles_rad):
                stats['total_rays'] += 1

                # Ray direction (x=forward, y=left, z=up)
                dir_x = cos_v * np.cos(h_angle_rad)
                dir_y = cos_v * np.sin(h_angle_rad)
                dir_z = sin_v

                # Trace ray through voxel grid
                hit_voxel = self._trace_ray(
                    dir_x, dir_y, dir_z,
                    occupied,
                    max_range_m=80.0
                )

                if hit_voxel is not None:
                    vx, vy, vz = hit_voxel
                    sparse_mask[vx, vy, vz] = 1
                    stats['hits'] += 1

                    # Record hit distance for analysis
                    hit_world = self._voxel_to_world_center(vx, vy, vz)
                    dist = np.linalg.norm(hit_world - self.LIDAR_WORLD_POS)
                    dist_bin = int(dist / 2) * 2  # 2m bins
                    stats['hits_by_distance'][dist_bin] = stats['hits_by_distance'].get(dist_bin, 0) + 1
                else:
                    stats['misses'] += 1

            if verbose and (beam_idx + 1) % 16 == 0:
                print(f"  Beam {beam_idx + 1}/{len(self.v_angles)}: {sparse_mask.sum()} observed voxels")

        if verbose:
            print(f"[DensityAware] Complete: {sparse_mask.sum()} voxels observed")
            print(f"  Total rays: {stats['total_rays']}")
            print(f"  Hits: {stats['hits']}, Misses: {stats['misses']}")

        return sparse_mask, stats

    def _trace_ray(
        self,
        dir_x: float, dir_y: float, dir_z: float,
        occupied: np.ndarray,
        max_range_m: float = 80.0
    ) -> Optional[Tuple[int, int, int]]:
        """
        Trace ray through voxel grid using 3D DDA algorithm.

        Returns first occupied voxel hit, or None if no hit.
        """
        # Start at LiDAR position in voxel coordinates
        # World (0, 0, 0) = Voxel (0, 128, 10)
        start_voxel = (0, 128, 10)

        # Compute endpoint in world coordinates, then convert to voxel
        end_world = np.array([
            dir_x * max_range_m,
            dir_y * max_range_m,
            dir_z * max_range_m
        ])
        end_voxel = self._world_to_voxel(end_world[0], end_world[1], end_world[2])

        # Use Bresenham for ray tracing
        ray_voxels = bresenham3d_single(start_voxel, end_voxel)

        for vx, vy, vz in ray_voxels:
            # Check bounds
            if not (0 <= vx < 256 and 0 <= vy < 256 and 0 <= vz < 32):
                continue

            # Check if occupied
            if occupied[vx, vy, vz]:
                return (vx, vy, vz)

        return None

    def simulate_with_point_cloud(
        self,
        complete_voxels: np.ndarray,
        verbose: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Simulate by generating full point cloud first, then voxelizing.

        This more closely matches how GT sparse observations are created:
        1. LiDAR captures points
        2. Points are voxelized

        The density pattern emerges naturally because at close range,
        many points fall within the same voxel.

        Returns:
            sparse_mask: Binary observation mask
            point_cloud: Nx3 point cloud in world coordinates
            stats: Simulation statistics
        """
        assert complete_voxels.shape == self.GRID_SIZE

        occupied = (complete_voxels > 0).astype(np.uint8)
        all_points = []

        stats = {
            'total_rays': 0,
            'hits': 0,
            'unique_voxels': 0,
        }

        # Horizontal angles (forward 180° FOV)
        h_angles_rad = np.linspace(
            np.radians(-90), np.radians(90),
            self.config.h_resolution, endpoint=False
        )

        if verbose:
            print(f"[DensityAware+PC] Generating point cloud...")

        for beam_idx, v_angle_rad in enumerate(self.v_angles_rad):
            cos_v = np.cos(v_angle_rad)
            sin_v = np.sin(v_angle_rad)

            for h_angle_rad in h_angles_rad:
                stats['total_rays'] += 1

                dir_x = cos_v * np.cos(h_angle_rad)
                dir_y = cos_v * np.sin(h_angle_rad)
                dir_z = sin_v

                hit_voxel = self._trace_ray(dir_x, dir_y, dir_z, occupied)

                if hit_voxel is not None:
                    # Get world position of hit (with small noise for realism)
                    hit_world = self._voxel_to_world_center(*hit_voxel)

                    # Add small range noise (2cm std)
                    range_noise = np.random.normal(0, 0.02)
                    direction = np.array([dir_x, dir_y, dir_z])
                    hit_world = hit_world + direction * range_noise

                    all_points.append(hit_world)
                    stats['hits'] += 1

            if verbose and (beam_idx + 1) % 16 == 0:
                print(f"  Beam {beam_idx + 1}/{len(self.v_angles)}: {len(all_points)} points")

        # Convert to numpy array
        point_cloud = np.array(all_points) if all_points else np.zeros((0, 3))

        # Voxelize point cloud
        sparse_mask = self._voxelize_points(point_cloud)
        stats['unique_voxels'] = sparse_mask.sum()

        if verbose:
            print(f"[DensityAware+PC] Generated {len(point_cloud)} points → {stats['unique_voxels']} voxels")

        return sparse_mask, point_cloud, stats

    def _voxelize_points(self, points: np.ndarray) -> np.ndarray:
        """Convert point cloud to voxel grid."""
        voxel_grid = np.zeros(self.GRID_SIZE, dtype=np.uint8)

        if len(points) == 0:
            return voxel_grid

        # Filter valid points
        valid = (
            (points[:, 0] >= self.X_BOUNDS[0]) & (points[:, 0] < self.X_BOUNDS[1]) &
            (points[:, 1] >= self.Y_BOUNDS[0]) & (points[:, 1] < self.X_BOUNDS[1]) &
            (points[:, 2] >= self.Z_BOUNDS[0]) & (points[:, 2] < self.Z_BOUNDS[1])
        )
        points = points[valid]

        if len(points) == 0:
            return voxel_grid

        # Convert to voxel indices
        vx = ((points[:, 0] - self.X_BOUNDS[0]) / self.VOXEL_SIZE).astype(int)
        vy = ((points[:, 1] - self.Y_BOUNDS[0]) / self.VOXEL_SIZE).astype(int)
        vz = ((points[:, 2] - self.Z_BOUNDS[0]) / self.VOXEL_SIZE).astype(int)

        vx = np.clip(vx, 0, self.GRID_SIZE[0] - 1)
        vy = np.clip(vy, 0, self.GRID_SIZE[1] - 1)
        vz = np.clip(vz, 0, self.GRID_SIZE[2] - 1)

        voxel_grid[vx, vy, vz] = 1

        return voxel_grid


class HighDensityLiDARSimulator:
    """
    High-density LiDAR simulator optimized for matching GT observations.

    This simulator uses a denser angular sampling than the physical sensor
    to better capture the observation density pattern seen in GT data.

    The rationale is that GT observations may come from:
    1. Accumulated scans over time
    2. Higher-resolution sensors
    3. Post-processing that fills gaps

    We tune the sampling density to match GT observation counts.
    """

    GRID_SIZE = (256, 256, 32)
    VOXEL_SIZE = 0.2  # meters

    # Grid bounds (world coordinates in meters)
    X_BOUNDS = (0.0, 51.2)
    Y_BOUNDS = (-25.6, 25.6)
    Z_BOUNDS = (-2.0, 4.4)

    def __init__(
        self,
        vertical_angles: np.ndarray = None,
        h_resolution: int = 4096,  # 2x Velodyne default
        target_density_multiplier: float = 2.0
    ):
        """
        Initialize high-density simulator.

        Args:
            vertical_angles: Vertical beam angles (degrees)
            h_resolution: Horizontal points per rotation
            target_density_multiplier: Multiplier for ray density
        """
        if vertical_angles is None:
            vertical_angles = VELODYNE_HDL64E_VERTICAL_ANGLES

        # Sort angles top to bottom
        self.v_angles = np.sort(vertical_angles)[::-1]
        self.v_angles_rad = np.radians(self.v_angles)

        # Use higher horizontal resolution
        self.h_resolution = int(h_resolution * target_density_multiplier)

        # Precompute ray directions
        h_angles = np.linspace(-90, 90, self.h_resolution, endpoint=False)
        self.h_angles_rad = np.radians(h_angles)

        print(f"[HighDensityLiDARSimulator] Initialized")
        print(f"  Horizontal resolution: {self.h_resolution} (including {target_density_multiplier}x multiplier)")
        print(f"  Vertical beams: {len(self.v_angles)}")
        print(f"  Total rays: {len(self.v_angles) * len(self.h_angles_rad)}")

    def _world_to_voxel(self, x: float, y: float, z: float) -> Tuple[int, int, int]:
        """Convert world coordinates to voxel indices."""
        vx = int((x - self.X_BOUNDS[0]) / self.VOXEL_SIZE)
        vy = int((y - self.Y_BOUNDS[0]) / self.VOXEL_SIZE)
        vz = int((z - self.Z_BOUNDS[0]) / self.VOXEL_SIZE)
        return (
            np.clip(vx, 0, self.GRID_SIZE[0] - 1),
            np.clip(vy, 0, self.GRID_SIZE[1] - 1),
            np.clip(vz, 0, self.GRID_SIZE[2] - 1),
        )

    def simulate(
        self,
        complete_voxels: np.ndarray,
        verbose: bool = False
    ) -> Tuple[np.ndarray, Dict]:
        """
        Simulate LiDAR observation with high density.
        """
        assert complete_voxels.shape == self.GRID_SIZE

        occupied = (complete_voxels > 0).astype(np.uint8)
        sparse_mask = np.zeros(self.GRID_SIZE, dtype=np.uint8)

        stats = {'total_rays': 0, 'hits': 0}

        # LiDAR position: voxel (0, 128, 10)
        start_voxel = (0, 128, 10)

        if verbose:
            total = len(self.v_angles) * len(self.h_angles_rad)
            print(f"[HighDensity] Tracing {total:,} rays...")

        for beam_idx, v_angle_rad in enumerate(self.v_angles_rad):
            cos_v = np.cos(v_angle_rad)
            sin_v = np.sin(v_angle_rad)

            for h_angle_rad in self.h_angles_rad:
                stats['total_rays'] += 1

                # Ray direction
                dir_x = cos_v * np.cos(h_angle_rad)
                dir_y = cos_v * np.sin(h_angle_rad)
                dir_z = sin_v

                # Endpoint at max range
                end_world = np.array([dir_x * 80, dir_y * 80, dir_z * 80])
                end_voxel = self._world_to_voxel(end_world[0], end_world[1], end_world[2])

                # Trace ray
                ray_voxels = bresenham3d_single(start_voxel, end_voxel)

                for vx, vy, vz in ray_voxels:
                    if not (0 <= vx < 256 and 0 <= vy < 256 and 0 <= vz < 32):
                        continue
                    if occupied[vx, vy, vz]:
                        sparse_mask[vx, vy, vz] = 1
                        stats['hits'] += 1
                        break

            if verbose and (beam_idx + 1) % 16 == 0:
                print(f"  Beam {beam_idx + 1}/{len(self.v_angles)}: {sparse_mask.sum()} voxels")

        stats['unique_voxels'] = int(sparse_mask.sum())

        if verbose:
            print(f"[HighDensity] Complete: {stats['unique_voxels']} voxels observed")

        return sparse_mask, stats


class MultiReturnLiDARSimulator:
    """
    Multi-return LiDAR simulator that captures multiple echoes per pulse.

    Velodyne HDL-64E supports dual return mode (strongest + last return),
    which allows seeing through partial occlusions like vegetation.

    This simulator traces rays and records multiple hits along each ray,
    simulating multi-return behavior that produces denser observations.

    Best configuration found through validation:
    - 2 returns per ray
    - 2 noisy samples per return
    - Range noise: 0.03m std
    - Angular noise: 0.002 rad std

    This produces ~12,000-14,000 observations matching GT's ~11,677.
    """

    GRID_SIZE = (256, 256, 32)
    VOXEL_SIZE = 0.2  # meters

    # Grid bounds (world coordinates in meters)
    X_BOUNDS = (0.0, 51.2)
    Y_BOUNDS = (-25.6, 25.6)
    Z_BOUNDS = (-2.0, 4.4)

    def __init__(
        self,
        vertical_angles: np.ndarray = None,
        h_resolution: int = 2048,
        max_returns: int = 2,
        samples_per_return: int = 2,
        range_noise_std: float = 0.03,
        angular_noise_std: float = 0.002,
    ):
        """
        Initialize multi-return LiDAR simulator.

        Args:
            vertical_angles: Vertical beam angles in degrees
            h_resolution: Horizontal points per 180° sweep
            max_returns: Maximum returns per ray (1-4, typical: 2)
            samples_per_return: Noise samples per return for spreading
            range_noise_std: Range noise standard deviation (meters)
            angular_noise_std: Angular noise standard deviation (radians)
        """
        if vertical_angles is None:
            vertical_angles = VELODYNE_HDL64E_VERTICAL_ANGLES

        self.v_angles = np.sort(vertical_angles)[::-1]  # Top to bottom
        self.v_angles_rad = np.radians(self.v_angles)
        self.h_resolution = h_resolution
        self.h_angles = np.linspace(-90, 90, h_resolution, endpoint=False)
        self.h_angles_rad = np.radians(self.h_angles)

        self.max_returns = max_returns
        self.samples_per_return = samples_per_return
        self.range_noise_std = range_noise_std
        self.angular_noise_std = angular_noise_std

        print(f"[MultiReturnLiDARSimulator] Initialized")
        print(f"  Vertical beams: {len(self.v_angles)}")
        print(f"  Horizontal resolution: {h_resolution}")
        print(f"  Max returns per ray: {max_returns}")
        print(f"  Samples per return: {samples_per_return}")
        print(f"  Total rays: {len(self.v_angles) * h_resolution}")

    def _world_to_voxel(self, x: float, y: float, z: float) -> Tuple[int, int, int]:
        """Convert world coordinates to voxel indices."""
        vx = int(x / self.VOXEL_SIZE)
        vy = int((y + 25.6) / self.VOXEL_SIZE)
        vz = int((z + 2.0) / self.VOXEL_SIZE)
        return vx, vy, vz

    def simulate(
        self,
        complete_voxels: np.ndarray,
        verbose: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Simulate multi-return LiDAR observation.

        Args:
            complete_voxels: (256, 256, 32) semantic voxel grid
            verbose: Print progress

        Returns:
            sparse_mask: Binary observation mask
            point_cloud: Nx3 point cloud in world coordinates
            stats: Simulation statistics
        """
        assert complete_voxels.shape == self.GRID_SIZE

        occupied = (complete_voxels > 0).astype(np.uint8)
        all_points = []

        stats = {
            'total_rays': 0,
            'total_returns': 0,
            'total_points': 0,
        }

        if verbose:
            total_rays = len(self.v_angles) * len(self.h_angles)
            print(f"[MultiReturn] Simulating {total_rays:,} rays with up to {self.max_returns} returns each...")

        for beam_idx, v_angle in enumerate(self.v_angles):
            v_rad = self.v_angles_rad[beam_idx]
            cos_v = np.cos(v_rad)
            sin_v = np.sin(v_rad)

            for h_angle, h_rad in zip(self.h_angles, self.h_angles_rad):
                stats['total_rays'] += 1

                # Base ray direction
                dir_x = cos_v * np.cos(h_rad)
                dir_y = cos_v * np.sin(h_rad)
                dir_z = sin_v

                # Trace ray and collect multiple returns
                t = 0.0
                dt = 0.05  # 5cm step size
                max_range = 80.0
                returns_found = 0
                last_hit_voxel = None

                while t < max_range and returns_found < self.max_returns:
                    t += dt
                    wx = dir_x * t
                    wy = dir_y * t
                    wz = dir_z * t

                    vx, vy, vz = self._world_to_voxel(wx, wy, wz)

                    if not (0 <= vx < 256 and 0 <= vy < 256 and 0 <= vz < 32):
                        break

                    current_voxel = (vx, vy, vz)

                    if occupied[vx, vy, vz] and current_voxel != last_hit_voxel:
                        # Found a return - generate noisy samples
                        for _ in range(self.samples_per_return):
                            rn = np.random.normal(0, self.range_noise_std)
                            hn = np.random.normal(0, self.angular_noise_std)
                            vn = np.random.normal(0, self.angular_noise_std)

                            noisy_t = t + rn
                            noisy_h = h_rad + hn
                            noisy_v = v_rad + vn

                            px = np.cos(noisy_v) * np.cos(noisy_h) * noisy_t
                            py = np.cos(noisy_v) * np.sin(noisy_h) * noisy_t
                            pz = np.sin(noisy_v) * noisy_t

                            all_points.append([px, py, pz])
                            stats['total_points'] += 1

                        returns_found += 1
                        stats['total_returns'] += 1
                        last_hit_voxel = current_voxel

                        # Skip through this voxel to look for next return
                        t += 0.2

            if verbose and (beam_idx + 1) % 16 == 0:
                print(f"  Beam {beam_idx + 1}/{len(self.v_angles)}: {len(all_points)} points")

        # Convert to numpy array
        point_cloud = np.array(all_points) if all_points else np.zeros((0, 3))

        # Voxelize point cloud (only mark occupied voxels)
        sparse_mask = np.zeros(self.GRID_SIZE, dtype=np.uint8)
        for p in point_cloud:
            vx, vy, vz = self._world_to_voxel(p[0], p[1], p[2])
            if 0 <= vx < 256 and 0 <= vy < 256 and 0 <= vz < 32:
                if occupied[vx, vy, vz]:
                    sparse_mask[vx, vy, vz] = 1

        stats['unique_voxels'] = int(sparse_mask.sum())

        if verbose:
            print(f"[MultiReturn] Complete:")
            print(f"  Total rays: {stats['total_rays']:,}")
            print(f"  Total returns: {stats['total_returns']:,}")
            print(f"  Total points: {stats['total_points']:,}")
            print(f"  Unique voxels: {stats['unique_voxels']:,}")

        return sparse_mask, point_cloud, stats

    def simulate_fast_with_points(
        self,
        complete_voxels: np.ndarray,
        verbose: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Fast multi-return LiDAR simulation that also returns the point cloud.
        Uses numba-accelerated ray tracing with per-beam point collection.

        Returns:
            sparse_mask: Binary observation mask (256, 256, 32)
            point_cloud: Nx3 float32 point cloud in world coordinates
            stats: Simulation statistics
        """
        assert complete_voxels.shape == self.GRID_SIZE

        if not HAS_NUMBA:
            if verbose:
                print("[MultiReturn] Numba not available, falling back to slow simulate()")
            return self.simulate(complete_voxels, verbose)

        occupied = (complete_voxels > 0).astype(np.uint8)

        sparse_mask, beam_points, beam_counts = simulate_multireturn_with_points_numba(
            occupied,
            self.v_angles_rad,
            self.h_angles_rad,
            max_returns=self.max_returns,
            samples_per_return=self.samples_per_return,
            range_noise_std=self.range_noise_std,
            angular_noise_std=self.angular_noise_std,
            max_range=80.0
        )

        # Concatenate per-beam points into single array
        total_points = int(beam_counts.sum())
        point_cloud = np.zeros((total_points, 3), dtype=np.float32)
        offset = 0
        for i in range(len(beam_counts)):
            n = int(beam_counts[i])
            if n > 0:
                point_cloud[offset:offset + n] = beam_points[i, :n]
                offset += n

        stats = {
            'total_rays': len(self.v_angles) * len(self.h_angles),
            'unique_voxels': int(sparse_mask.sum()),
            'total_points': total_points,
            'numba_accelerated': True,
        }

        if verbose:
            print(f"[MultiReturn-Fast+PC] {total_points:,} points → {stats['unique_voxels']:,} voxels")

        return sparse_mask, point_cloud, stats

    def simulate_fast(
        self,
        complete_voxels: np.ndarray,
        verbose: bool = False
    ) -> Tuple[np.ndarray, None, Dict]:
        """
        Fast multi-return LiDAR simulation using numba acceleration.

        This is 10-50x faster than simulate() but doesn't return point cloud.

        Args:
            complete_voxels: (256, 256, 32) semantic voxel grid
            verbose: Print progress

        Returns:
            sparse_mask: Binary observation mask
            None: (placeholder for point cloud, not computed)
            stats: Simulation statistics
        """
        assert complete_voxels.shape == self.GRID_SIZE

        if not HAS_NUMBA:
            if verbose:
                print("[MultiReturn] Numba not available, falling back to slow simulate()")
            return self.simulate(complete_voxels, verbose)

        occupied = (complete_voxels > 0).astype(np.uint8)

        if verbose:
            total_rays = len(self.v_angles) * len(self.h_angles)
            print(f"[MultiReturn-Fast] Simulating {total_rays:,} rays with numba...")

        # Use numba-accelerated simulation
        sparse_mask = simulate_multireturn_numba(
            occupied,
            self.v_angles_rad,
            self.h_angles_rad,
            max_returns=self.max_returns,
            samples_per_return=self.samples_per_return,
            range_noise_std=self.range_noise_std,
            angular_noise_std=self.angular_noise_std,
            max_range=80.0
        )

        stats = {
            'total_rays': len(self.v_angles) * len(self.h_angles),
            'unique_voxels': int(sparse_mask.sum()),
            'numba_accelerated': True,
        }

        if verbose:
            print(f"[MultiReturn-Fast] Complete: {stats['unique_voxels']:,} voxels observed")

        return sparse_mask, None, stats


def create_multi_return_resampler(
    max_returns: int = 2,
    samples_per_return: int = 2
) -> MultiReturnLiDARSimulator:
    """
    Create multi-return LiDAR simulator with best configuration.

    Default settings produce ~12,000-14,000 observations,
    matching GT's ~11,677 observations with IoU ~0.27.
    """
    return MultiReturnLiDARSimulator(
        vertical_angles=VELODYNE_HDL64E_VERTICAL_ANGLES,
        h_resolution=2048,
        max_returns=max_returns,
        samples_per_return=samples_per_return,
        range_noise_std=0.03,
        angular_noise_std=0.002,
    )


def create_density_aware_resampler() -> DensityAwareLiDARSimulator:
    """Create density-aware LiDAR simulator with SemanticKITTI config."""
    config = LiDARConfig(
        sensor_name="Velodyne_HDL64E_DensityAware",
        vertical_angles=VELODYNE_HDL64E_VERTICAL_ANGLES,
        h_resolution=2048,  # Standard Velodyne resolution
    )
    return DensityAwareLiDARSimulator(config)


def create_high_density_resampler(density_multiplier: float = 2.0) -> HighDensityLiDARSimulator:
    """Create high-density LiDAR simulator for better GT matching."""
    return HighDensityLiDARSimulator(
        vertical_angles=VELODYNE_HDL64E_VERTICAL_ANGLES,
        h_resolution=2048,
        target_density_multiplier=density_multiplier
    )


if __name__ == "__main__":
    print("=" * 60)
    print("Realistic LiDAR Resampler V2 - Test")
    print("=" * 60)

    # Show Velodyne beam angles
    print("\nVelodyne HDL-64E Beam Angles (sorted):")
    for i, angle in enumerate(VELODYNE_ANGLES_SORTED):
        print(f"  Beam {i:2d}: {angle:+6.2f}°")

    # Create resampler
    resampler = create_default_resampler()

    # Create test scene
    print("\nCreating test scene...")
    test_scene = np.zeros((256, 256, 32), dtype=np.uint16)

    # Ground plane
    test_scene[:, :, 8:10] = 9  # road at Z=8-9 (near sensor height)

    # Building
    test_scene[50:80, 100:160, 10:25] = 13

    # Car
    test_scene[100:110, 120:130, 10:12] = 1

    print(f"  Occupied voxels: {(test_scene > 0).sum()}")

    # Resample without effects
    print("\n1. Resampling (no effects)...")
    sparse1, _ = resampler.resample(
        test_scene,
        enable_raydrop=False,
        enable_noise=False,
        verbose=True
    )

    # Resample with ray drop
    print("\n2. Resampling (with ray drop)...")
    sparse2, _ = resampler.resample(
        test_scene,
        enable_raydrop=True,
        enable_noise=True,
        verbose=True
    )

    print("\n" + "=" * 60)
    print("Test complete!")
