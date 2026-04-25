#!/usr/bin/env python3
"""
LiDAR Ray Tracing Resampler

Converts complete 3D voxel scenes (256×256×32) into sparse LiDAR observations
using ray tracing. This simulates what a Velodyne-64 LiDAR would observe
if placed in the scene.

Based on the Bresenham3D algorithm from TALoS (NeurIPS 2024).

Usage:
    from gssc.data.lidar_simulator import LiDARResampler

    resampler = LiDARResampler()
    sparse_mask, invalid_mask = resampler.resample(complete_voxels)
"""

import numpy as np
from typing import Tuple, List, Optional
from pathlib import Path


def bresenham3d_single(start: Tuple[int, int, int], end: Tuple[int, int, int]) -> List[Tuple[int, int, int]]:
    """
    3D Bresenham line algorithm for a single ray.

    Generates all voxel coordinates along the line from start to end.
    Based on TALoS implementation.

    Args:
        start: (x, y, z) starting voxel
        end: (x, y, z) ending voxel

    Returns:
        List of (x, y, z) voxel coordinates along the ray
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

    # Driving axis is X
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

    # Driving axis is Y
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

    # Driving axis is Z
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


class LiDARResampler:
    """
    Simulates LiDAR observations from complete voxel scenes.

    Uses ray tracing to determine which voxels would be observed
    by a Velodyne-64 style LiDAR placed at the ego-vehicle position.

    SemanticKITTI Voxel Grid Coordinate System (from SSC-RS paper):
    - Voxel grid: 256×256×32 with 0.2m resolution
    - Physical extent: [0~51.2m, -25.6~25.6m, -2~4.4m] in (X, Y, Z)
    - X: Forward direction (0m to 51.2m from sensor)
    - Y: Lateral direction (-25.6m to +25.6m, centered on sensor)
    - Z: Vertical direction (-2m to +4.4m, sensor at ~0m)

    Voxel-to-World Coordinate Mapping:
    - X_world = X_voxel * 0.2m  (range: 0 to 51.2m)
    - Y_world = Y_voxel * 0.2m - 25.6m  (range: -25.6 to +25.6m)
    - Z_world = Z_voxel * 0.2m - 2.0m  (range: -2 to +4.4m)

    LiDAR Sensor Position (world coordinates): (0, 0, ~0)
    LiDAR Sensor Position (voxel coordinates): (0, 128, 10)
    """

    # LiDAR position in voxel coordinates
    # World (0, 0, 0) -> Voxel (0, 128, 10)
    # - X=0: sensor at front edge of grid (x_world=0m)
    # - Y=128: sensor centered laterally (y_world=128*0.2-25.6=0m)
    # - Z=10: sensor at ground level (z_world=10*0.2-2.0=0m)
    LIDAR_POSITION = (0, 128, 10)
    GRID_SIZE = (256, 256, 32)

    # Velodyne HDL-64E specifications
    NUM_BEAMS = 64  # Vertical channels
    H_RESOLUTION = 2048  # Horizontal resolution (360° / 0.176°)
    V_FOV_UP = 2.0  # degrees
    V_FOV_DOWN = -24.9  # degrees
    MAX_RANGE = 120.0  # meters

    # Voxel physical dimensions (approximate)
    VOXEL_SIZE = (0.2, 0.2, 0.2)  # meters per voxel

    def __init__(
        self,
        lidar_position: Tuple[int, int, int] = None,
        num_beams: int = 64,
        h_resolution: int = 2048,
        v_fov_up: float = 2.0,
        v_fov_down: float = -24.9,
    ):
        """
        Initialize the LiDAR resampler.

        Args:
            lidar_position: (x, y, z) voxel coordinates of LiDAR
            num_beams: Number of vertical laser beams
            h_resolution: Horizontal angular resolution (points per rotation)
            v_fov_up: Vertical field of view upward (degrees)
            v_fov_down: Vertical field of view downward (degrees)
        """
        self.lidar_pos = lidar_position or self.LIDAR_POSITION
        self.num_beams = num_beams
        self.h_resolution = h_resolution
        self.v_fov_up = np.radians(v_fov_up)
        self.v_fov_down = np.radians(v_fov_down)

        # Pre-compute ray directions
        self._compute_ray_directions()

    def _compute_ray_directions(self):
        """Pre-compute unit ray directions for all LiDAR beams."""
        # Vertical angles (elevation)
        v_angles = np.linspace(self.v_fov_down, self.v_fov_up, self.num_beams)

        # Horizontal angles (azimuth)
        # For SemanticKITTI grid where LiDAR is at X=0:
        # - 0° = forward (+X direction)
        # - 90° = left (+Y direction)
        # - Only use forward-facing angles since back rays exit grid immediately
        # Use ~180° FOV centered on forward direction: -90° to +90°
        h_angles = np.linspace(-np.pi/2, np.pi/2, self.h_resolution, endpoint=False)

        # Create meshgrid
        H, V = np.meshgrid(h_angles, v_angles)

        # Convert spherical to Cartesian (x=forward, y=left, z=up in vehicle frame)
        # In SemanticKITTI voxel grid: X=forward, Y=left, Z=up
        self.ray_dirs = np.stack([
            np.cos(V) * np.cos(H),  # x (forward, always positive for forward-facing)
            np.cos(V) * np.sin(H),  # y (left/right)
            np.sin(V),               # z (up/down)
        ], axis=-1)  # Shape: (num_beams, h_resolution, 3)

        # Flatten for easier iteration
        self.ray_dirs_flat = self.ray_dirs.reshape(-1, 3)

        print(f"[LiDARResampler] Initialized with {len(self.ray_dirs_flat)} rays")
        print(f"  LiDAR position: {self.lidar_pos}")
        print(f"  Beams: {self.num_beams}, H-resolution: {self.h_resolution}")
        print(f"  H-FOV: [-90°, +90°] (forward-facing)")
        print(f"  V-FOV: [{np.degrees(self.v_fov_down):.1f}°, {np.degrees(self.v_fov_up):.1f}°]")

    def _ray_to_voxel_endpoint(
        self,
        direction: np.ndarray,
        max_distance: float = 200.0
    ) -> Tuple[int, int, int]:
        """
        Convert ray direction to endpoint voxel coordinates.

        Args:
            direction: Unit direction vector (x, y, z)
            max_distance: Maximum ray distance in voxel units

        Returns:
            (x, y, z) endpoint voxel coordinates (clamped to grid)
        """
        endpoint = np.array(self.lidar_pos) + direction * max_distance

        # Clamp to grid bounds
        endpoint = np.clip(endpoint, [0, 0, 0],
                          [self.GRID_SIZE[0]-1, self.GRID_SIZE[1]-1, self.GRID_SIZE[2]-1])

        return tuple(endpoint.astype(int))

    def resample(
        self,
        complete_voxels: np.ndarray,
        return_invalid: bool = True,
        verbose: bool = False
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Convert complete voxel scene to sparse LiDAR observation.

        Args:
            complete_voxels: (256, 256, 32) array with semantic labels (0=empty, 1-19=classes)
            return_invalid: Whether to return invalid voxel mask
            verbose: Print progress information

        Returns:
            sparse_mask: (256, 256, 32) binary array (1=observed, 0=not observed)
            invalid_mask: (256, 256, 32) binary array (1=invalid/beyond LoS, 0=valid)
        """
        assert complete_voxels.shape == self.GRID_SIZE, \
            f"Expected shape {self.GRID_SIZE}, got {complete_voxels.shape}"

        # Create occupancy grid (binary: occupied or not)
        occupied = (complete_voxels > 0).astype(np.uint8)

        # Output masks
        sparse_mask = np.zeros(self.GRID_SIZE, dtype=np.uint8)
        traced_mask = np.zeros(self.GRID_SIZE, dtype=np.uint8)  # All voxels along rays

        if verbose:
            print(f"[Resample] Tracing {len(self.ray_dirs_flat)} rays...")

        # Trace each ray
        hit_count = 0
        for i, direction in enumerate(self.ray_dirs_flat):
            # Get ray endpoint (far away)
            endpoint = self._ray_to_voxel_endpoint(direction, max_distance=300.0)

            # Skip if endpoint is same as start (ray points outside grid immediately)
            if endpoint == self.lidar_pos:
                continue

            # Trace ray using Bresenham
            ray_voxels = bresenham3d_single(self.lidar_pos, endpoint)

            # Find first occupied voxel along ray
            first_hit = None
            for vx, vy, vz in ray_voxels:
                # Check bounds
                if not (0 <= vx < 256 and 0 <= vy < 256 and 0 <= vz < 32):
                    break

                # Mark as traced
                traced_mask[vx, vy, vz] = 1

                # Check for hit
                if occupied[vx, vy, vz]:
                    first_hit = (vx, vy, vz)
                    break

            # Mark first hit as observed
            if first_hit is not None:
                sparse_mask[first_hit[0], first_hit[1], first_hit[2]] = 1
                hit_count += 1

            if verbose and (i + 1) % 10000 == 0:
                print(f"  Traced {i+1}/{len(self.ray_dirs_flat)} rays, {hit_count} hits")

        if verbose:
            total_occupied = occupied.sum()
            observed = sparse_mask.sum()
            print(f"[Resample] Complete: {observed}/{total_occupied} occupied voxels observed "
                  f"({100*observed/total_occupied:.1f}%)")

        if return_invalid:
            # Invalid = occupied but not observed (occluded or beyond LoS)
            invalid_mask = (occupied & ~sparse_mask).astype(np.uint8)
            return sparse_mask, invalid_mask

        return sparse_mask, None

    def resample_fast(
        self,
        complete_voxels: np.ndarray,
        subsample_factor: int = 4,
        verbose: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fast resampling using subsampled rays.

        For quick testing - uses fewer rays for faster computation.

        Args:
            complete_voxels: (256, 256, 32) semantic voxel grid
            subsample_factor: Reduce rays by this factor (4 = 1/16 rays)
            verbose: Print progress

        Returns:
            sparse_mask, invalid_mask
        """
        assert complete_voxels.shape == self.GRID_SIZE

        occupied = (complete_voxels > 0).astype(np.uint8)
        sparse_mask = np.zeros(self.GRID_SIZE, dtype=np.uint8)

        # Subsample rays
        ray_indices = np.arange(0, len(self.ray_dirs_flat), subsample_factor)

        if verbose:
            print(f"[FastResample] Tracing {len(ray_indices)} rays (1/{subsample_factor**2} of full)...")

        hit_count = 0
        for idx in ray_indices:
            direction = self.ray_dirs_flat[idx]
            endpoint = self._ray_to_voxel_endpoint(direction, max_distance=300.0)

            if endpoint == self.lidar_pos:
                continue

            ray_voxels = bresenham3d_single(self.lidar_pos, endpoint)

            for vx, vy, vz in ray_voxels:
                if not (0 <= vx < 256 and 0 <= vy < 256 and 0 <= vz < 32):
                    break
                if occupied[vx, vy, vz]:
                    sparse_mask[vx, vy, vz] = 1
                    hit_count += 1
                    break

        if verbose:
            print(f"[FastResample] {hit_count} voxels observed")

        invalid_mask = (occupied & ~sparse_mask).astype(np.uint8)
        return sparse_mask, invalid_mask


def pack_voxels(voxel_grid: np.ndarray) -> np.ndarray:
    """
    Pack binary voxel grid into bit-compressed format (SemanticKITTI style).

    Args:
        voxel_grid: (256, 256, 32) binary array

    Returns:
        Packed array of shape (262144,) uint8
    """
    flat = voxel_grid.flatten().astype(np.uint8)

    # Pack 8 voxels into 1 byte (MSB first, matching SemanticKITTI)
    packed = np.zeros(flat.shape[0] // 8, dtype=np.uint8)
    for i in range(8):
        packed |= (flat[i::8] << (7 - i))

    return packed


def unpack_voxels(packed: np.ndarray, shape: Tuple[int, int, int] = (256, 256, 32)) -> np.ndarray:
    """
    Unpack bit-compressed voxels to full grid (SemanticKITTI style).

    Args:
        packed: Bit-packed array
        shape: Output shape

    Returns:
        Unpacked binary array
    """
    unpacked = np.zeros(packed.shape[0] * 8, dtype=np.uint8)
    for i in range(8):
        unpacked[i::8] = (packed >> (7 - i)) & 1

    return unpacked.reshape(shape)


def load_semantickitti_voxels(voxel_dir: Path, frame_id: str) -> dict:
    """
    Load SemanticKITTI voxel data for a frame.

    Args:
        voxel_dir: Path to voxels directory
        frame_id: Frame ID (e.g., "000000")

    Returns:
        Dict with 'sparse', 'labels', 'invalid' arrays
    """
    # Load sparse observation mask (.bin)
    sparse_path = voxel_dir / f"{frame_id}.bin"
    sparse_packed = np.fromfile(sparse_path, dtype=np.uint8)
    sparse = unpack_voxels(sparse_packed)

    # Load complete labels (.label)
    label_path = voxel_dir / f"{frame_id}.label"
    labels = np.fromfile(label_path, dtype=np.uint16).reshape(256, 256, 32)

    # Load invalid mask (.invalid)
    invalid_path = voxel_dir / f"{frame_id}.invalid"
    if invalid_path.exists():
        invalid_packed = np.fromfile(invalid_path, dtype=np.uint8)
        invalid = unpack_voxels(invalid_packed)
    else:
        invalid = None

    return {
        'sparse': sparse,
        'labels': labels,
        'invalid': invalid,
    }


if __name__ == "__main__":
    # Quick test
    print("Testing LiDAR Resampler...")

    # Create a simple test scene
    test_scene = np.zeros((256, 256, 32), dtype=np.uint16)

    # Add ground plane (class 9 = road)
    test_scene[:, :, 0:2] = 9

    # Add some buildings (class 13)
    test_scene[50:70, 50:70, 2:15] = 13
    test_scene[180:200, 180:200, 2:20] = 13

    # Add a car (class 1)
    test_scene[130:135, 130:133, 2:4] = 1

    print(f"Test scene: {test_scene.shape}")
    print(f"Occupied voxels: {(test_scene > 0).sum()}")

    # Create resampler with fewer rays for quick test
    resampler = LiDARResampler(num_beams=32, h_resolution=512)

    # Resample
    sparse, invalid = resampler.resample_fast(test_scene, subsample_factor=2, verbose=True)

    print(f"\nResults:")
    print(f"  Sparse observations: {sparse.sum()}")
    print(f"  Invalid (occluded): {invalid.sum()}")
    print(f"  Observation rate: {100 * sparse.sum() / (test_scene > 0).sum():.1f}%")
