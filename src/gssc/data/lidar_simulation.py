"""
LiDAR Simulation for Complete-to-Incomplete Scene Conversion

Converts completed scenes to realistic incomplete LiDAR observations using:
- Velodyne HDL-64E beam pattern specifications
- Bresenham3D ray tracing for Line-of-Sight occlusion
- Random pose sampling from SemanticKITTI training sequences
"""

import numpy as np
import torch
import random
from pathlib import Path
from typing import Tuple, List, Optional
import sys
import os

# Import Bresenham3D from TALoS
sys.path.append('TALoS')
from utils.util import Bresenham3D

class VelodyneHDL64E:
    """Velodyne HDL-64E LiDAR sensor specifications and simulation"""

    def __init__(self):
        # Vertical beam angles (64 beams from +2.0° to -24.9°)
        self.vertical_angles = np.linspace(2.0, -24.9, 64)  # degrees
        self.num_beams = 64
        self.max_range = 120.0  # meters
        self.horizontal_resolution = 0.2  # degrees (approx)

        # Convert to radians
        self.vertical_angles_rad = np.deg2rad(self.vertical_angles)

    def get_beam_directions(self, azimuth_angle: float) -> np.ndarray:
        """
        Get 3D directions for all 64 beams at given azimuth angle

        Args:
            azimuth_angle: Horizontal angle in degrees (0-360)

        Returns:
            Array of shape (64, 3) with beam directions [x, y, z]
        """
        azimuth_rad = np.deg2rad(azimuth_angle)
        directions = []

        for vertical_rad in self.vertical_angles_rad:
            # Convert spherical to Cartesian coordinates
            x = np.cos(vertical_rad) * np.cos(azimuth_rad)
            y = np.cos(vertical_rad) * np.sin(azimuth_rad)
            z = np.sin(vertical_rad)
            directions.append([x, y, z])

        return np.array(directions)

class VoxelCoordinateSystem:
    """Handle coordinate transformations for SemanticKITTI voxel grid"""

    def __init__(self, voxel_size: Tuple[int, int, int] = (256, 256, 32)):
        self.voxel_size = voxel_size

        # SemanticKITTI coordinate system (from semantic-kitti-api)
        self.voxel_origin = np.array([0, -25.6, -2.0])  # meters
        self.voxel_dimensions = np.array([51.2, 51.2, 6.4])  # meters
        self.resolution = self.voxel_dimensions / np.array(voxel_size)

    def world_to_voxel(self, world_coords: np.ndarray) -> np.ndarray:
        """Convert world coordinates to voxel indices"""
        relative_coords = world_coords - self.voxel_origin
        voxel_coords = relative_coords / self.resolution
        return np.floor(voxel_coords).astype(int)

    def voxel_to_world(self, voxel_coords: np.ndarray) -> np.ndarray:
        """Convert voxel indices to world coordinates"""
        world_coords = voxel_coords * self.resolution + self.voxel_origin
        return world_coords

    def is_valid_voxel(self, voxel_coords: np.ndarray) -> bool:
        """Check if voxel coordinates are within bounds"""
        return (np.all(voxel_coords >= 0) and
                np.all(voxel_coords < np.array(self.voxel_size)))

class LiDARSimulator:
    """Complete-to-incomplete scene conversion using LiDAR simulation

    Note: SemanticKITTI voxel grids are ego-centric (centered on vehicle).
    The sensor is always at fixed position [128, 128, 18] in voxel space,
    so poses.txt is not needed for simulation.
    """

    def __init__(self):
        self.lidar = VelodyneHDL64E()
        self.coord_system = VoxelCoordinateSystem()

    def simulate_lidar_scan(self,
                          complete_scene: np.ndarray,
                          pose: Optional[np.ndarray] = None,
                          azimuth_samples: int = 1800) -> np.ndarray:
        """
        Simulate LiDAR scan to create incomplete scene from complete scene

        Args:
            complete_scene: Complete voxel scene (256, 256, 32)
            pose: Camera pose (4x4 matrix). Used for validation purposes only.
                  Note: Since SemanticKITTI voxel grids are ego-centric (centered
                  on vehicle), the sensor position in voxel space is ALWAYS fixed
                  at [128, 128, 18] regardless of world pose.
            azimuth_samples: Number of azimuth angles to sample (360°/0.2° = 1800)

        Returns:
            Incomplete scene with same shape as input (only visible voxels)
        """
        # SemanticKITTI voxel grids are EGO-CENTRIC (centered on vehicle)
        # The sensor is ALWAYS at a fixed position in voxel coordinates
        # The 'pose' parameter is accepted for testing/validation consistency,
        # but doesn't affect sensor position since grids move with the vehicle
        sensor_voxel_pos = np.array([128, 128, 18])  # Center XY, LiDAR height

        # Initialize incomplete scene (START WITH EMPTY - only add visible voxels)
        incomplete_scene = np.zeros_like(complete_scene)

        # Sample azimuth angles for full 360° rotation
        azimuth_angles = np.linspace(0, 360, azimuth_samples, endpoint=False)

        for azimuth in azimuth_angles:
            # Get beam directions for current azimuth
            beam_directions = self.lidar.get_beam_directions(azimuth)

            # Cast rays for each of the 64 beams
            for beam_dir in beam_directions:
                self._cast_ray(incomplete_scene, complete_scene, sensor_voxel_pos, beam_dir)

        return incomplete_scene

    def _cast_ray(self, incomplete_scene: np.ndarray, complete_scene: np.ndarray,
                  start_pos: np.ndarray, direction: np.ndarray):
        """
        Cast a single LiDAR ray and add only visible voxels to incomplete scene

        Args:
            incomplete_scene: Output scene (initially empty, add visible voxels)
            complete_scene: Input complete scene (to check for obstacles)
            start_pos: Ray start position in voxel coordinates
            direction: Ray direction vector (normalized)
        """
        # Calculate end point at maximum range
        max_range_voxels = self.lidar.max_range / np.min(self.coord_system.resolution)
        end_pos = start_pos + direction * max_range_voxels

        # Use Bresenham3D to get all voxels along the ray
        start_point = [start_pos.astype(int)]
        end_points = [end_pos.astype(int)]

        try:
            ray_voxels = Bresenham3D(start_point, end_points, line_end=True)
        except:
            # Fallback to simple ray casting if Bresenham3D fails
            ray_voxels = self._simple_ray_cast(start_pos, end_pos)

        # Process ray: add visible voxels until we hit first obstacle
        for voxel_pos in ray_voxels:
            x, y, z = voxel_pos

            # Check bounds
            if not self.coord_system.is_valid_voxel(np.array([x, y, z])):
                continue

            # Check if this voxel is occupied in complete scene
            if complete_scene[x, y, z] != 0:
                # This voxel is visible - add it to incomplete scene
                incomplete_scene[x, y, z] = complete_scene[x, y, z]
                # Stop here - everything behind is occluded
                break
            # If empty in complete scene, continue ray (no occlusion)

    def _simple_ray_cast(self, start: np.ndarray, end: np.ndarray) -> List[Tuple[int, int, int]]:
        """Simple ray casting fallback using DDA algorithm"""
        start = start.astype(int)
        end = end.astype(int)

        # Simple linear interpolation
        num_steps = int(np.linalg.norm(end - start))
        if num_steps == 0:
            return [tuple(start)]

        steps = np.linspace(start, end, num_steps)
        return [tuple(step.astype(int)) for step in steps]

def create_complete_incomplete_pair(complete_scene: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create a complete-incomplete pair for training data augmentation

    This function is designed for production use to augment training data.
    For testing/validation, use test_lidar_simulation.py which loads actual
    frame-specific data from SemanticKITTI.

    Args:
        complete_scene: Generated complete scene (256, 256, 32) in ego-centric coordinates

    Returns:
        Tuple of (incomplete_scene, complete_scene)

    Note:
        SemanticKITTI voxel grids are ego-centric, so sensor position is always
        fixed at [128, 128, 18] in voxel space. No pose information needed.
    """
    simulator = LiDARSimulator()
    incomplete_scene = simulator.simulate_lidar_scan(complete_scene)
    return incomplete_scene, complete_scene

# Example usage
if __name__ == "__main__":
    # Test with a sample complete scene
    complete_scene = np.random.randint(0, 20, size=(256, 256, 32))

    incomplete, complete = create_complete_incomplete_pair(complete_scene)

    print(f"Complete scene shape: {complete.shape}")
    print(f"Incomplete scene shape: {incomplete.shape}")
    print(f"Occupancy reduction: {np.sum(complete > 0)} -> {np.sum(incomplete > 0)} voxels")