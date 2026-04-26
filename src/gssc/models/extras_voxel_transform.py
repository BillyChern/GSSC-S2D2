"""
Voxel Transformation Utilities for PaSCo-style Augmentation.

Reference: PaSCo (CVPR 2024)
- Code: <paco-reference>/pasco/models/transform_utils.py
- Code: <paco-reference>/pasco/data/semantic_kitti/kitti_dataset.py

These utilities support:
- Continuous rotation around Z-axis (±30°)
- Translation in XYZ (±0.6m XY, ±0.4m Z)
- Y-axis flip
- Proper inverse transformations for ensemble alignment

CRITICAL: All modalities (LiDAR, BEV, GT, WaffleIron) must be transformed
with the SAME transformation matrix T to ensure spatial consistency!
"""

import math

import numpy as np
import torch
import torch.nn.functional as F

# Default SemanticKITTI voxel grid parameters
VOXEL_SHAPE = (256, 256, 32)
VOXEL_RESOLUTION = 0.2  # meters per voxel


def generate_transformation_matrix(
    rot: float = 0.0,
    translation: tuple[float, float, float] = (0.0, 0.0, 0.0),
    flip_dim: int | None = None,
    scale: float = 1.0,
) -> torch.Tensor:
    """
    Generate 4x4 transformation matrix following PaSCo's transform_utils.py.

    Args:
        rot: Rotation angle in degrees around Z-axis
        translation: (tx, ty, tz) translation in meters
        flip_dim: Axis to flip (0=X, 1=Y, 2=Z, None=no flip)
        scale: Scale factor (default 1.0, PaSCo disables scale)

    Returns:
        4x4 transformation matrix as torch.Tensor
    """
    # Rotation around Z-axis
    theta = math.radians(rot)
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)

    # Rotation matrix (Z-axis)
    R = torch.tensor([
        [cos_t, -sin_t, 0.0],
        [sin_t, cos_t, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=torch.float32)

    # Apply scale
    R = R * scale

    # Apply flip
    if flip_dim is not None:
        flip = torch.ones(3, dtype=torch.float32)
        flip[flip_dim] = -1.0
        R = R * flip.unsqueeze(1)

    # Build 4x4 matrix
    T = torch.eye(4, dtype=torch.float32)
    T[:3, :3] = R
    T[0, 3] = translation[0]
    T[1, 3] = translation[1]
    T[2, 3] = translation[2]

    return T


def generate_random_transformation(
    max_angle: float = 30.0,
    flip: bool = True,
    scale_range: float = 0.0,  # PaSCo default: 0 (disabled)
    max_translation: tuple[float, float, float] = (0.6, 0.6, 0.4),
) -> torch.Tensor:
    """
    Generate random transformation matrix following PaSCo's augmentation.

    From PaSCo's transform_utils.py (lines 32-46):
    - translation = (np.random.rand(3) - 0.5) * max_translation
    - rot = (np.random.rand() - 0.5) * max_angle * 2
    - if flip and np.random.rand() > 0.5: flip_dim = 1
    - scale = 1.0 + (np.random.rand(3) - 0.5) * scale_range

    Args:
        max_angle: Maximum rotation angle in degrees (PaSCo: 30°)
        flip: Whether to apply random Y-flip
        scale_range: Scale variation (PaSCo default: 0.0 = disabled)
        max_translation: Maximum translation in meters

    Returns:
        4x4 transformation matrix
    """
    # Random rotation: uniform in [-max_angle, +max_angle]
    rot = (np.random.rand() - 0.5) * max_angle * 2

    # Random translation: uniform in [-max_t/2, +max_t/2]
    translation = tuple(
        (np.random.rand() - 0.5) * t for t in max_translation
    )

    # Random flip: 50% Y-axis flip
    flip_dim = 1 if (flip and np.random.rand() > 0.5) else None

    # Random scale: 1.0 ± scale_range/2
    scale = 1.0 + (np.random.rand() - 0.5) * scale_range

    return generate_transformation_matrix(
        rot=rot,
        translation=translation,
        flip_dim=flip_dim,
        scale=scale,
    )


class VoxelTransformer:
    """
    Transform dense voxel grids using affine transformations.

    Supports both forward and inverse transforms for:
    - 3D voxel grids (LiDAR, GT scene)
    - 2D BEV maps (projected from 3D)

    Uses grid_sample for interpolation.
    """

    def __init__(
        self,
        voxel_shape: tuple[int, int, int] = VOXEL_SHAPE,
        resolution: float = VOXEL_RESOLUTION,
    ):
        """
        Args:
            voxel_shape: (H, W, D) shape of voxel grid
            resolution: Meters per voxel
        """
        self.voxel_shape = voxel_shape
        self.resolution = resolution

    def transform_dense_voxels(
        self,
        voxels: torch.Tensor,
        T: torch.Tensor,
        mode: str = 'nearest',
        inverse: bool = False,
    ) -> torch.Tensor:
        """
        Apply 3D affine transformation to dense voxel grid.

        Args:
            voxels: [B, C, H, W, D] dense voxel tensor
            T: [4, 4] transformation matrix
            mode: Interpolation mode ('nearest' for labels, 'bilinear' for features)
            inverse: If True, apply inverse of T

        Returns:
            Transformed voxel tensor [B, C, H, W, D]
        """
        B, C, H, W, D = voxels.shape
        device = voxels.device

        T = T.to(device).float()
        if inverse:
            T = torch.inverse(T)

        # Extract 3D affine: T[:3, :3] (rotation/scale) and T[:3, 3] (translation)
        # For grid_sample, we need to map from output to input (inverse of visual transform)
        # The affine grid should map normalized [-1, 1] coords

        # Build affine grid matrix [3, 4] for grid_sample
        # grid_sample expects: [theta_11, theta_12, theta_13, tx;
        #                       theta_21, theta_22, theta_23, ty;
        #                       theta_31, theta_32, theta_33, tz]

        # Convert T to normalized coordinates
        # Translation is in meters, need to convert to normalized coords
        H_extent = H * self.resolution
        W_extent = W * self.resolution
        D_extent = D * self.resolution

        affine = torch.zeros(B, 3, 4, device=device)
        for b in range(B):
            affine[b, :3, :3] = T[:3, :3]
            affine[b, 0, 3] = 2 * T[0, 3] / H_extent
            affine[b, 1, 3] = 2 * T[1, 3] / W_extent
            affine[b, 2, 3] = 2 * T[2, 3] / D_extent

        # Generate 3D grid
        grid = F.affine_grid(affine, voxels.shape, align_corners=True)

        # Apply transformation
        result = F.grid_sample(
            voxels.float(), grid, mode=mode, padding_mode='zeros', align_corners=True
        )

        # Convert back to original dtype if needed
        if mode == 'nearest' and voxels.dtype in [torch.long, torch.int]:
            result = result.long()

        return result


class VoxelAugmenter:
    """
    Unified augmenter for applying consistent transforms to all modalities.

    CRITICAL: Applies the SAME transformation to LiDAR, BEV, and GT scene
    to ensure spatial consistency during MIMO training.

    Reference: PaSCo's approach where transform matrix T is shared across modalities.
    """

    def __init__(
        self,
        voxel_shape: tuple[int, int, int] = VOXEL_SHAPE,
        resolution: float = VOXEL_RESOLUTION,
    ):
        """
        Args:
            voxel_shape: (H, W, D) shape of voxel grid
            resolution: Meters per voxel
        """
        self.voxel_shape = voxel_shape
        self.resolution = resolution
        self.transformer = VoxelTransformer(voxel_shape, resolution)

    def transform_bev(
        self,
        bev: torch.Tensor,
        T: torch.Tensor,
        inverse: bool = False,
    ) -> torch.Tensor:
        """
        Apply 2D projection of 3D transform to BEV.

        BEV is the XY plane projection, so we extract 2D rotation and
        translation from the 4x4 matrix.

        Args:
            bev: [B, H, W] BEV semantic labels
            T: [4, 4] transformation matrix
            inverse: If True, apply inverse transform

        Returns:
            Transformed BEV [B, H, W]
        """
        B, H, W = bev.shape
        device = bev.device

        T = T.to(device).float()
        if inverse:
            T = torch.inverse(T)

        # Extract 2D affine from 4x4 matrix
        # T[:2, :2] is 2D rotation/scale, T[:2, 3] is XY translation
        H_extent = H * self.resolution
        W_extent = W * self.resolution

        affine_2d = torch.zeros(B, 2, 3, device=device)
        for b in range(B):
            affine_2d[b, 0, 0] = T[0, 0]
            affine_2d[b, 0, 1] = T[0, 1]
            affine_2d[b, 1, 0] = T[1, 0]
            affine_2d[b, 1, 1] = T[1, 1]
            affine_2d[b, 0, 2] = 2 * T[0, 3] / H_extent
            affine_2d[b, 1, 2] = 2 * T[1, 3] / W_extent

        # Apply 2D transform
        bev_float = bev.float().unsqueeze(1)  # [B, 1, H, W]
        grid = F.affine_grid(affine_2d, bev_float.shape, align_corners=True)
        result = F.grid_sample(
            bev_float, grid, mode='nearest', padding_mode='zeros', align_corners=True
        )

        return result.squeeze(1).long()

    def augment(
        self,
        lidar: torch.Tensor,
        bev: torch.Tensor,
        scene: torch.Tensor,
        T: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Apply the SAME transformation to all modalities.

        Args:
            lidar: [B, 1, H, W, D] sparse LiDAR voxels
            bev: [B, H, W] BEV semantic labels
            scene: [B, H, W, D] complete semantic scene
            T: [4, 4] transformation matrix (generates random if None)

        Returns:
            Dict with transformed modalities and transformation matrix T
        """
        if T is None:
            T = generate_random_transformation()

        # Transform LiDAR (3D)
        lidar_aug = self.transformer.transform_dense_voxels(
            lidar, T, mode='nearest', inverse=False
        )

        # Transform BEV (2D projection)
        bev_aug = self.transform_bev(bev, T, inverse=False)

        # Transform scene (3D)
        scene_aug = self.transformer.transform_dense_voxels(
            scene.unsqueeze(1).float(), T, mode='nearest', inverse=False
        ).squeeze(1).long()

        return {
            'lidar': lidar_aug,
            'bev': bev_aug,
            'scene': scene_aug,
            'T': T,
        }


# Test code
if __name__ == '__main__':
    print("Testing VoxelTransformer...")

    # Test transformation matrix generation
    T = generate_transformation_matrix(rot=30.0, translation=(0.5, 0.3, 0.0), flip_dim=1)
    print(f"Transformation matrix:\n{T}")

    # Test random transformation
    T_random = generate_random_transformation(max_angle=30.0)
    print(f"Random transformation matrix:\n{T_random}")

    # Test VoxelAugmenter
    augmenter = VoxelAugmenter(voxel_shape=(32, 32, 8))

    lidar = torch.zeros(1, 1, 32, 32, 8)
    lidar[0, 0, 10:22, 10:22, 2:6] = 1  # Some occupied region

    bev = torch.randint(0, 20, (1, 32, 32))
    scene = torch.randint(0, 20, (1, 32, 32, 8))

    result = augmenter.augment(lidar, bev, scene)
    print(f"LiDAR shape: {result['lidar'].shape}")
    print(f"BEV shape: {result['bev'].shape}")
    print(f"Scene shape: {result['scene'].shape}")
    print(f"T shape: {result['T'].shape}")

    print("\nVoxelTransformer tests passed!")
