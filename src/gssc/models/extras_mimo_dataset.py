"""
MIMO Dataset Wrapper for PaSCo-style Training.

Reference: PaSCo (CVPR 2024)
- Code: <paco-reference>/pasco/data/semantic_kitti/kitti_dataset.py

================================================================================
CRITICAL: PaSCo's TRUE MIMO Architecture (Jan 2026 Update)
================================================================================

After exhaustive analysis of PaSCo, the MIMO dataset must provide:

1. **Training**: N=3 DIFFERENT samples per batch item
2. **80% Random Crop**: Each sample independently cropped to 80% in X-Y plane
3. **Channel Concatenation**: Collate function merges samples along channel dim

Architecture Flow:
- Dataset.__getitem__() returns list of N samples (each with 80% crop applied)
- Collate function concatenates along channel dim: [B, N*C, H, W, D]
- Model processes merged tensor with single forward pass
- N separate decoder heads produce N outputs

Key Code References (PaSCo):
- kitti_dataset.py lines 463-490: 80% random crop
- kitti_dataset.py lines 126-140: Different samples for training
- augmenter.py merge(): Channel concatenation
================================================================================
"""

import torch
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Any, Tuple
from functools import partial


def apply_voxel_augmentation(
    lidar: torch.Tensor,
    bev: torch.Tensor,
    gt_scene: torch.Tensor,
    rotation_angle: float = 0.0,  # in degrees (0, 90, 180, 270)
    flip_x: bool = False,
    flip_y: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Apply geometric augmentation to voxel data (rotation, flip).

    Following PaSCo's inference augmentation strategy:
    - Random rotation (0°, 90°, 180°, 270°)
    - Random flip in X and/or Y axis

    Note: For voxel grids, we use discrete 90° rotations instead of
    continuous rotation to avoid interpolation artifacts.

    Args:
        lidar: [1, H, W, D] or [H, W, D] sparse LiDAR voxels
        bev: [H, W] or [1, H, W] BEV semantic labels
        gt_scene: [H, W, D] complete semantic scene
        rotation_angle: Rotation angle (0, 90, 180, 270 degrees)
        flip_x: Whether to flip along X axis
        flip_y: Whether to flip along Y axis

    Returns:
        Augmented (lidar, bev, gt_scene) tensors with consistent shapes
    """
    # Handle different LiDAR input shapes
    lidar_has_channel = lidar.dim() == 4
    if lidar_has_channel:
        lidar_3d = lidar[0]  # [H, W, D]
    else:
        lidar_3d = lidar

    # Handle different BEV input shapes - CRITICAL for TTA bug fix
    # BEV must be 2D [H, W] for rot90 with dims=(0, 1) to work correctly
    # Cases: [H, W] -> use as-is
    #        [1, H, W] -> squeeze dim 0
    #        [H, 1, W] -> squeeze dim 1 (edge case)
    #        [H, W, 1] -> squeeze dim 2 (edge case)
    bev_original_shape = bev.shape
    if bev.dim() > 2:
        # Squeeze all size-1 dimensions to get 2D
        bev = bev.squeeze()
        # If still 3D (e.g., [H, 1, W] where H=W), force to [H, W]
        if bev.dim() > 2:
            # Take the two largest dimensions as H and W
            bev = bev.reshape(bev.shape[0], -1) if bev.shape[0] >= bev.shape[1] else bev.reshape(-1, bev.shape[-1])
    bev_2d = bev  # Now guaranteed [H, W]

    # Apply rotation (0, 90, 180, 270 degrees)
    k = int(rotation_angle // 90) % 4
    if k > 0:
        # Rotate around Z axis: dims (0, 1) are X, Y
        lidar_3d = torch.rot90(lidar_3d, k, dims=(0, 1))
        bev_2d = torch.rot90(bev_2d, k, dims=(0, 1))
        gt_scene = torch.rot90(gt_scene, k, dims=(0, 1))

    # Apply flips
    if flip_x:
        lidar_3d = torch.flip(lidar_3d, dims=[0])
        bev_2d = torch.flip(bev_2d, dims=[0])
        gt_scene = torch.flip(gt_scene, dims=[0])

    if flip_y:
        lidar_3d = torch.flip(lidar_3d, dims=[1])
        bev_2d = torch.flip(bev_2d, dims=[1])
        gt_scene = torch.flip(gt_scene, dims=[1])

    # Restore channel dimension if needed
    if lidar_has_channel:
        lidar_3d = lidar_3d.unsqueeze(0)

    # BEV is always returned as 2D [H, W] for consistent stacking
    return lidar_3d, bev_2d, gt_scene


def apply_bev_feature_augmentation(
    bev_features: torch.Tensor,
    rotation_angle: float = 0.0,
    flip_x: bool = False,
    flip_y: bool = False,
) -> torch.Tensor:
    """
    Apply geometric augmentation to BEV feature maps (e.g., WaffleIron features).

    This function applies the SAME transformation as apply_voxel_augmentation
    to multi-channel BEV features, ensuring consistency during TTA.

    Args:
        bev_features: [C, H, W] BEV feature tensor (e.g., WaffleIron [64, 256, 256])
        rotation_angle: Rotation angle (0, 90, 180, 270 degrees)
        flip_x: Whether to flip along X axis
        flip_y: Whether to flip along Y axis

    Returns:
        [C, H, W] augmented BEV features
    """
    # Apply rotation (0, 90, 180, 270 degrees)
    # For [C, H, W] tensors, rotate in (H, W) plane which is dims (1, 2)
    k = int(rotation_angle // 90) % 4
    if k > 0:
        bev_features = torch.rot90(bev_features, k, dims=(1, 2))

    # Apply flips
    if flip_x:
        bev_features = torch.flip(bev_features, dims=[1])

    if flip_y:
        bev_features = torch.flip(bev_features, dims=[2])

    return bev_features


def apply_inverse_bev_feature_augmentation(
    bev_features: torch.Tensor,
    rotation_angle: float,
    flip_x: bool,
    flip_y: bool,
) -> torch.Tensor:
    """
    Apply inverse augmentation to BEV features to align with original coordinates.

    This is needed for MIMO ensemble: features from augmented inputs must be
    transformed back before averaging.

    Args:
        bev_features: [C, H, W] or [B, C, H, W] BEV features
        rotation_angle: Original rotation angle applied
        flip_x: Original flip_x applied
        flip_y: Original flip_y applied

    Returns:
        bev_features transformed back to original coordinates
    """
    # Handle batch dimension
    has_batch = bev_features.dim() == 4
    if has_batch:
        results = []
        for i in range(bev_features.shape[0]):
            result = apply_inverse_bev_feature_augmentation(
                bev_features[i], rotation_angle, flip_x, flip_y
            )
            results.append(result)
        return torch.stack(results, dim=0)

    # Inverse flip (flip is its own inverse)
    if flip_y:
        bev_features = torch.flip(bev_features, dims=[2])
    if flip_x:
        bev_features = torch.flip(bev_features, dims=[1])

    # Inverse rotation: rotate by (360 - angle)
    k = int(rotation_angle // 90) % 4
    if k > 0:
        inv_k = (4 - k) % 4
        bev_features = torch.rot90(bev_features, inv_k, dims=(1, 2))

    return bev_features


def get_random_augmentation_params(
    use_continuous: bool = True,
    max_angle: float = 30.0,
    max_translation: Tuple[float, float, float] = (0.6, 0.6, 0.4),
) -> Dict[str, Any]:
    """
    Generate random augmentation parameters for MIMO inference.

    Two modes:
    1. use_continuous=True (PaSCo default): Continuous rotation ±30°, translation ±0.6m
    2. use_continuous=False (legacy): Discrete 90° rotations only

    PaSCo's augmentation from augmenter.py:
    - max_angle: 30.0 (±30° continuous rotation around Z)
    - flip: True (50% Y-flip)
    - scale_range: 0.0 (DISABLED in PaSCo config)
    - max_translation: [0.6, 0.6, 0.4] meters

    Args:
        use_continuous: If True, use PaSCo-style continuous rotation
        max_angle: Maximum rotation angle in degrees (default ±30°)
        max_translation: Maximum translation in meters (x, y, z)

    Returns:
        Dict with transformation parameters
    """
    if use_continuous:
        # PaSCo-style: continuous rotation, translation, flip
        return {
            'rotation_angle': (np.random.rand() - 0.5) * max_angle * 2,  # Uniform in [-max, +max]
            'translation': tuple((np.random.rand(3) - 0.5) * np.array(max_translation)),
            'flip_y': np.random.rand() > 0.5,
            'flip_x': False,  # PaSCo only flips Y
            'use_continuous': True,
        }
    else:
        # Legacy: discrete 90° rotations
        return {
            'rotation_angle': np.random.choice([0, 90, 180, 270]),
            'translation': (0.0, 0.0, 0.0),
            'flip_x': np.random.rand() > 0.5,
            'flip_y': np.random.rand() > 0.5,
            'use_continuous': False,
        }


def apply_continuous_augmentation(
    lidar: torch.Tensor,
    bev: torch.Tensor,
    gt_scene: torch.Tensor,
    aug_params: Dict[str, Any],
    voxel_shape: Tuple[int, int, int] = (256, 256, 32),
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Apply PaSCo-style continuous augmentation to voxel data.

    Uses proper 3D affine transformation with:
    - Continuous rotation (±30°)
    - Translation (±0.6m)
    - Y-axis flip

    CRITICAL: All modalities are transformed with the SAME transformation matrix
    to ensure consistency!

    Args:
        lidar: [H, W, D] or [1, H, W, D] sparse LiDAR voxels
        bev: [H, W] BEV semantic labels
        gt_scene: [H, W, D] complete semantic scene
        aug_params: Dict from get_random_augmentation_params(use_continuous=True)
        voxel_shape: Shape of voxel grid (default 256x256x32)

    Returns:
        Augmented (lidar, bev, gt_scene) tensors
    """
    from .voxel_transform import VoxelAugmenter, generate_transformation_matrix

    # Extract parameters
    rotation_angle = aug_params['rotation_angle']
    translation = aug_params.get('translation', (0.0, 0.0, 0.0))
    flip_y = aug_params.get('flip_y', False)

    # Generate transformation matrix
    T = generate_transformation_matrix(
        rot=rotation_angle,
        translation=translation,
        flip_dim=1 if flip_y else None,  # Y-axis flip
        scale=1.0,  # No scale (PaSCo disables scale)
    )

    # Create augmenter
    augmenter = VoxelAugmenter(voxel_shape=voxel_shape)

    # Handle LiDAR dimensions
    lidar_has_channel = lidar.dim() == 4
    if not lidar_has_channel:
        lidar = lidar.unsqueeze(0)  # Add channel dim: [1, H, W, D]

    # Add batch dimension for VoxelAugmenter
    lidar_batch = lidar.unsqueeze(0)  # [1, 1, H, W, D]
    bev_batch = bev.unsqueeze(0)  # [1, H, W]
    gt_batch = gt_scene.unsqueeze(0)  # [1, H, W, D]

    # Apply augmentation with same T to all modalities
    result = augmenter.augment(
        lidar=lidar_batch,
        bev=bev_batch,
        scene=gt_batch,
        T=T,
    )

    # Extract results and remove batch dimension
    lidar_aug = result['lidar'].squeeze(0)  # [1, H, W, D] or [H, W, D]
    if not lidar_has_channel:
        lidar_aug = lidar_aug.squeeze(0)  # Remove channel dim

    bev_aug = result['bev'].squeeze(0)  # [H, W]
    gt_aug = result['scene'].squeeze(0)  # [H, W, D]

    return lidar_aug, bev_aug, gt_aug


def apply_inverse_augmentation(
    pred_scene: torch.Tensor,
    rotation_angle: float,
    flip_x: bool,
    flip_y: bool,
    translation: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    use_continuous: bool = False,
    voxel_shape: Tuple[int, int, int] = (256, 256, 32),
) -> torch.Tensor:
    """
    Apply inverse augmentation to prediction to align with original coordinates.

    This is needed for MIMO ensemble: predictions from augmented inputs
    must be transformed back before averaging.

    Supports two modes:
    1. use_continuous=True: Use proper inverse affine transformation
    2. use_continuous=False: Use discrete rot90 + flip (legacy)

    Args:
        pred_scene: [H, W, D] or [B, H, W, D] predicted scene
        rotation_angle: Original rotation angle applied (degrees)
        flip_x: Original flip_x applied
        flip_y: Original flip_y applied
        translation: Original translation applied (meters)
        use_continuous: Whether continuous transform was used
        voxel_shape: Shape of voxel grid

    Returns:
        pred_scene transformed back to original coordinates
    """
    # Handle batch dimension
    has_batch = pred_scene.dim() == 4
    if has_batch:
        results = []
        for i in range(pred_scene.shape[0]):
            result = apply_inverse_augmentation(
                pred_scene[i], rotation_angle, flip_x, flip_y,
                translation, use_continuous, voxel_shape
            )
            results.append(result)
        return torch.stack(results, dim=0)

    if use_continuous:
        # Use VoxelTransformer for proper inverse
        from .voxel_transform import VoxelTransformer, generate_transformation_matrix

        # Reconstruct the original transformation matrix
        T = generate_transformation_matrix(
            rot=rotation_angle,
            translation=translation,
            flip_dim=1 if flip_y else None,
            scale=1.0,
        )

        # Create transformer and apply inverse
        transformer = VoxelTransformer(voxel_shape=voxel_shape)

        # Add batch dim
        pred_batch = pred_scene.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W, D]

        # Apply inverse transform (inverse=True undoes the augmentation)
        result = transformer.transform_dense_voxels(pred_batch, T, mode='nearest', inverse=True)

        return result.squeeze(0).squeeze(0)  # [H, W, D]
    else:
        # Legacy discrete mode
        # Inverse flip (flip is its own inverse)
        if flip_y:
            pred_scene = torch.flip(pred_scene, dims=[1])
        if flip_x:
            pred_scene = torch.flip(pred_scene, dims=[0])

        # Inverse rotation: rotate by (360 - angle)
        k = int(rotation_angle // 90) % 4
        if k > 0:
            inv_k = (4 - k) % 4
            pred_scene = torch.rot90(pred_scene, inv_k, dims=(0, 1))

        return pred_scene


def apply_augmentation(
    lidar: torch.Tensor,
    bev: torch.Tensor,
    gt_scene: torch.Tensor,
    aug_params: Dict[str, Any],
    voxel_shape: Tuple[int, int, int] = (256, 256, 32),
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Unified augmentation function that dispatches to discrete or continuous mode.

    CRITICAL: All modalities (LiDAR, BEV, GT) are transformed with the SAME
    transformation parameters to ensure consistency!

    Args:
        lidar: [H, W, D] or [1, H, W, D] sparse LiDAR voxels
        bev: [H, W] BEV semantic labels
        gt_scene: [H, W, D] complete semantic scene
        aug_params: Dict from get_random_augmentation_params()
        voxel_shape: Shape of voxel grid

    Returns:
        Augmented (lidar, bev, gt_scene) tensors
    """
    use_continuous = aug_params.get('use_continuous', False)

    if use_continuous:
        # PaSCo-style continuous augmentation
        return apply_continuous_augmentation(lidar, bev, gt_scene, aug_params, voxel_shape)
    else:
        # Legacy discrete augmentation
        return apply_voxel_augmentation(
            lidar, bev, gt_scene,
            rotation_angle=aug_params['rotation_angle'],
            flip_x=aug_params['flip_x'],
            flip_y=aug_params['flip_y'],
        )


def apply_80_percent_crop(
    lidar: torch.Tensor,
    bev: torch.Tensor,
    gt_scene: torch.Tensor,
    crop_xy_only: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Apply 80% random crop following PaSCo's implementation EXACTLY.

    CRITICAL: PaSCo crops based on SEMANTIC (GT) coordinates, NOT input LiDAR!

    From PaSCo kitti_dataset.py lines 463-490:
    ```python
    min_coords = semantic_coords.min(dim=0)[0]  # SEMANTIC, not input!
    max_coords = semantic_coords.max(dim=0)[0]
    crop_size = (max_coords - min_coords) * 0.8
    new_min_coords = min_coords + (max_coords - min_coords - crop_size) * np.random.rand(3)
    new_max_coords = new_min_coords + crop_size
    # Keep voxels within [new_min_coords, new_max_coords)
    ```

    NOTE: PaSCo crops in X-Y plane only (not Z) for driving scenes.

    Args:
        lidar: [1, H, W, D] or [H, W, D] sparse LiDAR voxels
        bev: [H, W] BEV semantic labels
        gt_scene: [H, W, D] complete semantic scene
        crop_xy_only: If True, only crop X-Y (not Z) as PaSCo does

    Returns:
        Cropped tensors with values outside crop region set to 0
    """
    # Handle different input shapes
    has_channel = lidar.dim() == 4
    if has_channel:
        lidar_3d = lidar[0]  # [H, W, D]
    else:
        lidar_3d = lidar

    H, W, D = lidar_3d.shape

    # CRITICAL: Find occupied region from GT SCENE (semantic_coords), NOT LiDAR!
    # This is exactly how PaSCo does it (kitti_dataset.py line 465)
    gt_occupied = (gt_scene > 0)  # Use GT scene, not lidar!
    if not gt_occupied.any():
        # No occupied voxels in GT, return as-is
        return lidar, bev, gt_scene

    # Get bounding box of GT scene occupied voxels
    occupied_indices = torch.nonzero(gt_occupied)  # [N, 3]
    min_coords = occupied_indices.min(dim=0)[0].float()  # [3]
    max_coords = occupied_indices.max(dim=0)[0].float() + 1  # [3] (exclusive)

    # Compute 80% crop size
    scene_size = max_coords - min_coords
    crop_size = scene_size * 0.8

    # Random offset within remaining 20%
    offset_range = scene_size - crop_size
    random_offset = torch.tensor([np.random.rand() for _ in range(3)]) * offset_range

    new_min = min_coords + random_offset
    new_max = new_min + crop_size

    # Clamp to valid range
    new_min = torch.clamp(new_min, min=0)
    new_max[0] = torch.clamp(new_max[0], max=H)
    new_max[1] = torch.clamp(new_max[1], max=W)
    new_max[2] = torch.clamp(new_max[2], max=D)

    # Convert to integer indices
    x_min, y_min, z_min = new_min.int().tolist()
    x_max, y_max, z_max = new_max.int().tolist()

    # Ensure valid ranges
    x_max = max(x_min + 1, x_max)
    y_max = max(y_min + 1, y_max)
    z_max = max(z_min + 1, z_max)

    if crop_xy_only:
        # Only crop X-Y, keep full Z range (PaSCo behavior)
        z_min, z_max = 0, D

    # Create masks for cropping
    # Instead of actually cropping (changing size), we zero out regions outside crop
    # This maintains consistent tensor sizes for batching
    lidar_cropped = lidar.clone()
    bev_cropped = bev.clone()
    gt_scene_cropped = gt_scene.clone()

    # Zero out regions outside the crop window
    if has_channel:
        # X dimension
        if x_min > 0:
            lidar_cropped[:, :x_min, :, :] = 0
        if x_max < H:
            lidar_cropped[:, x_max:, :, :] = 0
        # Y dimension
        if y_min > 0:
            lidar_cropped[:, :, :y_min, :] = 0
        if y_max < W:
            lidar_cropped[:, :, :, y_max:] = 0
        # Z dimension (if not crop_xy_only)
        if not crop_xy_only:
            if z_min > 0:
                lidar_cropped[:, :, :, :z_min] = 0
            if z_max < D:
                lidar_cropped[:, :, :, z_max:] = 0
    else:
        # X dimension
        if x_min > 0:
            lidar_cropped[:x_min, :, :] = 0
        if x_max < H:
            lidar_cropped[x_max:, :, :] = 0
        # Y dimension
        if y_min > 0:
            lidar_cropped[:, :y_min, :] = 0
        if y_max < W:
            lidar_cropped[:, y_max:, :] = 0
        # Z dimension
        if not crop_xy_only:
            if z_min > 0:
                lidar_cropped[:, :, :z_min] = 0
            if z_max < D:
                lidar_cropped[:, :, z_max:] = 0

    # BEV (2D) - only X-Y crop
    if x_min > 0:
        bev_cropped[:x_min, :] = 0
    if x_max < H:
        bev_cropped[x_max:, :] = 0
    if y_min > 0:
        bev_cropped[:, :y_min] = 0
    if y_max < W:
        bev_cropped[:, y_max:] = 0

    # GT scene (3D)
    if x_min > 0:
        gt_scene_cropped[:x_min, :, :] = 0
    if x_max < H:
        gt_scene_cropped[x_max:, :, :] = 0
    if y_min > 0:
        gt_scene_cropped[:, :y_min, :] = 0
    if y_max < W:
        gt_scene_cropped[:, y_max:, :] = 0
    if not crop_xy_only:
        if z_min > 0:
            gt_scene_cropped[:, :, :z_min] = 0
        if z_max < D:
            gt_scene_cropped[:, :, z_max:] = 0

    return lidar_cropped, bev_cropped, gt_scene_cropped


class MIMODatasetWrapper(Dataset):
    """
    Wraps a base dataset to provide PaSCo-style MIMO training behavior.

    CRITICAL FEATURES (following PaSCo exactly):
    1. Training: N=3 DIFFERENT samples per __getitem__ call
    2. 80% Random Crop: Each sample cropped independently
    3. Validation: Same sample repeated N times WITH DIFFERENT AUGMENTATIONS

    Training Mode (split='train'):
        - Returns n_subnets DIFFERENT samples per __getitem__ call
        - Each sample gets 80% random crop independently
        - Quote from PaSCo: "at training {Xi} are distinct point clouds"

    Validation Mode (split='val') with val_aug=True (DEFAULT):
        - Returns the SAME sample repeated n_subnets times
        - EACH copy gets a DIFFERENT random augmentation (rotation, flip)
        - No 80% crop applied
        - Quote from PaSCo: "in inference, they are augmentations of the same point cloud"
        - The N augmented versions provide ensemble diversity

    Augmentation transforms (90° discrete rotations + flips):
        - Rotation: 0°, 90°, 180°, 270° (random choice)
        - Flip X: 50% chance
        - Flip Y: 50% chance

    Reference: <paco-reference>/pasco/data/semantic_kitti/kitti_dataset.py
               <paco-reference>/pasco/data/semantic_kitti/kitti_dm.py (val_aug=True default)
    """

    def __init__(
        self,
        base_dataset: Dataset,
        n_subnets: int = 3,  # PaSCo paper default: 3 (NOT 2!)
        split: str = 'train',
        apply_crop: bool = True,  # Apply 80% random crop (training only)
        crop_xy_only: bool = True,  # Only crop X-Y, not Z (PaSCo default)
        val_aug: bool = True,  # Apply augmentations during validation (PaSCo default: True)
        train_aug: bool = True,  # NEW: Apply augmentations during training (fixes train/inference mismatch)
        use_continuous_aug: bool = True,  # Use PaSCo-style continuous rotation (default: True)
        max_angle: float = 30.0,  # Maximum rotation angle (PaSCo: ±30°)
        max_translation: Tuple[float, float, float] = (0.6, 0.6, 0.4),  # Max translation (m)
        voxel_shape: Tuple[int, int, int] = (256, 256, 32),
    ):
        """
        Args:
            base_dataset: The underlying dataset to wrap
            n_subnets: Number of subnets (PaSCo paper default: 3)
            split: 'train' for different samples + crop, 'val' for same sample + augmentations
            apply_crop: Whether to apply 80% random crop (training only)
            crop_xy_only: Only crop in X-Y plane, not Z (PaSCo default)
            val_aug: Whether to apply random augmentations during validation (PaSCo default: True)
            train_aug: Whether to apply random augmentations during training (default: True)
                       CRITICAL: This fixes the train/inference mismatch where heads are trained
                       without augmentation but inference uses TTA augmentations
            use_continuous_aug: Use PaSCo-style continuous rotation instead of discrete 90°
            max_angle: Maximum rotation angle in degrees (default ±30°)
            max_translation: Maximum translation in meters (x, y, z)
            voxel_shape: Shape of voxel grid for transformation
        """
        super().__init__()
        self.base_dataset = base_dataset
        self.n_subnets = n_subnets
        self.split = split
        self.apply_crop = apply_crop and (split == 'train')
        self.crop_xy_only = crop_xy_only
        self.val_aug = val_aug and (split in ['val', 'test'])
        self.train_aug = train_aug and (split == 'train')  # NEW: Training augmentation
        self.use_continuous_aug = use_continuous_aug
        self.max_angle = max_angle
        self.max_translation = max_translation
        self.voxel_shape = voxel_shape

        print(f"[MIMODatasetWrapper] Configuration:")
        print(f"  - n_subnets: {n_subnets} (PaSCo paper default: 3)")
        print(f"  - split: {split}")
        print(f"  - 80% random crop: {'ENABLED' if self.apply_crop else 'DISABLED'}")
        print(f"  - crop_xy_only: {crop_xy_only}")
        print(f"  - val_aug (inference augmentation): {'ENABLED' if self.val_aug else 'DISABLED'}")
        print(f"  - train_aug (training augmentation): {'ENABLED' if self.train_aug else 'DISABLED'}")
        print(f"  - use_continuous_aug: {f'ENABLED (±{max_angle:.0f}°)' if use_continuous_aug else 'DISABLED (discrete 90°)'}")
        print(f"  - Training mode: {'Different samples + crop + aug' if split == 'train' else 'Same sample + augmentations'}")
        print(f"  - Base dataset size: {len(base_dataset)}")

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int) -> List[Dict[str, Any]]:
        """
        Return n_subnets samples following PaSCo behavior.

        From PaSCo's kitti_dataset.py (lines 126-140):
        ```python
        def __getitem__(self, idx):
            if self.split == "val":
                idx_list = [idx] * self.n_subnets  # Same sample n times
            else:
                idx_list = np.random.choice(
                    len(self.scans), self.n_subnets - 1, replace=False
                )
                idx_list = idx_list.tolist() + [idx]  # Different samples + current

            items = []
            random.shuffle(idx_list)
            for id in idx_list:
                items.append(self.get_individual(id))
        ```

        For validation with val_aug=True:
        - Same sample repeated N times
        - Each copy gets a DIFFERENT random augmentation (rotation, flip)
        - Augmentation params stored for inverse transform during ensemble

        Returns:
            List of n_subnets sample dicts
        """
        if self.split == 'val' or self.split == 'test':
            # Validation: Same sample repeated n_subnets times
            idx_list = [idx] * self.n_subnets
        else:
            # Training: Different samples for each subnet
            # n_subnets - 1 random samples + the requested sample
            if self.n_subnets > 1:
                # Choose n_subnets - 1 random indices (excluding current idx)
                available_indices = list(range(len(self.base_dataset)))
                available_indices.remove(idx)

                # Handle case where dataset is smaller than n_subnets
                n_random = min(self.n_subnets - 1, len(available_indices))
                random_indices = random.sample(available_indices, n_random)

                idx_list = random_indices + [idx]
            else:
                idx_list = [idx]

        # Shuffle the indices (as PaSCo does)
        random.shuffle(idx_list)

        # Get individual samples and apply transforms
        items = []
        for subnet_idx, sample_idx in enumerate(idx_list):
            item = self.base_dataset[sample_idx]
            item = dict(item)  # Make a copy

            # Convert to tensors if needed
            if 'lidar' in item and 'bev' in item and 'gt_scene' in item:
                lidar = item['lidar']
                bev = item['bev']
                gt_scene = item['gt_scene']

                if isinstance(lidar, np.ndarray):
                    lidar = torch.from_numpy(lidar)
                if isinstance(bev, np.ndarray):
                    bev = torch.from_numpy(bev)
                if isinstance(gt_scene, np.ndarray):
                    gt_scene = torch.from_numpy(gt_scene)

                # Apply 80% random crop (training only)
                if self.apply_crop:
                    lidar, bev, gt_scene = apply_80_percent_crop(
                        lidar, bev, gt_scene, crop_xy_only=self.crop_xy_only
                    )

                # Apply random augmentation
                # CRITICAL FIX: Apply augmentation during BOTH training and validation
                # This fixes the train/inference mismatch where heads trained without augmentation
                # but inference with TTA uses augmented inputs
                #
                # - train_aug (split='train'): Each subnet sees different scene with different augmentation
                # - val_aug (split='val'): Same scene repeated N times with different augmentations
                #
                # PaSCo uses continuous rotation (±30°), translation, and Y-flip
                should_augment = self.train_aug or self.val_aug
                if should_augment:
                    aug_params = get_random_augmentation_params(
                        use_continuous=self.use_continuous_aug,
                        max_angle=self.max_angle,
                        max_translation=self.max_translation,
                    )
                    lidar, bev, gt_scene = apply_augmentation(
                        lidar, bev, gt_scene, aug_params, self.voxel_shape
                    )
                    # Store augmentation params for inverse transform during ensemble
                    item['aug_params'] = aug_params
                else:
                    item['aug_params'] = None

                item['lidar'] = lidar
                item['bev'] = bev
                item['gt_scene'] = gt_scene

            # B6: Handle WaffleIron features if present
            # CRITICAL: Apply SAME augmentation to WaffleIron as to lidar/bev/gt_scene
            if 'waffleiron' in item and item['waffleiron'] is not None:
                waffleiron = item['waffleiron']
                if isinstance(waffleiron, np.ndarray):
                    waffleiron = torch.from_numpy(waffleiron)

                # Apply augmentation to WaffleIron features (same as other modalities)
                # WaffleIron is [C, H, W] BEV features, so use discrete rotation
                # CRITICAL FIX: Apply during BOTH training and validation
                if should_augment and item.get('aug_params') is not None:
                    aug_params = item['aug_params']
                    # For continuous mode, round rotation to nearest 90° for feature maps
                    # This avoids interpolation artifacts on feature maps
                    if aug_params.get('use_continuous', False):
                        raw_angle = aug_params['rotation_angle']
                        # Round to nearest 90° (0, 90, 180, 270)
                        discrete_angle = round(raw_angle / 90) * 90
                    else:
                        discrete_angle = aug_params['rotation_angle']

                    waffleiron = apply_bev_feature_augmentation(
                        waffleiron,
                        rotation_angle=discrete_angle,
                        flip_x=aug_params.get('flip_x', False),
                        flip_y=aug_params.get('flip_y', False),
                    )

                item['waffleiron'] = waffleiron

            items.append(item)

        return items


def mimo_collate_fn(
    batch: List[List[Dict[str, Any]]],
    n_subnets: int,
    concatenate_channels: bool = True,
) -> Dict[str, Any]:
    """
    Custom collate function for MIMO dataset.

    CRITICAL: PaSCo uses CHANNEL CONCATENATION, not separate tensors!

    From PaSCo augmenter.py merge():
    ```python
    in_feat_dense = torch.cat([in_feat_dense[t] for t in range(
        in_feat_dense.shape[0])], dim=0).unsqueeze(0)  # [1, N*C, H, W, D]
    ```

    Input batch structure:
        batch = [
            [sample_0_subnet_0, sample_0_subnet_1, sample_0_subnet_2],  # __getitem__(0)
            [sample_1_subnet_0, sample_1_subnet_1, sample_1_subnet_2],  # __getitem__(1)
            ...
        ]

    Output structure (concatenate_channels=True, PaSCo style):
        {
            'lidar': [B, N*C, H, W, D],  # Channel-concatenated
            'bev': [B, N, H, W],         # Stacked along new dim
            'gt_scene': [B, N, H, W, D], # Stacked for separate losses
            ...
        }

    Output structure (concatenate_channels=False, legacy style):
        {
            'lidar': [subnet_0_batch, subnet_1_batch, ...],  # List of [B, 1, H, W, D]
            'bev': [subnet_0_batch, subnet_1_batch, ...],
            ...
        }
    """
    batch_size = len(batch)

    if concatenate_channels:
        # PaSCo-style: Channel concatenation
        return _collate_channel_concat(batch, n_subnets, batch_size)
    else:
        # Legacy style: Separate tensors per subnet
        return _collate_separate(batch, n_subnets, batch_size)


def _collate_channel_concat(
    batch: List[List[Dict[str, Any]]],
    n_subnets: int,
    batch_size: int,
) -> Dict[str, Any]:
    """
    PaSCo-style collation with channel concatenation.

    Output format:
    - lidar: [B, N*C, H, W, D] - Channel concatenated for single forward pass
    - bev: [B, N, H, W] - Stacked (N separate BEV inputs)
    - gt_scene: [B, N, H, W, D] - Stacked (N separate GT targets)
    """
    result = {}

    if batch_size == 0 or len(batch[0]) == 0:
        return {'n_subnets': n_subnets, 'batch_size': 0}

    keys = batch[0][0].keys()

    for key in keys:
        # Gather all samples: [batch_idx][subnet_idx][key]
        all_samples = []
        for batch_idx in range(batch_size):
            subnet_samples = []
            for subnet_idx in range(n_subnets):
                val = batch[batch_idx][subnet_idx][key]
                # Convert to tensor if needed
                if isinstance(val, np.ndarray):
                    val = torch.from_numpy(val)
                subnet_samples.append(val)
            all_samples.append(subnet_samples)

        # Check first value type
        first_val = all_samples[0][0]

        if isinstance(first_val, torch.Tensor):
            if key == 'lidar':
                # Channel concatenation: [B, N*C, H, W, D]
                # Input: each is [C, H, W, D] or [1, H, W, D]
                batch_tensors = []
                for batch_idx in range(batch_size):
                    # Concatenate N samples along channel dim
                    subnet_tensors = all_samples[batch_idx]
                    # Ensure 4D: [C, H, W, D]
                    subnet_tensors = [t if t.dim() == 4 else t.unsqueeze(0) for t in subnet_tensors]
                    concatenated = torch.cat(subnet_tensors, dim=0)  # [N*C, H, W, D]
                    batch_tensors.append(concatenated)
                result[key] = torch.stack(batch_tensors, dim=0)  # [B, N*C, H, W, D]

            elif key in ['bev', 'gt_scene', 'waffleiron']:
                # Stack along new dimension: [B, N, ...]
                # For waffleiron: [B, N, 64, H, W]
                # Check if any sample has None waffleiron
                has_none = any(all_samples[b][s] is None
                              for b in range(batch_size)
                              for s in range(n_subnets))
                if has_none:
                    result[key] = None
                else:
                    batch_tensors = []
                    for batch_idx in range(batch_size):
                        subnet_tensors = all_samples[batch_idx]
                        stacked = torch.stack(subnet_tensors, dim=0)  # [N, ...]
                        batch_tensors.append(stacked)
                    result[key] = torch.stack(batch_tensors, dim=0)  # [B, N, ...]

            else:
                # Other tensors: stack per subnet then per batch
                batch_tensors = []
                for batch_idx in range(batch_size):
                    subnet_tensors = all_samples[batch_idx]
                    stacked = torch.stack(subnet_tensors, dim=0)
                    batch_tensors.append(stacked)
                result[key] = torch.stack(batch_tensors, dim=0)

        elif isinstance(first_val, (int, float)):
            # Scalars: nested list [batch][subnet]
            result[key] = [[all_samples[b][s] for s in range(n_subnets)] for b in range(batch_size)]

        elif isinstance(first_val, str):
            # Strings: nested list
            result[key] = [[all_samples[b][s] for s in range(n_subnets)] for b in range(batch_size)]

        else:
            # Other types: try to handle
            result[key] = [[all_samples[b][s] for s in range(n_subnets)] for b in range(batch_size)]

    # Add metadata
    result['n_subnets'] = n_subnets
    result['batch_size'] = batch_size
    result['channel_concatenated'] = True

    return result


def _collate_separate(
    batch: List[List[Dict[str, Any]]],
    n_subnets: int,
    batch_size: int,
) -> Dict[str, Any]:
    """
    Legacy collation with separate tensors per subnet.

    Output format:
    - lidar: [subnet_0_batch, subnet_1_batch, ...] where each is [B, C, H, W, D]
    - bev: [subnet_0_batch, subnet_1_batch, ...]
    - gt_scene: [subnet_0_batch, subnet_1_batch, ...]
    """
    # Reorganize: group by subnet
    subnet_batches = [[] for _ in range(n_subnets)]
    for batch_item in batch:
        for subnet_idx, sample in enumerate(batch_item):
            subnet_batches[subnet_idx].append(sample)

    result = {}

    if batch_size > 0 and len(batch[0]) > 0:
        keys = batch[0][0].keys()

        for key in keys:
            result[key] = []

            for subnet_idx in range(n_subnets):
                subnet_samples = subnet_batches[subnet_idx]
                first_val = subnet_samples[0][key]

                if isinstance(first_val, torch.Tensor):
                    stacked = torch.stack([s[key] for s in subnet_samples], dim=0)
                    result[key].append(stacked)
                elif isinstance(first_val, np.ndarray):
                    stacked = torch.from_numpy(np.stack([s[key] for s in subnet_samples], axis=0))
                    result[key].append(stacked)
                elif isinstance(first_val, (int, float)):
                    result[key].append([s[key] for s in subnet_samples])
                elif isinstance(first_val, str):
                    result[key].append([s[key] for s in subnet_samples])
                else:
                    result[key].append([s[key] for s in subnet_samples])

    result['n_subnets'] = n_subnets
    result['batch_size'] = batch_size
    result['channel_concatenated'] = False

    return result


def create_mimo_dataloader(
    base_dataset: Dataset,
    n_subnets: int = 3,  # PaSCo paper default: 3
    split: str = 'train',
    batch_size: int = 4,
    num_workers: int = 4,
    shuffle: bool = True,
    pin_memory: bool = True,
    drop_last: bool = True,
    apply_crop: bool = True,  # 80% random crop
    concatenate_channels: bool = True,  # PaSCo-style channel concat
) -> DataLoader:
    """
    Create a MIMO-compatible DataLoader following PaSCo exactly.

    Args:
        base_dataset: The underlying dataset
        n_subnets: Number of subnets (PaSCo paper default: 3)
        split: 'train' or 'val'
        batch_size: Batch size per GPU
        num_workers: Number of data loading workers
        shuffle: Whether to shuffle (typically True for train, False for val)
        pin_memory: Pin memory for faster GPU transfer
        drop_last: Drop incomplete batches
        apply_crop: Apply 80% random crop (training only)
        concatenate_channels: Use PaSCo-style channel concatenation

    Returns:
        DataLoader with MIMO collate function

    Output batch format (concatenate_channels=True):
        {
            'lidar': [B, N*C, H, W, D],  # Channel-concatenated for single forward
            'bev': [B, N, H, W],         # N separate BEV inputs
            'gt_scene': [B, N, H, W, D], # N separate GT targets
            'n_subnets': 3,
            'batch_size': B,
            'channel_concatenated': True,
        }
    """
    # Wrap the base dataset
    mimo_dataset = MIMODatasetWrapper(
        base_dataset=base_dataset,
        n_subnets=n_subnets,
        split=split,
        apply_crop=apply_crop,
    )

    # Create custom collate function
    collate_fn = partial(
        mimo_collate_fn,
        n_subnets=n_subnets,
        concatenate_channels=concatenate_channels,
    )

    return DataLoader(
        mimo_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        collate_fn=collate_fn,
    )


class MIMOBatchProcessor:
    """
    Helper class to process MIMO batches during training.

    Handles:
    - Channel-concatenated inputs (PaSCo-style)
    - N separate GT targets for N decoder heads
    - Loss computation averaged across heads
    """

    def __init__(self, model: torch.nn.Module, n_subnets: int = 3):
        """
        Args:
            model: The MIMO-aware scene completion model with N decoder heads
            n_subnets: Number of subnets (decoder heads)
        """
        self.model = model
        self.n_subnets = n_subnets

    def forward_mimo(
        self,
        batch: Dict[str, Any],
        device: torch.device,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Forward pass with PaSCo-style MIMO.

        SINGLE forward pass with channel-concatenated input → N outputs

        Args:
            batch: MIMO batch from create_mimo_dataloader
            device: Target device

        Returns:
            Tuple of (outputs_list, targets_list) for each decoder head
        """
        if batch.get('channel_concatenated', False):
            # PaSCo-style: Single forward pass
            lidar = batch['lidar'].to(device)  # [B, N*C, H, W, D]
            bev = batch['bev'].to(device)      # [B, N, H, W]
            gt_scene = batch['gt_scene'].to(device)  # [B, N, H, W, D]

            # Single forward pass, model returns N outputs from N heads
            outputs = self.model(bev, lidar)  # List of N outputs or dict with 'head_outputs'

            if isinstance(outputs, dict) and 'head_outputs' in outputs:
                outputs_list = outputs['head_outputs']
            elif isinstance(outputs, list):
                outputs_list = outputs
            else:
                # Single output, replicate for compatibility
                outputs_list = [outputs] * self.n_subnets

            # Separate GT targets for each head
            targets_list = [gt_scene[:, i] for i in range(self.n_subnets)]

            return outputs_list, targets_list
        else:
            # Legacy: N forward passes (less efficient)
            return self._forward_separate(batch, device)

    def _forward_separate(
        self,
        batch: Dict[str, Any],
        device: torch.device,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Legacy: N separate forward passes."""
        n_subnets = batch.get('n_subnets', self.n_subnets)
        outputs_list = []
        targets_list = []

        for subnet_idx in range(n_subnets):
            lidar = batch['lidar'][subnet_idx].to(device)
            bev = batch['bev'][subnet_idx].to(device)
            gt_scene = batch['gt_scene'][subnet_idx].to(device)

            # Forward pass
            base_model = self.model.base_model if hasattr(self.model, 'base_model') else self.model
            output = base_model(bev, lidar)

            outputs_list.append(output)
            targets_list.append(gt_scene)

        return outputs_list, targets_list

    def compute_loss(
        self,
        outputs_list: List[torch.Tensor],
        targets_list: List[torch.Tensor],
        loss_fn: torch.nn.Module,
    ) -> torch.Tensor:
        """
        Compute averaged loss across all decoder heads.

        Args:
            outputs_list: List of model outputs for each head
            targets_list: List of targets for each head
            loss_fn: Loss function

        Returns:
            Averaged loss across heads
        """
        total_loss = 0.0

        for output, target in zip(outputs_list, targets_list):
            if isinstance(output, dict):
                logits = output.get('logits', output.get('pred_scene'))
            else:
                logits = output

            loss = loss_fn(logits, target)
            total_loss += loss

        return total_loss / len(outputs_list)


# Example usage and testing
if __name__ == '__main__':
    print("Testing MIMODatasetWrapper with PaSCo-style features...")
    print("=" * 70)

    # Create a simple dummy dataset for testing
    class DummyDataset(Dataset):
        def __init__(self, size: int = 100):
            self.size = size

        def __len__(self):
            return self.size

        def __getitem__(self, idx):
            # Create data with some occupied voxels
            lidar = torch.zeros(1, 32, 32, 8)
            lidar[0, 8:24, 8:24, 2:6] = 1  # Some occupied region

            bev = torch.randint(0, 20, (32, 32))
            gt_scene = torch.randint(0, 20, (32, 32, 8))

            return {
                'lidar': lidar,
                'bev': bev,
                'gt_scene': gt_scene,
                'idx': idx,
                'seq': '00',
                'frame': f'{idx:06d}'
            }

    # Test with training split (different samples + 80% crop)
    print("\n=== Training Split (Different Samples + 80% Crop) ===")
    dummy_dataset = DummyDataset(100)
    mimo_dataset = MIMODatasetWrapper(
        dummy_dataset, n_subnets=3, split='train', apply_crop=True
    )

    items = mimo_dataset[0]
    print(f"Number of items returned: {len(items)}")
    print(f"Sample indices: {[item['idx'] for item in items]}")

    # Check crop was applied (some zeros outside crop region)
    print(f"LiDAR shape: {items[0]['lidar'].shape}")
    print(f"LiDAR occupancy (should be reduced by crop): {(items[0]['lidar'] > 0).sum().item()}")

    # Verify indices are different
    indices = [item['idx'] for item in items]
    assert len(set(indices)) == len(indices), "Training should return different samples!"
    print("✓ Training mode: Different samples per subnet")

    # Test with validation split (same sample, no crop)
    print("\n=== Validation Split (Same Sample, No Crop) ===")
    mimo_dataset_val = MIMODatasetWrapper(
        dummy_dataset, n_subnets=3, split='val', apply_crop=True
    )

    items_val = mimo_dataset_val[5]
    print(f"Number of items returned: {len(items_val)}")
    print(f"Sample indices: {[item['idx'] for item in items_val]}")

    indices_val = [item['idx'] for item in items_val]
    assert len(set(indices_val)) == 1, "Validation should return same sample!"
    print("✓ Validation mode: Same sample repeated")

    # Test DataLoader with channel concatenation
    print("\n=== Testing DataLoader (Channel Concatenation) ===")
    dataloader = create_mimo_dataloader(
        dummy_dataset,
        n_subnets=3,
        split='train',
        batch_size=4,
        num_workers=0,
        shuffle=True,
        pin_memory=False,
        concatenate_channels=True,
    )

    for batch in dataloader:
        print(f"Batch keys: {list(batch.keys())}")
        print(f"n_subnets: {batch['n_subnets']}")
        print(f"batch_size: {batch['batch_size']}")
        print(f"channel_concatenated: {batch['channel_concatenated']}")
        print(f"lidar shape: {batch['lidar'].shape} (expected [B, N*C, H, W, D])")
        print(f"bev shape: {batch['bev'].shape} (expected [B, N, H, W])")
        print(f"gt_scene shape: {batch['gt_scene'].shape} (expected [B, N, H, W, D])")

        # Verify shapes
        B = batch['batch_size']
        N = batch['n_subnets']
        assert batch['lidar'].shape == (B, N, 32, 32, 8), f"LiDAR shape mismatch"
        assert batch['bev'].shape == (B, N, 32, 32), f"BEV shape mismatch"
        assert batch['gt_scene'].shape == (B, N, 32, 32, 8), f"GT shape mismatch"
        print("✓ All shapes correct for PaSCo-style MIMO")
        break

    print("\n" + "=" * 70)
    print("MIMO Dataset tests passed!")
    print("=" * 70)
