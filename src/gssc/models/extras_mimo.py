"""
MIMO (Multi-Input Multi-Output) for Scene Completion

Reference: PaSCo (CVPR 2024)
- Paper: "Panoptic 3D Scene Completion"
- Code: /workspace/reference/PaSCo/pasco/models/

================================================================================
CRITICAL: PaSCo's TRUE MIMO Architecture (Jan 2026 Update)
================================================================================

After exhaustive code analysis, PaSCo uses a FUNDAMENTALLY DIFFERENT architecture
than our current implementation:

| Aspect              | PaSCo (CORRECT)                | Our Current (SIMPLIFIED)     |
|---------------------|--------------------------------|------------------------------|
| Forward passes      | 1 (single pass)                | N (multiple passes)          |
| Input format        | Channel-concatenated [1,N*C,H,W,D] | Separate tensors         |
| Encoder             | SHARED (processes merged)      | Same model N times           |
| Decoder heads       | N SEPARATE heads               | 1 head used N times          |
| 80% Random Crop     | YES (per sample)               | MISSING                      |
| n_subnets           | 3 (paper default)              | 3 (updated)                  |

PaSCo's Architecture Flow:
1. Dataset returns N=3 DIFFERENT samples (training) or N augmented views (inference)
2. Each sample gets 80% random crop independently
3. augmenter.merge() concatenates along channel dim: [1, 3*C, H, W, D]
4. SHARED encoder processes the merged tensor
5. SHARED decoder backbone upsamples
6. N SEPARATE completion heads produce N outputs (decoder_v3.py lines 130-136)
7. Ensemble via mean_probs (softmax first, then average)

Key Code References:
- augmenter.py merge(): Channel concatenation
- decoder_v3.py: self.completion_heads = nn.ModuleDict() with n_heads entries
- net_panoptic_sparse.py line 127: in_channels=f * n_infers (3*64=192)
- kitti_dataset.py lines 463-490: 80% random crop implementation

This file implements a SIMPLIFIED version using N forward passes instead of
the true channel-concat architecture. For full PaSCo compliance, implement:
1. Channel concatenation in data loading
2. N separate decoder heads in the UNet
3. 80% random crop per sample during training

================================================================================
PaSCo's MIMO Key Design (Section 3.2.1-3.2.2)
================================================================================

TRAINING PHASE (Dataset-Level MIMO):
- Each subnet receives a DIFFERENT sample from the dataset
- Each sample gets 80% RANDOM CROP (critical augmentation!)
- This is handled at the DATASET level, not model level
- Use `mimo_dataset.py:MIMODatasetWrapper` to wrap your dataset
- Quote from paper: "at training {Xi} are distinct point clouds"

INFERENCE PHASE (Model-Level MIMO):
- Each subnet receives the SAME sample with different augmentations
- Geometric augmentations: rotation ±30°, translation ±0.6m, y-flip
- Quote from paper: "in inference, they are augmentations of the same point cloud"

ENSEMBLE METHOD:
- Average PROBABILITIES (softmax outputs), NOT logits
- softmax first, then average: mean(softmax(L₀), softmax(L₁), softmax(L₂))
- Variance across subnets provides uncertainty estimate

================================================================================
Config Reference (PaSCo defaults from paper and code)
================================================================================
- n_subnets=3 (paper default - see net_panoptic_sparse.py)
- max_angle=30.0 (rotation range in degrees)
- translate_distance=0.2 (scale factor for max_translation=[0.6, 0.6, 0.4]m)
- scale_range=0.0 (DISABLED! PaSCo does not use scale augmentation)
- flip=True (50% chance of y-axis flip)
- ensemble_method='mean_probs' (average AFTER softmax, NOT logits!)
- 80% random crop per sample (training only)

CRITICAL: Both BEV and LiDAR must be transformed with the SAME matrix T.
- BEV: 2D projection of 3D transform (rotation around Z, XY translation)
- LiDAR: Full 3D voxel grid transformation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional
import numpy as np
import math

from .voxel_transform import (
    VoxelTransformer,
    generate_transformation_matrix,
    generate_random_transformation,
    VOXEL_SHAPE,
    VOXEL_RESOLUTION,
)


class BEVGeometricAugmenter(nn.Module):
    """
    Geometric augmenter for BEV inputs following PaSCo's approach.

    PaSCo applies these augmentations (Section 4, page 5):
    - Rotation: Random in [-30°, +30°] around z-axis
    - Translation: Random in [-0.6m, +0.6m] on x/y, [-0.4m, +0.4m] on z
    - Scale: Random in [0.95, 1.05]
    - Flip: 50% chance of y-axis flip

    For BEV (2D), we apply:
    - Rotation: Continuous rotation using grid_sample
    - Translation: Pixel-level shift
    - Scale: Resize and crop/pad
    - Flip: Horizontal flip (y-axis in BEV corresponds to horizontal)

    Reference: /workspace/reference/PaSCo/pasco/models/transform_utils.py
    """

    def __init__(
        self,
        max_angle: float = 30.0,  # degrees, PaSCo paper default
        max_translation: Tuple[float, float] = (0.6, 0.6),  # meters (x, y)
        scale_range: float = 0.0,  # PaSCo default: 0 (DISABLED!)
        flip_prob: float = 0.5,  # PaSCo: 50%
        voxel_size: float = 0.2,  # SemanticKITTI voxel size in meters
    ):
        """
        Args:
            max_angle: Maximum rotation angle in degrees (±30° for PaSCo)
            max_translation: Maximum translation in meters (x, y)
            scale_range: Scale variation (PaSCo default: 0.0 = disabled)
            flip_prob: Probability of horizontal flip
            voxel_size: Voxel size in meters for translation conversion
        """
        super().__init__()
        self.max_angle = max_angle
        self.max_translation = max_translation
        self.scale_range = scale_range
        self.flip_prob = flip_prob
        self.voxel_size = voxel_size

    def generate_random_transform(self) -> Dict[str, float]:
        """
        Generate random transformation parameters following PaSCo.

        From PaSCo's transform_utils.py (lines 32-46):
            translation = (np.random.rand(3) - 0.5) * max_translation
            rot = (np.random.rand() - 0.5) * max_angle * 2
            if flip and np.random.rand() > 0.5: flip_dim = 1
            scale = 1.0 + (np.random.rand(3) - 0.5) * scale_range
        """
        # Rotation: uniform in [-max_angle, +max_angle]
        angle = (np.random.rand() - 0.5) * 2 * self.max_angle

        # Translation: uniform in [-max_t/2, +max_t/2] for each axis
        # Convert from meters to voxels
        tx = (np.random.rand() - 0.5) * self.max_translation[0] / self.voxel_size
        ty = (np.random.rand() - 0.5) * self.max_translation[1] / self.voxel_size

        # Scale: uniform in [1 - scale_range/2, 1 + scale_range/2]
        scale = 1.0 + (np.random.rand() - 0.5) * self.scale_range

        # Flip: 50% probability
        flip = np.random.rand() < self.flip_prob

        return {
            'angle': angle,
            'tx': tx,
            'ty': ty,
            'scale': scale,
            'flip': flip,
        }

    def apply_transform(
        self,
        bev: torch.Tensor,
        transform: Dict[str, float],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Apply geometric transformation to BEV.

        Args:
            bev: [B, H, W] BEV semantic labels (long tensor)
            transform: Dict with angle, tx, ty, scale, flip

        Returns:
            transformed_bev: [B, H, W] transformed BEV
            inverse_transform: Dict for inverse transformation
        """
        B, H, W = bev.shape
        device = bev.device

        # Convert to float for grid_sample
        bev_float = bev.float().unsqueeze(1)  # [B, 1, H, W]

        angle = transform['angle']
        tx = transform['tx']
        ty = transform['ty']
        scale = transform['scale']
        flip = transform['flip']

        # Build affine transformation matrix
        # Note: grid_sample uses normalized coordinates [-1, 1]
        theta = math.radians(angle)
        cos_t, sin_t = math.cos(theta), math.sin(theta)

        # Combine rotation, scale, translation
        # For 2D affine: [s*cos, -s*sin, tx; s*sin, s*cos, ty]
        affine = torch.tensor([
            [scale * cos_t, -scale * sin_t, 2 * tx / W],
            [scale * sin_t, scale * cos_t, 2 * ty / H],
        ], dtype=torch.float32, device=device)

        if flip:
            # Flip along x-axis (horizontal)
            affine[0, 0] *= -1
            affine[0, 1] *= -1

        # Expand for batch
        affine = affine.unsqueeze(0).expand(B, -1, -1)

        # Generate grid
        grid = F.affine_grid(affine, bev_float.shape, align_corners=False)

        # Apply transformation (nearest neighbor for labels)
        transformed = F.grid_sample(
            bev_float, grid, mode='nearest', padding_mode='zeros', align_corners=False
        )

        # Convert back to long
        transformed_bev = transformed.squeeze(1).long()

        # Compute inverse transform for output
        inverse_transform = {
            'angle': -angle,
            'tx': -tx * cos_t - ty * sin_t,  # Inverse rotation of translation
            'ty': tx * sin_t - ty * cos_t,
            'scale': 1.0 / scale,
            'flip': flip,  # Flip is self-inverse
        }

        return transformed_bev, inverse_transform

    def apply_inverse_transform_3d(
        self,
        output_3d: torch.Tensor,
        inverse_transform: Dict[str, float],
    ) -> torch.Tensor:
        """
        Apply inverse transformation to 3D output.

        Args:
            output_3d: [B, C, H, W, D] 3D logits/probs
            inverse_transform: Dict with inverse transform parameters

        Returns:
            transformed_output: [B, C, H, W, D] inverse-transformed output
        """
        B, C, H, W, D = output_3d.shape
        device = output_3d.device

        angle = inverse_transform['angle']
        tx = inverse_transform['tx']
        ty = inverse_transform['ty']
        scale = inverse_transform['scale']
        flip = inverse_transform['flip']

        # Process each depth slice
        result = torch.zeros_like(output_3d)

        for d in range(D):
            slice_2d = output_3d[:, :, :, :, d]  # [B, C, H, W]

            # Build inverse affine
            theta = math.radians(angle)
            cos_t, sin_t = math.cos(theta), math.sin(theta)

            affine = torch.tensor([
                [scale * cos_t, -scale * sin_t, 2 * tx / W],
                [scale * sin_t, scale * cos_t, 2 * ty / H],
            ], dtype=torch.float32, device=device)

            if flip:
                affine[0, 0] *= -1
                affine[0, 1] *= -1

            affine = affine.unsqueeze(0).expand(B, -1, -1)
            grid = F.affine_grid(affine, slice_2d.shape, align_corners=False)

            # Bilinear interpolation for logits/probs
            transformed = F.grid_sample(
                slice_2d, grid, mode='bilinear', padding_mode='zeros', align_corners=False
            )
            result[:, :, :, :, d] = transformed

        return result

    def forward(
        self,
        bev: torch.Tensor,
        num_views: int = 3,
    ) -> List[Tuple[torch.Tensor, Dict[str, float]]]:
        """
        Generate multiple augmented views.

        Args:
            bev: [B, H, W] BEV semantic labels
            num_views: Number of augmented views

        Returns:
            List of (augmented_bev, inverse_transform) tuples
        """
        views = []

        # First view: identity (no augmentation)
        identity_transform = {'angle': 0, 'tx': 0, 'ty': 0, 'scale': 1.0, 'flip': False}
        views.append((bev.clone(), identity_transform))

        # Additional views: random augmentations
        for _ in range(num_views - 1):
            transform = self.generate_random_transform()
            aug_bev, inv_transform = self.apply_transform(bev, transform)
            views.append((aug_bev, inv_transform))

        return views


class Unified3DAugmenter(nn.Module):
    """
    PaSCo-style unified 3D augmenter for MIMO.

    CRITICAL: Transforms BOTH BEV and LiDAR with the SAME transformation matrix.
    This ensures spatial consistency between the 2D BEV and 3D LiDAR inputs.

    From PaSCo's kitti_dataset.py:
    - Line 396-398: transform_scene(semantic_coords, T, semantic_label) for labels
    - Line 429: in_coords = transform(in_coords, T) for LiDAR points
    - Same T matrix is applied to both!

    Reference:
    - /workspace/reference/PaSCo/pasco/models/transform_utils.py
    - /workspace/reference/PaSCo/pasco/data/semantic_kitti/kitti_dataset.py
    """

    def __init__(
        self,
        max_angle: float = 30.0,
        max_translation: Tuple[float, float, float] = (0.6, 0.6, 0.4),
        scale_range: float = 0.0,  # PaSCo default: 0 (DISABLED!)
        flip_prob: float = 0.5,
        voxel_shape: Tuple[int, int, int] = VOXEL_SHAPE,
        voxel_resolution: float = VOXEL_RESOLUTION,
    ):
        """
        Args:
            max_angle: Maximum rotation in degrees (PaSCo: 30°)
            max_translation: Maximum translation in meters (PaSCo: [0.6, 0.6, 0.4])
            scale_range: Scale variation (PaSCo default: 0.0 = disabled)
            flip_prob: Probability of Y-axis flip (PaSCo: 0.5)
            voxel_shape: Shape of voxel grid (X, Y, Z)
            voxel_resolution: Meters per voxel
        """
        super().__init__()
        self.max_angle = max_angle
        self.max_translation = max_translation
        self.scale_range = scale_range
        self.flip_prob = flip_prob
        self.voxel_resolution = voxel_resolution

        # 3D voxel transformer for LiDAR
        self.voxel_transformer = VoxelTransformer(
            voxel_shape=voxel_shape,
            resolution=voxel_resolution,
        )

    def generate_transform(self) -> torch.Tensor:
        """Generate random 4x4 transformation matrix."""
        return generate_random_transformation(
            max_angle=self.max_angle,
            flip=self.flip_prob > 0,
            scale_range=self.scale_range,
            max_translation=self.max_translation,
        )

    def transform_bev(
        self,
        bev: torch.Tensor,
        T: torch.Tensor,
        inverse: bool = False,
    ) -> torch.Tensor:
        """
        Transform BEV using 2D projection of 3D transformation.

        BEV is the XY plane projection, so we extract the 2D rotation and
        translation from the 4x4 matrix.

        Args:
            bev: [B, H, W] BEV semantic labels (H=X, W=Y in voxel coords)
            T: [4, 4] transformation matrix
            inverse: If True, apply inverse transform

        Returns:
            transformed_bev: [B, H, W] transformed BEV
        """
        B, H, W = bev.shape
        device = bev.device

        # Get 2D transform from 4x4 matrix
        # T is in the form: rotation around Z, translation in XY
        T = T.to(device).float()

        if inverse:
            T = torch.inverse(T)

        # Extract 2D affine: [cos, -sin, tx; sin, cos, ty]
        # Note: T[:2, :2] contains rotation+scale in XY plane
        #       T[:2, 3] contains translation in XY (in meters)
        affine_2d = torch.zeros(B, 2, 3, device=device)

        # Rotation and scale from 4x4 matrix
        affine_2d[:, 0, 0] = T[0, 0]
        affine_2d[:, 0, 1] = T[0, 1]
        affine_2d[:, 1, 0] = T[1, 0]
        affine_2d[:, 1, 1] = T[1, 1]

        # Translation: convert from meters to normalized coords [-1, 1]
        # Our voxel grid is H x W voxels, each voxel_resolution meters
        grid_extent_x = H * self.voxel_resolution  # Total meters in X
        grid_extent_y = W * self.voxel_resolution  # Total meters in Y
        affine_2d[:, 0, 2] = 2 * T[0, 3] / grid_extent_x
        affine_2d[:, 1, 2] = 2 * T[1, 3] / grid_extent_y

        # Apply transformation
        bev_float = bev.float().unsqueeze(1)  # [B, 1, H, W]
        grid = F.affine_grid(affine_2d, bev_float.shape, align_corners=True)
        transformed = F.grid_sample(
            bev_float, grid, mode='nearest', padding_mode='zeros', align_corners=True
        )

        return transformed.squeeze(1).long()

    def transform_lidar(
        self,
        lidar: torch.Tensor,
        T: torch.Tensor,
        inverse: bool = False,
    ) -> torch.Tensor:
        """
        Transform 3D LiDAR voxels using VoxelTransformer.

        Args:
            lidar: [B, 1, X, Y, Z] sparse LiDAR voxels
            T: [4, 4] transformation matrix
            inverse: If True, apply inverse transform

        Returns:
            transformed_lidar: [B, 1, X, Y, Z] transformed LiDAR
        """
        return self.voxel_transformer.transform_dense_voxels(
            lidar, T, mode='nearest', inverse=inverse
        )

    def transform_output_3d(
        self,
        output_3d: torch.Tensor,
        T: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply inverse transformation to 3D output for alignment.

        This is equivalent to PaSCo's sample_scene() function which
        maps the augmented output back to the original coordinate space.

        Args:
            output_3d: [B, C, X, Y, Z] 3D logits/probs
            T: [4, 4] original transformation matrix (will be inverted)

        Returns:
            aligned_output: [B, C, X, Y, Z] output in original coordinate space
        """
        # Apply inverse of T to bring output back to original space
        return self.voxel_transformer.transform_dense_voxels(
            output_3d, T, mode='bilinear', inverse=True
        )

    def augment(
        self,
        bev: torch.Tensor,
        lidar: torch.Tensor,
        T: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Augment both BEV and LiDAR with the SAME transformation.

        Args:
            bev: [B, H, W] BEV semantic labels
            lidar: [B, 1, X, Y, Z] sparse LiDAR voxels
            T: [4, 4] transformation matrix (generates random if None)

        Returns:
            Dict with:
            - 'bev': Transformed BEV
            - 'lidar': Transformed LiDAR
            - 'T': Transformation matrix used
        """
        if T is None:
            T = self.generate_transform()

        return {
            'bev': self.transform_bev(bev, T, inverse=False),
            'lidar': self.transform_lidar(lidar, T, inverse=False),
            'T': T,
        }

    def forward(
        self,
        bev: torch.Tensor,
        lidar: torch.Tensor,
        num_views: int = 3,
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Generate multiple augmented views.

        Args:
            bev: [B, H, W] BEV semantic labels
            lidar: [B, 1, X, Y, Z] sparse LiDAR voxels
            num_views: Number of augmented views (first is identity)

        Returns:
            List of dicts, each with 'bev', 'lidar', 'T'
        """
        views = []

        # First view: identity (no augmentation)
        T_identity = torch.eye(4)
        views.append({
            'bev': bev.clone(),
            'lidar': lidar.clone(),
            'T': T_identity,
        })

        # Additional views: random augmentations
        for _ in range(num_views - 1):
            T = self.generate_transform()
            views.append(self.augment(bev, lidar, T))

        return views


class BEVAugmenter(nn.Module):
    """
    Legacy augmenter for BEV inputs (non-geometric augmentations).

    Kept for backward compatibility. For PaSCo-style augmentation,
    use Unified3DAugmenter instead.
    """

    def __init__(
        self,
        num_classes: int = 20,
        noise_std: float = 0.1,
        smooth_factor: float = 0.1,
        dropout_prob: float = 0.1,
        dropout_block_size: int = 8,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.noise_std = noise_std
        self.smooth_factor = smooth_factor
        self.dropout_prob = dropout_prob
        self.dropout_block_size = dropout_block_size

    def identity(self, bev: torch.Tensor) -> torch.Tensor:
        """No augmentation - return as is."""
        return bev

    def add_noise(self, bev: torch.Tensor) -> torch.Tensor:
        """Add Gaussian noise to BEV one-hot representation."""
        B, H, W = bev.shape
        one_hot = F.one_hot(bev.long(), self.num_classes).float()
        noise = torch.randn_like(one_hot) * self.noise_std
        noisy = F.softmax(one_hot + noise, dim=-1)
        return noisy.argmax(dim=-1)

    def spatial_dropout(self, bev: torch.Tensor) -> torch.Tensor:
        """Drop random spatial blocks from BEV."""
        B, H, W = bev.shape
        result = bev.clone()
        if self.training:
            num_blocks_h = H // self.dropout_block_size
            num_blocks_w = W // self.dropout_block_size
            for b in range(B):
                for i in range(num_blocks_h):
                    for j in range(num_blocks_w):
                        if torch.rand(1).item() < self.dropout_prob:
                            h_start = i * self.dropout_block_size
                            h_end = min(h_start + self.dropout_block_size, H)
                            w_start = j * self.dropout_block_size
                            w_end = min(w_start + self.dropout_block_size, W)
                            result[b, h_start:h_end, w_start:w_end] = 0
        return result

    def rotate(self, bev: torch.Tensor, k: int = 1) -> torch.Tensor:
        """Rotate BEV by k*90 degrees."""
        return torch.rot90(bev, k, dims=[-2, -1])

    def flip_h(self, bev: torch.Tensor) -> torch.Tensor:
        """Horizontal flip."""
        return torch.flip(bev, dims=[-1])

    def flip_v(self, bev: torch.Tensor) -> torch.Tensor:
        """Vertical flip."""
        return torch.flip(bev, dims=[-2])

    def get_augmentations(self, aug_types: List[str] = None) -> List[callable]:
        """Get list of augmentation functions."""
        if aug_types is None:
            aug_types = ['identity', 'noise', 'dropout']

        aug_map = {
            'identity': self.identity,
            'noise': self.add_noise,
            'dropout': self.spatial_dropout,
            'rot90': lambda x: self.rotate(x, 1),
            'rot180': lambda x: self.rotate(x, 2),
            'rot270': lambda x: self.rotate(x, 3),
            'flip_h': self.flip_h,
            'flip_v': self.flip_v,
        }

        return [aug_map[t] for t in aug_types]

    def forward(
        self,
        bev: torch.Tensor,
        aug_types: List[str] = None,
    ) -> List[torch.Tensor]:
        """Apply augmentations to create multiple BEV views."""
        augmentations = self.get_augmentations(aug_types)
        return [aug(bev) for aug in augmentations]


class MIMOSceneCompletion(nn.Module):
    """
    MIMO wrapper for scene completion following PaSCo (CVPR 2024).

    Key Design from PaSCo Paper (Section 3.2.1-3.2.2):
    ================================================

    Training Phase:
    - Each subnet receives a DIFFERENT sample from the dataset
    - This teaches the model to be robust to input variations
    - Quote: "at training {Xi} are distinct point clouds"

    Inference Phase:
    - Each subnet receives the SAME sample with different augmentations
    - Augmentations: rotation (±30°), translation (±0.6m), flip, scale (±10%)
    - Quote: "in inference, they are augmentations of the same point cloud"

    CRITICAL FIX (Dec 2024):
    - Both BEV AND LiDAR must be transformed with the SAME matrix T
    - Previously only BEV was transformed, causing spatial misalignment
    - Now uses Unified3DAugmenter which transforms both consistently

    Ensemble Method:
    - Average PROBABILITIES (softmax outputs), NOT logits
    - Quote: "averaging the semantic probability p"
    - From ensembler.py: sem_probs = F.softmax(sem_logits, dim=-1); mean(sem_probs)

    Reference:
    - Paper Section 3.2.1-3.2.2 (page 4)
    - Code: /workspace/reference/PaSCo/pasco/models/ensembler.py (lines 159-187)
    - Code: /workspace/reference/PaSCo/pasco/data/semantic_kitti/kitti_dataset.py (lines 126-140)
    """

    def __init__(
        self,
        base_model: nn.Module,
        num_subnets: int = 3,
        num_classes: int = 20,
        aug_types: List[str] = None,
        ensemble_method: str = 'mean_probs',  # PaSCo default: mean_probs
        use_geometric_aug: bool = True,  # PaSCo-style geometric augmentation
        use_unified_3d_aug: bool = True,  # NEW: Use unified 3D augmenter
        max_angle: float = 30.0,  # PaSCo default
        max_translation: Tuple[float, float, float] = (0.6, 0.6, 0.4),  # PaSCo default
        scale_range: float = 0.1,  # PaSCo default
        voxel_shape: Tuple[int, int, int] = VOXEL_SHAPE,
    ):
        """
        Args:
            base_model: Base scene completion model (shared weights)
            num_subnets: Number of parallel forward passes (PaSCo uses 3)
            num_classes: Number of semantic classes
            aug_types: Legacy augmentation types (ignored if use_geometric_aug=True)
            ensemble_method: 'mean_probs' (PaSCo default), 'mean_logits', or 'vote'
            use_geometric_aug: Use PaSCo-style geometric augmentation
            use_unified_3d_aug: Use unified 3D augmenter (transforms both BEV and LiDAR)
            max_angle: Max rotation in degrees (PaSCo: 30°)
            max_translation: Max translation in meters (PaSCo: [0.6, 0.6, 0.4])
            scale_range: Scale variation (PaSCo: 0.1 → [0.95, 1.05])
            voxel_shape: Shape of voxel grid (X, Y, Z)
        """
        super().__init__()
        self.base_model = base_model
        self.num_subnets = num_subnets
        self.num_classes = num_classes
        self.ensemble_method = ensemble_method
        self.use_geometric_aug = use_geometric_aug
        self.use_unified_3d_aug = use_unified_3d_aug

        # NEW: Unified 3D augmenter (transforms both BEV and LiDAR)
        if use_unified_3d_aug:
            self.unified_augmenter = Unified3DAugmenter(
                max_angle=max_angle,
                max_translation=max_translation,
                scale_range=scale_range,
                voxel_shape=voxel_shape,
            )
            print("[MIMO] Using Unified3DAugmenter (transforms both BEV and LiDAR)")
        else:
            self.unified_augmenter = None

        # Legacy geometric augmenter (BEV only - has bug)
        if use_geometric_aug and not use_unified_3d_aug:
            self.augmenter = BEVGeometricAugmenter(
                max_angle=max_angle,
                max_translation=max_translation[:2],  # Only XY for 2D
                scale_range=scale_range,
            )
            print("[MIMO] WARNING: Using BEVGeometricAugmenter (BEV only, LiDAR not transformed)")
        elif not use_unified_3d_aug:
            # Legacy augmenter
            self.augmenter = BEVAugmenter(num_classes=num_classes)
            if aug_types is None:
                aug_types = ['identity', 'noise', 'dropout'][:num_subnets]
            self.aug_types = aug_types

        # Inverse transforms for legacy 90-degree rotations
        self.inverse_transforms = {
            'identity': lambda x: x,
            'noise': lambda x: x,
            'dropout': lambda x: x,
            'rot90': lambda x: torch.rot90(x, -1, dims=[-3, -2]),
            'rot180': lambda x: torch.rot90(x, -2, dims=[-3, -2]),
            'rot270': lambda x: torch.rot90(x, -3, dims=[-3, -2]),
            'flip_h': lambda x: torch.flip(x, dims=[-2]),
            'flip_v': lambda x: torch.flip(x, dims=[-3]),
        }

    def forward(
        self,
        bev: torch.Tensor,
        lidar: torch.Tensor,
        t: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with MIMO following PaSCo.

        PaSCo Inference Behavior:
        - All subnets receive the SAME input sample
        - Each subnet applies a DIFFERENT random augmentation
        - Outputs are inverse-transformed to align spatially
        - Probabilities (not logits) are averaged

        CRITICAL: Both BEV and LiDAR are now transformed with the SAME matrix T
        when using use_unified_3d_aug=True (default).

        Args:
            bev: [B, H, W] BEV semantic labels
            lidar: [B, 1, H, W, D] sparse LiDAR voxels
            t: [B] diffusion timestep (if using diffusion)

        Returns:
            Dict with:
            - 'logits': [B, C, H, W, D] ensembled logits (log of avg probs)
            - 'predictions': [B, H, W, D] ensembled predictions
            - 'subnet_logits': List of [B, C, H, W, D] per-subnet logits
            - 'uncertainty': [B, H, W, D] prediction variance
        """
        B, H, W = bev.shape

        subnet_logits = []
        subnet_probs = []

        # NEW: Unified 3D augmentation (transforms both BEV and LiDAR)
        if self.use_unified_3d_aug and self.unified_augmenter is not None:
            # Generate augmented views with consistent BEV and LiDAR transforms
            views = self.unified_augmenter(bev, lidar, num_views=self.num_subnets)

            for view in views:
                aug_bev = view['bev']
                aug_lidar = view['lidar']
                T = view['T']

                # Forward through base model with BOTH inputs transformed
                if t is not None:
                    output = self.base_model(aug_bev, aug_lidar, t, **kwargs)
                else:
                    output = self.base_model(aug_bev, aug_lidar, **kwargs)

                # Get logits
                if isinstance(output, dict):
                    logits = output.get('logits', output.get('pred_scene', None))
                else:
                    logits = output

                # Apply inverse transform to align output back to original space
                # Skip for identity transform (first view)
                if not torch.allclose(T, torch.eye(4).to(T.device)):
                    logits = self.unified_augmenter.transform_output_3d(logits, T)

                subnet_logits.append(logits)
                subnet_probs.append(F.softmax(logits, dim=1))

        elif self.use_geometric_aug:
            # DEPRECATED: BEV-only augmentation (has spatial misalignment bug)
            # Kept for backward compatibility
            views = self.augmenter(bev, num_views=self.num_subnets)

            for aug_bev, inv_transform in views:
                # Forward through base model
                # WARNING: lidar is NOT transformed, causing misalignment
                if t is not None:
                    output = self.base_model(aug_bev, lidar, t, **kwargs)
                else:
                    output = self.base_model(aug_bev, lidar, **kwargs)

                # Get logits
                if isinstance(output, dict):
                    logits = output.get('logits', output.get('pred_scene', None))
                else:
                    logits = output

                # Apply inverse transform to align outputs
                if inv_transform['angle'] != 0 or inv_transform['flip']:
                    logits = self.augmenter.apply_inverse_transform_3d(logits, inv_transform)

                subnet_logits.append(logits)
                subnet_probs.append(F.softmax(logits, dim=1))
        else:
            # Legacy augmentation
            aug_bevs = self.augmenter(bev, self.aug_types)

            for i, (aug_bev, aug_type) in enumerate(zip(aug_bevs, self.aug_types)):
                aug_lidar = lidar
                if aug_type in ['rot90', 'rot180', 'rot270', 'flip_h', 'flip_v']:
                    if aug_type == 'rot90':
                        aug_lidar = torch.rot90(lidar, 1, dims=[-3, -2])
                    elif aug_type == 'rot180':
                        aug_lidar = torch.rot90(lidar, 2, dims=[-3, -2])
                    elif aug_type == 'rot270':
                        aug_lidar = torch.rot90(lidar, 3, dims=[-3, -2])
                    elif aug_type == 'flip_h':
                        aug_lidar = torch.flip(lidar, dims=[-2])
                    elif aug_type == 'flip_v':
                        aug_lidar = torch.flip(lidar, dims=[-3])

                if t is not None:
                    output = self.base_model(aug_bev, aug_lidar, t, **kwargs)
                else:
                    output = self.base_model(aug_bev, aug_lidar, **kwargs)

                if isinstance(output, dict):
                    logits = output.get('logits', output.get('pred_scene', None))
                else:
                    logits = output

                if aug_type in ['rot90', 'rot180', 'rot270', 'flip_h', 'flip_v']:
                    logits = self.inverse_transforms[aug_type](logits)

                subnet_logits.append(logits)
                subnet_probs.append(F.softmax(logits, dim=1))

        # Ensemble predictions - PaSCo uses mean_probs
        if self.ensemble_method == 'mean_probs':
            # PaSCo's approach: Average probabilities after softmax
            # From ensembler.py: sem_probs.append(torch.stack(sem_probs, dim=0).mean(0))
            stacked_probs = torch.stack(subnet_probs, dim=0)  # [N, B, C, H, W, D]
            ensemble_probs = stacked_probs.mean(dim=0)  # [B, C, H, W, D]
            ensemble_logits = torch.log(ensemble_probs + 1e-10)
            ensemble_preds = ensemble_probs.argmax(dim=1)

        elif self.ensemble_method == 'mean_logits':
            stacked_logits = torch.stack(subnet_logits, dim=0)
            ensemble_logits = stacked_logits.mean(dim=0)
            ensemble_preds = ensemble_logits.argmax(dim=1)

        elif self.ensemble_method == 'vote':
            subnet_preds = [l.argmax(dim=1) for l in subnet_logits]
            stacked_preds = torch.stack(subnet_preds, dim=0)
            ensemble_preds = stacked_preds.mode(dim=0).values
            stacked_logits = torch.stack(subnet_logits, dim=0)
            ensemble_logits = stacked_logits.mean(dim=0)

        # Compute uncertainty (variance across subnets)
        # PaSCo uses variance of probabilities
        if len(subnet_probs) > 1:
            stacked_probs = torch.stack(subnet_probs, dim=0)
            uncertainty = stacked_probs.var(dim=0).mean(dim=1)  # [B, H, W, D]
        else:
            uncertainty = torch.zeros_like(ensemble_preds).float()

        return {
            'logits': ensemble_logits,
            'predictions': ensemble_preds,
            'subnet_logits': subnet_logits,
            'subnet_probs': subnet_probs,
            'uncertainty': uncertainty,
        }


class MIMOEnsembler(nn.Module):
    """
    Ensembler for combining MIMO subnet outputs.

    Reference: /workspace/reference/PaSCo/pasco/models/ensembler.py
    """

    def __init__(
        self,
        num_classes: int = 20,
        iou_threshold: float = 0.2,
        method: str = 'simple_avg',
    ):
        super().__init__()
        self.num_classes = num_classes
        self.iou_threshold = iou_threshold
        self.method = method

    def forward(
        self,
        predictions: List[torch.Tensor],
        probs: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Ensemble multiple predictions."""
        if len(predictions) == 1:
            return predictions[0], torch.zeros_like(predictions[0]).float()

        if self.method == 'simple_avg' and probs is not None:
            stacked_probs = torch.stack(probs, dim=0)
            avg_probs = stacked_probs.mean(dim=0)
            ensemble_pred = avg_probs.argmax(dim=1)
            uncertainty = stacked_probs.var(dim=0).mean(dim=1)
        else:
            stacked_preds = torch.stack(predictions, dim=0)
            ensemble_pred = stacked_preds.mode(dim=0).values
            agreement = (stacked_preds == ensemble_pred.unsqueeze(0)).float().mean(dim=0)
            uncertainty = 1 - agreement

        return ensemble_pred, uncertainty

    @staticmethod
    def compute_uncertainty_map(probs: List[torch.Tensor]) -> torch.Tensor:
        """Compute per-voxel uncertainty from probability predictions."""
        stacked = torch.stack(probs, dim=0)
        avg_probs = stacked.mean(dim=0)
        entropy = -(avg_probs * torch.log(avg_probs + 1e-10)).sum(dim=1)
        max_entropy = np.log(probs[0].shape[1])
        uncertainty = entropy / max_entropy
        return uncertainty


def create_mimo_wrapper(
    base_model: nn.Module,
    num_subnets: int = 3,  # PaSCo paper default: 3 (NOT 2!)
    num_classes: int = 20,
    aug_types: List[str] = None,
    use_geometric_aug: bool = True,
    use_unified_3d_aug: bool = True,
    ensemble_method: str = 'mean_probs',  # PaSCo default: mean_probs (NOT mean_logits!)
    max_angle: float = 30.0,  # PaSCo paper: 30 degrees
    scale_range: float = 0.0,  # PaSCo default: 0 (disabled)
    voxel_shape: Tuple[int, int, int] = VOXEL_SHAPE,
) -> MIMOSceneCompletion:
    """
    Convenience function to wrap a model with MIMO.

    Default settings match PaSCo (CVPR 2024):
    - num_subnets=3 (PaSCo paper default - see net_panoptic_sparse.py)
    - ensemble_method='mean_probs' (average probabilities AFTER softmax, NOT logits!)
    - scale_range=0.0 (disabled in PaSCo)
    - max_angle=30.0 (from PaSCo paper)
    - use_unified_3d_aug=True (transforms BOTH BEV and LiDAR with same T)
    - Augmentations: rotation ±30°, translation ±0.6m, flip, NO scale

    NOTE: The TRUE PaSCo architecture uses:
    - Channel concatenation (single forward pass with N*C input channels)
    - N separate decoder heads (not N forward passes)
    - 80% random crop per sample during training

    This wrapper currently uses N forward passes as a simpler approximation.
    For full PaSCo-style MIMO, implement the channel-concat architecture.

    Args:
        base_model: Scene completion model
        num_subnets: Number of subnets (forward passes), PaSCo paper=3
        num_classes: Number of semantic classes
        aug_types: Legacy augmentation types (ignored if use_geometric_aug=True)
        use_geometric_aug: Use PaSCo-style geometric augmentation
        use_unified_3d_aug: Use unified 3D augmenter (transforms both BEV and LiDAR)
        ensemble_method: 'mean_probs' (default), 'mean_logits', or 'vote'
        max_angle: Max rotation in degrees (PaSCo: 30°)
        scale_range: Scale variation (PaSCo: 0.0 = disabled)
        voxel_shape: Shape of voxel grid (X, Y, Z)

    Returns:
        MIMOSceneCompletion wrapper
    """
    return MIMOSceneCompletion(
        base_model=base_model,
        num_subnets=num_subnets,
        num_classes=num_classes,
        aug_types=aug_types,
        use_geometric_aug=use_geometric_aug,
        use_unified_3d_aug=use_unified_3d_aug,
        ensemble_method=ensemble_method,
        max_angle=max_angle,
        scale_range=scale_range,
        voxel_shape=voxel_shape,
    )
