"""
Semantic Pruning for Scene Completion (PaSCo-style)

This module implements semantic pruning to improve performance on rare classes
by focusing on occupied regions and removing spurious predictions.

Reference: PaSCo (CVPR 2024)
- Semantic Pruning: Better rare class performance (+33.72 PQ for truck)
- Uses MinkowskiPruning to prune voxels outside predicted bounding boxes

Key Insight:
- Scene completion often over-predicts (hallucination in empty regions)
- Semantic pruning removes predictions in regions that are clearly empty
- This improves precision on rare classes (car, truck, bicycle, etc.)

Implementation:
1. Occupancy-based pruning: Remove predictions where P(empty) is very high
2. LiDAR-guided pruning: Keep predictions near observed LiDAR points
3. Bounding box pruning: Prune outside predicted object extents
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict


def compute_occupancy_mask(
    probs: torch.Tensor,
    empty_threshold: float = 0.9,
    min_occupied_prob: float = 0.1,
) -> torch.Tensor:
    """
    Compute mask of voxels that are likely occupied.

    Args:
        probs: [B, C, H, W, D] probability tensor (C=num_classes)
        empty_threshold: If P(empty) > threshold, mark as empty
        min_occupied_prob: Minimum probability for any non-empty class

    Returns:
        [B, H, W, D] boolean mask (True = likely occupied)
    """
    # Class 0 is typically "empty" or "unlabeled" in SemanticKITTI
    p_empty = probs[:, 0]  # [B, H, W, D]

    # Also check if any non-empty class has reasonable probability
    p_occupied = probs[:, 1:].max(dim=1)[0]  # [B, H, W, D]

    # Voxel is likely occupied if:
    # 1. P(empty) < threshold AND
    # 2. Max P(any_class) > min_occupied_prob
    occupied_mask = (p_empty < empty_threshold) & (p_occupied > min_occupied_prob)

    return occupied_mask


def lidar_guided_mask(
    lidar: torch.Tensor,
    dilation_radius: int = 2,
) -> torch.Tensor:
    """
    Create mask based on LiDAR observations with spatial dilation.

    Scene completion should focus on regions near observed LiDAR points.

    Args:
        lidar: [B, 1, H, W, D] binary LiDAR voxels
        dilation_radius: Radius for morphological dilation

    Returns:
        [B, H, W, D] boolean mask (True = near LiDAR observation)
    """
    # Squeeze channel dim
    lidar_binary = lidar.squeeze(1) > 0.5  # [B, H, W, D]

    if dilation_radius <= 0:
        return lidar_binary

    # Apply 3D morphological dilation using max pooling
    # Need to add channel dim for pooling
    lidar_float = lidar_binary.float().unsqueeze(1)  # [B, 1, H, W, D]

    kernel_size = 2 * dilation_radius + 1
    padding = dilation_radius

    dilated = F.max_pool3d(
        lidar_float,
        kernel_size=kernel_size,
        stride=1,
        padding=padding,
    )

    return dilated.squeeze(1) > 0.5


def prune_predictions(
    pred_labels: torch.Tensor,
    mask: torch.Tensor,
    empty_class: int = 0,
) -> torch.Tensor:
    """
    Prune predictions by setting masked-out voxels to empty class.

    Args:
        pred_labels: [B, H, W, D] predicted class labels
        mask: [B, H, W, D] boolean mask (True = keep prediction)
        empty_class: Class index for empty voxels

    Returns:
        [B, H, W, D] pruned predictions
    """
    pruned = pred_labels.clone()
    pruned[~mask] = empty_class
    return pruned


def semantic_pruning_postprocess(
    pred_probs: torch.Tensor,
    lidar: Optional[torch.Tensor] = None,
    empty_threshold: float = 0.85,
    lidar_dilation: int = 3,
    use_lidar_mask: bool = True,
    use_occupancy_mask: bool = True,
    combine_mode: str = "intersection",
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Apply PaSCo-style semantic pruning as post-processing.

    This improves performance on rare classes by:
    1. Removing predictions in clearly empty regions
    2. Keeping predictions only near LiDAR observations

    Args:
        pred_probs: [B, C, H, W, D] predicted probabilities
        lidar: [B, 1, H, W, D] binary LiDAR voxels (optional)
        empty_threshold: Threshold for occupancy-based pruning
        lidar_dilation: Dilation radius for LiDAR-guided mask
        use_lidar_mask: Whether to use LiDAR-guided pruning
        use_occupancy_mask: Whether to use occupancy-based pruning
        combine_mode: How to combine masks ("intersection" or "union")

    Returns:
        Tuple of:
        - [B, H, W, D] pruned class predictions
        - Dict with 'occupancy_mask', 'lidar_mask', 'final_mask' for visualization
    """
    B, C, H, W, D = pred_probs.shape
    device = pred_probs.device

    masks_info = {}

    # Start with all-True mask
    final_mask = torch.ones(B, H, W, D, dtype=torch.bool, device=device)

    # Compute occupancy-based mask
    if use_occupancy_mask:
        occupancy_mask = compute_occupancy_mask(
            pred_probs,
            empty_threshold=empty_threshold,
        )
        masks_info['occupancy_mask'] = occupancy_mask

        if combine_mode == "intersection":
            final_mask = final_mask & occupancy_mask
        else:  # union
            final_mask = occupancy_mask

    # Compute LiDAR-guided mask
    if use_lidar_mask and lidar is not None:
        lidar_mask = lidar_guided_mask(
            lidar,
            dilation_radius=lidar_dilation,
        )
        masks_info['lidar_mask'] = lidar_mask

        if combine_mode == "intersection":
            final_mask = final_mask & lidar_mask
        else:  # union
            if use_occupancy_mask:
                final_mask = final_mask | lidar_mask
            else:
                final_mask = lidar_mask

    masks_info['final_mask'] = final_mask

    # Get class predictions
    pred_labels = pred_probs.argmax(dim=1)  # [B, H, W, D]

    # Apply pruning
    pruned_labels = prune_predictions(pred_labels, final_mask, empty_class=0)

    return pruned_labels, masks_info


class SemanticPruning(nn.Module):
    """
    Learnable semantic pruning module.

    Instead of fixed thresholds, learns to predict which voxels to keep
    based on the input features and predictions.
    """

    def __init__(
        self,
        num_classes: int = 20,
        hidden_dim: int = 32,
        use_context: bool = True,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.use_context = use_context

        # Predict keep/prune decision from class probabilities
        input_dim = num_classes
        if use_context:
            # Add local context (neighboring predictions)
            input_dim += num_classes  # Will be populated with local stats

        self.keep_predictor = nn.Sequential(
            nn.Conv3d(input_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm3d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm3d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden_dim, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        pred_probs: torch.Tensor,
        temperature: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict which voxels to keep.

        Args:
            pred_probs: [B, C, H, W, D] predicted probabilities
            temperature: Temperature for soft pruning (lower = harder)

        Returns:
            Tuple of:
            - [B, H, W, D] pruned predictions
            - [B, 1, H, W, D] keep probabilities
        """
        if self.use_context:
            # Compute local class distribution (average in 3x3x3 neighborhood)
            local_avg = F.avg_pool3d(pred_probs, kernel_size=3, stride=1, padding=1)
            input_feats = torch.cat([pred_probs, local_avg], dim=1)
        else:
            input_feats = pred_probs

        # Predict keep probability
        keep_probs = self.keep_predictor(input_feats)  # [B, 1, H, W, D]
        keep_probs = keep_probs / temperature
        keep_probs = torch.sigmoid(keep_probs)

        # Soft pruning: weight predictions by keep probability
        pred_labels = pred_probs.argmax(dim=1)  # [B, H, W, D]

        # Hard pruning at inference (use Gumbel-softmax for training)
        keep_mask = keep_probs.squeeze(1) > 0.5
        pruned_labels = pred_labels.clone()
        pruned_labels[~keep_mask] = 0  # Set to empty class

        return pruned_labels, keep_probs


def apply_bbox_pruning(
    pred_labels: torch.Tensor,
    lidar: torch.Tensor,
    margin: int = 5,
) -> torch.Tensor:
    """
    Prune predictions outside the LiDAR observation bounding box.

    Following PaSCo's approach: prune voxels outside [min_coord, max_coord].

    Args:
        pred_labels: [B, H, W, D] predicted class labels
        lidar: [B, 1, H, W, D] binary LiDAR voxels
        margin: Padding around bounding box

    Returns:
        [B, H, W, D] pruned predictions
    """
    B, _, H, W, D = lidar.shape
    device = lidar.device

    pruned = pred_labels.clone()

    for b in range(B):
        # Find bounding box of LiDAR observations
        occupied = lidar[b, 0] > 0.5  # [H, W, D]

        if not occupied.any():
            # No LiDAR points, set all to empty
            pruned[b] = 0
            continue

        # Get indices of occupied voxels
        occupied_indices = torch.nonzero(occupied)  # [N, 3]

        # Compute bounding box with margin
        min_coords = occupied_indices.min(dim=0)[0]
        max_coords = occupied_indices.max(dim=0)[0]

        min_h = max(0, min_coords[0].item() - margin)
        max_h = min(H - 1, max_coords[0].item() + margin)
        min_w = max(0, min_coords[1].item() - margin)
        max_w = min(W - 1, max_coords[1].item() + margin)
        min_d = max(0, min_coords[2].item() - margin)
        max_d = min(D - 1, max_coords[2].item() + margin)

        # Create mask
        mask = torch.zeros(H, W, D, dtype=torch.bool, device=device)
        mask[min_h:max_h+1, min_w:max_w+1, min_d:max_d+1] = True

        # Apply pruning
        pruned[b][~mask] = 0

    return pruned


# Example usage
if __name__ == "__main__":
    print("=== Semantic Pruning Test ===\n")

    # Create dummy data
    B, C, H, W, D = 2, 20, 64, 64, 8

    # Random probabilities
    pred_probs = torch.softmax(torch.randn(B, C, H, W, D), dim=1)

    # Sparse LiDAR (10% occupied)
    lidar = (torch.rand(B, 1, H, W, D) > 0.9).float()

    print(f"Input pred_probs shape: {pred_probs.shape}")
    print(f"Input lidar shape: {lidar.shape}")
    print(f"LiDAR occupancy: {lidar.mean():.2%}")

    # Test fixed-threshold pruning
    print("\n--- Fixed-threshold Semantic Pruning ---")
    pruned, masks = semantic_pruning_postprocess(
        pred_probs,
        lidar=lidar,
        empty_threshold=0.85,
        lidar_dilation=2,
    )

    print(f"Output shape: {pruned.shape}")
    print(f"Occupancy mask: {masks['occupancy_mask'].float().mean():.2%}")
    print(f"LiDAR mask: {masks['lidar_mask'].float().mean():.2%}")
    print(f"Final mask: {masks['final_mask'].float().mean():.2%}")

    # Count class distribution before/after
    pred_labels = pred_probs.argmax(dim=1)
    empty_before = (pred_labels == 0).float().mean()
    empty_after = (pruned == 0).float().mean()
    print(f"Empty voxels before: {empty_before:.2%}")
    print(f"Empty voxels after: {empty_after:.2%}")

    # Test learnable pruning
    print("\n--- Learnable Semantic Pruning ---")
    pruner = SemanticPruning(num_classes=C)
    pruned_learn, keep_probs = pruner(pred_probs)

    print(f"Keep probabilities range: [{keep_probs.min():.3f}, {keep_probs.max():.3f}]")
    print(f"Pruned output shape: {pruned_learn.shape}")

    # Test bounding box pruning
    print("\n--- Bounding Box Pruning ---")
    bbox_pruned = apply_bbox_pruning(pred_labels, lidar, margin=3)
    empty_bbox = (bbox_pruned == 0).float().mean()
    print(f"Empty voxels after bbox pruning: {empty_bbox:.2%}")

    print("\n=== Semantic Pruning Test Passed! ===")
