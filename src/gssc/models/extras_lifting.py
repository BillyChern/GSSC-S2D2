"""
S2: 2D→3D Lifting for Scene Completion

Lifts 2D BEV semantic predictions to rough 3D scenes.
Based on S3CNet algorithm (+5.28% mIoU improvement reported).

The lifted 3D scene serves as:
1. Initialization for SDEdit sampling (instead of pure noise)
2. Additional conditioning for the diffusion model

Key insight: Each semantic class has a typical height range.
E.g., buildings span full height (z=0-32), traffic signs are elevated (z=3-25),
road extends higher than expected due to terrain slope (z=0-23), etc.

Reference:
- S3CNet (CoRL 2020): "2D→3D Lifting" section
- Architecture plan: <repo>/docs/architecture_improvement_plan.md
"""


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Z-ranges for each class (in voxel units, grid is 256x256x32)
# Computed from actual SemanticKITTI data: 2nd-98th percentile of z-coordinates
# Class 0 = unlabeled, 1-19 = semantic classes
CLASS_Z_RANGES = {
    0: (0, 0),      # unlabeled - no fill
    1: (0, 21),     # car
    2: (0, 16),     # bicycle
    3: (0, 16),     # motorcycle
    4: (0, 23),     # truck
    5: (0, 22),     # other-vehicle
    6: (0, 19),     # person
    7: (0, 16),     # bicyclist
    8: (0, 16),     # motorcyclist
    9: (0, 23),     # road
    10: (0, 21),    # parking
    11: (0, 25),    # sidewalk
    12: (0, 22),    # other-ground
    13: (0, 32),    # building
    14: (0, 26),    # fence
    15: (0, 32),    # vegetation
    16: (0, 30),    # trunk
    17: (0, 27),    # terrain
    18: (0, 28),    # pole
    19: (3, 25),    # traffic-sign
}


class BEVTo3DLifter(nn.Module):
    """
    Lifts 2D BEV semantic predictions to rough 3D scenes.

    Uses class-specific height priors to fill voxels vertically.
    """

    def __init__(self,
                 num_classes: int = 20,
                 num_z: int = 32,
                 class_z_ranges: dict[int, tuple[int, int]] | None = None,
                 learnable: bool = False):
        """
        Args:
            num_classes: Number of semantic classes
            num_z: Number of Z slices (height)
            class_z_ranges: Dict mapping class -> (z_min, z_max)
            learnable: Whether to make z-ranges learnable
        """
        super().__init__()

        self.num_classes = num_classes
        self.num_z = num_z
        self.z_ranges = class_z_ranges or CLASS_Z_RANGES

        if learnable:
            # Learnable z-range parameters
            z_mins = torch.zeros(num_classes)
            z_maxs = torch.zeros(num_classes)
            for c in range(num_classes):
                z_min, z_max = self.z_ranges.get(c, (0, num_z))
                z_mins[c] = z_min
                z_maxs[c] = z_max

            self.z_mins = nn.Parameter(z_mins)
            self.z_maxs = nn.Parameter(z_maxs)
        else:
            self.register_buffer('z_mins', None)
            self.register_buffer('z_maxs', None)

    def forward(self, bev_pred: torch.Tensor) -> torch.Tensor:
        """
        Lift BEV prediction to 3D.

        Args:
            bev_pred: [B, H, W] BEV semantic prediction (class indices)
                      or [B, C, H, W] BEV logits

        Returns:
            scene_3d: [B, H, W, Z] 3D semantic prediction
        """
        # Handle logits input
        if bev_pred.dim() == 4:
            bev_pred = bev_pred.argmax(dim=1)  # [B, H, W]

        B, H, W = bev_pred.shape
        device = bev_pred.device

        # Initialize output
        scene_3d = torch.zeros(B, H, W, self.num_z, dtype=torch.long, device=device)

        # Fill each class
        for c in range(1, self.num_classes):  # Skip class 0 (unlabeled)
            z_min, z_max = self.z_ranges.get(c, (0, self.num_z))

            # Clamp to valid range
            z_min = max(0, min(z_min, self.num_z - 1))
            z_max = max(z_min + 1, min(z_max, self.num_z))

            # Find (x, y) where BEV predicts class c
            mask = (bev_pred == c)  # [B, H, W]

            # Fill z-range for those locations
            for z in range(z_min, z_max):
                scene_3d[:, :, :, z][mask] = c

        return scene_3d

    def forward_s3cnet(self, bev_pred: torch.Tensor, scene_3d_pred: torch.Tensor) -> torch.Tensor:
        """
        S3CNet-style per-sample dynamic lifting + fusion.

        From S3CNet paper (CoRL 2020), page 6:
        "(1) split the 3D volume into layers along the z-dimension (32 layers);
         (2) locate the slice with the maximum occurrence of the desired class;
         (3) fill each voxel with the associated 2D pixel prediction if there exists
             such a class in the voxel's n×n (n=3) neighborhood, otherwise attempt
             to fill in the above voxel until the highest layer is exceeded;
         (4) repeat for all classes"

        Args:
            bev_pred: [B, H, W] BEV class predictions (argmax), or [B, C, H, W] logits
            scene_3d_pred: [B, H, W, D] 3D predictions from the network (class indices)

        Returns:
            fused: [B, H, W, D] fused 3D predictions (3D pred + lifted BEV filling empties)
        """
        # Handle logits input
        if bev_pred.dim() == 4:
            bev_pred = bev_pred.argmax(dim=1)  # [B, H, W]

        B, H, W = bev_pred.shape
        D = scene_3d_pred.shape[-1]  # 32
        device = bev_pred.device

        # Start from 3D prediction, fill empty voxels with lifted BEV
        fused = scene_3d_pred.clone()

        for b in range(B):
            pred_3d = scene_3d_pred[b]  # [H, W, D]
            bev = bev_pred[b]           # [H, W]

            for c in range(1, self.num_classes):  # Skip class 0 (empty/unlabeled)
                # Step 1: Find z-slice with maximum occurrence of class c in 3D prediction
                class_mask_3d = (pred_3d == c)  # [H, W, D]
                counts_per_z = class_mask_3d.sum(dim=(0, 1))  # [D]

                if counts_per_z.sum() == 0:
                    # Class not present in 3D prediction at all — use static fallback
                    z_best = self.z_ranges.get(c, (0, D))[0]
                else:
                    z_best = counts_per_z.argmax().item()

                # Step 2: Find (x, y) where BEV predicts class c
                bev_class_mask = (bev == c)  # [H, W]
                if not bev_class_mask.any():
                    continue

                # Step 3: For each such (x,y), check 3×3 neighborhood at z_best
                # If class c exists in neighborhood, fill; otherwise propagate upward
                # Dilate the 3D class mask at z_best with 3×3 kernel to get neighborhood
                if counts_per_z.sum() > 0:
                    slice_at_z = class_mask_3d[:, :, z_best].float().unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
                    # 3×3 max-pool = dilation (check if class c exists in 3×3 neighborhood)
                    neighborhood = F.max_pool2d(slice_at_z, kernel_size=3, stride=1, padding=1)
                    has_neighbor = neighborhood.squeeze() > 0  # [H, W]
                else:
                    has_neighbor = torch.zeros(H, W, dtype=torch.bool, device=device)

                # Pixels where BEV says class c AND (neighbor exists OR we'll propagate up)
                fill_mask = bev_class_mask  # [H, W]

                # Fill at z_best where neighbor check passes
                direct_fill = fill_mask & has_neighbor
                # For remaining pixels, propagate upward from z_best
                propagate_fill = fill_mask & ~has_neighbor

                # Fill direct matches at z_best (only where 3D pred is empty)
                empty_at_z = (fused[b, :, :, z_best] == 0)
                fused[b, :, :, z_best][direct_fill & empty_at_z] = c

                # Propagate upward for non-neighbor matches
                if propagate_fill.any():
                    for z in range(z_best, D):
                        empty_at_z = (fused[b, :, :, z] == 0)
                        can_fill = propagate_fill & empty_at_z
                        if can_fill.any():
                            fused[b, :, :, z][can_fill] = c
                            break  # S3CNet: "fill in the above voxel" — one voxel up

        return fused

    def forward_column_mvf(self, bev_pred: torch.Tensor, scene_3d_pred: torch.Tensor,
                           pred_probs: torch.Tensor | None = None) -> torch.Tensor:
        """
        Column-consistency MVF: only modify columns where BEV and 3D disagree.

        For each (x,y) column where BEV predicts class c:
        - If 3D prediction already has class c somewhere in that column → no change
        - If 3D prediction has NO class c in that column → BEV-3D disagreement →
          find the best z to place class c using the column's occupancy pattern

        This is much more targeted than S3CNet's lifting (which fills all empties).
        Designed for dense prediction models that already predict all voxels.

        Args:
            bev_pred: [B, H, W] BEV class predictions (argmax), or [B, C, H, W] logits
            scene_3d_pred: [B, H, W, D] 3D predictions (class indices)
            pred_probs: [B, C, H, W, D] optional prediction probabilities for confidence

        Returns:
            fused: [B, H, W, D] fused 3D predictions
        """
        if bev_pred.dim() == 4:
            bev_pred = bev_pred.argmax(dim=1)

        B, H, W = bev_pred.shape
        D = scene_3d_pred.shape[-1]
        device = bev_pred.device
        fused = scene_3d_pred.clone()

        for b in range(B):
            pred_3d = scene_3d_pred[b]  # [H, W, D]
            bev = bev_pred[b]           # [H, W]

            for c in range(1, self.num_classes):
                bev_has_c = (bev == c)  # [H, W] — BEV says class c
                if not bev_has_c.any():
                    continue

                # Check which columns already have class c in 3D prediction
                col_has_c = (pred_3d == c).any(dim=-1)  # [H, W]

                # Disagreement: BEV says c, but 3D has zero c in column
                disagree = bev_has_c & ~col_has_c  # [H, W]
                if not disagree.any():
                    continue

                # For disagreeing columns, find best z to place class c
                # Strategy: use the lowest-confidence z in the column
                if pred_probs is not None:
                    # Find the z with lowest max confidence in disagreeing columns
                    col_probs = pred_probs[b]  # [C, H, W, D]
                    max_conf_per_z = col_probs.max(dim=0).values  # [H, W, D]
                    # Mask non-disagreeing columns with inf so they're never selected
                    masked_conf = max_conf_per_z.clone()
                    masked_conf[~disagree] = float('inf')
                    # Also prefer z where prediction is empty (class 0)
                    is_empty = (pred_3d == 0)  # [H, W, D]
                    # Boost confidence of non-empty voxels (don't override them)
                    masked_conf[~is_empty] = masked_conf[~is_empty] + 1.0
                    z_best = masked_conf.argmin(dim=-1)  # [H, W]
                else:
                    # Fallback: use static class z-range midpoint
                    z_min, z_max = self.z_ranges.get(c, (0, D))
                    z_mid = (z_min + z_max) // 2
                    # Find any empty z near midpoint in the column
                    z_best = torch.full((H, W), z_mid, dtype=torch.long, device=device)

                # Apply: only at disagreeing columns, at z_best
                rows, cols = torch.where(disagree)
                if len(rows) > 0:
                    zs = z_best[rows, cols]
                    fused[b, rows, cols, zs] = c

        return fused

    def forward_soft(self, bev_probs_or_logits: torch.Tensor) -> torch.Tensor:
        """
        Soft lifting using probabilities instead of hard labels.

        Args:
            bev_probs_or_logits: [B, C, H, W] BEV class probabilities (one-hot or soft)

        Returns:
            scene_3d_probs: [B, C, H, W, Z] 3D class probabilities
        """
        B, C, H, W = bev_probs_or_logits.shape
        device = bev_probs_or_logits.device

        # Input is already valid probabilities (one-hot from F.one_hot or soft predictions)
        # Do NOT apply softmax — it would destroy peaked distributions
        # (e.g., one-hot [0,0,1,0,...] → softmax → [0.046,...,0.125,...,0.046])
        bev_probs = bev_probs_or_logits  # [B, C, H, W]

        # Initialize output
        scene_3d_probs = torch.zeros(B, C, H, W, self.num_z, device=device)

        # For each class, spread its probability across its z-range
        for c in range(C):
            z_min, z_max = self.z_ranges.get(c, (0, self.num_z))
            if z_min >= z_max:
                continue  # Skip classes with no z-range (e.g., class 0 = unlabeled)
            z_min = max(0, min(z_min, self.num_z - 1))
            z_max = max(z_min + 1, min(z_max, self.num_z))

            # Uniform distribution across z-range
            for z in range(z_min, z_max):
                scene_3d_probs[:, c, :, :, z] = bev_probs[:, c, :, :]

        # Normalize along class dimension
        scene_3d_probs = scene_3d_probs / (scene_3d_probs.sum(dim=1, keepdim=True) + 1e-8)

        return scene_3d_probs


class LearnableLifter(nn.Module):
    """
    Learnable 2D→3D lifting network.

    Instead of fixed height ranges, learns to lift BEV to 3D.
    """

    def __init__(self,
                 num_classes: int = 20,
                 num_z: int = 32,
                 hidden_dim: int = 128):
        super().__init__()

        self.num_classes = num_classes
        self.num_z = num_z

        # Per-class height predictor
        # Input: class embedding, Output: z-distribution
        self.class_embed = nn.Embedding(num_classes, hidden_dim)

        self.z_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_z),
            nn.Softmax(dim=-1)
        )

        # 2D convolution for context
        self.context_net = nn.Sequential(
            nn.Conv2d(num_classes, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, bev_logits: torch.Tensor) -> torch.Tensor:
        """
        Learnable lifting.

        Args:
            bev_logits: [B, C, H, W] BEV class logits

        Returns:
            scene_3d: [B, C, H, W, Z] 3D class probabilities
        """
        B, C, H, W = bev_logits.shape

        # BEV probabilities
        bev_probs = F.softmax(bev_logits, dim=1)

        # Context features
        context = self.context_net(bev_probs)  # [B, hidden, H, W]

        # For each class, predict z-distribution
        class_ids = torch.arange(C, device=bev_logits.device)
        class_embeds = self.class_embed(class_ids)  # [C, hidden]

        # Z-distribution per class
        z_dist = self.z_predictor(class_embeds)  # [C, Z]

        # Combine: [B, C, H, W] x [C, Z] -> [B, C, H, W, Z]
        scene_3d = bev_probs.unsqueeze(-1) * z_dist.view(1, C, 1, 1, -1)

        return scene_3d


class LiftingModule(nn.Module):
    """
    Complete lifting module for scene completion.

    Combines BEV prediction lifting with feature encoding for conditioning.
    Optionally uses PaSCo's SPCDense3Dv2 for dense hallucination at coarse resolution.
    """

    def __init__(self,
                 num_classes: int = 20,
                 num_z: int = 32,
                 feature_dim: int = 64,
                 learnable: bool = False,
                 use_dense3d: bool = False,
                 dense3d_dropout: float = 0.1):
        """
        Args:
            num_classes: Number of semantic classes
            num_z: Number of Z slices (height)
            feature_dim: Output feature dimension
            learnable: Use learnable lifting vs fixed height priors
            use_dense3d: Enable PaSCo's SPCDense3Dv2 for dense hallucination
            dense3d_dropout: Dropout for dense3d module
        """
        super().__init__()

        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.use_dense3d = use_dense3d

        if learnable:
            self.lifter = LearnableLifter(num_classes, num_z)
        else:
            self.lifter = BEVTo3DLifter(num_classes, num_z)

        # Encode lifted 3D for conditioning
        self.encoder = nn.Sequential(
            nn.Conv3d(num_classes, feature_dim, 3, padding=1),
            nn.BatchNorm3d(feature_dim),
            nn.ReLU(inplace=True),
            nn.Conv3d(feature_dim, feature_dim, 3, padding=1),
            nn.BatchNorm3d(feature_dim),
            nn.ReLU(inplace=True),
        )

        # Optional: PaSCo's SPCDense3Dv2 at coarse resolution for hallucination
        # After encoding, we downsample to 32×32×4, apply dense3d, then upsample back
        self.dense3d = None
        if use_dense3d:
            from gssc.models.dense_3d_cnn import SPCDense3Dv2
            self.dense3d = nn.Sequential(
                SPCDense3Dv2(init_size=feature_dim),
                nn.Dropout3d(dense3d_dropout),
            )
            print(f"[LiftingModule] Enabled SPCDense3Dv2 (feature_dim={feature_dim}, dropout={dense3d_dropout})")

    def forward(self, bev_logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Lift BEV and encode for conditioning.

        Args:
            bev_logits: [B, C, H, W] BEV class logits

        Returns:
            lifted_3d: [B, H, W, Z] lifted semantic labels
            lifted_features: [B, feature_dim, H, W, Z] encoded features
        """
        # Lift to 3D
        if hasattr(self.lifter, 'forward_soft'):
            lifted_probs = self.lifter.forward_soft(bev_logits)  # [B, C, H, W, Z]
        else:
            lifted_probs = self.lifter(bev_logits)

        # Hard labels
        if lifted_probs.dim() == 5:
            lifted_3d = lifted_probs.argmax(dim=1)  # [B, H, W, Z]
            # Permute for 3D conv: [B, C, H, W, Z]
            lifted_features = self.encoder(lifted_probs)
        else:
            lifted_3d = lifted_probs
            # One-hot encode for features
            one_hot = F.one_hot(lifted_3d.long(), self.lifter.num_classes)
            one_hot = one_hot.permute(0, 4, 1, 2, 3).float()  # [B, C, H, W, Z]
            lifted_features = self.encoder(one_hot)

        # Optional: Apply PaSCo's SPCDense3Dv2 at coarse resolution
        # This helps hallucinate features in occluded regions
        if self.use_dense3d and self.dense3d is not None:
            # Save original for skip connection
            original_features = lifted_features

            # Downsample to coarse resolution (1:8 → 32×32×4 for 256×256×32)
            B, C, H, W, D = lifted_features.shape
            coarse_h, coarse_w, coarse_d = H // 8, W // 8, max(1, D // 8)
            coarse_features = F.adaptive_avg_pool3d(lifted_features, (coarse_h, coarse_w, coarse_d))

            # Apply SPCDense3Dv2 for multi-scale dense hallucination
            hallucinated = self.dense3d(coarse_features)

            # Upsample back to original resolution
            hallucinated = F.interpolate(hallucinated, size=(H, W, D), mode='trilinear', align_corners=False)

            # Combine with skip connection (residual)
            lifted_features = original_features + hallucinated

        return lifted_3d, lifted_features


def compute_class_z_statistics(data_root: str, sequences: list) -> dict[int, tuple[float, float]]:
    """
    Compute z-range statistics for each class from data.

    Useful for calibrating the CLASS_Z_RANGES dict.
    """
    from pathlib import Path

    from tqdm import tqdm

    class_z_min = {c: float('inf') for c in range(20)}
    class_z_max = {c: float('-inf') for c in range(20)}
    class_counts = {c: 0 for c in range(20)}

    data_root = Path(data_root)

    for seq in sequences:
        voxels_path = data_root / 'sequences' / seq / 'voxels'
        if not voxels_path.exists():
            continue

        for label_file in tqdm(list(voxels_path.glob('*.label')), desc=f"Seq {seq}"):
            labels = np.fromfile(str(label_file), dtype=np.uint16).reshape(256, 256, 32)

            for c in range(1, 20):
                mask = labels == c
                if mask.any():
                    z_coords = np.where(mask)[2]
                    class_z_min[c] = min(class_z_min[c], z_coords.min())
                    class_z_max[c] = max(class_z_max[c], z_coords.max())
                    class_counts[c] += mask.sum()

    # Print results
    print("\nClass Z-Range Statistics:")
    for c in range(1, 20):
        if class_counts[c] > 0:
            print(f"  Class {c}: z=[{class_z_min[c]}, {class_z_max[c]}], count={class_counts[c]}")

    return {c: (int(class_z_min[c]), int(class_z_max[c]) + 1)
            for c in range(20) if class_counts[c] > 0}
