"""
3D Lovász-Softmax Loss for Semantic Scene Completion.

Lovász-Softmax directly optimizes the mean IoU metric via a smooth surrogate.
Ported from the 2D BEV implementation in bev_diffusion/sparse_bev_net.py.

Reference: Berman et al., "The Lovász-Softmax loss: A tractable surrogate
for the optimization of the intersection-over-union measure in neural networks"
CVPR 2018.

Usage:
    loss_fn = LovaszSoftmax3D(ignore_index=0)
    loss = loss_fn(logits, targets)
    # logits: [B, C, H, W, D] raw model output
    # targets: [B, H, W, D] class indices
"""

import torch
import torch.nn.functional as F


class LovaszSoftmax3D(torch.nn.Module):
    """Multi-class Lovász-Softmax loss for 3D voxel predictions.

    Optimizes a differentiable surrogate of mean IoU.
    Only computes loss on non-empty voxels (class > 0) for efficiency.
    """

    def __init__(self, ignore_index: int = 0):
        super().__init__()
        self.ignore_index = ignore_index

    def lovasz_grad(self, gt_sorted: torch.Tensor) -> torch.Tensor:
        """Compute gradient of the Lovász extension w.r.t sorted errors.

        Args:
            gt_sorted: [P] sorted binary ground truth
        Returns:
            [P] Lovász gradient weights
        """
        gts = gt_sorted.sum()
        intersection = gts - gt_sorted.float().cumsum(0)
        union = gts + (1 - gt_sorted).float().cumsum(0)
        jaccard = 1.0 - intersection / union
        if len(jaccard) > 1:
            jaccard[1:] = jaccard[1:] - jaccard[:-1]
        return jaccard

    def lovasz_softmax_flat(self, probas: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Multi-class Lovász-Softmax loss on flattened predictions.

        Args:
            probas: [P, C] class probabilities (softmax output)
            labels: [P] ground truth class indices
        Returns:
            scalar loss
        """
        C = probas.size(1)
        losses = []
        for c in range(C):
            if c == self.ignore_index:
                continue
            fg = (labels == c).float()
            if fg.sum() == 0:
                continue
            errors = (fg - probas[:, c]).abs()
            errors_sorted, perm = torch.sort(errors, descending=True)
            fg_sorted = fg[perm]
            grad = self.lovasz_grad(fg_sorted)
            losses.append((errors_sorted * grad).sum())

        if len(losses) == 0:
            return probas.sum() * 0.0  # Return 0 with gradient
        return torch.stack(losses).mean()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute 3D Lovász-Softmax loss.

        Args:
            logits: [B, C, H, W, D] raw model output (before softmax)
            targets: [B, H, W, D] ground truth class indices
        Returns:
            scalar loss value
        """
        probas = F.softmax(logits, dim=1)

        # Flatten: [B, C, H, W, D] → [B*H*W*D, C]
        # Note: This creates a 2M×20 tensor (40M floats) before filtering to ~100K
        # non-empty voxels. The temporary allocation is acceptable (~160MB) and the
        # boolean masking is fast. Alternative (sparse-first) would complicate the API.
        B, C, H, W, D = probas.shape
        probas = probas.permute(0, 2, 3, 4, 1).contiguous().view(-1, C)
        targets = targets.view(-1)

        # Remove ignored voxels (class 0 = empty/unlabeled)
        valid = targets != self.ignore_index
        probas = probas[valid]
        targets = targets[valid]

        if len(targets) == 0:
            return logits.sum() * 0.0

        return self.lovasz_softmax_flat(probas, targets)
