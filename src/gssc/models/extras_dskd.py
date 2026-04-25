"""
S3: Dense-to-Sparse Knowledge Distillation (DSKD) for Scene Completion.

Matches pairwise feature relationships between teacher and student models.

Reference:
- SCPNet (CVPR 2023): Semantic Scene Completion with Knowledge Distillation
- Paper formula (Eq. 1-2):
    P(i,j) = F(i)^T·F(j) / (||F(i)||₂·||F(j)||₂)  # Cosine similarity
    L_dskd = (1/N²) Σᵢ Σⱼ ||P_S(i,j) - P_T(i,j)||²₂  # L2/MSE loss

Key SCPNet insight:
- Teacher features: F_T ∈ ℝ^(N_m × C_f) where N_m = non-empty voxels in multi-frame
- Student features: F_S ∈ ℝ^(N_s × C_f) where N_s = non-empty voxels in single-frame
- CRITICAL: "indices are sorted and ℐ_S(i,j) = ℐ_T(i,j)"
- Only compute similarity at SHARED occupied locations (intersection)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class DSKDLoss3D(nn.Module):
    """
    Dense-to-Sparse Knowledge Distillation loss for 3D features.

    SCPNet-style: Computes pairwise similarity only at occupied locations.
    This focuses the learning signal on meaningful regions.

    Reference: SCPNet paper (CVPR 2023)
    """

    def __init__(
        self,
        normalize: bool = True,
        temperature: float = 1.0,
        use_occupancy_mask: bool = True,
        max_samples: int = 4096,  # Max voxels to sample (for memory)
    ):
        """
        Args:
            normalize: Whether to L2-normalize features (required for cosine similarity)
            temperature: Temperature for similarity scaling (default 1.0, no scaling)
            use_occupancy_mask: Whether to compute similarity only at occupied locations
            max_samples: Maximum number of voxels to sample (for memory efficiency)
        """
        super().__init__()
        self.normalize = normalize
        self.temperature = temperature
        self.use_occupancy_mask = use_occupancy_mask
        self.max_samples = max_samples

    def forward(
        self,
        student_features: torch.Tensor,
        teacher_features: torch.Tensor,
        student_occupancy: Optional[torch.Tensor] = None,
        teacher_occupancy: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute DSKD loss using SCPNet's pairwise similarity approach.

        Args:
            student_features: [B, C, H, W, D] student bottleneck features
            teacher_features: [B, C, H, W, D] teacher bottleneck features (detached)
            student_occupancy: [B, 1, H', W', D'] student lidar occupancy (optional)
            teacher_occupancy: [B, 1, H', W', D'] teacher lidar occupancy (optional)

        Returns:
            loss: scalar DSKD loss
        """
        B, C, H, W, D = student_features.shape

        # If occupancy masks provided, compute at shared occupied locations (SCPNet-style)
        if self.use_occupancy_mask and student_occupancy is not None and teacher_occupancy is not None:
            loss = self._forward_with_occupancy(
                student_features, teacher_features,
                student_occupancy, teacher_occupancy
            )
        else:
            # Dense computation (all locations)
            loss = self._forward_dense(student_features, teacher_features)

        return loss

    def _forward_dense(
        self,
        student_features: torch.Tensor,
        teacher_features: torch.Tensor
    ) -> torch.Tensor:
        """Dense DSKD: compute similarity over all spatial locations."""
        if student_features.dim() == 5:
            B, C, H, W, D = student_features.shape
            S = student_features.flatten(2)  # [B, C, N] where N = H*W*D
            T = teacher_features.flatten(2)
        elif student_features.dim() == 4:
            B, C, H, W = student_features.shape
            S = student_features.flatten(2)
            T = teacher_features.flatten(2)
        else:
            raise ValueError(f"Expected 4D or 5D features, got {student_features.dim()}D")

        # L2 normalize along channel dimension
        if self.normalize:
            S = F.normalize(S, dim=1)
            T = F.normalize(T, dim=1)

        # Pairwise cosine similarity: P = S^T @ S
        P_S = torch.bmm(S.transpose(1, 2), S) / self.temperature  # [B, N, N]
        P_T = torch.bmm(T.transpose(1, 2), T) / self.temperature

        # MSE loss on similarity matrices
        loss = F.mse_loss(P_S, P_T)

        return loss

    def _forward_with_occupancy(
        self,
        student_features: torch.Tensor,
        teacher_features: torch.Tensor,
        student_occupancy: torch.Tensor,
        teacher_occupancy: torch.Tensor,
    ) -> torch.Tensor:
        """
        SCPNet-style DSKD: compute similarity only at shared occupied locations.

        This focuses the learning signal on semantically meaningful regions.
        """
        B, C, H, W, D = student_features.shape

        # Downsample occupancy masks to match feature resolution
        # Features are at bottleneck (e.g., 16×16×2), occupancy at input (256×256×32)
        occ_H, occ_W, occ_D = student_occupancy.shape[2:]

        if (occ_H, occ_W, occ_D) != (H, W, D):
            # Downsample with max pooling (any occupied → occupied)
            scale_factor = (occ_H // H, occ_W // W, occ_D // D)
            student_occ_down = F.max_pool3d(
                student_occupancy.float(),
                kernel_size=scale_factor,
                stride=scale_factor
            ) > 0.5
            teacher_occ_down = F.max_pool3d(
                teacher_occupancy.float(),
                kernel_size=scale_factor,
                stride=scale_factor
            ) > 0.5
        else:
            student_occ_down = student_occupancy > 0.5
            teacher_occ_down = teacher_occupancy > 0.5

        # Compute intersection: occupied in BOTH teacher and student
        # SCPNet: "indices are sorted and ℐ_S(i,j) = ℐ_T(i,j)"
        shared_mask = (student_occ_down & teacher_occ_down).squeeze(1)  # [B, H, W, D]

        total_loss = 0.0
        valid_batches = 0

        for b in range(B):
            mask_b = shared_mask[b]  # [H, W, D]
            num_occupied = mask_b.sum().item()

            if num_occupied < 2:
                # Need at least 2 voxels for pairwise similarity
                continue

            # Extract features at occupied locations
            # Flatten mask and features
            mask_flat = mask_b.flatten()  # [N]
            indices = torch.where(mask_flat)[0]  # [num_occupied]

            # Sample if too many (for memory)
            if len(indices) > self.max_samples:
                perm = torch.randperm(len(indices), device=indices.device)[:self.max_samples]
                indices = indices[perm]

            # Extract features: [C, H, W, D] -> [C, N] -> [C, num_occupied]
            S_flat = student_features[b].flatten(1)  # [C, H*W*D]
            T_flat = teacher_features[b].flatten(1)

            S_occ = S_flat[:, indices]  # [C, num_occupied]
            T_occ = T_flat[:, indices]

            # Normalize
            if self.normalize:
                S_occ = F.normalize(S_occ, dim=0)  # Normalize along C
                T_occ = F.normalize(T_occ, dim=0)

            # Pairwise similarity: [num_occupied, num_occupied]
            P_S = torch.mm(S_occ.t(), S_occ) / self.temperature
            P_T = torch.mm(T_occ.t(), T_occ) / self.temperature

            # MSE loss for this batch
            loss_b = F.mse_loss(P_S, P_T)
            total_loss += loss_b
            valid_batches += 1

        if valid_batches == 0:
            # Fallback to dense if no valid batches
            return self._forward_dense(student_features, teacher_features)

        return total_loss / valid_batches


def create_dskd_loss(
    normalize: bool = True,
    temperature: float = 1.0,
    use_occupancy_mask: bool = True,
    max_samples: int = 4096,
) -> DSKDLoss3D:
    """Create DSKD loss for scene completion."""
    return DSKDLoss3D(
        normalize=normalize,
        temperature=temperature,
        use_occupancy_mask=use_occupancy_mask,
        max_samples=max_samples,
    )
