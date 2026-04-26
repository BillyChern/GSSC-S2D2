"""
Class-Balancing Diffusion Models (CBDM)

Implementation based on "Class-Balancing Diffusion Models" (CVPR 2023).

The key insight is that standard diffusion models trained on imbalanced data:
1. Generate fewer samples of rare classes
2. Show mode collapse for tail classes
3. Have lower diversity in rare class generations

This module provides:
1. ClassBalancingRegularizer - Regularization loss for training
2. RareClassGuidance - Sampling guidance for rare classes
3. Inverse frequency class weighting utilities

Reference: https://arxiv.org/abs/2305.00562
"""


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# SemanticKITTI class statistics (from analysis)
SEMANTICKITTI_CLASS_COUNTS = {
    0: 5780876,   # unlabeled
    1: 526657,    # car
    2: 10,        # bicycle (EXTREME)
    3: 48,        # motorcycle (EXTREME)
    4: 0,         # truck (EXTREME)
    5: 11500,     # other-vehicle
    6: 248,       # person (EXTREME)
    7: 0,         # bicyclist (EXTREME)
    8: 4944,      # motorcyclist (EXTREME)
    9: 3568653,   # road (HEAD)
    10: 342892,   # parking
    11: 1378581,  # sidewalk (HEAD)
    12: 138036,   # other-ground
    13: 313756,   # building
    14: 138344,   # fence
    15: 2407829,  # vegetation (HEAD)
    16: 33960,    # trunk
    17: 1716345,  # terrain (HEAD)
    18: 12980,    # pole
    19: 8341,     # traffic-sign
}

# Classes with extremely low counts
EXTREME_RARE_CLASSES = [2, 3, 4, 6, 7, 8]  # bicycle, motorcycle, truck, person, bicyclist, motorcyclist
RARE_CLASSES = [2, 3, 4, 5, 6, 7, 8, 16, 18, 19]  # All rare classes
HEAD_CLASSES = [9, 11, 15, 17]  # road, sidewalk, vegetation, terrain


def compute_class_weights(
    class_counts: dict[int, int],
    num_classes: int = 20,
    method: str = 'inverse_freq',
    smoothing: float = 1.0,
    min_weight: float = 0.1,
    max_weight: float = 10.0,
) -> torch.Tensor:
    """
    Compute class weights for balancing.

    Args:
        class_counts: Dict of {class_id: count}
        num_classes: Total number of classes
        method: 'inverse_freq', 'sqrt_inverse', 'effective_num'
        smoothing: Smoothing factor
        min_weight: Minimum weight
        max_weight: Maximum weight

    Returns:
        weights: [num_classes] tensor of weights
    """
    counts = torch.zeros(num_classes)
    for cls, cnt in class_counts.items():
        counts[cls] = max(cnt, 1)  # Avoid division by zero

    total = counts.sum()

    if method == 'inverse_freq':
        # Standard inverse frequency
        weights = total / (counts * num_classes)

    elif method == 'sqrt_inverse':
        # Square root dampening (less aggressive)
        freq = counts / total
        weights = 1.0 / torch.sqrt(freq + smoothing)

    elif method == 'effective_num':
        # Effective number of samples (from Class-Balanced Loss paper)
        beta = 0.9999
        effective_num = 1.0 - torch.pow(beta, counts)
        weights = (1.0 - beta) / (effective_num + 1e-8)

    else:
        raise ValueError(f"Unknown method: {method}")

    # Normalize
    weights = weights / weights.mean()

    # Clip to range
    weights = weights.clamp(min_weight, max_weight)

    return weights


class ClassBalancingRegularizer(nn.Module):
    """
    Class-balancing regularizer for diffusion training.

    Adds a KL divergence term that encourages broader class distribution
    in generated samples, especially at later diffusion timesteps.
    """

    def __init__(
        self,
        num_classes: int = 20,
        beta: float = 0.1,
        class_weights: torch.Tensor | None = None,
        target_distribution: str = 'weighted_uniform',
        timestep_scaling: str = 'linear',
    ):
        """
        Args:
            num_classes: Number of semantic classes
            beta: Regularization strength
            class_weights: Optional precomputed class weights
            target_distribution: 'uniform' or 'weighted_uniform'
            timestep_scaling: 'linear', 'cosine', or 'constant'
        """
        super().__init__()
        self.num_classes = num_classes
        self.beta = beta
        self.timestep_scaling = timestep_scaling

        # Compute or use provided weights
        if class_weights is None:
            class_weights = compute_class_weights(
                SEMANTICKITTI_CLASS_COUNTS,
                num_classes=num_classes,
                method='sqrt_inverse'
            )

        self.register_buffer('class_weights', class_weights)

        # Target distribution
        if target_distribution == 'uniform':
            target = torch.ones(num_classes) / num_classes
        else:
            # Weighted uniform - higher probability for rare classes
            target = class_weights / class_weights.sum()

        self.register_buffer('target_distribution', target)

    def compute_timestep_weight(self, t: torch.Tensor, T: int) -> torch.Tensor:
        """
        Compute timestep-dependent weight.

        Higher weight at later timesteps (when structure is forming).
        """
        t_normalized = t.float() / T

        if self.timestep_scaling == 'linear':
            return t_normalized

        elif self.timestep_scaling == 'cosine':
            return 0.5 * (1 + torch.cos(np.pi * (1 - t_normalized)))

        elif self.timestep_scaling == 'constant':
            return torch.ones_like(t_normalized)

        else:
            return t_normalized

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        T: int,
        ignore_class: int = 0,
    ) -> torch.Tensor:
        """
        Compute class-balancing regularization loss.

        Args:
            x_t: [B, X, Y, Z] or [B, C, X, Y, Z] - current samples (labels or logits)
            t: [B] - timesteps
            T: Total timesteps
            ignore_class: Class to ignore (usually 0 = unlabeled)

        Returns:
            loss: Scalar regularization loss
        """
        # Handle both label and logit inputs
        if x_t.dim() == 5:
            # Logits: [B, C, X, Y, Z] -> get predicted labels
            x_labels = x_t.argmax(dim=1)
        else:
            x_labels = x_t

        # Compute batch class distribution
        batch_dist = torch.zeros(self.num_classes, device=x_t.device)
        total_valid = 0

        for cls in range(self.num_classes):
            if cls == ignore_class:
                continue
            count = (x_labels == cls).sum().float()
            batch_dist[cls] = count
            total_valid += count

        # Normalize to probability
        if total_valid > 0:
            batch_dist = batch_dist / total_valid

        # Add small epsilon for numerical stability
        batch_dist = batch_dist + 1e-8
        batch_dist = batch_dist / batch_dist.sum()

        # KL divergence from target distribution
        kl_div = F.kl_div(
            batch_dist.log(),
            self.target_distribution.to(x_t.device),
            reduction='batchmean'
        )

        # Scale by timestep
        timestep_weight = self.compute_timestep_weight(t, T).mean()

        return self.beta * timestep_weight * kl_div


class RareClassGuidance:
    """
    Guidance for boosting rare class generation during sampling.
    """

    def __init__(
        self,
        rare_classes: list[int] = None,
        boost_factor: float = 2.0,
        temperature: float = 1.0,
    ):
        """
        Args:
            rare_classes: List of rare class IDs to boost
            boost_factor: Multiplicative factor for rare class logits
            temperature: Sampling temperature
        """
        self.rare_classes = rare_classes or RARE_CLASSES
        self.boost_factor = boost_factor
        self.temperature = temperature

    def apply(
        self,
        logits: torch.Tensor,
        timestep: int | None = None,
        total_timesteps: int | None = None,
    ) -> torch.Tensor:
        """
        Apply rare class guidance to logits.

        Args:
            logits: [B, C, ...] prediction logits
            timestep: Current timestep (optional, for adaptive boosting)
            total_timesteps: Total timesteps (optional)

        Returns:
            guided_logits: [B, C, ...] with boosted rare classes
        """
        guided = logits.clone()

        # Adaptive boost factor based on timestep
        if timestep is not None and total_timesteps is not None:
            # Stronger boost at later timesteps (structure forming)
            t_ratio = timestep / total_timesteps
            adaptive_boost = 1.0 + (self.boost_factor - 1.0) * t_ratio
        else:
            adaptive_boost = self.boost_factor

        # Boost rare class logits
        for cls in self.rare_classes:
            guided[:, cls] = guided[:, cls] * adaptive_boost

        # Apply temperature
        guided = guided / self.temperature

        return guided


class ClassAwareLoss(nn.Module):
    """
    Combined loss with class-balancing for diffusion training.

    L_total = L_diffusion + β * L_cbdm + γ * L_focal
    """

    def __init__(
        self,
        num_classes: int = 20,
        cbdm_beta: float = 0.1,
        focal_gamma: float = 2.0,
        use_focal: bool = True,
        class_weights: torch.Tensor | None = None,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.use_focal = use_focal

        # Class-balancing regularizer
        self.cbdm = ClassBalancingRegularizer(
            num_classes=num_classes,
            beta=cbdm_beta,
            class_weights=class_weights,
        )

        # Focal loss parameters
        self.focal_gamma = focal_gamma

        # Class weights for CE loss
        if class_weights is None:
            class_weights = compute_class_weights(
                SEMANTICKITTI_CLASS_COUNTS,
                num_classes=num_classes,
            )
        self.register_buffer('class_weights', class_weights)

    def focal_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        ignore_index: int = 255,
    ) -> torch.Tensor:
        """
        Focal loss for handling class imbalance.

        FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
        """
        ce_loss = F.cross_entropy(
            logits,
            targets,
            weight=self.class_weights,
            ignore_index=ignore_index,
            reduction='none'
        )

        # Get probabilities
        probs = F.softmax(logits, dim=1)
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        pt = torch.where(targets == ignore_index, torch.ones_like(pt), pt)

        # Focal weight
        focal_weight = (1 - pt) ** self.focal_gamma

        loss = focal_weight * ce_loss
        return loss.mean()

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        t: torch.Tensor,
        T: int,
        diffusion_loss: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Compute combined loss.

        Args:
            logits: [B, C, X, Y, Z] prediction logits
            targets: [B, X, Y, Z] target labels
            t: [B] timesteps
            T: Total timesteps
            diffusion_loss: Optional precomputed diffusion loss

        Returns:
            Dict with loss components
        """
        losses = {}

        # Standard CE or Focal loss
        if self.use_focal:
            losses['ce'] = self.focal_loss(logits, targets)
        else:
            losses['ce'] = F.cross_entropy(
                logits,
                targets,
                weight=self.class_weights,
                ignore_index=255,
            )

        # Class-balancing regularizer
        losses['cbdm'] = self.cbdm(logits, t, T)

        # Total
        losses['total'] = losses['ce'] + losses['cbdm']

        if diffusion_loss is not None:
            losses['diffusion'] = diffusion_loss
            losses['total'] = losses['total'] + diffusion_loss

        return losses


def get_class_balanced_sampler(
    dataset,
    class_counts: dict[int, int] = None,
    num_samples: int | None = None,
) -> torch.utils.data.Sampler:
    """
    Create a class-balanced sampler for DataLoader.

    Samples are weighted by inverse class frequency so rare classes
    are sampled more often.
    """
    if class_counts is None:
        class_counts = SEMANTICKITTI_CLASS_COUNTS

    # Compute sample weights
    weights = compute_class_weights(class_counts, method='inverse_freq')

    # Create per-sample weights based on dominant class in each sample
    sample_weights = []
    for idx in range(len(dataset)):
        # Get sample's dominant class (or average weight)
        # This is dataset-specific implementation
        try:
            _, label = dataset[idx]
            dominant_class = label.mode().values.item()
            sample_weights.append(weights[dominant_class].item())
        except (IndexError, KeyError, RuntimeError, AttributeError):
            sample_weights.append(1.0)

    sample_weights = torch.tensor(sample_weights)

    if num_samples is None:
        num_samples = len(dataset)

    return torch.utils.data.WeightedRandomSampler(
        weights=sample_weights,
        num_samples=num_samples,
        replacement=True
    )


if __name__ == '__main__':
    # Test class weights computation
    print("Testing class balancing...")

    weights = compute_class_weights(SEMANTICKITTI_CLASS_COUNTS, method='sqrt_inverse')
    print("\nClass weights (sqrt_inverse):")
    for i, w in enumerate(weights):
        print(f"  Class {i:2d}: {w:.4f}")

    # Test regularizer
    reg = ClassBalancingRegularizer(beta=0.1)
    x_t = torch.randint(0, 20, (4, 32, 32, 8))
    t = torch.randint(0, 100, (4,))

    loss = reg(x_t, t, T=100)
    print(f"\nCBDM regularization loss: {loss.item():.6f}")

    # Test guidance
    guidance = RareClassGuidance(boost_factor=2.0)
    logits = torch.randn(2, 20, 32, 32, 8)
    guided = guidance.apply(logits, timestep=50, total_timesteps=100)
    print(f"\nRare class boost applied. Shape: {guided.shape}")

    print("\nClass balancing tests passed!")
