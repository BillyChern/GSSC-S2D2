"""
D3PM: Discrete Denoising Diffusion Probabilistic Models

Implements discrete diffusion for categorical data with:
- Absorbing state (mask) transition
- Uniform transition (optional)
- ELBO + auxiliary CE loss

Reference: https://arxiv.org/abs/2107.03006
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Dict


class D3PM(nn.Module):
    """
    Base D3PM class for discrete diffusion on categorical data.

    Args:
        num_classes: Number of semantic classes (e.g., 20 for SemanticKITTI)
        num_timesteps: Number of diffusion steps (default: 1000)
        transition_type: 'absorbing' (mask) or 'uniform'
        schedule_type: 'linear' or 'cosine' noise schedule
        aux_loss_weight: Weight for auxiliary cross-entropy loss (default: 0.001)
    """

    def __init__(
        self,
        num_classes: int = 20,
        num_timesteps: int = 1000,
        transition_type: str = 'absorbing',
        schedule_type: str = 'cosine',
        aux_loss_weight: float = 0.001,
        class_weights: Optional[torch.Tensor] = None,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.num_timesteps = num_timesteps
        self.transition_type = transition_type
        self.schedule_type = schedule_type
        self.aux_loss_weight = aux_loss_weight

        # For absorbing state, we add a [MASK] token as class num_classes
        if transition_type == 'absorbing':
            self.num_tokens = num_classes + 1  # +1 for [MASK]
            self.mask_token = num_classes
        else:
            self.num_tokens = num_classes
            self.mask_token = None

        # Class weights for imbalanced data
        if class_weights is not None:
            self.register_buffer('class_weights', class_weights)
        else:
            self.register_buffer('class_weights', torch.ones(num_classes))

        # Build noise schedule
        self._build_schedule()

    def _build_schedule(self):
        """Build the noise schedule (beta_t values)."""
        if self.schedule_type == 'linear':
            # Linear schedule
            betas = np.linspace(1e-4, 0.02, self.num_timesteps)
        elif self.schedule_type == 'cosine':
            # Cosine schedule (better for discrete data)
            steps = np.arange(self.num_timesteps + 1)
            alpha_bar = np.cos((steps / self.num_timesteps + 0.008) / 1.008 * np.pi / 2) ** 2
            alpha_bar = alpha_bar / alpha_bar[0]
            betas = 1 - alpha_bar[1:] / alpha_bar[:-1]
            betas = np.clip(betas, 0, 0.999)
        else:
            raise ValueError(f"Unknown schedule type: {self.schedule_type}")

        # Convert to cumulative probabilities
        # For absorbing: probability of being masked by time t
        # For uniform: probability of being corrupted by time t
        alphas = 1 - betas
        alpha_cumprod = np.cumprod(alphas)

        # Register as buffers
        self.register_buffer('betas', torch.tensor(betas, dtype=torch.float32))
        self.register_buffer('alphas', torch.tensor(alphas, dtype=torch.float32))
        self.register_buffer('alpha_cumprod', torch.tensor(alpha_cumprod, dtype=torch.float32))

        # For absorbing state: probability of being masked
        # q(x_t = [MASK] | x_0) = 1 - alpha_bar_t
        self.register_buffer('mask_prob', 1 - torch.tensor(alpha_cumprod, dtype=torch.float32))

    def q_sample(
        self,
        x_0: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward diffusion: sample x_t given x_0 and timestep t.

        For absorbing state: randomly mask tokens with probability mask_prob[t]
        For uniform: randomly replace tokens with uniform distribution

        Args:
            x_0: Original labels [B, H, W], values in [0, num_classes-1]
            t: Timesteps [B]
            noise: Optional pre-generated noise

        Returns:
            x_t: Noised labels [B, H, W]
        """
        batch_size = x_0.shape[0]

        if self.transition_type == 'absorbing':
            # Get mask probability for each sample in batch
            mask_prob = self.mask_prob[t]  # [B]
            mask_prob = mask_prob.view(batch_size, 1, 1)  # [B, 1, 1]

            # Generate random mask
            if noise is None:
                noise = torch.rand_like(x_0.float())
            mask = noise < mask_prob

            # Apply mask
            x_t = torch.where(mask, torch.full_like(x_0, self.mask_token), x_0)

        elif self.transition_type == 'uniform':
            # Get corruption probability
            corrupt_prob = 1 - self.alpha_cumprod[t]
            corrupt_prob = corrupt_prob.view(batch_size, 1, 1)

            # Generate random corruption
            if noise is None:
                noise = torch.rand_like(x_0.float())
            corrupt_mask = noise < corrupt_prob

            # Random replacement with uniform distribution
            random_labels = torch.randint(
                0, self.num_classes, x_0.shape,
                device=x_0.device, dtype=x_0.dtype
            )
            x_t = torch.where(corrupt_mask, random_labels, x_0)
        else:
            raise ValueError(f"Unknown transition type: {self.transition_type}")

        return x_t

    def compute_loss(
        self,
        model: nn.Module,
        x_0: torch.Tensor,
        condition: torch.Tensor,
        t: Optional[torch.Tensor] = None,
        prev_pred: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute D3PM training loss.

        Args:
            model: The denoising network (predicts x_0 from x_t)
            x_0: Ground truth labels [B, H, W]
            condition: LiDAR conditioning features [B, C, H, W]
            t: Optional timesteps (randomly sampled if None)
            prev_pred: Previous prediction for self-conditioning

        Returns:
            Dictionary with loss components
        """
        batch_size = x_0.shape[0]
        device = x_0.device

        # Sample timesteps if not provided
        if t is None:
            t = torch.randint(0, self.num_timesteps, (batch_size,), device=device)

        # Forward diffusion: get noised labels
        x_t = self.q_sample(x_0, t)

        # Self-conditioning: 50% of time use previous prediction
        if prev_pred is None and self.training and torch.rand(1).item() < 0.5:
            with torch.no_grad():
                prev_pred = model(x_t, t, condition, prev_pred=None)
                if isinstance(prev_pred, tuple):
                    prev_pred = prev_pred[0]
                prev_pred = prev_pred.argmax(dim=1)

        # Model prediction
        logits = model(x_t, t, condition, prev_pred=prev_pred)
        if isinstance(logits, tuple):
            logits, aux_logits = logits
        else:
            aux_logits = None

        # Compute cross-entropy loss (x_0 parameterization)
        # Use uniform timestep weighting (no mask_prob weighting)
        # Previous mask_prob weighting caused near-zero gradients at low timesteps
        if self.transition_type == 'absorbing':
            # The model predicts over num_classes (not num_tokens)
            ce_loss = F.cross_entropy(
                logits.view(-1, self.num_classes),
                x_0.view(-1),
                weight=self.class_weights,
                reduction='mean'  # Uniform weighting across all timesteps
            )
        else:
            ce_loss = F.cross_entropy(
                logits.view(-1, self.num_classes),
                x_0.view(-1),
                weight=self.class_weights,
            )

        # ELBO approximation loss
        elbo_loss = ce_loss  # Simplified ELBO

        # Auxiliary loss
        aux_loss = torch.tensor(0.0, device=device)
        if aux_logits is not None:
            aux_loss = F.cross_entropy(
                aux_logits.view(-1, self.num_classes),
                x_0.view(-1),
                weight=self.class_weights,
            )

        # Total loss
        total_loss = elbo_loss + self.aux_loss_weight * aux_loss

        return {
            'loss': total_loss,
            'elbo_loss': elbo_loss,
            'aux_loss': aux_loss,
            'ce_loss': ce_loss,
        }

    @torch.no_grad()
    def sample(
        self,
        model: nn.Module,
        condition: torch.Tensor,
        num_steps: Optional[int] = None,
        temperature: float = 1.0,
        use_self_conditioning: bool = True,
    ) -> torch.Tensor:
        """
        Generate samples via reverse diffusion.

        Args:
            model: The denoising network
            condition: LiDAR conditioning [B, C, H, W]
            num_steps: Number of denoising steps (default: num_timesteps)
            temperature: Sampling temperature
            use_self_conditioning: Whether to use self-conditioning

        Returns:
            Predicted labels [B, H, W]
        """
        if num_steps is None:
            num_steps = self.num_timesteps

        batch_size = condition.shape[0]
        H, W = condition.shape[2], condition.shape[3]
        device = condition.device

        # Start from fully masked (absorbing) or random (uniform)
        if self.transition_type == 'absorbing':
            x_t = torch.full((batch_size, H, W), self.mask_token, device=device, dtype=torch.long)
        else:
            x_t = torch.randint(0, self.num_classes, (batch_size, H, W), device=device)

        # Timesteps for sampling (can use fewer steps)
        if num_steps < self.num_timesteps:
            timesteps = torch.linspace(
                self.num_timesteps - 1, 0, num_steps,
                device=device
            ).long()
        else:
            timesteps = torch.arange(self.num_timesteps - 1, -1, -1, device=device)

        prev_pred = None

        for t_idx, t in enumerate(timesteps):
            t_batch = t.expand(batch_size)

            # Model prediction
            logits = model(x_t, t_batch, condition, prev_pred=prev_pred)
            if isinstance(logits, tuple):
                logits = logits[0]

            # Apply temperature
            logits = logits / temperature

            # Sample from predicted distribution
            probs = F.softmax(logits, dim=1)  # [B, C, H, W]

            # Gumbel-max sampling
            gumbel = -torch.log(-torch.log(torch.rand_like(probs) + 1e-8) + 1e-8)
            pred = (torch.log(probs + 1e-8) + gumbel).argmax(dim=1)  # [B, H, W]

            # For absorbing: only update masked positions
            if self.transition_type == 'absorbing':
                # Compute how many tokens to unmask at this step
                if t_idx < len(timesteps) - 1:
                    t_next = timesteps[t_idx + 1]
                    unmask_prob = self.mask_prob[t] - self.mask_prob[t_next]
                else:
                    unmask_prob = self.mask_prob[t]

                # Only unmask a subset
                unmask_mask = (x_t == self.mask_token) & (torch.rand_like(x_t.float()) < unmask_prob)
                x_t = torch.where(unmask_mask, pred, x_t)
            else:
                x_t = pred

            # Self-conditioning
            if use_self_conditioning:
                prev_pred = pred

        # Final prediction
        if self.transition_type == 'absorbing':
            # Any remaining [MASK] tokens: use final prediction
            mask = x_t == self.mask_token
            logits = model(x_t, torch.zeros(batch_size, device=device, dtype=torch.long), condition, prev_pred=prev_pred)
            if isinstance(logits, tuple):
                logits = logits[0]
            final_pred = logits.argmax(dim=1)
            x_t = torch.where(mask, final_pred, x_t)

        return x_t


class AbsorbingD3PM(D3PM):
    """Convenience class for absorbing state D3PM."""

    def __init__(self, num_classes: int = 20, num_timesteps: int = 1000, **kwargs):
        super().__init__(
            num_classes=num_classes,
            num_timesteps=num_timesteps,
            transition_type='absorbing',
            **kwargs
        )


class UniformD3PM(D3PM):
    """Convenience class for uniform transition D3PM."""

    def __init__(self, num_classes: int = 20, num_timesteps: int = 1000, **kwargs):
        super().__init__(
            num_classes=num_classes,
            num_timesteps=num_timesteps,
            transition_type='uniform',
            **kwargs
        )
