"""
Multinomial Diffusion for 3D Semantic Voxel Generation

Based on "Argmax Flows and Multinomial Diffusion" (Hoogeboom et al., 2021)
and the architecture description in the notes.

Key formulas:
- Forward: q(x_t|x_{t-1}) = Cat(x_t | (1-β_t)x_{t-1} + β_t/K)
- q(x_t|x_0) = Cat(x_t | α_t*x_0 + (1-α_t)/K)
- Neural net predicts x̂_0 directly (not noise)
- Loss: KL(q(x_{t-1}|x_t, x_0) || p(x_{t-1}|x_t))

Training with 100 timesteps (as per paper).
"""


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


def log_1_min_a(a: torch.Tensor) -> torch.Tensor:
    """Compute log(1 - exp(a)) numerically stable."""
    return torch.log1p(-torch.exp(a))


def log_add_exp(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Compute log(exp(a) + exp(b)) numerically stable."""
    maximum = torch.max(a, b)
    return maximum + torch.log(torch.exp(a - maximum) + torch.exp(b - maximum))


def extract(a: torch.Tensor, t: torch.Tensor, x_shape: tuple[int, ...]) -> torch.Tensor:
    """Extract values from a at timestep t and reshape for broadcasting."""
    B = t.shape[0]
    out = a.gather(-1, t)
    return out.reshape(B, *((1,) * (len(x_shape) - 1)))


class MultinomialDiffusion3D(nn.Module):
    """
    Multinomial Diffusion for 3D semantic voxels.

    The diffusion process is defined on one-hot categorical distributions:
    - Forward: gradually mix one-hot with uniform distribution
    - Reverse: neural network predicts clean x_0, then compute posterior

    Args:
        num_classes: Number of semantic classes (K)
        num_timesteps: Number of diffusion steps (T)
        loss_type: 'kl' for KL divergence, 'ce' for cross-entropy
    """

    def __init__(
        self,
        num_classes: int = 20,
        num_timesteps: int = 100,
        loss_type: str = 'kl',
    ):
        super().__init__()

        self.num_classes = num_classes
        self.num_timesteps = num_timesteps
        self.loss_type = loss_type

        # Create beta schedule (linear)
        # β goes from 0.0001 to 0.02 (standard for multinomial diffusion)
        betas = torch.linspace(0.0001, 0.02, num_timesteps, dtype=torch.float64)

        # Compute α values
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        # Convert to log space for numerical stability
        log_alpha = torch.log(alphas)
        log_cumprod_alpha = torch.log(alphas_cumprod)
        log_1_min_alpha = log_1_min_a(log_alpha)
        log_1_min_cumprod_alpha = log_1_min_a(log_cumprod_alpha)

        # Register buffers
        self.register_buffer('betas', betas.float())
        self.register_buffer('alphas', alphas.float())
        self.register_buffer('alphas_cumprod', alphas_cumprod.float())
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev.float())

        self.register_buffer('log_alpha', log_alpha.float())
        self.register_buffer('log_cumprod_alpha', log_cumprod_alpha.float())
        self.register_buffer('log_1_min_alpha', log_1_min_alpha.float())
        self.register_buffer('log_1_min_cumprod_alpha', log_1_min_cumprod_alpha.float())

    def _prepare_bev(self, bev: torch.Tensor) -> torch.Tensor:
        """Convert BEV to class probabilities [B, K, H, W].

        Handles both:
        - Hard BEV: [B, H, W] int64 class indices -> one-hot [B, K, H, W]
        - Soft BEV: [B, K, H, W] float probabilities -> passthrough
        """
        if bev.dim() == 4:
            return bev  # Already soft probabilities
        bev_onehot = F.one_hot(bev.long(), num_classes=self.num_classes).float()
        return bev_onehot.permute(0, 3, 1, 2)

    def q_probs(self, x_0: torch.Tensor, t: torch.Tensor, x_scpnet: torch.Tensor = None) -> torch.Tensor:
        """
        Compute q(x_t | x_0) - the forward diffusion distribution.

        Standard: q(x_t | x_0) = α_t * x_0 + (1 - α_t) / K
        Cold:     q(x_t | x_0) = α_t * x_0 + (1 - α_t) * x_scpnet

        Args:
            x_0: One-hot encoded clean data [B, K, ...]
            t: Timesteps [B]
            x_scpnet: One-hot SCPNet prediction [B, K, ...] for cold diffusion (None = standard)
        """
        alpha_t = extract(self.alphas_cumprod, t, x_0.shape)  # [B, 1, 1, 1, 1]

        if x_scpnet is not None:
            probs = alpha_t * x_0 + (1.0 - alpha_t) * x_scpnet
        else:
            probs = alpha_t * x_0 + (1.0 - alpha_t) / self.num_classes

        return probs

    def q_sample(self, x_0: torch.Tensor, t: torch.Tensor, x_scpnet: torch.Tensor = None) -> torch.Tensor:
        """
        Sample from q(x_t | x_0) - the forward diffusion process.

        Args:
            x_0: One-hot encoded clean data [B, K, ...]
            t: Timesteps [B]
            x_scpnet: One-hot SCPNet prediction for cold diffusion (None = standard)

        Returns:
            Sampled x_t as one-hot [B, K, ...]
        """
        # Get probabilities
        probs = self.q_probs(x_0, t, x_scpnet=x_scpnet)  # [B, K, H, W, D]

        # Reshape for sampling: [B, K, H*W*D] -> [B*H*W*D, K]
        shape = probs.shape
        K = shape[1]
        probs_flat = probs.permute(0, 2, 3, 4, 1).reshape(-1, K)  # [B*H*W*D, K]

        # Sample from categorical
        samples = torch.multinomial(probs_flat, num_samples=1).squeeze(-1)  # [B*H*W*D]

        # Convert to one-hot and reshape back
        x_t = F.one_hot(samples, num_classes=K).float()  # [B*H*W*D, K]
        x_t = x_t.reshape(shape[0], shape[2], shape[3], shape[4], K)
        x_t = x_t.permute(0, 4, 1, 2, 3)  # [B, K, H, W, D]

        return x_t

    def q_posterior_logits(
        self,
        x_0: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor,
        x_scpnet: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Compute log posterior q(x_{t-1} | x_t, x_0).

        Standard: noise target = 1/K (uniform)
        Cold:     noise target = x_scpnet (SCPNet prediction)

        Args:
            x_0: Clean data probabilities [B, K, H, W, D]
            x_t: Noisy data (one-hot) [B, K, H, W, D]
            t: Timesteps [B]
            x_scpnet: SCPNet prediction for cold diffusion (None = standard uniform)
        """
        K = self.num_classes
        noise = x_scpnet if x_scpnet is not None else (1.0 / K)

        alpha_t = extract(self.alphas, t, x_0.shape)
        alpha_bar_t_minus_1 = extract(self.alphas_cumprod_prev, t, x_0.shape)

        # q(x_t | x_{t-1}=k): transition uses noise target
        log_q_xt_given_xtm1 = torch.log(
            alpha_t * x_t + (1.0 - alpha_t) * noise + 1e-10
        )

        # q(x_{t-1}=k | x_0): marginal uses noise target
        log_q_xtm1_given_x0 = torch.log(
            alpha_bar_t_minus_1 * x_0 + (1.0 - alpha_bar_t_minus_1) * noise + 1e-10
        )

        log_posterior_unnorm = log_q_xt_given_xtm1 + log_q_xtm1_given_x0
        return log_posterior_unnorm

    def q_posterior(
        self,
        x_0: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor,
        x_scpnet: torch.Tensor = None,
    ) -> torch.Tensor:
        """Compute normalized posterior q(x_{t-1} | x_t, x_0)."""
        log_posterior_unnorm = self.q_posterior_logits(x_0, x_t, t, x_scpnet=x_scpnet)

        # Normalize
        log_posterior = log_posterior_unnorm - torch.logsumexp(
            log_posterior_unnorm, dim=1, keepdim=True
        )
        posterior = torch.exp(log_posterior)

        return posterior

    def p_probs(
        self,
        model: nn.Module,
        x_t: torch.Tensor,
        t: torch.Tensor,
        bev: torch.Tensor,
        lidar: torch.Tensor,
        lifted_features: torch.Tensor | None = None,
        nn_indices: torch.Tensor | None = None,
        **model_kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute p(x_{t-1} | x_t) using neural network prediction.

        The network predicts x̂_0, then we compute the posterior
        p(x_{t-1} | x_t) = p(x_{t-1} | x_t, x̂_0)

        Args:
            model: Neural network that predicts x_0 logits
            x_t: Current noisy sample [B, K, H, W, D]
            t: Timesteps [B]
            bev: BEV conditioning [B, K, H, W]
            lidar: LiDAR conditioning [B, 1, H, W, D]
            lifted_features: Optional S1/S2 lifted 3D features [B, feat_dim, H, W, D]
            nn_indices: Optional precomputed NN indices for densification [B, 3, H, W, D]
            **model_kwargs: Additional kwargs passed to model (e.g., geom_bev)

        Returns:
            x_0_pred: Predicted clean data (logits) [B, K, H, W, D]
            probs: Reverse process probabilities [B, K, H, W, D]
        """
        # Extract x_scpnet before passing to model (model doesn't accept it)
        x_scpnet = model_kwargs.pop('x_scpnet', None)

        # Get model prediction (logits for x_0)
        x_0_logits = model(x_t, t, bev, lidar, lifted_features=lifted_features, nn_indices=nn_indices, **model_kwargs)  # [B, K, H, W, D]

        # Restore x_scpnet for future calls
        if x_scpnet is not None:
            model_kwargs['x_scpnet'] = x_scpnet

        # Convert to probabilities
        x_0_pred = F.softmax(x_0_logits, dim=1)

        # For t > 0, compute posterior q(x_{t-1}|x_t, x_hat_0) — matches training loss
        # For t = 0, use x_0_pred directly (no further denoising needed)
        t_is_zero = (t == 0).reshape(-1, *([1] * (x_0_pred.dim() - 1)))  # [B, 1, 1, 1, 1]
        posterior = self.q_posterior(x_0_pred, x_t, t, x_scpnet=x_scpnet)
        probs = torch.where(t_is_zero, x_0_pred, posterior)

        return x_0_logits, probs

    def training_losses(
        self,
        model: nn.Module,
        x_0: torch.Tensor,
        t: torch.Tensor,
        bev: torch.Tensor,
        lidar: torch.Tensor,
        lifted_features: torch.Tensor | None = None,
        nn_indices: torch.Tensor | None = None,
        **model_kwargs,
    ) -> dict[str, torch.Tensor]:
        """
        Compute training loss following the reference paper.

        For t >= 1: L_{t-1} = KL(q(x_{t-1}|x_t, x_0) || p(x_{t-1}|x_t))
        For t = 0: L_0 = -log p(x_0 | x_1) (reconstruction loss)

        Args:
            model: Neural network
            x_0: Clean data (class indices) [B, H, W, D]
            t: Timesteps [B]
            bev: BEV conditioning (class indices) [B, H, W]
            lidar: LiDAR conditioning [B, 1, H, W, D]
            lifted_features: Optional S2 lifted 3D features [B, feat_dim, H, W, D]
            nn_indices: Optional precomputed NN indices for densification [B, 3, H, W, D]
            **model_kwargs: Additional kwargs passed to model (e.g., geom_bev)

        Returns:
            Dictionary with 'loss' and other metrics
        """
        B = x_0.shape[0]

        # Convert x_0 to one-hot
        x_0_onehot = F.one_hot(x_0.long(), num_classes=self.num_classes).float()
        x_0_onehot = x_0_onehot.permute(0, 4, 1, 2, 3)  # [B, K, H, W, D]

        # Convert BEV to class probabilities (handles both hard [B,H,W] and soft [B,K,H,W])
        bev_onehot = self._prepare_bev(bev)  # [B, K, H, W]

        # Sample x_t from q(x_t | x_0) — cold diffusion uses x_scpnet instead of uniform
        x_scpnet = model_kwargs.get('x_scpnet', None)
        x_t = self.q_sample(x_0_onehot, t, x_scpnet=x_scpnet)  # [B, K, H, W, D]

        # Get model prediction (x̂_0)
        x_0_logits, _ = self.p_probs(model, x_t, t, bev_onehot, lidar, lifted_features=lifted_features, nn_indices=nn_indices, **model_kwargs)
        x_0_pred_probs = F.softmax(x_0_logits, dim=1)  # [B, K, H, W, D]

        # Compute loss based on loss_type
        if self.loss_type == 'ce':
            # Cross-entropy loss (simpler approximation)
            # L = -log p(x_0 | x̂_0) = CE(x̂_0, x_0)
            x_0_logits_flat = x_0_logits.reshape(B, self.num_classes, -1)
            x_0_flat = x_0.reshape(B, -1).long()

            loss = F.cross_entropy(
                x_0_logits_flat,
                x_0_flat,
                reduction='none'
            ).mean()

        else:  # 'kl' - proper multinomial diffusion loss
            # L_{t-1} = KL(q(x_{t-1}|x_t, x_0) || p(x_{t-1}|x_t))
            # where p(x_{t-1}|x_t) = q(x_{t-1}|x_t, x̂_0)

            # True posterior: q(x_{t-1}|x_t, x_0)
            q_posterior_true = self.q_posterior(x_0_onehot, x_t, t, x_scpnet=x_scpnet)  # [B, K, H, W, D]

            # Predicted posterior: p(x_{t-1}|x_t) = q(x_{t-1}|x_t, x̂_0)
            p_posterior_pred = self.q_posterior(x_0_pred_probs, x_t, t, x_scpnet=x_scpnet)  # [B, K, H, W, D]

            # KL(q || p) = sum_k q[k] * log(q[k] / p[k])
            # = sum_k q[k] * (log q[k] - log p[k])
            kl_per_class = q_posterior_true * (
                torch.log(q_posterior_true + 1e-10) - torch.log(p_posterior_pred + 1e-10)
            )  # [B, K, H, W, D]

            # Sum over classes, mean over voxels and batch
            kl_per_voxel = kl_per_class.sum(dim=1)  # [B, H, W, D]
            loss = kl_per_voxel.mean()

        # Compute accuracy for monitoring
        x_0_pred = x_0_logits.argmax(dim=1)  # [B, H, W, D]
        accuracy = (x_0_pred == x_0).float().mean()

        return {
            'loss': loss,
            'accuracy': accuracy,
            'x_t_onehot': x_t,  # S3: Return x_t for DSKD feature extraction
            'bev_onehot': bev_onehot,  # S3: Return bev_onehot for consistency
            'x_0_logits': x_0_logits,  # Reuse for KD to avoid double forward pass
        }

    @torch.no_grad()
    def p_sample(
        self,
        model: nn.Module,
        x_t: torch.Tensor,
        t: torch.Tensor,
        bev: torch.Tensor,
        lidar: torch.Tensor,
        lifted_features: torch.Tensor | None = None,
        nn_indices: torch.Tensor | None = None,
        **model_kwargs,
    ) -> torch.Tensor:
        """
        Sample x_{t-1} from p(x_{t-1} | x_t).

        Args:
            model: Neural network
            x_t: Current noisy sample (one-hot) [B, K, H, W, D]
            t: Current timestep [B]
            bev: BEV conditioning (one-hot) [B, K, H, W]
            lidar: LiDAR conditioning [B, 1, H, W, D]
            lifted_features: Optional S1/S2 lifted 3D features [B, feat_dim, H, W, D]
            nn_indices: Optional precomputed NN indices for densification [B, 3, H, W, D]
            **model_kwargs: Additional kwargs passed to model (e.g., geom_bev)

        Returns:
            x_{t-1} as one-hot [B, K, H, W, D]
        """
        # Get predicted probabilities
        _, probs = self.p_probs(model, x_t, t, bev, lidar, lifted_features=lifted_features, nn_indices=nn_indices, **model_kwargs)  # [B, K, H, W, D]

        # Sample from categorical
        shape = probs.shape
        K = shape[1]
        probs_flat = probs.permute(0, 2, 3, 4, 1).reshape(-1, K)  # [B*H*W*D, K]

        # For t=0, use argmax; for t>0, sample
        if (t == 0).all():
            samples = probs_flat.argmax(dim=-1)
        else:
            # Add small epsilon for numerical stability
            probs_flat = probs_flat.clamp(min=1e-10)
            probs_flat = probs_flat / probs_flat.sum(dim=-1, keepdim=True)
            samples = torch.multinomial(probs_flat, num_samples=1).squeeze(-1)

        # Convert to one-hot
        x_t_minus_1 = F.one_hot(samples, num_classes=K).float()
        x_t_minus_1 = x_t_minus_1.reshape(shape[0], shape[2], shape[3], shape[4], K)
        x_t_minus_1 = x_t_minus_1.permute(0, 4, 1, 2, 3)  # [B, K, H, W, D]

        return x_t_minus_1

    @torch.no_grad()
    def sample(
        self,
        model: nn.Module,
        bev: torch.Tensor,
        lidar: torch.Tensor,
        shape: tuple[int, int, int, int],  # (B, H, W, D)
        device: torch.device,
        lifted_features: torch.Tensor | None = None,
        show_progress: bool = True,
        nn_indices: torch.Tensor | None = None,
        **model_kwargs,
    ) -> torch.Tensor:
        """
        Generate samples using the reverse diffusion process.

        Args:
            model: Neural network
            bev: BEV conditioning (class indices) [B, H, W]
            lidar: LiDAR conditioning [B, 1, H, W, D]
            shape: Output shape (B, H, W, D)
            device: Device to use
            lifted_features: Optional S1/S2 lifted 3D features [B, feat_dim, H, W, D]
            show_progress: Whether to show progress bar
            nn_indices: Optional precomputed NN indices for densification [B, 3, H, W, D]
            **model_kwargs: Additional kwargs passed to model (e.g., geom_bev)

        Returns:
            Generated samples (class indices) [B, H, W, D]
        """
        B, H, W, D = shape
        model.eval()

        # Convert BEV to class probabilities (handles both hard [B,H,W] and soft [B,K,H,W])
        bev_onehot = self._prepare_bev(bev).to(device)  # [B, K, H, W]

        # Start from uniform noise
        x_t = torch.ones(B, self.num_classes, H, W, D, device=device) / self.num_classes

        # Sample from uniform
        probs_flat = x_t.permute(0, 2, 3, 4, 1).reshape(-1, self.num_classes)
        samples = torch.multinomial(probs_flat, num_samples=1).squeeze(-1)
        x_t = F.one_hot(samples, num_classes=self.num_classes).float()
        x_t = x_t.reshape(B, H, W, D, self.num_classes)
        x_t = x_t.permute(0, 4, 1, 2, 3).to(device)  # [B, K, H, W, D]

        # Reverse diffusion
        timesteps = list(range(self.num_timesteps))[::-1]
        if show_progress:
            timesteps = tqdm(timesteps, desc="Sampling")

        for t in timesteps:
            t_batch = torch.full((B,), t, device=device, dtype=torch.long)
            x_t = self.p_sample(model, x_t, t_batch, bev_onehot, lidar, lifted_features=lifted_features, nn_indices=nn_indices, **model_kwargs)

        # Convert to class indices
        x_0 = x_t.argmax(dim=1)  # [B, H, W, D]

        return x_0

    @torch.no_grad()
    def sample_repaint(
        self,
        model: nn.Module,
        bev: torch.Tensor,
        lidar: torch.Tensor,
        known_labels: torch.Tensor,
        obs_mask: torch.Tensor,
        shape: tuple[int, int, int, int],
        device: torch.device,
        lifted_features: torch.Tensor | None = None,
        show_progress: bool = True,
        nn_indices: torch.Tensor | None = None,
        repaint_jumps: int = 1,
        **model_kwargs,
    ) -> torch.Tensor:
        """Repaint sampling: anchor observed voxels at each denoising step.

        At each step t, after denoising, observed voxels are replaced with
        properly re-noised known labels. Only unobserved voxels come from
        the model's generation. This implements inpainting-style diffusion.

        Args:
            known_labels: [B, H, W, D] int64 class labels to anchor (e.g., SCPNet pred)
            obs_mask: [B, 1, H, W, D] binary mask (1 = observed/anchored)
        """
        B, H, W, D = shape
        K = self.num_classes
        model.eval()

        bev_onehot = self._prepare_bev(bev).to(device)

        # Known labels to one-hot
        known_onehot = F.one_hot(known_labels.long(), K).float()
        known_onehot = known_onehot.permute(0, 4, 1, 2, 3).to(device)  # [B, K, H, W, D]

        # Expand obs_mask to match one-hot channels
        obs_mask_k = obs_mask.expand(-1, K, -1, -1, -1).to(device)  # [B, K, H, W, D]

        # Pass obs_mask to model via model_kwargs (for obs_mask_channel input)
        model_kwargs['obs_mask'] = obs_mask

        # Start from uniform noise
        probs_flat = torch.ones(B * H * W * D, K, device=device) / K
        samples = torch.multinomial(probs_flat, num_samples=1).squeeze(-1)
        x_t = F.one_hot(samples, num_classes=K).float()
        x_t = x_t.reshape(B, H, W, D, K).permute(0, 4, 1, 2, 3)  # [B, K, H, W, D]

        timesteps = list(range(self.num_timesteps))[::-1]
        if show_progress:
            timesteps = tqdm(timesteps, desc="Repaint")

        for t in timesteps:
            t_batch = torch.full((B,), t, device=device, dtype=torch.long)

            # Step 1: Denoise all voxels
            x_t = self.p_sample(model, x_t, t_batch, bev_onehot, lidar,
                                lifted_features=lifted_features,
                                nn_indices=nn_indices, **model_kwargs)

            # Step 2: Re-noise known labels to current level and replace
            if t > 0:
                t_prev = torch.full((B,), t - 1, device=device, dtype=torch.long)
                known_noisy = self.q_sample(known_onehot, t_prev)
            else:
                known_noisy = known_onehot  # At t=0: clean labels

            # Step 3: Replace observed voxels with re-noised known
            x_t = obs_mask_k * known_noisy + (1.0 - obs_mask_k) * x_t

        return x_t.argmax(dim=1)  # [B, H, W, D]

    @torch.no_grad()
    def sample_cold(
        self,
        model: nn.Module,
        bev: torch.Tensor,
        lidar: torch.Tensor,
        scpnet_pred: torch.Tensor,
        shape: tuple[int, int, int, int],
        device: torch.device,
        lifted_features: torch.Tensor | None = None,
        show_progress: bool = True,
        nn_indices: torch.Tensor | None = None,
        start_timestep: int | None = None,
        start_pred: torch.Tensor | None = None,
        **model_kwargs,
    ) -> torch.Tensor:
        """Cold Diffusion: flow from SCPNet prediction to GT.

        Forward: q(x_t|x_0) = α_t·GT + (1-α_t)·SCPNet (not uniform noise)
        Inference: start from SCPNet (t=T), iteratively refine toward GT.

        Optional few-step refinement:
            start_pred: [B, H, W, D] better starting point (e.g., direct prediction)
            start_timestep: start from this t instead of T-1 (e.g., 5 or 10)
        """
        B, H, W, D = shape
        K = self.num_classes
        model.eval()

        bev_onehot = self._prepare_bev(bev).to(device)
        scpnet_oh = F.one_hot(scpnet_pred.long(), K).float()
        scpnet_oh = scpnet_oh.permute(0, 4, 1, 2, 3).to(device)  # [B, K, H, W, D]

        # Pass x_scpnet through model_kwargs for posterior computation
        model_kwargs['x_scpnet'] = scpnet_oh

        if start_pred is not None and start_timestep is not None:
            # Few-step refinement: start from a better prediction at a low t
            start_oh = F.one_hot(start_pred.long(), K).float()
            start_oh = start_oh.permute(0, 4, 1, 2, 3).to(device)
            t_start = torch.full((B,), start_timestep, device=device, dtype=torch.long)
            x_t = self.q_sample(start_oh, t_start, x_scpnet=scpnet_oh)
            timesteps = list(range(start_timestep, -1, -1))
        else:
            # Full cold sampling: start from SCPNet prediction (t=T)
            x_t = scpnet_oh.clone()
            timesteps = list(range(self.num_timesteps - 1, -1, -1))

        if show_progress:
            timesteps = tqdm(timesteps, desc="Cold Diffusion")

        for t in timesteps:
            t_batch = torch.full((B,), t, device=device, dtype=torch.long)
            x_t = self.p_sample(model, x_t, t_batch, bev_onehot, lidar,
                                lifted_features=lifted_features,
                                nn_indices=nn_indices, **model_kwargs)

        return x_t.argmax(dim=1)  # [B, H, W, D]

    @torch.no_grad()
    def sample_cfg(
        self,
        model: nn.Module,
        bev: torch.Tensor,
        lidar: torch.Tensor,
        shape: tuple[int, int, int, int],
        device: torch.device,
        lifted_features: torch.Tensor | None = None,
        guidance_scale: float = 1.5,
        cfg_target: str = 'lifted',
        show_progress: bool = True,
        nn_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Sample with Classifier-Free Guidance.

        Args:
            cfg_target: What to zero for unconditional pass:
                'lifted' — zero lifted_features (V1-style, for CFG on lifted condition)
                'lidar' — zero lidar (V3-style, for CFG on LSK3DNet features)
        """
        B, H, W, D = shape
        model.eval()

        # Convert BEV to class probabilities (handles both hard [B,H,W] and soft [B,K,H,W])
        bev_onehot = self._prepare_bev(bev).to(device)

        # Start from uniform noise
        x_t = torch.ones(B, self.num_classes, H, W, D, device=device) / self.num_classes
        probs_flat = x_t.permute(0, 2, 3, 4, 1).reshape(-1, self.num_classes)
        samples = torch.multinomial(probs_flat, num_samples=1).squeeze(-1)
        x_t = F.one_hot(samples, num_classes=self.num_classes).float()
        x_t = x_t.reshape(B, H, W, D, self.num_classes)
        x_t = x_t.permute(0, 4, 1, 2, 3).to(device)

        timesteps = list(range(self.num_timesteps))[::-1]
        if show_progress:
            timesteps = tqdm(timesteps, desc="CFG Sampling")

        for t in timesteps:
            t_batch = torch.full((B,), t, device=device, dtype=torch.long)

            if cfg_target == 'lidar':
                # V3-style CFG: zero out lidar conditioning (primary condition for V3 models)
                # Use force_uncond=True to bypass sparse encoder (avoids empty tensor crash)
                cond_logits = model(x_t, t_batch, bev_onehot, lidar, lifted_features=lifted_features, nn_indices=nn_indices)
                uncond_logits = model(x_t, t_batch, bev_onehot, lidar, lifted_features=lifted_features, force_uncond=True, nn_indices=nn_indices)
            elif lifted_features is not None:
                # V1-style CFG: zero out lifted features
                cond_logits = model(x_t, t_batch, bev_onehot, lidar, lifted_features=lifted_features, nn_indices=nn_indices)
                uncond_logits = model(x_t, t_batch, bev_onehot, lidar, lifted_features=torch.zeros_like(lifted_features), nn_indices=nn_indices)
            else:
                # No CFG possible — standard forward pass
                cond_logits = model(x_t, t_batch, bev_onehot, lidar, lifted_features=lifted_features, nn_indices=nn_indices)
                uncond_logits = cond_logits  # no-op CFG

            # CFG combination
            guided_logits = uncond_logits + guidance_scale * (cond_logits - uncond_logits)
            x_0_pred = F.softmax(guided_logits, dim=1)

            # Use posterior q(x_{t-1}|x_t, x_hat_0) for t > 0 — matches training loss
            if t > 0:
                sampling_probs = self.q_posterior(x_0_pred, x_t, t_batch)
            else:
                sampling_probs = x_0_pred

            # Sample from posterior (or x_0_pred at t=0)
            K = sampling_probs.shape[1]
            probs_flat = sampling_probs.permute(0, 2, 3, 4, 1).reshape(-1, K)

            if t == 0:
                samples = probs_flat.argmax(dim=-1)
            else:
                probs_flat = probs_flat.clamp(min=1e-10)
                probs_flat = probs_flat / probs_flat.sum(dim=-1, keepdim=True)
                samples = torch.multinomial(probs_flat, num_samples=1).squeeze(-1)

            x_t = F.one_hot(samples, num_classes=K).float()
            x_t = x_t.reshape(B, H, W, D, K).permute(0, 4, 1, 2, 3)

        x_0 = x_t.argmax(dim=1)
        return x_0

    @torch.no_grad()
    def sample_sdedit(
        self,
        model: nn.Module,
        bev: torch.Tensor,
        lidar: torch.Tensor,
        lifted_3d: torch.Tensor,  # [B, H, W, D] rough 3D estimate from lifting
        shape: tuple[int, int, int, int],  # (B, H, W, D)
        device: torch.device,
        lifted_features: torch.Tensor | None = None,
        start_timestep: int = 50,  # Start from t=50 instead of t=T
        show_progress: bool = True,
        **model_kwargs,
    ) -> torch.Tensor:
        """
        S2: SDEdit-style sampling starting from lifted 3D estimate.

        Instead of starting from pure noise (t=T), starts from the lifted 3D
        estimate with noise added at t=start_timestep. This preserves structure
        from the lifted estimate while allowing refinement.

        Reference: SDEdit (Meng et al., 2021)

        Args:
            model: Neural network
            bev: BEV conditioning (class indices) [B, H, W]
            lidar: LiDAR conditioning [B, 1, H, W, D]
            lifted_3d: Lifted 3D estimate (class indices) [B, H, W, D]
            shape: Output shape (B, H, W, D)
            device: Device to use
            lifted_features: Optional S1/S2 lifted 3D features [B, feat_dim, H, W, D]
            start_timestep: Timestep to start denoising from (default: 50 out of 100)
            show_progress: Whether to show progress bar

        Returns:
            Generated samples (class indices) [B, H, W, D]
        """
        B, H, W, D = shape
        model.eval()

        # Convert BEV to class probabilities (handles both hard [B,H,W] and soft [B,K,H,W])
        bev_onehot = self._prepare_bev(bev).to(device)  # [B, K, H, W]

        # Convert lifted_3d to one-hot
        lifted_onehot = F.one_hot(lifted_3d.long(), num_classes=self.num_classes).float()
        lifted_onehot = lifted_onehot.permute(0, 4, 1, 2, 3).to(device)  # [B, K, H, W, D]

        # Add noise to lifted estimate at start_timestep using forward diffusion
        # q(x_t | x_0) = Cat(x_t | α_t * x_0 + (1 - α_t) / K)
        t_start = torch.full((B,), start_timestep, device=device, dtype=torch.long)
        x_t = self.q_sample(lifted_onehot, t_start)  # [B, K, H, W, D]

        # Reverse diffusion from start_timestep to 0
        # Bug fix: Must include start_timestep in the loop!
        # Previously: list(range(start_timestep))[::-1] = [start_timestep-1, ..., 0] (WRONG!)
        # This caused timestep mismatch: x_t at level start_timestep, but model sees t=start_timestep-1
        timesteps = list(range(start_timestep, -1, -1))  # [start_timestep, start_timestep-1, ..., 0]
        if show_progress:
            timesteps = tqdm(timesteps, desc=f"SDEdit sampling (t={start_timestep}→0)")

        for t in timesteps:
            t_batch = torch.full((B,), t, device=device, dtype=torch.long)
            # Pass lifted_features and any extra model_kwargs (e.g., pt_fea, grid_ind, scpnet_pred)
            x_t = self.p_sample(model, x_t, t_batch, bev_onehot, lidar, lifted_features=lifted_features, **model_kwargs)

        # Convert to class indices
        x_0 = x_t.argmax(dim=1)  # [B, H, W, D]

        return x_0


    @torch.no_grad()
    def sample_mimo(
        self,
        model: nn.Module,
        bev: torch.Tensor,  # [B, H, W] single BEV
        lidar: torch.Tensor,  # [B, 1, H, W, D] single LiDAR
        waffleiron: torch.Tensor,  # [B, C, H, W] WaffleIron BEV features (COMPULSORY like PaSCo)
        n_subnets: int,
        shape: tuple[int, int, int, int],  # (B, H, W, D)
        device: torch.device,
        show_progress: bool = True,
        use_tta: bool = False,  # Test-time augmentation (PaSCo val_aug=True)
        use_semantic_pruning: bool = False,  # PaSCo-style semantic pruning
        pruning_empty_threshold: float = 0.85,
        pruning_lidar_dilation: int = 3,
        # PaSCo-style TTA parameters (only used when use_tta=True)
        use_continuous_tta: bool = True,  # Use continuous rotation + translation
        tta_max_angle: float = 30.0,  # Max rotation ±30° (PaSCo default)
        tta_max_translation: tuple[float, float, float] = (0.6, 0.6, 0.4),  # Max translation (m)
    ) -> torch.Tensor:
        """
        S4: MIMO sampling for PaSCo-style ensemble inference.

        Two modes:
        1. use_tta=False (default): Same sample for all N subnets
           - Diversity comes from N decoder heads trained on different data
           - Simpler, no inverse transforms needed

        2. use_tta=True: Apply different augmentations to each subnet
           - PaSCo-style: Continuous rotation (±30°) + translation (±0.6m)
           - Inverse transforms applied before ensemble averaging
           - Matches PaSCo's val_aug=True behavior

        Optional: Semantic Pruning (PaSCo-style)
        - Removes predictions in clearly empty regions
        - Keeps predictions only near LiDAR observations
        - Improves performance on rare classes

        Optional: WaffleIron Features (PaSCo-style B6)
        - Precomputed semantic features from WaffleIron model
        - Augmented with same transform as lidar/bev during TTA
        - Provides rich semantic priors for scene completion

        Reference: PaSCo CVPR 2024, Section 3.2.2 (Inference)
                   reference/PaSCo/pasco/data/semantic_kitti/kitti_dm.py (val_aug=True)

        Args:
            model: MIMO model (MIMOSceneCompletionUNet)
            bev: BEV conditioning (class indices) [B, H, W]
            lidar: LiDAR conditioning [B, 1, H, W, D]
            waffleiron: WaffleIron BEV features [B, C, H, W] (COMPULSORY like PaSCo)
            n_subnets: Number of subnets (N)
            shape: Output shape (B, H, W, D)
            device: Device to use
            show_progress: Whether to show progress bar
            use_tta: Whether to use test-time augmentation (PaSCo val_aug)
            use_semantic_pruning: Whether to apply PaSCo-style semantic pruning
            pruning_empty_threshold: Threshold for empty class probability
            pruning_lidar_dilation: Dilation radius for LiDAR-guided mask
            use_continuous_tta: Use PaSCo-style continuous rotation (True) or discrete 90° (False)
            tta_max_angle: Maximum rotation angle in degrees (PaSCo: 30°)
            tta_max_translation: Maximum translation in meters (PaSCo: 0.6m XY, 0.4m Z)

        Returns:
            Generated samples (class indices) [B, H, W, D]
        """
        B, H, W, D = shape
        model.eval()

        if use_tta:
            return self._sample_mimo_with_tta(
                model, bev, lidar, waffleiron, n_subnets, shape, device, show_progress,
                use_semantic_pruning=use_semantic_pruning,
                pruning_empty_threshold=pruning_empty_threshold,
                pruning_lidar_dilation=pruning_lidar_dilation,
                # PaSCo-style TTA parameters
                use_continuous_tta=use_continuous_tta,
                tta_max_angle=tta_max_angle,
                tta_max_translation=tta_max_translation,
            )

        # Standard MIMO: Same sample for all subnets (diversity from N heads)
        # BEV: [B, H, W] → [B, N, H, W]
        bev_stacked = bev.unsqueeze(1).expand(-1, n_subnets, -1, -1)

        # LiDAR: [B, 1, H, W, D] → [B, N, H, W, D]
        lidar_stacked = lidar.expand(-1, n_subnets, -1, -1, -1)

        # WaffleIron: [B, C, H, W] → [B, N, C, H, W]
        waffleiron_stacked = waffleiron.unsqueeze(1).expand(-1, n_subnets, -1, -1, -1)

        # Start from uniform noise (replicated N times)
        x_t = torch.ones(B, self.num_classes, H, W, D, device=device) / self.num_classes
        probs_flat = x_t.permute(0, 2, 3, 4, 1).reshape(-1, self.num_classes)
        samples = torch.multinomial(probs_flat, num_samples=1).squeeze(-1)
        x_t_single = F.one_hot(samples, num_classes=self.num_classes).float()
        x_t_single = x_t_single.reshape(B, H, W, D, self.num_classes)
        x_t_single = x_t_single.permute(0, 4, 1, 2, 3).to(device)

        x_t_list = [x_t_single.clone() for _ in range(n_subnets)]
        x_t = torch.cat(x_t_list, dim=1)

        timesteps = list(range(self.num_timesteps))[::-1]
        if show_progress:
            timesteps = tqdm(timesteps, desc="MIMO Sampling")

        for t in timesteps:
            t_batch = torch.full((B,), t, device=device, dtype=torch.long)
            outputs = model(x_t, t_batch, bev_stacked, lidar_stacked, waffleiron_stacked)
            ensemble_probs = outputs['ensemble_probs']

            probs_flat = ensemble_probs.permute(0, 2, 3, 4, 1).reshape(-1, self.num_classes)
            if t == 0:
                samples = probs_flat.argmax(dim=-1)
            else:
                probs_flat = probs_flat.clamp(min=1e-10)
                probs_flat = probs_flat / probs_flat.sum(dim=-1, keepdim=True)
                samples = torch.multinomial(probs_flat, num_samples=1).squeeze(-1)

            x_t_single = F.one_hot(samples, num_classes=self.num_classes).float()
            x_t_single = x_t_single.reshape(B, H, W, D, self.num_classes)
            x_t_single = x_t_single.permute(0, 4, 1, 2, 3).to(device)
            x_t_list = [x_t_single.clone() for _ in range(n_subnets)]
            x_t = torch.cat(x_t_list, dim=1)

        # Apply semantic pruning if enabled (PaSCo-style)
        if use_semantic_pruning:
            from gssc.models.semantic_pruning import semantic_pruning_postprocess
            x_0, _ = semantic_pruning_postprocess(
                pred_probs=ensemble_probs,  # [B, C, H, W, D]
                lidar=lidar,  # [B, 1, H, W, D]
                empty_threshold=pruning_empty_threshold,
                lidar_dilation=pruning_lidar_dilation,
                use_lidar_mask=True,
                use_occupancy_mask=True,
                combine_mode="intersection",
            )
        else:
            x_0 = ensemble_probs.argmax(dim=1)
        return x_0

    @torch.no_grad()
    def _sample_mimo_with_tta(
        self,
        model: nn.Module,
        bev: torch.Tensor,
        lidar: torch.Tensor,
        waffleiron: torch.Tensor,  # [B, C, H, W] WaffleIron features (COMPULSORY like PaSCo)
        n_subnets: int,
        shape: tuple[int, int, int, int],
        device: torch.device,
        show_progress: bool,
        use_semantic_pruning: bool = False,
        pruning_empty_threshold: float = 0.85,
        pruning_lidar_dilation: int = 3,
        # PaSCo-style TTA parameters
        use_continuous_tta: bool = True,  # Use continuous rotation + translation
        tta_max_angle: float = 30.0,  # Max rotation ±30° (PaSCo default)
        tta_max_translation: tuple[float, float, float] = (0.6, 0.6, 0.4),  # Max translation in meters
    ) -> torch.Tensor:
        """
        MIMO sampling with test-time augmentation (PaSCo val_aug=True).

        Each subnet sees a different augmented version of the input:
        - PaSCo-style (use_continuous_tta=True): Continuous rotation (±30°) + translation (±0.6m)
        - Legacy (use_continuous_tta=False): Discrete rotation (0°, 90°, 180°, 270°) + flips

        Predictions are inverse-transformed before ensemble averaging.
        Optionally applies semantic pruning (PaSCo-style) after sampling.
        WaffleIron BEV features are COMPULSORY (like PaSCo) with matching augmentation.

        PaSCo Reference:
        - max_angle: 30.0 (±30° continuous rotation around Z)
        - max_translation: [0.6, 0.6, 0.4] meters (XY, Z)
        - flip: True (50% Y-flip)
        - scale: 0.0 (DISABLED)
        """
        from gssc.models.mimo_dataset import (
            apply_augmentation,
            apply_bev_feature_augmentation,
            apply_inverse_augmentation,
            get_random_augmentation_params,
        )

        B, H, W, D = shape
        voxel_shape = (H, W, D)

        # Generate augmentation params for each subnet with PaSCo-style settings
        aug_params_list = [
            get_random_augmentation_params(
                use_continuous=use_continuous_tta,
                max_angle=tta_max_angle,
                max_translation=tta_max_translation,
            )
            for _ in range(n_subnets)
        ]

        # Apply augmentations to create N augmented inputs
        # Process each batch item separately (augmentation expects unbatched inputs)
        bev_aug_batch = []  # Will be [B, N, H, W]
        lidar_aug_batch = []  # Will be [B, N, H, W, D]
        waffleiron_aug_batch = []  # Will be [B, N, C, H, W] if waffleiron provided

        for b in range(B):
            bev_aug_list = []
            lidar_aug_list = []
            waffleiron_aug_list = []
            for params in aug_params_list:
                # Extract single batch item (unbatched)
                lidar_3d = lidar[b, 0]  # [H, W, D]
                bev_single = bev[b]  # [H, W] or [1, H, W] if extra channel dim
                # Handle extra channel dimension if present - ensure 2D [H, W]
                # Bug fix: squeeze() ALL size-1 dims, not just dim 0
                bev_single = bev_single.squeeze()
                if bev_single.dim() > 2:
                    # Force to 2D by keeping only the two largest dims
                    bev_single = bev_single.view(bev_single.shape[0], -1)
                dummy_gt = torch.zeros_like(lidar_3d)  # [H, W, D]

                # Apply augmentation using the unified function that handles:
                # - Continuous rotation (±30°) with translation (±0.6m) if use_continuous=True
                # - Discrete 90° rotation + flips if use_continuous=False
                lidar_aug, bev_aug, _ = apply_augmentation(
                    lidar_3d, bev_single, dummy_gt,
                    aug_params=params,
                    voxel_shape=voxel_shape,
                )
                # Ensure BEV is 2D [H, W] by squeezing all size-1 dimensions
                bev_aug = bev_aug.squeeze()
                if bev_aug.dim() > 2:
                    bev_aug = bev_aug.view(bev_aug.shape[0], -1)
                elif bev_aug.dim() < 2:
                    # Should never happen but safety check
                    bev_aug = bev_aug.view(H, W)
                bev_aug_list.append(bev_aug)
                lidar_aug_list.append(lidar_aug)

                # Apply same augmentation to WaffleIron features (optional - may be None)
                if waffleiron is not None:
                    waffleiron_single = waffleiron[b]  # [C, H, W]
                    if params.get('use_continuous', False):
                        # Round to nearest 90° for feature maps
                        discrete_angle = round(params['rotation_angle'] / 90) * 90
                    else:
                        discrete_angle = params['rotation_angle']
                    waffleiron_aug = apply_bev_feature_augmentation(
                        waffleiron_single,
                        rotation_angle=discrete_angle,
                        flip_x=params.get('flip_x', False),
                        flip_y=params.get('flip_y', False),
                    )
                    waffleiron_aug_list.append(waffleiron_aug)

            # Stack for this batch item: [N, H, W] and [N, H, W, D]
            bev_aug_batch.append(torch.stack(bev_aug_list, dim=0))
            lidar_aug_batch.append(torch.stack(lidar_aug_list, dim=0))
            # WaffleIron is optional - only append if provided
            if waffleiron is not None:
                waffleiron_aug_batch.append(torch.stack(waffleiron_aug_list, dim=0))

        # Stack across batch: [B, N, H, W], [B, N, H, W, D], [B, N, C, H, W]
        bev_stacked = torch.stack(bev_aug_batch, dim=0)
        lidar_stacked = torch.stack(lidar_aug_batch, dim=0)
        waffleiron_stacked = torch.stack(waffleiron_aug_batch, dim=0) if waffleiron is not None else None

        # Initialize noise - EACH HEAD GETS INDEPENDENT NOISE
        # This matches training where each head has noise added to a DIFFERENT scene
        # Using same noise for all heads reduces ensemble diversity and doesn't match training
        x_t_list = []
        for _ in range(n_subnets):
            x_t_i = torch.ones(B, self.num_classes, H, W, D, device=device) / self.num_classes
            probs_flat = x_t_i.permute(0, 2, 3, 4, 1).reshape(-1, self.num_classes)
            samples = torch.multinomial(probs_flat, num_samples=1).squeeze(-1)
            x_t_i = F.one_hot(samples, num_classes=self.num_classes).float()
            x_t_i = x_t_i.reshape(B, H, W, D, self.num_classes)
            x_t_i = x_t_i.permute(0, 4, 1, 2, 3).to(device)
            x_t_list.append(x_t_i)
        x_t = torch.cat(x_t_list, dim=1)

        timesteps = list(range(self.num_timesteps))[::-1]
        if show_progress:
            timesteps = tqdm(timesteps, desc="MIMO TTA Sampling")

        for t in timesteps:
            t_batch = torch.full((B,), t, device=device, dtype=torch.long)
            outputs = model(x_t, t_batch, bev_stacked, lidar_stacked, waffleiron_stacked)

            # Get individual head outputs
            # Store BOTH: raw probs (augmented space) for x_t sampling, aligned probs for final ensemble
            head_probs_raw = []  # Raw probs in each head's augmented space (for x_t sampling)
            head_probs_aligned = []  # Inverse-transformed probs (for final ensemble)

            for i, head_logits in enumerate(outputs['head_outputs']):
                head_probs = F.softmax(head_logits, dim=1)  # [B, C, H, W, D]
                head_probs_raw.append(head_probs)  # Keep raw probs for independent x_t sampling

                params = aug_params_list[i]

                # Apply inverse transform to each batch item and class channel
                # NOTE: apply_inverse_augmentation expects [H, W, D] 3D tensors,
                # so we must process each class channel separately
                aligned = []
                for b in range(B):
                    probs_b = head_probs[b]  # [C, H, W, D]
                    # Apply inverse to each class channel separately
                    class_aligned = []
                    for c in range(probs_b.shape[0]):
                        probs_c = probs_b[c]  # [H, W, D]
                        probs_c_aligned = apply_inverse_augmentation(
                            probs_c,  # [H, W, D]
                            params['rotation_angle'],
                            params.get('flip_x', False),
                            params.get('flip_y', False),
                            translation=params.get('translation', (0.0, 0.0, 0.0)),
                            use_continuous=params.get('use_continuous', False),
                            voxel_shape=voxel_shape,
                        )  # [H, W, D]
                        class_aligned.append(probs_c_aligned)
                    probs_b_aligned = torch.stack(class_aligned, dim=0)  # [C, H, W, D]
                    aligned.append(probs_b_aligned)
                head_probs_aligned.append(torch.stack(aligned, dim=0))

            if t == 0:
                # FINAL STEP: Ensemble inverse-transformed probabilities for output
                ensemble_probs = torch.stack(head_probs_aligned, dim=0).mean(dim=0)
                # samples will be computed after loop for final output
            else:
                # INTERMEDIATE STEPS: Each head samples its OWN x_t from its OWN raw probs
                # This maintains independent denoising trajectories per head (matching training)
                x_t_list = []
                for head_probs in head_probs_raw:
                    # Sample x_t for this head from its raw (augmented space) probs
                    probs_flat = head_probs.permute(0, 2, 3, 4, 1).reshape(-1, self.num_classes)
                    probs_flat = probs_flat.clamp(min=1e-10)
                    probs_flat = probs_flat / probs_flat.sum(dim=-1, keepdim=True)
                    samples_i = torch.multinomial(probs_flat, num_samples=1).squeeze(-1)

                    x_t_i = F.one_hot(samples_i, num_classes=self.num_classes).float()
                    x_t_i = x_t_i.reshape(B, H, W, D, self.num_classes)
                    x_t_i = x_t_i.permute(0, 4, 1, 2, 3).to(device)
                    x_t_list.append(x_t_i)
                x_t = torch.cat(x_t_list, dim=1)

        # Apply semantic pruning if enabled (PaSCo-style)
        # NOTE: Use original (un-augmented) lidar for pruning since ensemble_probs
        # is already in the original coordinate frame after inverse transforms
        if use_semantic_pruning:
            from gssc.models.semantic_pruning import semantic_pruning_postprocess
            x_0, _ = semantic_pruning_postprocess(
                pred_probs=ensemble_probs,  # [B, C, H, W, D]
                lidar=lidar,  # [B, 1, H, W, D] - original lidar
                empty_threshold=pruning_empty_threshold,
                lidar_dilation=pruning_lidar_dilation,
                use_lidar_mask=True,
                use_occupancy_mask=True,
                combine_mode="intersection",
            )
        else:
            x_0 = ensemble_probs.argmax(dim=1)
        return x_0


class MultinomialDiffusion3DV2(MultinomialDiffusion3D):
    """
    Enhanced Multinomial Diffusion with improved loss function.

    Improvements over V1:
    1. Configurable beta_max for stronger noise schedule
    2. Focal loss for hard example mining
    3. Class-balanced weights for rare class handling
    4. Occupied voxel weighting (95% empty → need strong rebalancing)
    5. Very low weight for class 0 (empty) to focus on semantics
    6. Lovász loss component for direct IoU optimization

    Based on analysis of BEV perception loss (L3-deeper: 24.01% mIoU).
    """

    def __init__(
        self,
        num_classes: int = 20,
        num_timesteps: int = 100,
        beta_min: float = 0.0001,
        beta_max: float = 0.1,  # Much stronger than default 0.02!
        focal_gamma: float = 2.0,
        class_0_weight: float = 0.02,  # Very low weight for empty class
        occupied_weight: float = 10.0,  # Weight for non-empty voxels
        lovasz_weight: float = 0.3,  # Weight for Lovász loss
        obs_weight_factor: float = 2.0,  # Extra weight for LiDAR-observed voxels
        class_weights: torch.Tensor = None,  # Optional per-class weights
        auxiliary_loss_weight: float = 0.0,  # Auxiliary x0 reconstruction loss (paper: 0.05)
        adaptive_auxiliary_loss: bool = True,  # Weight auxiliary loss by (1-t/T)+1
        completion_weight: float = 0.0,  # Extra weight for UNOBSERVED voxels (completion task)
        loss_type: str = 'kl',  # 'kl' (posterior-based) or 'ce_direct' (direct CE on x_0)
    ):
        # Don't call parent __init__ - we'll set up our own beta schedule
        nn.Module.__init__(self)

        self.num_classes = num_classes
        self.num_timesteps = num_timesteps
        self.focal_gamma = focal_gamma
        self.class_0_weight = class_0_weight
        self.occupied_weight = occupied_weight
        self.lovasz_weight = lovasz_weight
        self.obs_weight_factor = obs_weight_factor
        self.completion_weight = completion_weight
        self.auxiliary_loss_weight = auxiliary_loss_weight
        self.adaptive_auxiliary_loss = adaptive_auxiliary_loss
        self.loss_type = loss_type

        # Register class weights
        if class_weights is not None:
            self.register_buffer('class_weights', class_weights)
        else:
            # Default: use SemanticKITTI balanced weights
            weights = self._compute_class_balanced_weights()
            self.register_buffer('class_weights', weights)

        # Create STRONGER beta schedule
        # Original: 0.0001 to 0.02 → alpha_T ≈ 0.36 (too easy!)
        # New: 0.0001 to 0.1 → alpha_T ≈ 0.0001 (much harder!)
        betas = torch.linspace(beta_min, beta_max, num_timesteps, dtype=torch.float64)

        # Compute α values
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        # Log for debugging
        print("[MultinomialDiffusion3DV2] Noise schedule:")
        print(f"  beta: {beta_min} → {beta_max}")
        print(f"  alpha_cumprod[0] = {alphas_cumprod[0]:.4f}")
        print(f"  alpha_cumprod[50] = {alphas_cumprod[50]:.4f}")
        print(f"  alpha_cumprod[99] = {alphas_cumprod[99]:.6f}")
        print(f"  At t=99: P(correct) = {alphas_cumprod[99] + (1-alphas_cumprod[99])/num_classes:.4f}")

        # Convert to log space for numerical stability
        log_alpha = torch.log(alphas)
        log_cumprod_alpha = torch.log(alphas_cumprod)
        log_1_min_alpha = log_1_min_a(log_alpha)
        log_1_min_cumprod_alpha = log_1_min_a(log_cumprod_alpha)

        # Register buffers
        self.register_buffer('betas', betas.float())
        self.register_buffer('alphas', alphas.float())
        self.register_buffer('alphas_cumprod', alphas_cumprod.float())
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev.float())

        self.register_buffer('log_alpha', log_alpha.float())
        self.register_buffer('log_cumprod_alpha', log_cumprod_alpha.float())
        self.register_buffer('log_1_min_alpha', log_1_min_alpha.float())
        self.register_buffer('log_1_min_cumprod_alpha', log_1_min_cumprod_alpha.float())

    def _compute_class_balanced_weights(self, beta: float = 0.999, max_weight: float = 20.0) -> torch.Tensor:
        """
        Compute Class-Balanced weights for SemanticKITTI.
        Formula: α_c = (1 - β) / (1 - β^n_c)
        """
        # SemanticKITTI 3D voxel frequencies (from analysis)
        frequencies = [
            0.9501,   # 0: empty (will be downweighted separately)
            0.0053,   # 1: car
            0.0001,   # 2: bicycle
            0.0001,   # 3: motorcycle
            0.0001,   # 4: truck
            0.0005,   # 5: other-vehicle
            0.0005,   # 6: person
            0.0001,   # 7: bicyclist
            0.0001,   # 8: motorcyclist
            0.0109,   # 9: road
            0.0010,   # 10: parking
            0.0108,   # 11: sidewalk
            0.0001,   # 12: other-ground
            0.0037,   # 13: building
            0.0002,   # 14: fence
            0.0076,   # 15: vegetation
            0.0010,   # 16: trunk
            0.0081,   # 17: terrain
            0.0004,   # 18: pole
            0.0003,   # 19: traffic-sign
        ]

        total_samples = 2_000_000
        sample_counts = [int(f * total_samples) for f in frequencies]

        raw_weights = []
        for n_c in sample_counts:
            if n_c == 0:
                raw_weights.append(0.0)
            else:
                raw_weights.append((1 - beta) / (1 - beta ** n_c))

        # Normalize non-zero weights
        nonzero = [w for i, w in enumerate(raw_weights) if w > 0 and i > 0]  # Exclude class 0
        if nonzero:
            mean_weight = sum(nonzero) / len(nonzero)
            normalized = [w / mean_weight if w > 0 else 0.0 for w in raw_weights]
        else:
            normalized = raw_weights

        # Clip and set class 0 weight very low
        clipped = [min(w, max_weight) for w in normalized]
        clipped[0] = self.class_0_weight  # Very low weight for empty class

        return torch.tensor(clipped, dtype=torch.float32)

    def _lovasz_grad(self, gt_sorted: torch.Tensor) -> torch.Tensor:
        """Compute gradient of the Lovász extension w.r.t sorted errors."""
        gts = gt_sorted.sum()
        intersection = gts - gt_sorted.float().cumsum(0)
        union = gts + (1 - gt_sorted).float().cumsum(0)
        jaccard = 1. - intersection / union
        if len(jaccard) > 1:
            jaccard[1:] = jaccard[1:] - jaccard[:-1]
        return jaccard

    def _lovasz_softmax_flat(self, probas: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Multi-class Lovász-Softmax loss.
        Args:
            probas: [P, C] class probabilities
            labels: [P] ground truth labels
        """
        C = probas.size(1)
        losses = []
        for c in range(1, C):  # Skip class 0 (empty)
            fg = (labels == c).float()
            if fg.sum() == 0:
                continue
            errors = (fg - probas[:, c]).abs()
            errors_sorted, perm = torch.sort(errors, descending=True)
            fg_sorted = fg[perm]
            grad = self._lovasz_grad(fg_sorted)
            losses.append((errors_sorted * grad).sum())

        if len(losses) == 0:
            return probas.sum() * 0.0
        return torch.stack(losses).mean()

    def training_losses(
        self,
        model: nn.Module,
        x_0: torch.Tensor,
        t: torch.Tensor,
        bev: torch.Tensor,
        lidar: torch.Tensor,
        lifted_features: torch.Tensor | None = None,
        nn_indices: torch.Tensor | None = None,
        **model_kwargs,
    ) -> dict[str, torch.Tensor]:
        """
        Compute enhanced training loss with PROPER KL base + enhancements:
        1. KL loss as base (proper multinomial diffusion, not CE!)
        2. Focal modulation (gamma=2) for hard example mining
        3. Class-balanced weights for rare classes
        4. Occupied voxel weighting (10x for non-empty)
        5. Very low weight for class 0 (empty)
        6. Lovász loss for IoU optimization
        7. Observation weighting (LiDAR-observed voxels get higher weight)

        Args:
            model: Neural network
            x_0: Clean data (class indices) [B, H, W, D]
            t: Timesteps [B]
            bev: BEV conditioning (class indices) [B, H, W]
            lidar: LiDAR conditioning [B, 1, H, W, D] - sparse binary input
            lifted_features: Optional S2 lifted 3D features [B, feat_dim, H, W, D]
            nn_indices: Optional precomputed NN indices for densification [B, 3, H, W, D]
            **model_kwargs: Additional kwargs passed to model (e.g., geom_bev)

        Returns:
            Dictionary with 'loss' and metrics
        """
        B = x_0.shape[0]
        device = x_0.device

        # Convert x_0 to one-hot
        x_0_onehot = F.one_hot(x_0.long(), num_classes=self.num_classes).float()
        x_0_onehot = x_0_onehot.permute(0, 4, 1, 2, 3)  # [B, K, H, W, D]

        # Convert BEV to class probabilities (handles both hard [B,H,W] and soft [B,K,H,W])
        bev_onehot = self._prepare_bev(bev)  # [B, K, H, W]

        # Sample x_t from q(x_t | x_0) — cold diffusion uses x_scpnet instead of uniform
        x_scpnet = model_kwargs.get('x_scpnet', None)
        x_t = self.q_sample(x_0_onehot, t, x_scpnet=x_scpnet)  # [B, K, H, W, D]

        # Get model prediction (x̂_0)
        x_0_logits, _ = self.p_probs(model, x_t, t, bev_onehot, lidar, lifted_features=lifted_features, nn_indices=nn_indices, **model_kwargs)
        x_0_pred_probs = F.softmax(x_0_logits, dim=1)  # [B, K, H, W, D]

        # Flatten for weighting: [B, N] where N = H*W*D
        x_0_flat = x_0.reshape(B, -1).long()  # [B, N]
        probs_flat = x_0_pred_probs.reshape(B, self.num_classes, -1)  # [B, K, N]

        if getattr(self, 'loss_type', 'kl') == 'ce_direct':
            # === DIRECT CE LOSS (bypasses posterior — matches Algorithm 2 sampling) ===
            # CE(x_0_logits, x_0) per voxel, with class weights
            x_0_logits_flat = x_0_logits.reshape(B, self.num_classes, -1)  # [B, K, N]
            ce_per_voxel = F.cross_entropy(
                x_0_logits_flat, x_0_flat,
                weight=self.class_weights.to(device),
                reduction='none',
            )  # [B, N]
            base_loss_flat = ce_per_voxel
        else:
            # === PROPER KL LOSS (reference paper) ===
            # L_{t-1} = KL(q(x_{t-1}|x_t, x_0) || p(x_{t-1}|x_t))

            # True posterior: q(x_{t-1}|x_t, x_0) — cold diffusion uses x_scpnet
            q_posterior_true = self.q_posterior(x_0_onehot, x_t, t, x_scpnet=x_scpnet)  # [B, K, H, W, D]

            # Predicted posterior: p(x_{t-1}|x_t) = q(x_{t-1}|x_t, x̂_0)
            p_posterior_pred = self.q_posterior(x_0_pred_probs, x_t, t, x_scpnet=x_scpnet)  # [B, K, H, W, D]

            # KL(q || p) per voxel = sum_k q[k] * log(q[k] / p[k])
            kl_per_class = q_posterior_true * (
                torch.log(q_posterior_true + 1e-10) - torch.log(p_posterior_pred + 1e-10)
            )  # [B, K, H, W, D]
            kl_per_voxel = kl_per_class.sum(dim=1)  # [B, H, W, D]
            base_loss_flat = kl_per_voxel.reshape(B, -1)  # [B, N]

        # === FOCAL MODULATION ===
        # Hard examples = low confidence on true x_0
        p_t = probs_flat.gather(1, x_0_flat.unsqueeze(1)).squeeze(1)  # [B, N]
        focal_weight = (1 - p_t) ** self.focal_gamma

        focal_loss = focal_weight * base_loss_flat  # [B, N]

        # === CLASS-BALANCED WEIGHTING ===
        # For ce_direct, class weights already applied in CE; for KL, apply here
        if getattr(self, 'loss_type', 'kl') == 'ce_direct':
            weighted_loss = focal_loss  # class weights already in CE
        else:
            class_weight_per_voxel = self.class_weights.to(device)[x_0_flat]  # [B, N]
            weighted_loss = focal_loss * class_weight_per_voxel  # [B, N]

        # === OCCUPIED VOXEL WEIGHTING ===
        # Weight occupied voxels (class > 0) higher
        occupied_mask = (x_0_flat > 0).float()  # [B, N]
        voxel_weights = 1.0 + (self.occupied_weight - 1.0) * occupied_mask  # [B, N]

        # === OBSERVATION WEIGHTING (from LiDAR) ===
        # Voxels directly observed by LiDAR get higher weight
        if self.obs_weight_factor > 0:
            # Handle both binary [B,1,H,W,D] and multi-channel [B,C,H,W,D] lidar
            if lidar.shape[1] == 1:
                lidar_obs = lidar.squeeze(1)  # [B, H, W, D]
            else:
                # Multi-channel (e.g. LSK3DNet 20ch): observed = any channel > 0
                lidar_obs = (lidar.abs().sum(dim=1) > 0).float()  # [B, H, W, D]
            lidar_flat = lidar_obs.reshape(B, -1)  # [B, N]
            obs_weights = 1.0 + self.obs_weight_factor * lidar_flat
            voxel_weights = voxel_weights * obs_weights

        # === COMPLETION WEIGHTING (unobserved voxels = main completion target) ===
        if self.completion_weight > 0:
            if lidar.shape[1] == 1:
                lidar_obs_cw = lidar.squeeze(1)
            else:
                lidar_obs_cw = (lidar.abs().sum(dim=1) > 0).float()
            unobs_flat = (1.0 - lidar_obs_cw).reshape(B, -1)
            voxel_weights = voxel_weights * (1.0 + self.completion_weight * unobs_flat)

        final_weighted_loss = (weighted_loss * voxel_weights).mean()

        # === LOVÁSZ LOSS (for IoU optimization) ===
        # Operates on x̂_0 predictions directly
        if self.lovasz_weight > 0:
            probs_for_lovasz = probs_flat.permute(0, 2, 1).reshape(-1, self.num_classes)  # [B*N, K]
            labels_for_lovasz = x_0_flat.reshape(-1)  # [B*N]
            lovasz_loss = self._lovasz_softmax_flat(probs_for_lovasz, labels_for_lovasz)
            total_loss = final_weighted_loss + self.lovasz_weight * lovasz_loss
        else:
            lovasz_loss = torch.tensor(0.0, device=device)
            total_loss = final_weighted_loss

        # === AUXILIARY LOSS (from pyramid diffusion paper) ===
        # Direct KL between true x_0 and predicted x_0 (excluding mask class)
        # This provides direct supervision on x_0 prediction quality
        if self.auxiliary_loss_weight > 0:
            # KL(true_x0 || pred_x0) - direct reconstruction guidance
            log_x0_true = torch.log(x_0_onehot + 1e-10)  # [B, K, H, W, D]
            log_x0_pred = torch.log(x_0_pred_probs + 1e-10)  # [B, K, H, W, D]

            # Exclude last class (mask token if any) - use all K classes for 3D SSC
            kl_aux = (x_0_onehot * (log_x0_true - log_x0_pred)).sum(dim=1)  # [B, H, W, D]
            kl_aux = kl_aux.reshape(B, -1).mean(dim=1)  # [B]

            # Adaptive weighting: higher weight at lower timesteps (closer to x_0)
            if self.adaptive_auxiliary_loss:
                aux_weight = (1.0 - t.float() / self.num_timesteps) + 1.0  # [B]
            else:
                aux_weight = torch.ones_like(t.float())

            auxiliary_loss = (aux_weight * kl_aux).mean()
            total_loss = total_loss + self.auxiliary_loss_weight * auxiliary_loss
        else:
            auxiliary_loss = torch.tensor(0.0, device=device)

        # Compute metrics for monitoring
        x_0_pred = x_0_logits.argmax(dim=1)  # [B, H, W, D]
        accuracy = (x_0_pred == x_0).float().mean()

        # Accuracy on occupied voxels only
        occupied_voxels = (x_0 > 0)
        if occupied_voxels.sum() > 0:
            occupied_accuracy = ((x_0_pred == x_0) & occupied_voxels).float().sum() / occupied_voxels.float().sum()
        else:
            occupied_accuracy = torch.tensor(0.0, device=device)

        return {
            'loss': total_loss,
            'kl_loss': final_weighted_loss,
            'lovasz_loss': lovasz_loss,
            'auxiliary_loss': auxiliary_loss,
            'accuracy': accuracy,
            'occupied_accuracy': occupied_accuracy,
            'x_t_onehot': x_t,  # S3: Return x_t for DSKD feature extraction
            'bev_onehot': bev_onehot,  # S3: Return bev_onehot for consistency
            'x_0_logits': x_0_logits,  # Reuse for KD to avoid double forward pass
        }


    @torch.no_grad()
    def sample_algo2(
        self,
        model: nn.Module,
        bev: torch.Tensor,
        lidar: torch.Tensor,
        scpnet_pred: torch.Tensor,
        shape: tuple,
        device: torch.device,
        n_steps: int = 100,
        show_progress: bool = False,
        return_softmax: bool = False,
        **model_kwargs,
    ) -> torch.Tensor:
        """Algorithm 2 (Bansal et al.) correction-based sampling.

        x_{t-1} = x_t + (α_{t-1} - α_t)·(x̂_0 - x_scpnet)
        Monotonically improves with more steps, no posterior bottleneck.

        Args:
            model: Denoising model
            bev: BEV conditioning [B, H, W] or [B, K, H, W]
            lidar: LiDAR conditioning [B, 1, H, W, D]
            scpnet_pred: SCPNet predictions (class indices) [B, H, W, D]
            shape: (B, H, W, D) output shape
            device: Device
            n_steps: Number of sampling steps (2-100)
            show_progress: Show progress bar
            **model_kwargs: Extra kwargs for model (e.g., ssc_pred)
        Returns:
            pred_scene: [B, H, W, D] class indices
        """
        K = self.num_classes
        alphas = self.alphas_cumprod.to(device)

        # Prepare inputs
        bev_oh = self._prepare_bev(bev).to(device)
        scp_oh = F.one_hot(scpnet_pred.long(), K).float().permute(0, 4, 1, 2, 3).to(device)
        x_t = scp_oh.clone()

        # Ensure ssc_pred is in model_kwargs
        if 'ssc_pred' not in model_kwargs:
            model_kwargs['ssc_pred'] = scp_oh

        # Build timestep schedule
        timesteps = list(range(99, -1, -1)) if n_steps >= 100 else \
                    list(np.linspace(99, 0, n_steps, dtype=int))

        for idx, t in enumerate(timesteps):
            B = x_t.shape[0]
            t_batch = torch.full((B,), t, device=device, dtype=torch.long)

            # Pop x_scpnet before model call (model doesn't accept it)
            x_scpnet_saved = model_kwargs.pop('x_scpnet', None)
            x_0_logits = model(x_t, t_batch, bev_oh, lidar, **model_kwargs)
            if x_scpnet_saved is not None:
                model_kwargs['x_scpnet'] = x_scpnet_saved

            x_0_pred = F.softmax(x_0_logits, dim=1)

            if idx < len(timesteps) - 1:
                # Intermediate step: apply correction toward x_0_pred
                t_next = timesteps[idx + 1]
                alpha_t = alphas[t]
                alpha_next = alphas[t_next]
                delta_alpha = alpha_next - alpha_t
                correction = x_0_pred - scp_oh
                x_t = x_t + delta_alpha * correction
                x_t = x_t.clamp(min=0.0)
                x_t = x_t / (x_t.sum(dim=1, keepdim=True) + 1e-10)
            else:
                # Last step: use direct x_0 prediction
                x_t = x_0_pred

        if return_softmax:
            return x_t  # [B, K, H, W, D] simplex-valued
        return x_t.argmax(dim=1)


class MultinomialDiffusion3DEMA:
    """
    Exponential Moving Average for model parameters.
    Used during training to get smoother model for inference.
    """

    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

        # Initialize shadow weights
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        """Update shadow weights."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = (
                    self.decay * self.shadow[name] + (1 - self.decay) * param.data
                )

    def apply_shadow(self):
        """Apply shadow weights to model (for inference)."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self):
        """Restore original weights (after inference)."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}


if __name__ == "__main__":
    # Test the diffusion process
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create diffusion
    diffusion = MultinomialDiffusion3D(
        num_classes=20,
        num_timesteps=100,
    ).to(device)

    print(f"Diffusion created with {diffusion.num_timesteps} timesteps")

    # Test forward diffusion
    B, H, W, D = 2, 32, 32, 8
    x_0 = torch.randint(0, 20, (B, H, W, D), device=device)
    x_0_onehot = F.one_hot(x_0.long(), num_classes=20).float()
    x_0_onehot = x_0_onehot.permute(0, 4, 1, 2, 3)  # [B, K, H, W, D]

    t = torch.randint(0, 100, (B,), device=device)

    print(f"Input shape: {x_0.shape}")
    print(f"One-hot shape: {x_0_onehot.shape}")

    # Sample x_t
    x_t = diffusion.q_sample(x_0_onehot, t)
    print(f"Sampled x_t shape: {x_t.shape}")

    # Check that x_t is one-hot
    assert x_t.sum(dim=1).allclose(torch.ones_like(x_t[:, 0])), "x_t should be one-hot"
    print("Forward diffusion test passed!")
