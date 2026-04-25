"""
Variance Exploding (VE) Gaussian Diffusion for 3D Semantic Voxel Completion.

Key differences from standard VP (Variance Preserving) DDPM:

1. FORWARD PROCESS:
   - VP:  x_t = sqrt(α̅_t) * x_0 + sqrt(1-α̅_t) * ε   (signal scaled DOWN)
   - VE:  x_t = x_0 + σ_t * ε                         (signal PRESERVED)

2. RECOVERY:
   - VP:  x_0 = (x_t - σ_t * ε_pred) / sqrt(α̅_t)     (division by tiny α → AMPLIFICATION!)
   - VE:  x_0 = x_t - σ_t * ε_pred                    (simple subtraction → NO AMPLIFICATION)

3. REPRESENTATION:
   - Old: 21 channels, logit-scaled one-hot {-5, +5} → bimodal, hostile to Gaussian noise
   - New: 20 channels, soft probabilities [0.005, 0.905] → quasi-continuous, smooth

4. IMPLICIT OCCUPANCY:
   - Channel 0 = P(empty/class_0)
   - Channels 1-19 = P(class_1), ..., P(class_19)
   - No separate occupancy channel needed!

This matches DiffSSC's approach more closely:
- DiffSSC uses VE-style: x_t = x_0 + σ_t * ε
- DiffSSC never samples from pure noise (uses SDEdit from partial points)
- DiffSSC operates on continuous xyz coordinates

Reference: DiffSSC (IROS 2025) — diffssc/models/models.py
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
from tqdm import tqdm


def extract(a: torch.Tensor, t: torch.Tensor, x_shape: Tuple[int, ...]) -> torch.Tensor:
    """Extract values from a at timestep t and reshape for broadcasting."""
    B = t.shape[0]
    out = a.gather(-1, t.clamp(0, a.shape[0] - 1))
    return out.reshape(B, *((1,) * (len(x_shape) - 1)))


def linear_sigma_schedule(T: int, sigma_min: float = 0.01, sigma_max: float = 1.0) -> torch.Tensor:
    """Linear sigma schedule for VE diffusion.

    σ_t increases linearly from sigma_min to sigma_max.
    At t=0: σ_0 = sigma_min (almost no noise)
    At t=T-1: σ_{T-1} = sigma_max (full noise)
    """
    return torch.linspace(sigma_min, sigma_max, T, dtype=torch.float64)


def exponential_sigma_schedule(T: int, sigma_min: float = 0.01, sigma_max: float = 1.0) -> torch.Tensor:
    """Exponential sigma schedule (used by Song et al. score-based models).

    σ_t = sigma_min * (sigma_max / sigma_min)^(t / (T-1))
    """
    t = torch.arange(T, dtype=torch.float64)
    return sigma_min * (sigma_max / sigma_min) ** (t / (T - 1))


def cosine_sigma_schedule(T: int, sigma_min: float = 0.01, sigma_max: float = 1.0) -> torch.Tensor:
    """Cosine sigma schedule — smoother transitions.

    σ_t follows a cosine curve from sigma_min to sigma_max.
    """
    t = torch.arange(T, dtype=torch.float64)
    # Cosine goes from 1 to -1 as t goes from 0 to T-1
    # We want sigma to go from sigma_min to sigma_max
    cos_t = torch.cos(math.pi * t / (T - 1))  # 1 → -1
    # Map from [1, -1] to [sigma_min, sigma_max]
    return sigma_min + 0.5 * (1 - cos_t) * (sigma_max - sigma_min)


class GaussianDiffusionVE(nn.Module):
    """Variance Exploding Gaussian diffusion for 3D semantic voxel completion.

    Representation: 20 channels per voxel (soft probabilities over 20 classes).
    Channel 0 = P(empty), Channels 1-19 = P(semantic class k).

    Args:
        num_classes: Number of semantic classes (including empty = class 0)
        num_timesteps: Number of diffusion steps
        sigma_min: Minimum noise level (at t=0)
        sigma_max: Maximum noise level (at t=T-1)
        sigma_schedule: Schedule type ('linear', 'exponential', 'cosine')
        label_smoothing: Smoothing factor for one-hot encoding (0.0 = hard, 0.1 = soft)
        lambda_reg: Regularization weight for noise prediction
    """

    def __init__(
        self,
        num_classes: int = 20,
        num_timesteps: int = 1000,
        sigma_min: float = 0.01,
        sigma_max: float = 1.0,
        sigma_schedule: str = 'linear',
        label_smoothing: float = 0.1,
        lambda_reg: float = 5.0,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.num_timesteps = num_timesteps
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.label_smoothing = label_smoothing
        self.lambda_reg = lambda_reg

        # Build sigma schedule
        if sigma_schedule == 'linear':
            sigmas = linear_sigma_schedule(num_timesteps, sigma_min, sigma_max)
        elif sigma_schedule == 'exponential':
            sigmas = exponential_sigma_schedule(num_timesteps, sigma_min, sigma_max)
        elif sigma_schedule == 'cosine':
            sigmas = cosine_sigma_schedule(num_timesteps, sigma_min, sigma_max)
        else:
            raise ValueError(f"Unknown sigma_schedule: {sigma_schedule}")

        self.register_buffer('sigmas', sigmas.float())

        # For compatibility with DPM-Solver (which expects alpha/beta)
        # In VE, we can fake these: α̅_t = 1, σ_t = sqrt(1 - α̅_t) doesn't apply
        # Instead, we directly use sigmas

        # Precompute for DDPM-style sampling
        # In VE, the posterior is different from VP
        # x_{t-1} = x_0 + σ_{t-1} * ε_new, where ε_new ~ N(0, I)
        # But we can also do DDPM-style with coefficients...
        # For simplicity, we'll use the direct VE formulation

    def encode_x0(self, gt_labels: torch.Tensor) -> torch.Tensor:
        """Encode GT class indices [B, H, W, D] → soft probs [B, 20, H, W, D].

        Uses label smoothing to create quasi-continuous distribution:
        - Correct class gets probability (1 - ε)
        - Other classes get probability ε / (K - 1)

        With ε = 0.1, K = 20:
        - Correct class: 0.9
        - Other classes: 0.1 / 19 ≈ 0.00526

        This avoids the bimodal {0, 1} spike problem!
        """
        B, H, W, D = gt_labels.shape

        # One-hot encoding
        one_hot = F.one_hot(gt_labels.long(), self.num_classes).float()  # [B,H,W,D,20]

        # Label smoothing
        if self.label_smoothing > 0:
            smooth = self.label_smoothing / (self.num_classes - 1)
            one_hot = one_hot * (1 - self.label_smoothing - smooth) + smooth
            # Correct class: 1 - ε + ε/(K-1) - ε/(K-1) = 1 - ε
            # Hmm, let me redo this more carefully
            # We want: correct class = 1 - ε, others = ε / (K-1)
            # one_hot starts as: correct = 1, others = 0
            # After: one_hot * (1 - ε) + (1 - one_hot) * ε / (K-1)
            #      = one_hot - one_hot * ε + ε/(K-1) - one_hot * ε/(K-1)
            #      = one_hot * (1 - ε - ε/(K-1)) + ε/(K-1)

        # Actually, let's do it correctly:
        if self.label_smoothing > 0:
            eps = self.label_smoothing
            K = self.num_classes
            # correct class: 1 - eps + eps/K (we spread eps uniformly, then add back the one-hot)
            # Actually standard label smoothing:
            # soft = (1 - eps) * one_hot + eps / K
            one_hot = F.one_hot(gt_labels.long(), self.num_classes).float()
            soft = (1 - eps) * one_hot + eps / K
        else:
            soft = one_hot

        # Permute to [B, 20, H, W, D]
        return soft.permute(0, 4, 1, 2, 3)

    def decode_x0(self, x_0: torch.Tensor) -> torch.Tensor:
        """Decode soft probs [B, 20, H, W, D] → class indices [B, H, W, D].

        Simply takes argmax over the class dimension.
        Class 0 = empty, Classes 1-19 = semantic.
        """
        # Clamp to valid range (noise can push outside [0, 1])
        x_0_clamped = x_0.clamp(0, 1)

        # Argmax over class dimension
        return x_0_clamped.argmax(dim=1)

    def q_sample(self, x_0: torch.Tensor, t: torch.Tensor,
                 noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """VE Forward process: x_t = x_0 + σ_t * ε

        Signal is PRESERVED, only noise is added.
        """
        if noise is None:
            noise = torch.randn_like(x_0)

        sigma = extract(self.sigmas, t, x_0.shape)
        return x_0 + sigma * noise

    def _predict_x0(self, x_t: torch.Tensor, t: torch.Tensor,
                    eps_pred: torch.Tensor) -> torch.Tensor:
        """VE Recovery: x_0 = x_t - σ_t * ε_pred

        NO division by sqrt(alpha_bar)! NO amplification!
        """
        sigma = extract(self.sigmas, t, x_t.shape)
        return x_t - sigma * eps_pred

    def _global_regularization(self, eps_pred: torch.Tensor) -> torch.Tensor:
        """Regularization on predicted noise: mean → 0, std → 1.

        This encourages the model to predict proper Gaussian noise.
        """
        mean = eps_pred.mean()
        std = eps_pred.std()
        return mean ** 2 + (std - 1.0) ** 2

    def training_losses(
        self,
        model: nn.Module,
        x_0: torch.Tensor,  # GT labels [B, H, W, D]
        t: torch.Tensor,
        bev: torch.Tensor,  # BEV condition [B, H, W] (may be ignored)
        lidar: torch.Tensor,  # LiDAR condition [B, 20, H, W, D]
        lifted_features: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute VE diffusion training loss.

        L = MSE(ε̂, ε) + λ_reg * (mean² + (std-1)²)
        """
        # Encode GT labels to soft probabilities
        x_0_soft = self.encode_x0(x_0)  # [B, 20, H, W, D]

        # Sample noise
        noise = torch.randn_like(x_0_soft)

        # VE Forward: x_t = x_0 + σ_t * ε
        x_t = self.q_sample(x_0_soft, t, noise)

        # Model predicts noise
        # Convert BEV to one-hot for interface compatibility
        bev_onehot = F.one_hot(bev.long(), num_classes=self.num_classes).float()
        bev_onehot = bev_onehot.permute(0, 3, 1, 2)  # [B, K, H, W]

        eps_pred = model(x_t, t, bev_onehot, lidar, lifted_features=lifted_features)

        # MSE loss on noise
        mse_loss = F.mse_loss(eps_pred, noise)

        # Regularization
        reg_loss = self._global_regularization(eps_pred)

        loss = mse_loss + self.lambda_reg * reg_loss

        # Monitoring: predict x_0 and compute accuracy
        with torch.no_grad():
            x_0_pred = self._predict_x0(x_t, t, eps_pred)
            pred_labels = self.decode_x0(x_0_pred)
            accuracy = (pred_labels == x_0).float().mean()

        return {
            'loss': loss,
            'mse_loss': mse_loss.detach(),
            'reg_loss': reg_loss.detach(),
            'accuracy': accuracy,
        }

    @torch.no_grad()
    def _p_sample_ve(self, model: nn.Module, x_t: torch.Tensor, t: torch.Tensor,
                     t_prev: torch.Tensor, bev: torch.Tensor, lidar: torch.Tensor,
                     lifted_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Single VE reverse step: x_{t-1} from x_t.

        VE reverse process:
        1. Predict noise: ε̂ = model(x_t, t)
        2. Estimate x_0: x̂_0 = x_t - σ_t * ε̂
        3. Re-noise to t-1: x_{t-1} = x̂_0 + σ_{t-1} * ε_new

        This is different from VP-DDPM which uses posterior mean/variance.
        """
        B = x_t.shape[0]

        # Predict noise
        eps_pred = model(x_t, t, bev, lidar, lifted_features=lifted_features)

        # Estimate x_0
        x_0_pred = self._predict_x0(x_t, t, eps_pred)

        # Get sigma values
        sigma_t = extract(self.sigmas, t, x_t.shape)
        sigma_prev = extract(self.sigmas, t_prev.clamp(min=0), x_t.shape)

        # At t=0, we just return x_0_pred
        # For t > 0, we re-noise to sigma_{t-1}
        # x_{t-1} = x_0_pred + sigma_{t-1} * z, where z ~ N(0, I)
        noise = torch.randn_like(x_t)

        # Mask out noise for t=0
        nonzero_mask = (t != 0).float().view(B, *([1] * (x_t.dim() - 1)))

        x_prev = x_0_pred + nonzero_mask * sigma_prev * noise

        return x_prev

    @torch.no_grad()
    def sample(
        self,
        model: nn.Module,
        bev: torch.Tensor,
        lidar: torch.Tensor,
        shape: Tuple[int, int, int, int],
        device: torch.device,
        lifted_features: Optional[torch.Tensor] = None,
        show_progress: bool = True,
    ) -> torch.Tensor:
        """Sample from VE diffusion (full reverse process).

        Starts from x_T = x_0_init + σ_T * noise, where x_0_init is uniform or zeros.

        Note: For best results, use sample_sdedit with a good initialization!
        """
        B, H, W, D = shape
        model.eval()

        bev_onehot = F.one_hot(bev.long(), num_classes=self.num_classes).float()
        bev_onehot = bev_onehot.permute(0, 3, 1, 2).to(device)

        # Start from noise (VE: x_T = 0 + σ_T * ε, assuming x_0 ≈ 0)
        # Or we could start from uniform distribution (1/K for each class)
        x_0_init = torch.ones(B, self.num_classes, H, W, D, device=device) / self.num_classes
        x_t = x_0_init + self.sigmas[-1] * torch.randn_like(x_0_init)

        timesteps = list(range(self.num_timesteps))[::-1]
        if show_progress:
            timesteps = tqdm(timesteps, desc="VE Sampling")

        for i, t_val in enumerate(timesteps):
            t_batch = torch.full((B,), t_val, device=device, dtype=torch.long)
            t_prev = torch.full((B,), max(t_val - 1, 0), device=device, dtype=torch.long)
            x_t = self._p_sample_ve(model, x_t, t_batch, t_prev, bev_onehot, lidar,
                                    lifted_features=lifted_features)

        return self.decode_x0(x_t)

    @torch.no_grad()
    def sample_sdedit(
        self,
        model: nn.Module,
        bev: torch.Tensor,
        lidar: torch.Tensor,
        init_probs: torch.Tensor,  # [B, 20, H, W, D] soft probs from LSK3DNet/BEV
        shape: Tuple[int, int, int, int],
        device: torch.device,
        lifted_features: Optional[torch.Tensor] = None,
        start_timestep: int = 500,
        show_progress: bool = True,
    ) -> torch.Tensor:
        """SDEdit-style sampling: start from initialization + noise.

        This is how DiffSSC actually works:
        1. Start from partial observation (here: LSK3DNet soft probs)
        2. Add noise at start_timestep
        3. Denoise from start_timestep → 0

        Args:
            init_probs: [B, 20, H, W, D] soft probabilities (e.g., from LSK3DNet)
            start_timestep: Where to start denoising (0 = no change, T = full generation)
        """
        B, H, W, D = shape
        model.eval()

        bev_onehot = F.one_hot(bev.long(), num_classes=self.num_classes).float()
        bev_onehot = bev_onehot.permute(0, 3, 1, 2).to(device)

        # Add noise at start_timestep
        t_start = torch.full((B,), start_timestep, device=device, dtype=torch.long)
        noise = torch.randn_like(init_probs)
        x_t = self.q_sample(init_probs, t_start, noise)

        # Denoise from start_timestep → 0
        timesteps = list(range(start_timestep, -1, -1))
        if show_progress:
            timesteps = tqdm(timesteps, desc=f"SDEdit (t={start_timestep}→0)")

        for t_val in timesteps:
            t_batch = torch.full((B,), t_val, device=device, dtype=torch.long)
            t_prev = torch.full((B,), max(t_val - 1, 0), device=device, dtype=torch.long)
            x_t = self._p_sample_ve(model, x_t, t_batch, t_prev, bev_onehot, lidar,
                                    lifted_features=lifted_features)

        return self.decode_x0(x_t)

    @torch.no_grad()
    def sample_dpm_solver(
        self,
        model: nn.Module,
        bev: torch.Tensor,
        lidar: torch.Tensor,
        shape: Tuple[int, int, int, int],
        device: torch.device,
        lifted_features: Optional[torch.Tensor] = None,
        init_probs: Optional[torch.Tensor] = None,
        start_timestep: Optional[int] = None,
        num_steps: int = 50,
        show_progress: bool = True,
    ) -> torch.Tensor:
        """Fast sampling with DPM-Solver++.

        For VE diffusion, we need to configure the scheduler appropriately.
        """
        from diffusers import DPMSolverMultistepScheduler

        B, H, W, D = shape
        model.eval()

        bev_onehot = F.one_hot(bev.long(), num_classes=self.num_classes).float()
        bev_onehot = bev_onehot.permute(0, 3, 1, 2).to(device)

        # For VE diffusion with DPM-Solver, we need to be careful
        # DPM-Solver expects VP-style betas, but we can use custom sigma schedule
        # Let's use the "sample" prediction type and pass x_0 directly

        # Create scheduler with our sigma schedule
        # Note: diffusers DPMSolverMultistepScheduler doesn't directly support VE
        # We'll use a simple manual implementation for now

        # Initialize
        if init_probs is not None and start_timestep is not None:
            # SDEdit mode
            t_start = torch.full((B,), start_timestep, device=device, dtype=torch.long)
            noise = torch.randn_like(init_probs)
            x_t = self.q_sample(init_probs, t_start, noise)
            actual_start = start_timestep
        else:
            # Full generation from uniform
            x_0_init = torch.ones(B, self.num_classes, H, W, D, device=device) / self.num_classes
            x_t = x_0_init + self.sigmas[-1] * torch.randn_like(x_0_init)
            actual_start = self.num_timesteps - 1

        # Create timestep sequence (fewer steps for fast sampling)
        step_ratio = actual_start / num_steps
        timesteps = [int(actual_start - i * step_ratio) for i in range(num_steps)]
        timesteps = [max(0, t) for t in timesteps]  # Ensure non-negative
        timesteps.append(0)  # Always end at 0

        if show_progress:
            timesteps_iter = tqdm(range(len(timesteps) - 1), desc=f"DPM++ ({num_steps} steps)")
        else:
            timesteps_iter = range(len(timesteps) - 1)

        for i in timesteps_iter:
            t_val = timesteps[i]
            t_next = timesteps[i + 1]

            t_batch = torch.full((B,), t_val, device=device, dtype=torch.long)
            t_prev = torch.full((B,), t_next, device=device, dtype=torch.long)

            # Predict noise
            eps_pred = model(x_t, t_batch, bev_onehot, lidar, lifted_features=lifted_features)

            # Estimate x_0
            x_0_pred = self._predict_x0(x_t, t_batch, eps_pred)

            # Re-noise to t_next (if t_next > 0)
            if t_next > 0:
                sigma_next = self.sigmas[t_next]
                noise = torch.randn_like(x_t)
                x_t = x_0_pred + sigma_next * noise
            else:
                x_t = x_0_pred

        return self.decode_x0(x_t)


class VEModelWrapper(nn.Module):
    """Wrapper to adapt existing UNet for VE diffusion.

    Changes:
    - Input: 20 channels (soft probs) instead of 21 (logit-encoded)
    - Output: 20 channels (noise prediction)

    The wrapped model should handle the new channel count.
    """

    def __init__(self, base_model: nn.Module, num_classes: int = 20):
        super().__init__()
        self.base_model = base_model
        self.num_classes = num_classes

    def forward(
        self,
        x_t: torch.Tensor,  # [B, 20, H, W, D]
        t: torch.Tensor,
        bev: torch.Tensor,  # [B, 20, H, W] one-hot (may be ignored)
        lidar: torch.Tensor,  # [B, 20, H, W, D] one-hot LiDAR semantics
        lifted_features: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Forward pass: predict noise given noisy input and conditions."""
        return self.base_model(x_t, t, lidar, lifted_features=lifted_features, **kwargs)
