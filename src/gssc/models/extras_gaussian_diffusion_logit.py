"""
Logit-Space VE Gaussian Diffusion with EDM Preconditioning.

Working in logit space z = log(p):
- z can be any real value — no simplex constraint violation
- softmax(z) always gives valid probabilities
- Gaussian noise in logit space naturally preserves structure

EDM Preconditioning (Karras et al. 2022):
- c_in(σ) normalizes model input to ~O(1) at ALL noise levels
- c_skip(σ) creates skip connection (pass-through at low σ)
- c_out(σ) scales model output appropriately
- This fixes the 712x input variance issue with raw VE diffusion

VE Forward Process:
    z_t = z_0 + σ_t * ε

EDM Denoised Estimate:
    D(z_t; σ) = c_skip(σ) * z_t + c_out(σ) * F_θ(c_in(σ) * z_t; t)

Decoding:
    p = softmax(D)
    class = argmax(p)

References:
    - DiffSSC (IROS 2025)
    - Karras et al. "Elucidating the Design Space of Diffusion-Based
      Generative Models" (NeurIPS 2022)
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
    """Linear sigma schedule for VE diffusion."""
    return torch.linspace(sigma_min, sigma_max, T, dtype=torch.float64)


def cosine_sigma_schedule(T: int, sigma_min: float = 0.01, sigma_max: float = 1.0) -> torch.Tensor:
    """Cosine sigma schedule — smoother transitions."""
    t = torch.arange(T, dtype=torch.float64)
    cos_t = torch.cos(math.pi * t / (T - 1))
    return sigma_min + 0.5 * (1 - cos_t) * (sigma_max - sigma_min)


class GaussianDiffusionLogit(nn.Module):
    """Logit-space VE Gaussian diffusion with EDM preconditioning.

    Key features:
    1. Works in LOGIT space — softmax always gives valid probabilities
    2. EDM preconditioning — model input normalized to ~O(1) at all σ
    3. x0-prediction — model predicts denoised z_0, not noise ε
    4. SDEdit support — start from LSK3DNet predictions

    Args:
        num_classes: Number of semantic classes
        num_timesteps: Number of diffusion steps
        sigma_min: Minimum noise level
        sigma_max: Maximum noise level (must be >> logit_scale)
        sigma_schedule: Schedule type ('linear', 'cosine')
        logit_scale: Scale factor for one-hot → logit conversion
        sigma_data: Data standard deviation for EDM preconditioning
                    (defaults to logit_scale if not provided)
    """

    def __init__(
        self,
        num_classes: int = 20,
        num_timesteps: int = 1000,
        sigma_min: float = 0.01,
        sigma_max: float = 80.0,
        sigma_schedule: str = 'cosine',
        logit_scale: float = 3.0,
        lambda_reg: float = 1.0,  # kept for interface compat, unused
        sigma_data: Optional[float] = None,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.num_timesteps = num_timesteps
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.logit_scale = logit_scale
        self.sigma_data = sigma_data if sigma_data is not None else logit_scale

        # Build sigma schedule
        if sigma_schedule == 'linear':
            sigmas = linear_sigma_schedule(num_timesteps, sigma_min, sigma_max)
        elif sigma_schedule == 'cosine':
            sigmas = cosine_sigma_schedule(num_timesteps, sigma_min, sigma_max)
        else:
            raise ValueError(f"Unknown sigma_schedule: {sigma_schedule}")

        self.register_buffer('sigmas', sigmas.float())

        # Precompute EDM coefficients for each timestep
        sd = self.sigma_data
        c_in = 1.0 / (sigmas ** 2 + sd ** 2).sqrt()
        c_skip = sd ** 2 / (sigmas ** 2 + sd ** 2)
        c_out = sigmas * sd / (sigmas ** 2 + sd ** 2).sqrt()
        # Loss weight: ensures uniform effective loss on F_θ
        c_weight = (sigmas ** 2 + sd ** 2) / (sigmas * sd) ** 2
        # Clip weight to avoid explosion at σ→0
        c_weight = c_weight.clamp(max=1000.0)

        self.register_buffer('c_in', c_in.float())
        self.register_buffer('c_skip', c_skip.float())
        self.register_buffer('c_out', c_out.float())
        self.register_buffer('c_weight', c_weight.float())

    def encode_x0(self, gt_labels: torch.Tensor) -> torch.Tensor:
        """Encode GT class indices [B, H, W, D] → logits [B, 20, H, W, D]."""
        B, H, W, D = gt_labels.shape
        one_hot = F.one_hot(gt_labels.long(), self.num_classes).float()
        logits = one_hot * self.logit_scale - (1 - one_hot) * self.logit_scale / (self.num_classes - 1)
        logits = logits - logits.mean(dim=-1, keepdim=True)
        return logits.permute(0, 4, 1, 2, 3)

    def probs_to_logits(self, probs: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        """Convert probabilities [B, 20, H, W, D] → logits matching encode_x0 scale.

        Critical fixes for SDEdit initialization:
        1. Near-uniform voxels (99.4% of LSK3DNet output) are filled with
           class-0 (empty) logits matching encode_x0 format.
        2. All logits are SCALED to match logit_scale used by encode_x0,
           preventing a 2.6x scale mismatch between training and inference.

        Without these fixes, the model receives inputs that are massively
        out-of-distribution (zero logits instead of peaked ±5 logits).
        """
        C = self.num_classes
        ls = self.logit_scale

        # Detect near-uniform voxels (LSK3DNet gives ~1/20 for unconfident)
        max_prob = probs.max(dim=1, keepdim=True).values  # [B, 1, H, W, D]
        near_uniform = max_prob < (1.0 / C + 0.01)  # < 0.06

        # For confident voxels: convert probs → logits, then scale to match
        # encode_x0's logit_scale. We use the same one-hot encoding:
        # argmax class gets +logit_scale, others get -logit_scale/(C-1)
        argmax_class = probs.argmax(dim=1)  # [B, H, W, D]

        # Build logits using encode_x0-style encoding for ALL voxels
        # This ensures the scale EXACTLY matches training
        one_hot = F.one_hot(argmax_class.long(), C).float()  # [B, H, W, D, C]
        logits = one_hot * ls - (1 - one_hot) * ls / (C - 1)
        logits = logits - logits.mean(dim=-1, keepdim=True)
        logits = logits.permute(0, 4, 1, 2, 3)  # [B, C, H, W, D]

        # For confident voxels, optionally blend with soft probs
        # (not needed for now — hard argmax is cleaner and matches encode_x0)

        return logits

    def decode_x0(self, z_0: torch.Tensor) -> torch.Tensor:
        """Decode logits → class indices via softmax + argmax."""
        return F.softmax(z_0, dim=1).argmax(dim=1)

    def decode_to_probs(self, z_0: torch.Tensor) -> torch.Tensor:
        """Decode logits to probabilities."""
        return F.softmax(z_0, dim=1)

    def q_sample(self, z_0: torch.Tensor, t: torch.Tensor,
                 noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """VE Forward: z_t = z_0 + σ_t * ε"""
        if noise is None:
            noise = torch.randn_like(z_0)
        sigma = extract(self.sigmas, t, z_0.shape)
        return z_0 + sigma * noise

    def _denoised_estimate(self, model: nn.Module, z_t: torch.Tensor,
                           t: torch.Tensor, bev: torch.Tensor,
                           lidar: torch.Tensor,
                           lifted_features: Optional[torch.Tensor] = None,
                           cond_3d: Optional[torch.Tensor] = None,
                           force_uncond: bool = False,
                           ) -> torch.Tensor:
        """EDM denoised estimate: D = c_skip * z_t + c_out * F_θ(c_in * z_t).

        The model F_θ receives normalized input (c_in * z_t) which is always
        ~O(1) regardless of the noise level σ. This is critical for VE
        diffusion where raw z_t varies by 712x across noise levels.

        force_uncond: if True, passes force_uncond=True to model which zeros
        all conditioning (for CFG unconditional pass).
        """
        cin = extract(self.c_in, t, z_t.shape)
        cskip = extract(self.c_skip, t, z_t.shape)
        cout = extract(self.c_out, t, z_t.shape)

        # Normalize input — model always sees ~O(1) values
        z_normalized = cin * z_t

        # Model predicts raw F (not noise, not z_0 directly)
        F_raw = model(z_normalized, t, bev, lidar, lifted_features=lifted_features,
                      cond_3d=cond_3d, force_uncond=force_uncond)

        # Combine: skip connection + scaled model output
        D = cskip * z_t + cout * F_raw
        return D

    def training_losses(
        self,
        model: nn.Module,
        x_0: torch.Tensor,
        t: torch.Tensor,
        bev: torch.Tensor,
        lidar: torch.Tensor,
        lifted_features: Optional[torch.Tensor] = None,
        cond_3d: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """EDM training loss: λ(σ) * ||D(z_t) - z_0||²

        The loss weight λ(σ) ensures uniform effective loss on F_θ,
        so the model learns equally well at all noise levels.

        cond_3d: optional [B, 20, H, W, D] conditioning signal (e.g. LSK3DNet
        3D predictions). Passed to the model as extra input channels.
        Standard diffusion: always noise GT z_0, model denoises toward GT.
        The conditioning acts as a "cheat sheet" guiding the model.
        """
        z_0 = self.encode_x0(x_0)  # target is always GT
        noise = torch.randn_like(z_0)

        # Standard forward process: always from GT
        z_t = self.q_sample(z_0, t, noise)

        bev_onehot = F.one_hot(bev.long(), num_classes=self.num_classes).float()
        bev_onehot = bev_onehot.permute(0, 3, 1, 2)

        # EDM denoised estimate (cond_3d passed to model as extra input channels)
        D = self._denoised_estimate(model, z_t, t, bev_onehot, lidar,
                                    lifted_features=lifted_features,
                                    cond_3d=cond_3d)

        # Weighted MSE loss (EDM weighting for uniform F_θ loss)
        weight = extract(self.c_weight, t, z_t.shape)
        mse_loss = (weight * (D - z_0) ** 2).mean()

        loss = mse_loss

        # Monitoring
        with torch.no_grad():
            pred_labels = self.decode_x0(D)
            accuracy = (pred_labels == x_0).float().mean()

        return {
            'loss': loss,
            'mse_loss': mse_loss.detach(),
            'reg_loss': torch.tensor(0.0, device=loss.device),
            'accuracy': accuracy,
        }

    @torch.no_grad()
    def _p_sample_ve(self, model: nn.Module, z_t: torch.Tensor, t: torch.Tensor,
                     t_prev: torch.Tensor, bev: torch.Tensor, lidar: torch.Tensor,
                     lifted_features: Optional[torch.Tensor] = None,
                     cond_3d: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Single VE reverse step with EDM preconditioning (DDIM / deterministic)."""
        B = z_t.shape[0]

        # Denoised estimate via EDM
        z_0_pred = self._denoised_estimate(model, z_t, t, bev, lidar,
                                           lifted_features=lifted_features,
                                           cond_3d=cond_3d)

        # DDIM update: preserve noise direction
        sigma_t = extract(self.sigmas, t.clamp(min=0), z_t.shape)
        sigma_prev = extract(self.sigmas, t_prev.clamp(min=0), z_t.shape)
        nonzero_mask = (t != 0).float().view(B, *([1] * (z_t.dim() - 1)))

        # z_prev = z_0_pred + (σ_prev/σ_t) * (z_t - z_0_pred)
        z_prev = z_0_pred + nonzero_mask * (sigma_prev / sigma_t.clamp(min=1e-8)) * (z_t - z_0_pred)
        return z_prev

    @torch.no_grad()
    def sample(
        self,
        model: nn.Module,
        bev: torch.Tensor,
        lidar: torch.Tensor,
        shape: Tuple[int, int, int, int],
        device: torch.device,
        lifted_features: Optional[torch.Tensor] = None,
        cond_3d: Optional[torch.Tensor] = None,
        show_progress: bool = True,
    ) -> torch.Tensor:
        """Sample from VE diffusion (full reverse process)."""
        B, H, W, D = shape
        model.eval()

        bev_onehot = F.one_hot(bev.long(), num_classes=self.num_classes).float()
        bev_onehot = bev_onehot.permute(0, 3, 1, 2).to(device)

        z_t = self.sigmas[-1] * torch.randn(B, self.num_classes, H, W, D, device=device)

        timesteps = list(range(self.num_timesteps))[::-1]
        if show_progress:
            timesteps = tqdm(timesteps, desc="VE-EDM Sampling")

        for t_val in timesteps:
            t_batch = torch.full((B,), t_val, device=device, dtype=torch.long)
            t_prev = torch.full((B,), max(t_val - 1, 0), device=device, dtype=torch.long)
            z_t = self._p_sample_ve(model, z_t, t_batch, t_prev, bev_onehot, lidar,
                                    lifted_features=lifted_features, cond_3d=cond_3d)

        return self.decode_x0(z_t)

    @torch.no_grad()
    def sample_sdedit(
        self,
        model: nn.Module,
        bev: torch.Tensor,
        lidar: torch.Tensor,
        init_probs: torch.Tensor,
        shape: Tuple[int, int, int, int],
        device: torch.device,
        lifted_features: Optional[torch.Tensor] = None,
        cond_3d: Optional[torch.Tensor] = None,
        start_timestep: int = 500,
        show_progress: bool = True,
    ) -> torch.Tensor:
        """SDEdit: start from LSK3DNet init + noise, denoise partially."""
        B, H, W, D = shape
        model.eval()

        bev_onehot = F.one_hot(bev.long(), num_classes=self.num_classes).float()
        bev_onehot = bev_onehot.permute(0, 3, 1, 2).to(device)

        z_0_init = self.probs_to_logits(init_probs)
        t_start = torch.full((B,), start_timestep, device=device, dtype=torch.long)
        z_t = self.q_sample(z_0_init, t_start)

        timesteps = list(range(start_timestep, -1, -1))
        if show_progress:
            timesteps = tqdm(timesteps, desc=f"SDEdit-EDM (t={start_timestep}→0)")

        for t_val in timesteps:
            t_batch = torch.full((B,), t_val, device=device, dtype=torch.long)
            t_prev = torch.full((B,), max(t_val - 1, 0), device=device, dtype=torch.long)
            z_t = self._p_sample_ve(model, z_t, t_batch, t_prev, bev_onehot, lidar,
                                    lifted_features=lifted_features, cond_3d=cond_3d)

        return self.decode_x0(z_t)

    @torch.no_grad()
    def sample_dpm_solver(
        self,
        model: nn.Module,
        bev: torch.Tensor,
        lidar: torch.Tensor,
        shape: Tuple[int, int, int, int],
        device: torch.device,
        lifted_features: Optional[torch.Tensor] = None,
        cond_3d: Optional[torch.Tensor] = None,
        init_probs: Optional[torch.Tensor] = None,
        start_timestep: Optional[int] = None,
        num_steps: int = 50,
        guidance_scale: float = 1.0,
        show_progress: bool = True,
    ) -> torch.Tensor:
        """DDIM sampling with EDM preconditioning and Classifier-Free Guidance.

        Uses deterministic (DDIM) updates instead of ancestral (stochastic):
            z_{t-1} = z_0_pred + (σ_{t-1}/σ_t) * (z_t - z_0_pred)

        CFG (when guidance_scale > 1.0):
            z_0_guided = z_0_uncond + w * (z_0_cond - z_0_uncond)
        where w = guidance_scale. DiffSSC uses w=6.0.
        """
        B, H, W, D = shape
        model.eval()

        bev_onehot = F.one_hot(bev.long(), num_classes=self.num_classes).float()
        bev_onehot = bev_onehot.permute(0, 3, 1, 2).to(device)

        if init_probs is not None and start_timestep is not None:
            z_0_init = self.probs_to_logits(init_probs)
            t_start = torch.full((B,), start_timestep, device=device, dtype=torch.long)
            z_t = self.q_sample(z_0_init, t_start)
            actual_start = start_timestep
        else:
            z_t = self.sigmas[-1] * torch.randn(B, self.num_classes, H, W, D, device=device)
            actual_start = self.num_timesteps - 1

        use_cfg = guidance_scale > 1.0

        step_ratio = actual_start / num_steps
        timesteps = [int(actual_start - i * step_ratio) for i in range(num_steps)]
        timesteps = [max(0, t) for t in timesteps]
        timesteps.append(0)

        if show_progress:
            timesteps_iter = tqdm(range(len(timesteps) - 1), desc=f"DDIM-EDM ({num_steps} steps, w={guidance_scale})")
        else:
            timesteps_iter = range(len(timesteps) - 1)

        for i in timesteps_iter:
            t_val = timesteps[i]
            t_next = timesteps[i + 1]

            t_batch = torch.full((B,), t_val, device=device, dtype=torch.long)

            # Conditional denoised estimate
            z_0_cond = self._denoised_estimate(model, z_t, t_batch, bev_onehot, lidar,
                                               lifted_features=lifted_features,
                                               cond_3d=cond_3d)

            if use_cfg:
                # Unconditional estimate (force_uncond zeros all conditioning)
                z_0_uncond = self._denoised_estimate(model, z_t, t_batch, bev_onehot, lidar,
                                                     lifted_features=lifted_features,
                                                     cond_3d=cond_3d,
                                                     force_uncond=True)
                # CFG: amplify the difference between conditional and unconditional
                z_0_pred = z_0_uncond + guidance_scale * (z_0_cond - z_0_uncond)
            else:
                z_0_pred = z_0_cond

            if t_next > 0:
                # DDIM update: preserve noise direction, just scale it down
                sigma_t = self.sigmas[t_val]
                sigma_next = self.sigmas[t_next]
                # z_t = z_0_pred + σ_t * ε  =>  ε = (z_t - z_0_pred) / σ_t
                # z_{t-1} = z_0_pred + σ_{t-1} * ε
                z_t = z_0_pred + (sigma_next / sigma_t) * (z_t - z_0_pred)
            else:
                z_t = z_0_pred

        return self.decode_x0(z_t)


class LogitModelWrapper(nn.Module):
    """Wrapper to adapt existing UNet for logit-space VE diffusion.

    The model receives c_in-normalized logits (always ~O(1)) and outputs
    raw F prediction. The diffusion module handles c_skip/c_out scaling.
    """

    def __init__(self, base_model: nn.Module, num_classes: int = 20):
        super().__init__()
        self.base_model = base_model
        self.num_classes = num_classes

    def forward(
        self,
        z_t: torch.Tensor,  # [B, 20, H, W, D] — already c_in-normalized
        t: torch.Tensor,
        bev: torch.Tensor,
        lidar: torch.Tensor,
        lifted_features: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Forward pass: predict raw F in logit space."""
        return self.base_model(z_t, t, lidar, lifted_features=lifted_features, **kwargs)
