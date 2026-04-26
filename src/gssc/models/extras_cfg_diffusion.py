"""
S1: Classifier-Free Guidance (CFG) for Scene Completion

Implements hierarchical classifier-free guidance where only the LIFTED
condition is dropped during training (BEV and LiDAR always present).

At inference, guided prediction:
    ε_guided = ε_uncond + w·(ε_cond - ε_uncond)

where w is the guidance scale (typically 1.5-2.0).

Key design decision (from architecture plan):
- Drop ONLY the lifted condition (p=0.1)
- Keep BEV and LiDAR conditions always present
- This is hierarchical CFG - some conditions more important than others

Reference:
- Classifier-Free Diffusion Guidance (Ho & Salimans, 2021)
- Architecture plan: <repo>/docs/architecture_improvement_plan.md
"""

import random

import torch
import torch.nn as nn
import torch.nn.functional as F


class CFGWrapper(nn.Module):
    """
    Wrapper that adds CFG support to any diffusion model.

    During training: randomly drops the lifted condition
    During inference: computes guided predictions
    """

    def __init__(self,
                 model: nn.Module,
                 p_uncond_lifted: float = 0.1,
                 guidance_scale: float = 1.5):
        """
        Args:
            model: Base diffusion model
            p_uncond_lifted: Probability of dropping lifted condition (training)
            guidance_scale: CFG guidance scale (inference)
        """
        super().__init__()
        self.model = model
        self.p_uncond_lifted = p_uncond_lifted
        self.guidance_scale = guidance_scale

    def forward(self,
                x_t: torch.Tensor,
                t: torch.Tensor,
                bev_cond: torch.Tensor,
                lidar_cond: torch.Tensor,
                lifted_cond: torch.Tensor,
                cfg_training: bool = True) -> torch.Tensor:
        """
        Forward pass with optional CFG dropout.

        Args:
            x_t: Noisy input at timestep t
            t: Timestep
            bev_cond: BEV condition (always present)
            lidar_cond: LiDAR condition (always present)
            lifted_cond: Lifted 3D condition (dropped with p=0.1)
            cfg_training: Whether to apply CFG dropout

        Returns:
            Model prediction
        """
        if self.training and cfg_training:
            # Randomly drop lifted condition
            if random.random() < self.p_uncond_lifted:
                lifted_cond = torch.zeros_like(lifted_cond)

        return self.model(x_t, t, bev_cond, lidar_cond, lifted_cond)

    def guided_forward(self,
                       x_t: torch.Tensor,
                       t: torch.Tensor,
                       bev_cond: torch.Tensor,
                       lidar_cond: torch.Tensor,
                       lifted_cond: torch.Tensor,
                       guidance_scale: float | None = None) -> torch.Tensor:
        """
        CFG-guided inference.

        Computes: ε_guided = ε_uncond + w·(ε_cond - ε_uncond)

        Args:
            x_t, t, bev_cond, lidar_cond, lifted_cond: Same as forward
            guidance_scale: Override default guidance scale

        Returns:
            Guided prediction
        """
        w = guidance_scale if guidance_scale is not None else self.guidance_scale

        # Unconditional prediction (lifted = zeros)
        eps_uncond = self.model(x_t, t, bev_cond, lidar_cond,
                               torch.zeros_like(lifted_cond))

        # Conditional prediction
        eps_cond = self.model(x_t, t, bev_cond, lidar_cond, lifted_cond)

        # Guided prediction
        eps_guided = eps_uncond + w * (eps_cond - eps_uncond)

        return eps_guided


class SDEditSampler:
    """
    SDEdit-style sampling that starts from a rough estimate instead of pure noise.

    Key insight: Instead of starting from pure noise (t=T), start from
    the lifted 3D estimate with added noise at t=t*.

    This preserves structure from the lifted estimate while allowing
    the diffusion model to refine details.

    Reference: SDEdit (Meng et al., 2021)
    """

    def __init__(self,
                 diffusion_model,
                 num_timesteps: int = 1000,
                 start_timestep: int = 500):
        """
        Args:
            diffusion_model: The diffusion model
            num_timesteps: Total timesteps (T)
            start_timestep: Where to start denoising (t*)
        """
        self.model = diffusion_model
        self.T = num_timesteps
        self.t_start = start_timestep

        # Setup noise schedule
        self._setup_schedule()

    def _setup_schedule(self):
        """Setup diffusion noise schedule."""
        betas = torch.linspace(0.0001, 0.02, self.T)
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)

        self.register_buffer = lambda n, t: setattr(self, n, t)
        self.sqrt_alpha_bar = torch.sqrt(alpha_bar)
        self.sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - alpha_bar)

    def add_noise(self,
                  x_0: torch.Tensor,
                  t: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Add noise to x_0 at timestep t."""
        noise = torch.randn_like(x_0.float())

        sqrt_ab = self.sqrt_alpha_bar[t].to(x_0.device)
        sqrt_1_ab = self.sqrt_one_minus_alpha_bar[t].to(x_0.device)

        x_t = sqrt_ab * x_0.float() + sqrt_1_ab * noise

        return x_t, noise

    def sample(self,
               lifted_3d: torch.Tensor,
               bev_cond: torch.Tensor,
               lidar_cond: torch.Tensor,
               use_cfg: bool = True,
               guidance_scale: float = 1.5) -> torch.Tensor:
        """
        SDEdit sampling starting from lifted 3D estimate.

        Args:
            lifted_3d: [B, H, W, D] rough 3D estimate from BEV lifting
            bev_cond: BEV condition
            lidar_cond: LiDAR condition
            use_cfg: Whether to use CFG
            guidance_scale: CFG guidance scale

        Returns:
            Final denoised sample
        """
        device = lifted_3d.device

        # Add noise at t_start
        x_t, _ = self.add_noise(lifted_3d, self.t_start)

        # Denoise from t_start to 0
        for t in reversed(range(self.t_start)):
            t_tensor = torch.tensor([t], device=device).long()

            # Get prediction
            if use_cfg and hasattr(self.model, 'guided_forward'):
                eps_pred = self.model.guided_forward(
                    x_t, t_tensor, bev_cond, lidar_cond, lifted_3d,
                    guidance_scale=guidance_scale
                )
            else:
                eps_pred = self.model(x_t, t_tensor, bev_cond, lidar_cond, lifted_3d)

            # Denoise step (simplified DDPM)
            x_t = self._denoise_step(x_t, eps_pred, t)

        return x_t

    def _denoise_step(self,
                      x_t: torch.Tensor,
                      eps_pred: torch.Tensor,
                      t: int) -> torch.Tensor:
        """Single DDPM denoising step."""
        # Simplified - actual implementation should use proper DDPM formula
        alpha_bar_t = self.sqrt_alpha_bar[t] ** 2
        alpha_bar_t_1 = self.sqrt_alpha_bar[t-1] ** 2 if t > 0 else 1.0

        # Predict x_0
        x_0_pred = (x_t - self.sqrt_one_minus_alpha_bar[t].to(x_t.device) * eps_pred) / \
                   self.sqrt_alpha_bar[t].to(x_t.device)

        # Sample x_{t-1}
        if t > 0:
            noise = torch.randn_like(x_t)
            sigma = torch.sqrt((1 - alpha_bar_t_1) / (1 - alpha_bar_t) * (1 - alpha_bar_t / alpha_bar_t_1))
            x_t_1 = torch.sqrt(alpha_bar_t_1) * x_0_pred + \
                    torch.sqrt(1 - alpha_bar_t_1 - sigma**2) * eps_pred + \
                    sigma * noise
        else:
            x_t_1 = x_0_pred

        return x_t_1


class CFGTrainingMixin:
    """
    Mixin class to add CFG training support to existing diffusion trainers.

    Usage:
        class MyTrainer(CFGTrainingMixin, BaseTrainer):
            ...
    """

    def cfg_training_step(self,
                          x_0: torch.Tensor,
                          bev_cond: torch.Tensor,
                          lidar_cond: torch.Tensor,
                          lifted_cond: torch.Tensor,
                          p_uncond: float = 0.1) -> dict[str, torch.Tensor]:
        """
        CFG training step with hierarchical dropout.

        Only drops lifted_cond, keeps bev_cond and lidar_cond.
        """
        # Sample timestep
        t = torch.randint(0, self.T, (x_0.shape[0],), device=x_0.device)

        # Add noise
        noise = torch.randn_like(x_0.float())
        x_t = self.q_sample(x_0, t, noise)

        # Hierarchical dropout: only drop lifted
        if random.random() < p_uncond:
            lifted_cond = torch.zeros_like(lifted_cond)

        # Predict noise
        pred = self.model(x_t, t, bev_cond, lidar_cond, lifted_cond)

        # Loss
        loss = F.mse_loss(pred, noise)

        return {'loss': loss, 'noise_pred': pred, 'noise_gt': noise}
