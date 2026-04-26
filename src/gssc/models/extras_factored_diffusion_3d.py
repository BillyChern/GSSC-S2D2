"""
Factored Discrete Diffusion for 3D Semantic Voxel Completion.

Decomposes the single K=20 multinomial diffusion into two parallel processes:
1. OccupancyDiffusion (K=2): Binary {empty, occupied} per voxel
2. SemanticDiffusion (K=20): Semantic class per voxel (meaningful at occupied)

Each has its own noise schedule — occupancy gets more aggressive noise
(needs to explore geometry changes), semantics gets gentler noise
(preserve class info longer). This mimics DiffSSC's anisotropic noise
(σ_occ >> σ_sem) but in discrete space.

Coupled at sampling time: final prediction = sem_pred * occ_pred.
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


def extract(a: torch.Tensor, t: torch.Tensor, x_shape: tuple[int, ...]) -> torch.Tensor:
    """Extract values from a at timestep t and reshape for broadcasting."""
    B = t.shape[0]
    out = a.gather(-1, t.clamp(0, a.shape[0] - 1))
    return out.reshape(B, *((1,) * (len(x_shape) - 1)))


class BinaryDiffusion(nn.Module):
    """Multinomial diffusion with K=2 for occupancy.

    Forward: q(o_t|o_0) = Cat(o_t | α_t · o_0 + (1-α_t)/2)
    More aggressive schedule (beta_max=0.15) to explore geometry.
    """

    def __init__(self, num_timesteps: int = 100, beta_max: float = 0.15):
        super().__init__()
        self.K = 2
        self.num_timesteps = num_timesteps

        betas = torch.linspace(0.0001, beta_max, num_timesteps, dtype=torch.float64)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        self.register_buffer('alphas', alphas.float())  # step alphas (NOT cumulative)
        self.register_buffer('alphas_cumprod', alphas_cumprod.float())
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev.float())

    def q_sample(self, x_0_onehot: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """q(o_t | o_0) = ᾱ_t · o_0 + (1-ᾱ_t) / 2"""
        alpha_bar_t = extract(self.alphas_cumprod, t, x_0_onehot.shape)
        return alpha_bar_t * x_0_onehot + (1.0 - alpha_bar_t) / self.K

    def q_posterior(self, x_0_probs: torch.Tensor, x_t: torch.Tensor,
                    t: torch.Tensor) -> torch.Tensor:
        """q(o_{t-1} | o_t, o_0) — posterior for reverse step."""
        alpha_t = extract(self.alphas, t, x_t.shape)  # step alpha (NOT cumulative)
        alpha_bar_prev = extract(self.alphas_cumprod_prev, t, x_t.shape)

        # Unnormalized: q(x_t|x_{t-1}) * q(x_{t-1}|x_0)
        # q(x_t|x_{t-1}=k) = α_t * x_t[k] + (1-α_t)/K  (single step transition)
        # q(x_{t-1}=k|x_0) = ᾱ_{t-1} * x_0[k] + (1-ᾱ_{t-1})/K  (cumulative)
        prior = alpha_bar_prev * x_0_probs + (1.0 - alpha_bar_prev) / self.K
        likelihood = alpha_t * x_t + (1.0 - alpha_t) / self.K

        # Posterior via Bayes' rule
        unnorm = prior * likelihood
        return unnorm / (unnorm.sum(dim=1, keepdim=True) + 1e-10)

    def kl_loss(self, x_0_onehot: torch.Tensor, x_t: torch.Tensor,
                x_0_pred_probs: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """KL(q(o_{t-1}|o_t, o_0) || p(o_{t-1}|o_t))."""
        q_post = self.q_posterior(x_0_onehot, x_t, t)
        p_post = self.q_posterior(x_0_pred_probs, x_t, t)
        kl = q_post * (torch.log(q_post + 1e-10) - torch.log(p_post + 1e-10))
        return kl.sum(dim=1).mean()


class SemanticDiffusion(nn.Module):
    """Multinomial diffusion with K=20 for semantics.

    Gentler schedule (beta_max=0.05) to preserve class info longer.
    Loss is MASKED: only computed at GT-occupied voxels.
    """

    def __init__(self, num_classes: int = 20, num_timesteps: int = 100,
                 beta_max: float = 0.05):
        super().__init__()
        self.K = num_classes
        self.num_timesteps = num_timesteps

        betas = torch.linspace(0.0001, beta_max, num_timesteps, dtype=torch.float64)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        self.register_buffer('alphas', alphas.float())  # step alphas (NOT cumulative)
        self.register_buffer('alphas_cumprod', alphas_cumprod.float())
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev.float())

    def q_sample(self, x_0_onehot: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """q(s_t | s_0) = ᾱ_t · s_0 + (1-ᾱ_t) / K"""
        alpha_bar_t = extract(self.alphas_cumprod, t, x_0_onehot.shape)
        return alpha_bar_t * x_0_onehot + (1.0 - alpha_bar_t) / self.K

    def q_posterior(self, x_0_probs: torch.Tensor, x_t: torch.Tensor,
                    t: torch.Tensor) -> torch.Tensor:
        """q(s_{t-1} | s_t, s_0)."""
        alpha_t = extract(self.alphas, t, x_t.shape)  # step alpha (NOT cumulative)
        alpha_bar_prev = extract(self.alphas_cumprod_prev, t, x_t.shape)

        prior = alpha_bar_prev * x_0_probs + (1.0 - alpha_bar_prev) / self.K
        likelihood = alpha_t * x_t + (1.0 - alpha_t) / self.K
        unnorm = prior * likelihood
        return unnorm / (unnorm.sum(dim=1, keepdim=True) + 1e-10)

    def masked_kl_loss(self, x_0_onehot: torch.Tensor, x_t: torch.Tensor,
                       x_0_pred_probs: torch.Tensor, t: torch.Tensor,
                       mask: torch.Tensor) -> torch.Tensor:
        """KL loss only at occupied voxels (mask=True)."""
        q_post = self.q_posterior(x_0_onehot, x_t, t)
        p_post = self.q_posterior(x_0_pred_probs, x_t, t)
        kl = q_post * (torch.log(q_post + 1e-10) - torch.log(p_post + 1e-10))
        kl_per_voxel = kl.sum(dim=1)  # [B, H, W, D]

        if mask.sum() > 0:
            return (kl_per_voxel * mask.float()).sum() / mask.float().sum()
        return kl_per_voxel.mean()


class FactoredDiffusion3D(nn.Module):
    """Factored discrete diffusion: OccupancyDiffusion + SemanticDiffusion.

    Model input: 22 channels = 2 (occ one-hot) + 20 (sem one-hot)
    Model output: 22 channels = 2 (occ logits) + 20 (sem logits)

    Different noise schedules: occ is more aggressive (explore geometry),
    sem is gentler (preserve class info). Mimics DiffSSC σ_occ >> σ_sem.
    """

    def __init__(
        self,
        num_classes: int = 20,
        num_timesteps: int = 100,
        beta_max_occ: float = 0.15,
        beta_max_sem: float = 0.05,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_timesteps = num_timesteps

        self.occ_diff = BinaryDiffusion(
            num_timesteps=num_timesteps, beta_max=beta_max_occ)
        self.sem_diff = SemanticDiffusion(
            num_classes=num_classes, num_timesteps=num_timesteps, beta_max=beta_max_sem)

    def training_losses(
        self,
        model: nn.Module,
        x_0: torch.Tensor,
        t: torch.Tensor,
        bev: torch.Tensor,
        lidar: torch.Tensor,
        lifted_features: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute factored training loss.

        Args:
            model: Neural net, outputs [B, 22, H, W, D] — split into 2+20
            x_0: Clean data (class indices) [B, H, W, D]
            t: Timesteps [B]
            bev: BEV conditioning [B, H, W]
            lidar: LiDAR conditioning [B, 20, H, W, D]
        """
        device = x_0.device

        # Decompose GT into occupancy and semantics
        occ_0 = (x_0 > 0).long()  # [B, H, W, D] binary
        occ_mask = (x_0 > 0)  # occupied voxel mask

        # One-hot encode
        occ_0_oh = F.one_hot(occ_0, 2).float().permute(0, 4, 1, 2, 3)  # [B,2,H,W,D]
        sem_0_oh = F.one_hot(x_0.long(), self.num_classes).float().permute(0, 4, 1, 2, 3)

        # Independent forward noise (shared timestep t)
        occ_t = self.occ_diff.q_sample(occ_0_oh, t)  # [B,2,H,W,D]
        sem_t = self.sem_diff.q_sample(sem_0_oh, t)  # [B,20,H,W,D]

        # Concatenate: [B, 22, H, W, D]
        x_t = torch.cat([occ_t, sem_t], dim=1)

        # BEV one-hot for interface compatibility
        bev_onehot = F.one_hot(bev.long(), num_classes=self.num_classes).float()
        bev_onehot = bev_onehot.permute(0, 3, 1, 2)

        # Model predicts 22 channels
        logits = model(x_t, t, bev_onehot, lidar, lifted_features=lifted_features)
        occ_logits = logits[:, :2]  # [B,2,H,W,D]
        sem_logits = logits[:, 2:]  # [B,20,H,W,D]

        occ_probs = F.softmax(occ_logits, dim=1)
        sem_probs = F.softmax(sem_logits, dim=1)

        # Occupancy loss (all voxels)
        occ_loss = self.occ_diff.kl_loss(occ_0_oh, occ_t, occ_probs, t)

        # Semantic loss (occupied voxels ONLY)
        sem_loss = self.sem_diff.masked_kl_loss(sem_0_oh, sem_t, sem_probs, t, occ_mask)

        loss = occ_loss + sem_loss

        # Monitoring
        with torch.no_grad():
            occ_pred = occ_probs.argmax(dim=1)  # [B,H,W,D] binary
            sem_pred = sem_probs.argmax(dim=1)  # [B,H,W,D] 0-19
            combined = sem_pred * occ_pred
            accuracy = (combined == x_0).float().mean()

        return {
            'loss': loss,
            'occ_loss': occ_loss.detach(),
            'sem_loss': sem_loss.detach(),
            'accuracy': accuracy,
            'x_0_logits': logits,
            'x_t_onehot': x_t,
            'bev_onehot': bev_onehot,
        }

    @torch.no_grad()
    def sample(
        self,
        model: nn.Module,
        bev: torch.Tensor,
        lidar: torch.Tensor,
        shape: tuple[int, int, int, int],
        device: torch.device,
        lifted_features: torch.Tensor | None = None,
        show_progress: bool = True,
    ) -> torch.Tensor:
        """Coupled ancestral sampling."""
        B, H, W, D = shape
        model.eval()

        bev_onehot = F.one_hot(bev.long(), num_classes=self.num_classes).float()
        bev_onehot = bev_onehot.permute(0, 3, 1, 2).to(device)

        # Start from uniform noise for both
        occ_t = torch.ones(B, 2, H, W, D, device=device) / 2
        probs_flat = occ_t.permute(0, 2, 3, 4, 1).reshape(-1, 2)
        occ_samples = torch.multinomial(probs_flat, 1).squeeze(-1)
        occ_t = F.one_hot(occ_samples, 2).float().reshape(B, H, W, D, 2).permute(0, 4, 1, 2, 3)

        sem_t = torch.ones(B, self.num_classes, H, W, D, device=device) / self.num_classes
        probs_flat = sem_t.permute(0, 2, 3, 4, 1).reshape(-1, self.num_classes)
        sem_samples = torch.multinomial(probs_flat, 1).squeeze(-1)
        sem_t = F.one_hot(sem_samples, self.num_classes).float().reshape(B, H, W, D, self.num_classes).permute(0, 4, 1, 2, 3)

        timesteps = list(range(self.num_timesteps))[::-1]
        if show_progress:
            timesteps = tqdm(timesteps, desc="Factored Sampling")

        for t_val in timesteps:
            t_batch = torch.full((B,), t_val, device=device, dtype=torch.long)
            x_t = torch.cat([occ_t, sem_t], dim=1)

            logits = model(x_t, t_batch, bev_onehot, lidar,
                          lifted_features=lifted_features)
            occ_logits = logits[:, :2]
            sem_logits = logits[:, 2:]

            occ_probs = F.softmax(occ_logits, dim=1)
            sem_probs = F.softmax(sem_logits, dim=1)

            # Independent posterior sampling
            t_is_zero = (t_batch == 0).view(-1, 1, 1, 1, 1)

            # Occupancy posterior
            occ_posterior = self.occ_diff.q_posterior(occ_probs, occ_t, t_batch)
            occ_step = torch.where(t_is_zero, occ_probs, occ_posterior)
            if t_val == 0:
                occ_t = F.one_hot(occ_step.argmax(1), 2).float().permute(0, 4, 1, 2, 3)
            else:
                flat = occ_step.permute(0, 2, 3, 4, 1).reshape(-1, 2).clamp(min=1e-10)
                flat = flat / flat.sum(-1, keepdim=True)
                s = torch.multinomial(flat, 1).squeeze(-1)
                occ_t = F.one_hot(s, 2).float().reshape(B, H, W, D, 2).permute(0, 4, 1, 2, 3)

            # Semantic posterior
            sem_posterior = self.sem_diff.q_posterior(sem_probs, sem_t, t_batch)
            sem_step = torch.where(t_is_zero, sem_probs, sem_posterior)
            if t_val == 0:
                sem_t = F.one_hot(sem_step.argmax(1), self.num_classes).float().permute(0, 4, 1, 2, 3)
            else:
                flat = sem_step.permute(0, 2, 3, 4, 1).reshape(-1, self.num_classes).clamp(min=1e-10)
                flat = flat / flat.sum(-1, keepdim=True)
                s = torch.multinomial(flat, 1).squeeze(-1)
                sem_t = F.one_hot(s, self.num_classes).float().reshape(B, H, W, D, self.num_classes).permute(0, 4, 1, 2, 3)

        # Combine: argmax occupancy × argmax semantics
        occ_pred = occ_t.argmax(dim=1)  # [B,H,W,D] binary
        sem_pred = sem_t.argmax(dim=1)  # [B,H,W,D] 0-19
        return sem_pred * occ_pred  # empty where occ=0

    @torch.no_grad()
    def sample_cfg(
        self,
        model: nn.Module,
        bev: torch.Tensor,
        lidar: torch.Tensor,
        shape: tuple[int, int, int, int],
        device: torch.device,
        lifted_features: torch.Tensor | None = None,
        guidance_scale: float = 3.0,
        cfg_target: str = 'lidar',
        show_progress: bool = True,
    ) -> torch.Tensor:
        """Sample with CFG for factored diffusion."""
        B, H, W, D = shape
        model.eval()

        bev_onehot = F.one_hot(bev.long(), num_classes=self.num_classes).float()
        bev_onehot = bev_onehot.permute(0, 3, 1, 2).to(device)

        # Start from uniform noise
        occ_t = torch.ones(B, 2, H, W, D, device=device) / 2
        flat = occ_t.permute(0, 2, 3, 4, 1).reshape(-1, 2)
        occ_t = F.one_hot(torch.multinomial(flat, 1).squeeze(-1), 2).float().reshape(B, H, W, D, 2).permute(0, 4, 1, 2, 3)

        sem_t = torch.ones(B, self.num_classes, H, W, D, device=device) / self.num_classes
        flat = sem_t.permute(0, 2, 3, 4, 1).reshape(-1, self.num_classes)
        sem_t = F.one_hot(torch.multinomial(flat, 1).squeeze(-1), self.num_classes).float().reshape(B, H, W, D, self.num_classes).permute(0, 4, 1, 2, 3)

        timesteps = list(range(self.num_timesteps))[::-1]
        if show_progress:
            timesteps = tqdm(timesteps, desc="Factored CFG Sampling")

        for t_val in timesteps:
            t_batch = torch.full((B,), t_val, device=device, dtype=torch.long)
            x_t = torch.cat([occ_t, sem_t], dim=1)

            # Two forward passes
            # Use force_uncond=True to bypass sparse encoder (avoids empty tensor crash)
            cond_logits = model(x_t, t_batch, bev_onehot, lidar,
                              lifted_features=lifted_features)
            uncond_logits = model(x_t, t_batch, bev_onehot, lidar,
                                lifted_features=lifted_features, force_uncond=True)

            # CFG blend on logits
            guided_logits = uncond_logits + guidance_scale * (cond_logits - uncond_logits)
            occ_logits = guided_logits[:, :2]
            sem_logits = guided_logits[:, 2:]

            occ_probs = F.softmax(occ_logits, dim=1)
            sem_probs = F.softmax(sem_logits, dim=1)

            t_is_zero = (t_batch == 0).view(-1, 1, 1, 1, 1)

            # Occupancy posterior
            occ_posterior = self.occ_diff.q_posterior(occ_probs, occ_t, t_batch)
            occ_step = torch.where(t_is_zero, occ_probs, occ_posterior)
            if t_val == 0:
                occ_t = F.one_hot(occ_step.argmax(1), 2).float().permute(0, 4, 1, 2, 3)
            else:
                flat = occ_step.permute(0, 2, 3, 4, 1).reshape(-1, 2).clamp(min=1e-10)
                flat = flat / flat.sum(-1, keepdim=True)
                occ_t = F.one_hot(torch.multinomial(flat, 1).squeeze(-1), 2).float().reshape(B, H, W, D, 2).permute(0, 4, 1, 2, 3)

            # Semantic posterior
            sem_posterior = self.sem_diff.q_posterior(sem_probs, sem_t, t_batch)
            sem_step = torch.where(t_is_zero, sem_probs, sem_posterior)
            if t_val == 0:
                sem_t = F.one_hot(sem_step.argmax(1), self.num_classes).float().permute(0, 4, 1, 2, 3)
            else:
                flat = sem_step.permute(0, 2, 3, 4, 1).reshape(-1, self.num_classes).clamp(min=1e-10)
                flat = flat / flat.sum(-1, keepdim=True)
                sem_t = F.one_hot(torch.multinomial(flat, 1).squeeze(-1), self.num_classes).float().reshape(B, H, W, D, self.num_classes).permute(0, 4, 1, 2, 3)

        occ_pred = occ_t.argmax(dim=1)
        sem_pred = sem_t.argmax(dim=1)
        return sem_pred * occ_pred

    @torch.no_grad()
    def sample_sdedit(
        self,
        model: nn.Module,
        bev: torch.Tensor,
        lidar: torch.Tensor,
        lsk3d_init: torch.Tensor,
        shape: tuple[int, int, int, int],
        device: torch.device,
        lifted_features: torch.Tensor | None = None,
        start_timestep: int = 50,
        show_progress: bool = True,
    ) -> torch.Tensor:
        """SDEdit from LSK3DNet hard labels (discrete limitation: must argmax)."""
        B, H, W, D = shape
        model.eval()

        bev_onehot = F.one_hot(bev.long(), num_classes=self.num_classes).float()
        bev_onehot = bev_onehot.permute(0, 3, 1, 2).to(device)

        # Convert LSK3DNet to hard labels (argmax — loses uncertainty)
        sem_init = lsk3d_init.argmax(dim=1)  # [B,H,W,D]
        occ_init = (lsk3d_init.sum(dim=1) > 0.5).long()  # [B,H,W,D]

        occ_0 = F.one_hot(occ_init, 2).float().permute(0, 4, 1, 2, 3)
        sem_0 = F.one_hot(sem_init.long(), self.num_classes).float().permute(0, 4, 1, 2, 3)

        # Add noise to start_timestep
        t_start = torch.full((B,), start_timestep, device=device, dtype=torch.long)
        occ_t = self.occ_diff.q_sample(occ_0, t_start)
        sem_t = self.sem_diff.q_sample(sem_0, t_start)

        # Sample from noisy categorical
        flat = occ_t.permute(0, 2, 3, 4, 1).reshape(-1, 2).clamp(min=1e-10)
        flat = flat / flat.sum(-1, keepdim=True)
        occ_t = F.one_hot(torch.multinomial(flat, 1).squeeze(-1), 2).float().reshape(B, H, W, D, 2).permute(0, 4, 1, 2, 3)

        flat = sem_t.permute(0, 2, 3, 4, 1).reshape(-1, self.num_classes).clamp(min=1e-10)
        flat = flat / flat.sum(-1, keepdim=True)
        sem_t = F.one_hot(torch.multinomial(flat, 1).squeeze(-1), self.num_classes).float().reshape(B, H, W, D, self.num_classes).permute(0, 4, 1, 2, 3)

        timesteps = list(range(start_timestep, -1, -1))
        if show_progress:
            timesteps = tqdm(timesteps, desc=f"Factored SDEdit (t={start_timestep}→0)")

        for t_val in timesteps:
            t_batch = torch.full((B,), t_val, device=device, dtype=torch.long)
            x_t = torch.cat([occ_t, sem_t], dim=1)

            logits = model(x_t, t_batch, bev_onehot, lidar,
                          lifted_features=lifted_features)
            occ_probs = F.softmax(logits[:, :2], dim=1)
            sem_probs = F.softmax(logits[:, 2:], dim=1)

            t_is_zero = (t_batch == 0).view(-1, 1, 1, 1, 1)

            occ_posterior = self.occ_diff.q_posterior(occ_probs, occ_t, t_batch)
            occ_step = torch.where(t_is_zero, occ_probs, occ_posterior)
            if t_val == 0:
                occ_t = F.one_hot(occ_step.argmax(1), 2).float().permute(0, 4, 1, 2, 3)
            else:
                flat = occ_step.permute(0, 2, 3, 4, 1).reshape(-1, 2).clamp(min=1e-10)
                flat = flat / flat.sum(-1, keepdim=True)
                occ_t = F.one_hot(torch.multinomial(flat, 1).squeeze(-1), 2).float().reshape(B, H, W, D, 2).permute(0, 4, 1, 2, 3)

            sem_posterior = self.sem_diff.q_posterior(sem_probs, sem_t, t_batch)
            sem_step = torch.where(t_is_zero, sem_probs, sem_posterior)
            if t_val == 0:
                sem_t = F.one_hot(sem_step.argmax(1), self.num_classes).float().permute(0, 4, 1, 2, 3)
            else:
                flat = sem_step.permute(0, 2, 3, 4, 1).reshape(-1, self.num_classes).clamp(min=1e-10)
                flat = flat / flat.sum(-1, keepdim=True)
                sem_t = F.one_hot(torch.multinomial(flat, 1).squeeze(-1), self.num_classes).float().reshape(B, H, W, D, self.num_classes).permute(0, 4, 1, 2, 3)

        occ_pred = occ_t.argmax(dim=1)
        sem_pred = sem_t.argmax(dim=1)
        return sem_pred * occ_pred
