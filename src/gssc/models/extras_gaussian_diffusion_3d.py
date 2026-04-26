"""
Continuous Gaussian Diffusion for 3D Semantic Voxel Completion.

Follows DiffSSC's approach with ALL techniques from paper + code:

FROM CODE (proven, 26.7% mIoU):
- ε-prediction with MSE loss
- Global regularization: mean²+(std-1)², weight=5.0
- CFG: uncond_prob=0.1, w=6.0
- SDE-DPM-Solver++ for 50-step inference

FROM PAPER (Eq. 3-6, validated by ablation):
- Anisotropic noise: W = diag(σ_p, σ_s) with σ_p=1.0, σ_s=0.2
- Cosine beta schedule (Eq. 6): +6% mIoU over linear
- Separate λ_p=5.0, λ_s=4.0 for spatial/semantic regularization
- Semantic skewness regularization (Eq. 5): handles long-tailed distribution

NOVEL FOR OUR PIPELINE:
- SDEdit from soft LSK3DNet TTA probabilities (70.25% accuracy)
- Factored encode/decode with logit_scale for meaningful Gaussian range
- 21-channel representation: 1 occ_logit + 20 sem_logits per voxel

Reference: DiffSSC (CVPR 2024) — diffssc/models/models.py, minkunet.py
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


def extract(a: torch.Tensor, t: torch.Tensor, x_shape: tuple[int, ...]) -> torch.Tensor:
    """Extract values from a at timestep t and reshape for broadcasting."""
    B = t.shape[0]
    out = a.gather(-1, t.clamp(0, a.shape[0] - 1))
    return out.reshape(B, *((1,) * (len(x_shape) - 1)))


def diffssc_cosine_schedule(T: int, beta_start: float = 3.5e-5,
                            beta_end: float = 0.007) -> torch.Tensor:
    """DiffSSC Paper Eq. 6 — cosine beta schedule.

    β_t = β_0 + 0.5 * (1 + cos(πt/T)) * (β_T - β_0)

    NOT the Nichol & Dhariwal variant (which modulates ᾱ_t, not β_t).
    Paper ablation: +6% mIoU over linear schedule (20.7% → 26.7%).
    """
    t = torch.arange(T, dtype=torch.float64)
    betas = beta_start + 0.5 * (1 + torch.cos(math.pi * t / T)) * (beta_end - beta_start)
    return betas


def linear_diffssc_schedule(T: int, beta_start: float = 1e-4,
                            beta_end: float = 0.02) -> torch.Tensor:
    """DiffSSC ACTUAL CODE schedule — linear in sqrt(β) space.

    From DiffSSC scheduling.py:get_named_beta_schedule('linear', T):
        scale = 1000 / T
        betas = linspace(scale * 1e-4, scale * 0.02, T)
    But their default T=1000, so scale=1 → betas = linspace(1e-4, 0.02, T).

    This gives:
    - Total Σβ ≈ 10.0 (vs our cosine_diffssc Σβ ≈ 3.5)
    - ᾱ_T ≈ 0.0007 (vs our 0.029) — 40× less signal retention
    - Proper noise destruction at t=T — model MUST learn denoising

    Despite the paper claiming cosine (+6% mIoU over linear in ablation),
    the released code that achieves 26.7% mIoU uses THIS linear schedule.
    """
    betas = torch.linspace(beta_start, beta_end, T, dtype=torch.float64)
    return betas


def nichol_dhariwal_cosine_schedule(T: int, s: float = 0.008) -> torch.Tensor:
    """Nichol & Dhariwal cosine schedule — modulates ᾱ_t.

    ᾱ_t = f(t) / f(0), where f(t) = cos²((t/T + s) / (1 + s) * π/2)
    β_t = 1 - ᾱ_t / ᾱ_{t-1}
    """
    t = torch.arange(T + 1, dtype=torch.float64)
    f = torch.cos((t / T + s) / (1 + s) * math.pi / 2) ** 2
    alphas_cumprod = f / f[0]
    betas = 1 - alphas_cumprod[1:] / alphas_cumprod[:-1]
    return betas.clamp(max=0.999)


class GaussianDiffusion3D(nn.Module):
    """Continuous Gaussian diffusion for 3D semantic voxel completion.

    Supports two encoding modes:
    - 'factored_21ch': 1 occ_logit + 20 sem_logits (original DiffSSC-style)
    - 'logit_20ch': 20ch logit encoding (one_hot * scale, mean-centered)

    Args:
        num_channels: Total channels per voxel (21 for factored, 20 for logit)
        num_timesteps: Number of diffusion steps
        logit_scale: Scale for encoding GT labels into logit space
        sigma_occ: Anisotropic noise scale for occupancy (DiffSSC Paper Eq. 3)
        sigma_sem: Anisotropic noise scale for semantics (DiffSSC Paper Eq. 3)
        lambda_p: Spatial (occupancy) regularization weight (DiffSSC Paper Eq. 5)
        lambda_s: Semantic regularization weight (DiffSSC Paper Eq. 5)
        use_skewness_reg: Add skewness term to semantic reg (DiffSSC Paper Eq. 5)
        anisotropic: Use W matrix for noise (DiffSSC Paper Eq. 3)
        beta_schedule: Schedule type ('cosine_diffssc', 'cosine_nd', 'linear')
        num_classes: Number of semantic classes
        encoding: 'factored_21ch' (default) or 'logit_20ch'
    """

    def __init__(
        self,
        num_channels: int = 21,
        num_timesteps: int = 1000,
        logit_scale: float = 5.0,
        sigma_occ: float = 1.0,
        sigma_sem: float = 0.2,
        lambda_p: float = 5.0,
        lambda_s: float = 4.0,
        use_skewness_reg: bool = True,
        anisotropic: bool = False,
        beta_schedule: str = 'linear_diffssc',
        num_classes: int = 20,
        encoding: str = 'factored_21ch',
        # Loss weighting (matching D3PM V2 best practices)
        class_0_weight: float = 1.0,
        occupied_weight: float = 1.0,
        lovasz_weight: float = 0.0,
        obs_weight_factor: float = 0.0,
    ):
        super().__init__()

        self.encoding = encoding
        if encoding == 'logit_20ch':
            num_channels = num_classes  # Force 20ch

        self.num_channels = num_channels
        self.num_timesteps = num_timesteps
        self.logit_scale = logit_scale
        self.lambda_p = lambda_p
        self.lambda_s = lambda_s
        self.use_skewness_reg = use_skewness_reg
        self.anisotropic = anisotropic
        self.num_classes = num_classes

        # Loss weighting for class imbalance
        self.class_0_weight = class_0_weight
        self.occupied_weight = occupied_weight
        self.lovasz_weight = lovasz_weight
        self.obs_weight_factor = obs_weight_factor

        # Build anisotropic sigma vector [21]: [σ_occ, σ_sem, ..., σ_sem]
        sigma = torch.ones(num_channels)
        sigma[0] = sigma_occ
        sigma[1:] = sigma_sem
        self.register_buffer('sigma', sigma)

        # Beta schedule
        if beta_schedule == 'linear_diffssc':
            # DiffSSC actual code: linear(1e-4, 0.02) — proven at 26.7% mIoU
            betas = linear_diffssc_schedule(num_timesteps)
        elif beta_schedule == 'cosine_diffssc':
            # DiffSSC paper Eq. 6 — CAUTION: produces much weaker noise than code
            betas = diffssc_cosine_schedule(num_timesteps)
        elif beta_schedule == 'cosine_nd':
            betas = nichol_dhariwal_cosine_schedule(num_timesteps)
        elif beta_schedule == 'linear':
            betas = torch.linspace(3.5e-5, 0.007, num_timesteps, dtype=torch.float64)
        else:
            raise ValueError(f"Unknown beta_schedule: {beta_schedule}")

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        # Precompute coefficients
        self.register_buffer('betas', betas.float())
        self.register_buffer('alphas', alphas.float())
        self.register_buffer('alphas_cumprod', alphas_cumprod.float())
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod).float())
        self.register_buffer('sqrt_one_minus_alphas_cumprod',
                             torch.sqrt(1.0 - alphas_cumprod).float())
        # For x_0 prediction from x_t and eps
        self.register_buffer('sqrt_recip_alphas_cumprod',
                             torch.sqrt(1.0 / alphas_cumprod).float())
        self.register_buffer('sqrt_recipm1_alphas_cumprod',
                             torch.sqrt(1.0 / alphas_cumprod - 1).float())
        # For posterior q(x_{t-1} | x_t, x_0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance.float())
        self.register_buffer('posterior_log_variance_clipped',
                             torch.log(posterior_variance.clamp(min=1e-20)).float())
        self.register_buffer('posterior_mean_coef1',
                             (torch.sqrt(alphas_cumprod_prev) * betas / (1.0 - alphas_cumprod)).float())
        self.register_buffer('posterior_mean_coef2',
                             (torch.sqrt(alphas) * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)).float())

    def encode_x0(self, gt_labels: torch.Tensor) -> torch.Tensor:
        """Encode GT class indices [B, H, W, D] → continuous representation.

        factored_21ch: [B, 21, H, W, D] — 1 occ_logit + 20 sem_logits
        logit_20ch: [B, 20, H, W, D] — one_hot * scale, mean-centered
        """
        if self.encoding == 'logit_20ch':
            return self._encode_x0_logit(gt_labels)

        B, H, W, D = gt_labels.shape

        # Occupancy: class 0 = empty, classes 1-19 = occupied
        occ = (gt_labels > 0).float()  # [B, H, W, D]
        occ_logit = self.logit_scale * (2.0 * occ - 1.0)  # +scale or -scale

        # Semantics: one-hot encode the 20 classes
        sem_onehot = F.one_hot(gt_labels.long(), self.num_classes).float()  # [B,H,W,D,20]
        sem_logits = self.logit_scale * sem_onehot  # +scale at GT class
        sem_logits = sem_logits.permute(0, 4, 1, 2, 3)  # [B, 20, H, W, D]

        # Concatenate: [B, 21, H, W, D]
        x_0 = torch.cat([occ_logit.unsqueeze(1), sem_logits], dim=1)
        return x_0

    def _encode_x0_logit(self, gt_labels: torch.Tensor) -> torch.Tensor:
        """20ch logit encoding: one_hot * scale, mean-centered. [B, 20, H, W, D]."""
        one_hot = F.one_hot(gt_labels.long(), self.num_classes).float()
        logits = one_hot * self.logit_scale - (1 - one_hot) * self.logit_scale / (self.num_classes - 1)
        logits = logits - logits.mean(dim=-1, keepdim=True)
        return logits.permute(0, 4, 1, 2, 3)

    def decode_x0(self, x_0: torch.Tensor) -> torch.Tensor:
        """Decode continuous representation → class indices [B, H, W, D].

        factored_21ch: occ_logit thresholding + sem argmax
        logit_20ch: softmax + argmax
        """
        if self.encoding == 'logit_20ch':
            return F.softmax(x_0, dim=1).argmax(dim=1)

        occ_logit = x_0[:, 0]  # [B, H, W, D]
        sem_logits = x_0[:, 1:]  # [B, 20, H, W, D]

        occ_pred = (occ_logit > 0).long()  # binary
        sem_pred = sem_logits.argmax(dim=1)  # [B, H, W, D] in range [0, 19]

        # Empty where occ=0
        return sem_pred * occ_pred

    def q_sample(self, x_0: torch.Tensor, t: torch.Tensor,
                 noise: torch.Tensor | None = None) -> torch.Tensor:
        """Forward process: x_t = √ᾱ_t · x_0 + √(1-ᾱ_t) · W · ε

        If anisotropic=True, applies W = diag(sigma) to noise (Paper Eq. 3).
        If anisotropic=False, uses raw ε (matching DiffSSC released code).
        """
        if noise is None:
            noise = torch.randn_like(x_0)

        sqrt_alpha = extract(self.sqrt_alphas_cumprod, t, x_0.shape)
        sqrt_one_minus_alpha = extract(self.sqrt_one_minus_alphas_cumprod, t, x_0.shape)

        if self.anisotropic:
            # W · ε: scale noise per channel
            sigma = self.sigma.view(1, -1, *([1] * (x_0.dim() - 2)))
            return sqrt_alpha * x_0 + sqrt_one_minus_alpha * sigma * noise
        else:
            return sqrt_alpha * x_0 + sqrt_one_minus_alpha * noise

    def _predict_x0(self, x_t: torch.Tensor, t: torch.Tensor,
                    eps_pred: torch.Tensor) -> torch.Tensor:
        """Predict x_0 from x_t and predicted noise.

        Forward: x_t = √ᾱ_t · x_0 + √(1-ᾱ_t) · W · ε
        Rearrange: x_0 = (x_t - √(1-ᾱ_t) · eps_pred) / √ᾱ_t

        At high noise levels (t→T), 1/sqrt(alpha_bar) → large, amplifying
        eps prediction errors into extreme x_0 values. We clip x_0 to a
        generous range to prevent catastrophic divergence during sampling.
        """
        sqrt_recip = extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape)
        sqrt_recipm1 = extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

        x_0_pred = sqrt_recip * x_t - sqrt_recipm1 * eps_pred

        # Clip to prevent divergence at high noise levels where 1/sqrt(alpha_bar)
        # amplifies prediction errors. Range ±2*logit_scale is generous enough
        # to not constrain valid predictions.
        clip_range = self.logit_scale * 2.0
        x_0_pred = x_0_pred.clamp(-clip_range, clip_range)

        return x_0_pred

    def _q_posterior_mean_variance(self, x_0: torch.Tensor, x_t: torch.Tensor,
                                   t: torch.Tensor):
        """Compute posterior q(x_{t-1} | x_t, x_0) mean and variance."""
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_0
            + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_var = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_var = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_var, posterior_log_var

    def _global_regularization(self, eps_pred: torch.Tensor) -> torch.Tensor:
        """DiffSSC Paper Eq. 5: separate spatial/semantic + skewness.

        factored_21ch:
            Spatial (channel 0): λ_p · (mean² + (std-1)²)
            Semantic (channels 1-20): λ_s · (mean² + (std-1)² + skew²)
        logit_20ch:
            Spatial (channel 0 = empty class): λ_p · (mean² + (std-1)²)
            Semantic (channels 1-19 = occupied classes): λ_s · (mean² + (std-1)² + skew²)

        Without global reg: 26.7% → 10.9% mIoU (-15.8%) per DiffSSC ablation.
        """
        # Split: channel 0 as "spatial/occupancy", rest as "semantic"
        # For logit_20ch, channel 0 = empty class (analogous to occupancy)
        eps_occ = eps_pred[:, 0:1]
        mean_p = eps_occ.mean()
        std_p = eps_occ.std()
        reg_p = mean_p ** 2 + (std_p - 1.0) ** 2

        eps_sem = eps_pred[:, 1:]
        mean_s = eps_sem.mean()
        std_s = eps_sem.std()
        reg_s = mean_s ** 2 + (std_s - 1.0) ** 2

        if self.use_skewness_reg:
            skew_s = ((eps_sem - mean_s) ** 3).mean() / (std_s ** 3 + 1e-8)
            reg_s = reg_s + skew_s ** 2

        return self.lambda_p * reg_p + self.lambda_s * reg_s

    def _lovasz_grad(self, gt_sorted: torch.Tensor) -> torch.Tensor:
        """Compute gradient of the Lovász extension w.r.t sorted errors."""
        gts = gt_sorted.sum()
        intersection = gts - gt_sorted.float().cumsum(0)
        union = gts + (1 - gt_sorted).float().cumsum(0)
        jaccard = 1.0 - intersection / union
        if len(jaccard) > 1:
            jaccard[1:] = jaccard[1:] - jaccard[:-1]
        return jaccard

    def _lovasz_softmax_3d(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Lovász-Softmax for 3D volumes. Directly optimizes mIoU, ignoring class 0.

        Args:
            logits: [B, C, H, W, D] raw predictions (before softmax)
            labels: [B, H, W, D] ground truth class indices
        Returns:
            Scalar Lovász-Softmax loss (mean over non-zero classes)
        """
        probas = F.softmax(logits, dim=1)
        # Flatten: [B, C, H, W, D] → [N, C], [B, H, W, D] → [N]
        probas_flat = probas.permute(0, 2, 3, 4, 1).contiguous().view(-1, self.num_classes)
        labels_flat = labels.view(-1)

        losses = []
        for c in range(self.num_classes):
            if c == 0:  # ignore empty class
                continue
            fg = (labels_flat == c).float()
            if fg.sum() == 0:
                continue
            errors = (fg - probas_flat[:, c]).abs()
            errors_sorted, perm = torch.sort(errors, descending=True)
            fg_sorted = fg[perm]
            grad = self._lovasz_grad(fg_sorted)
            losses.append((errors_sorted * grad).sum())

        if len(losses) == 0:
            return logits.sum() * 0.0  # zero with grad
        return torch.stack(losses).mean()

    def training_losses(
        self,
        model: nn.Module,
        x_0: torch.Tensor,
        t: torch.Tensor,
        bev: torch.Tensor,
        lidar: torch.Tensor,
        lifted_features: torch.Tensor | None = None,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        """Compute training loss with occupancy-weighted MSE + Lovász auxiliary.

        L = weighted_MSE(ε̂, target) + λ_p·reg_p + λ_s·reg_s + lovasz_w·Lovász(x̂_0, x_0)

        Occupancy weighting matches D3PM V2 best practices (43.8% mIoU):
        - Empty voxels (class 0): weight = class_0_weight (default 0.02)
        - Occupied voxels: weight = occupied_weight (default 10.0)
        - LiDAR-observed voxels: extra obs_weight_factor boost
        - Lovász: direct IoU optimization on predicted x_0 (ignores class 0)

        Args:
            model: Neural network that predicts noise ε̂
            x_0: Clean data (class indices) [B, H, W, D]
            t: Timesteps [B]
            bev: BEV conditioning (class indices) [B, H, W]
            lidar: LiDAR conditioning [B, 20, H, W, D]
        """
        # Encode GT labels to continuous representation
        x_0_cont = self.encode_x0(x_0)  # [B, C, H, W, D]

        # Sample noise
        noise = torch.randn_like(x_0_cont)

        # Forward: x_t = √ᾱ_t · x_0 + √(1-ᾱ_t) · [W ·] ε
        x_t = self.q_sample(x_0_cont, t, noise)

        # Model predicts noise ε̂
        bev_onehot = F.one_hot(bev.long(), num_classes=self.num_classes).float()
        bev_onehot = bev_onehot.permute(0, 3, 1, 2)  # [B, K, H, W]
        eps_pred = model(x_t, t, bev_onehot, lidar, lifted_features=lifted_features)

        # MSE target
        if self.anisotropic:
            sigma = self.sigma.view(1, -1, *([1] * (x_0_cont.dim() - 2)))
            target = sigma * noise
        else:
            target = noise

        # === OCCUPANCY-WEIGHTED MSE ===
        # Without weighting, 95% of gradient goes to empty voxels (class 0).
        # The model learns to denoise empty space and ignores occupied voxels.
        per_element_mse = (eps_pred - target) ** 2  # [B, C, H, W, D]

        if self.occupied_weight != 1.0 or self.class_0_weight != 1.0:
            occupied = (x_0 > 0).float()  # [B, H, W, D]
            voxel_weights = torch.where(
                occupied > 0,
                torch.tensor(self.occupied_weight, device=x_0.device),
                torch.tensor(self.class_0_weight, device=x_0.device),
            )  # [B, H, W, D]

            # Observation weighting: LSK3DNet-observed voxels get extra weight
            if self.obs_weight_factor > 0:
                obs_mask = (lidar.sum(dim=1) > 0).float()  # [B, H, W, D]
                voxel_weights = voxel_weights * (1.0 + self.obs_weight_factor * obs_mask)

            # Normalize to mean=1 so loss scale (and effective LR) stays constant
            voxel_weights = voxel_weights / voxel_weights.mean().clamp(min=1e-8)

            # Expand to [B, C, H, W, D] and apply
            voxel_weights = voxel_weights.unsqueeze(1)
            mse_loss = (voxel_weights * per_element_mse).mean()
        else:
            mse_loss = per_element_mse.mean()

        # Global regularization (Paper Eq. 5)
        reg_loss = self._global_regularization(eps_pred)

        loss = mse_loss + reg_loss

        # === LOVÁSZ AUXILIARY on predicted x_0 ===
        # Direct IoU optimization: softmax(x̂_0) vs GT labels, ignoring class 0.
        # This gives the model direct supervision on semantic correctness,
        # complementing the noise-prediction MSE which is indirect.
        lovasz_loss = torch.tensor(0.0, device=x_0.device)
        if self.lovasz_weight > 0:
            x_0_pred = self._predict_x0(x_t, t, eps_pred)
            lovasz_loss = self._lovasz_softmax_3d(x_0_pred, x_0)
            loss = loss + self.lovasz_weight * lovasz_loss

        # Monitoring: predict x_0, decode to labels, compute accuracy
        with torch.no_grad():
            if self.lovasz_weight > 0:
                # Already computed x_0_pred above
                pred_labels = self.decode_x0(x_0_pred.detach())
            else:
                x_0_pred = self._predict_x0(x_t, t, eps_pred)
                pred_labels = self.decode_x0(x_0_pred)
            accuracy = (pred_labels == x_0).float().mean()

        return {
            'loss': loss,
            'mse_loss': mse_loss.detach(),
            'reg_loss': reg_loss.detach(),
            'lovasz_loss': lovasz_loss.detach() if torch.is_tensor(lovasz_loss) else lovasz_loss,
            'accuracy': accuracy,
            # WARNING: This is eps_pred (noise prediction), NOT x_0 logits.
            # Named 'x_0_logits' for API compatibility with multinomial diffusion.
            # Do NOT use for Lovász loss or other losses that expect class logits.
            # Only multinomial_diffusion_3d.py returns actual x_0 logits here.
            'x_0_logits': eps_pred,
            'x_t_onehot': x_t,
            'bev_onehot': bev_onehot,
        }

    @torch.no_grad()
    def _p_sample_ddpm(self, model: nn.Module, x_t: torch.Tensor, t: torch.Tensor,
                       bev: torch.Tensor, lidar: torch.Tensor,
                       lifted_features: torch.Tensor | None = None) -> torch.Tensor:
        """Single DDPM reverse step: x_{t-1} ~ p(x_{t-1} | x_t)."""
        B = x_t.shape[0]

        eps_pred = model(x_t, t, bev, lidar, lifted_features=lifted_features)
        x_0_pred = self._predict_x0(x_t, t, eps_pred)

        # Posterior mean and variance
        posterior_mean, posterior_var, posterior_log_var = \
            self._q_posterior_mean_variance(x_0_pred, x_t, t)

        # Sample — noise must match forward process anisotropic scaling
        noise = torch.randn_like(x_t)
        if self.anisotropic:
            sigma = self.sigma.view(1, -1, *([1] * (x_t.dim() - 2)))
            noise = sigma * noise
        # No noise at t=0
        nonzero_mask = (t != 0).float().view(B, *([1] * (x_t.dim() - 1)))
        x_prev = posterior_mean + nonzero_mask * torch.exp(0.5 * posterior_log_var) * noise

        return x_prev

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
        """Sample with DDPM (standard, no fast solver).

        Args:
            model: Neural network
            bev: BEV conditioning (class indices) [B, H, W]
            lidar: LiDAR conditioning [B, 20, H, W, D]
            shape: (B, H, W, D)
            device: Device

        Returns:
            Generated samples (class indices) [B, H, W, D]
        """
        B, H, W, D = shape
        model.eval()

        bev_onehot = F.one_hot(bev.long(), num_classes=self.num_classes).float()
        bev_onehot = bev_onehot.permute(0, 3, 1, 2).to(device)

        # Start from Gaussian noise
        x_t = torch.randn(B, self.num_channels, H, W, D, device=device)
        if self.anisotropic:
            sigma = self.sigma.view(1, -1, 1, 1, 1)
            x_t = x_t * sigma

        timesteps = list(range(self.num_timesteps))[::-1]
        if show_progress:
            timesteps = tqdm(timesteps, desc="DDPM Sampling")

        for t_val in timesteps:
            t_batch = torch.full((B,), t_val, device=device, dtype=torch.long)
            x_t = self._p_sample_ddpm(model, x_t, t_batch, bev_onehot, lidar,
                                       lifted_features=lifted_features)

        return self.decode_x0(x_t)

    @torch.no_grad()
    def sample_dpm_solver(
        self,
        model: nn.Module,
        bev: torch.Tensor,
        lidar: torch.Tensor,
        shape: tuple[int, int, int, int],
        device: torch.device,
        lifted_features: torch.Tensor | None = None,
        num_steps: int = 50,
        show_progress: bool = True,
    ) -> torch.Tensor:
        """Fast sampling with DPM-Solver++ (ODE variant).

        50 steps instead of 1000 for ~20× speedup.

        Uses prediction_type='sample' (x_0 prediction) so we can clip
        x_0 estimates to [-logit_scale, logit_scale] before passing to
        the scheduler. This prevents extreme x_0 values (caused by
        1/sqrt(alpha_bar) amplification at high noise levels) from
        causing sampling divergence.
        """
        from diffusers import DPMSolverMultistepScheduler

        B, H, W, D = shape
        model.eval()

        bev_onehot = F.one_hot(bev.long(), num_classes=self.num_classes).float()
        bev_onehot = bev_onehot.permute(0, 3, 1, 2).to(device)

        scheduler = DPMSolverMultistepScheduler(
            num_train_timesteps=self.num_timesteps,
            trained_betas=self.betas.cpu().numpy(),
            prediction_type='sample',  # We pass clipped x_0, not raw eps
            algorithm_type='dpmsolver++',
            solver_order=2,
        )
        scheduler.set_timesteps(num_steps, device=device)

        # Start from noise
        x_t = torch.randn(B, self.num_channels, H, W, D, device=device)
        if self.anisotropic:
            sigma = self.sigma.view(1, -1, 1, 1, 1)
            x_t = x_t * sigma

        steps = scheduler.timesteps
        if show_progress:
            steps = tqdm(steps, desc=f"DPM++ ({num_steps} steps)")

        for t_val in steps:
            t_batch = t_val.unsqueeze(0).expand(B).to(device)
            eps_pred = model(x_t, t_batch, bev_onehot, lidar,
                           lifted_features=lifted_features)

            # Compute clipped x_0 prediction (prevents divergence)
            x_0_pred = self._predict_x0(x_t, t_batch, eps_pred)

            # DPM-Solver operates on flattened spatial dims
            x_flat = x_t.flatten(2)  # [B, 21, H*W*D]
            x_0_flat = x_0_pred.flatten(2)
            result = scheduler.step(x_0_flat, t_val, x_flat)
            x_t = result.prev_sample.view(B, self.num_channels, H, W, D)

        return self.decode_x0(x_t)

    @torch.no_grad()
    def sample_cfg(
        self,
        model: nn.Module,
        bev: torch.Tensor,
        lidar: torch.Tensor,
        shape: tuple[int, int, int, int],
        device: torch.device,
        lifted_features: torch.Tensor | None = None,
        guidance_scale: float = 6.0,
        num_steps: int = 50,
        cfg_target: str = 'lidar',
        show_progress: bool = True,
    ) -> torch.Tensor:
        """CFG + SDE-DPM-Solver++ (DiffSSC: w=6.0, proven in code).

        Two forward passes per step: conditional + unconditional.
        Blend: eps = eps_uncond + w * (eps_cond - eps_uncond)
        """
        from diffusers import DPMSolverMultistepScheduler

        B, H, W, D = shape
        model.eval()

        bev_onehot = F.one_hot(bev.long(), num_classes=self.num_classes).float()
        bev_onehot = bev_onehot.permute(0, 3, 1, 2).to(device)

        scheduler = DPMSolverMultistepScheduler(
            num_train_timesteps=self.num_timesteps,
            trained_betas=self.betas.cpu().numpy(),
            prediction_type='sample',  # We pass clipped x_0, not raw eps
            algorithm_type='dpmsolver++',
            solver_order=2,
        )
        scheduler.set_timesteps(num_steps, device=device)

        x_t = torch.randn(B, self.num_channels, H, W, D, device=device)
        if self.anisotropic:
            sigma = self.sigma.view(1, -1, 1, 1, 1)
            x_t = x_t * sigma

        steps = scheduler.timesteps
        if show_progress:
            steps = tqdm(steps, desc=f"CFG+DPM++ (w={guidance_scale})")

        for t_val in steps:
            t_batch = t_val.unsqueeze(0).expand(B).to(device)

            # Two forward passes (DiffSSC models.py:104-109)
            # Use force_uncond=True to bypass sparse encoder (avoids empty tensor crash)
            cond_eps = model(x_t, t_batch, bev_onehot, lidar,
                           lifted_features=lifted_features)
            uncond_eps = model(x_t, t_batch, bev_onehot, lidar,
                             lifted_features=lifted_features, force_uncond=True)

            # CFG blend
            guided_eps = uncond_eps + guidance_scale * (cond_eps - uncond_eps)

            # Compute clipped x_0 prediction (prevents divergence)
            x_0_pred = self._predict_x0(x_t, t_batch, guided_eps)

            x_flat = x_t.flatten(2)
            x_0_flat = x_0_pred.flatten(2)
            result = scheduler.step(x_0_flat, t_val, x_flat)
            x_t = result.prev_sample.view(B, self.num_channels, H, W, D)

        return self.decode_x0(x_t)

    def probs_to_logits(self, probs: torch.Tensor) -> torch.Tensor:
        """Convert LSK3DNet probabilities [B, 20, H, W, D] → encoding-matched x_0.

        For logit_20ch: argmax → one_hot → encode_x0-style logits.
        Exactly matches encode_x0 scale and distribution. For unobserved voxels
        (all zeros), argmax defaults to class 0 → correct "empty" encoding.
        For factored_21ch: probs → occ_logit + sem_logits (21ch).
        """
        if self.encoding == 'logit_20ch':
            C = self.num_classes
            ls = self.logit_scale
            # Argmax encoding: exactly matches encode_x0 for any input
            # Zero probs (unobserved) → argmax = class 0 → "empty" logits
            # High-prob voxels → argmax = predicted class → peaked logits
            argmax_class = probs.argmax(dim=1)  # [B, H, W, D]
            one_hot = F.one_hot(argmax_class.long(), C).float()  # [B, H, W, D, C]
            logits = one_hot * ls - (1 - one_hot) * ls / (C - 1)
            logits = logits - logits.mean(dim=-1, keepdim=True)
            return logits.permute(0, 4, 1, 2, 3)  # [B, C, H, W, D]
        else:
            # factored_21ch encoding
            occ_prob = probs[:, 1:].sum(dim=1, keepdim=True).clamp(0, 1)
            occ_logit = self.logit_scale * (2.0 * occ_prob - 1.0)
            sem_logits = self.logit_scale * probs
            return torch.cat([occ_logit, sem_logits], dim=1)

    @torch.no_grad()
    def sample_sdedit(
        self,
        model: nn.Module,
        bev: torch.Tensor,
        lidar: torch.Tensor,
        lsk3d_probs: torch.Tensor,
        shape: tuple[int, int, int, int],
        device: torch.device,
        lifted_features: torch.Tensor | None = None,
        start_timestep: int = 200,
        guidance_scale: float = 6.0,
        num_steps: int = 50,
        show_progress: bool = True,
    ) -> torch.Tensor:
        """SDEdit from LSK3DNet TTA probabilities using DDIM with subsampled steps.

        Uses DDIM (deterministic) with subsampled timesteps for speed (50 steps
        instead of 200 full DDPM steps = 4x speedup).

        Args:
            lsk3d_probs: [B, 20, H, W, D] soft probs from LSK3DNet TTA
            start_timestep: Where to start denoising (e.g. 200/1000 = mild noise)
            guidance_scale: CFG strength (1.0 = no CFG, 6.0 = DiffSSC default)
            num_steps: Number of DDIM steps (default 50)
        """
        B, H, W, D = shape
        model.eval()

        bev_onehot = F.one_hot(bev.long(), num_classes=self.num_classes).float()
        bev_onehot = bev_onehot.permute(0, 3, 1, 2).to(device)

        # Convert LSK3DNet probs to continuous x_0 matching encoding format
        x_0_init = self.probs_to_logits(lsk3d_probs)

        # Add noise to start_timestep (VP forward)
        noise = torch.randn_like(x_0_init)
        t_start = torch.full((B,), start_timestep, device=device, dtype=torch.long)
        x_t = self.q_sample(x_0_init, t_start, noise)

        # Subsample timesteps: start_timestep → 0 in num_steps DDIM steps
        step_ratio = start_timestep / num_steps
        timesteps = [int(start_timestep - i * step_ratio) for i in range(num_steps)]
        timesteps = [max(0, t) for t in timesteps]
        timesteps.append(0)

        use_cfg = guidance_scale > 1.0

        if show_progress:
            step_iter = tqdm(range(len(timesteps) - 1),
                           desc=f"SDEdit-DDIM ({num_steps} steps, t={start_timestep}→0)")
        else:
            step_iter = range(len(timesteps) - 1)

        for i in step_iter:
            t_val = timesteps[i]
            t_next = timesteps[i + 1]
            t_batch = torch.full((B,), t_val, device=device, dtype=torch.long)

            # Get eps prediction (with optional CFG)
            if use_cfg:
                cond_eps = model(x_t, t_batch, bev_onehot, lidar,
                               lifted_features=lifted_features)
                uncond_eps = model(x_t, t_batch, bev_onehot, lidar,
                                 lifted_features=lifted_features,
                                 force_uncond=True)
                eps_pred = uncond_eps + guidance_scale * (cond_eps - uncond_eps)
            else:
                eps_pred = model(x_t, t_batch, bev_onehot, lidar,
                               lifted_features=lifted_features)

            # Predict x_0 from eps
            x_0_pred = self._predict_x0(x_t, t_batch, eps_pred)

            # DDIM update: deterministic, no noise
            # After x_0 clipping in _predict_x0, raw eps_pred is inconsistent
            # with clipped x_0_pred. Recompute corrected direction from clipped x_0.
            if t_next > 0:
                sqrt_alpha_cur = self.sqrt_alphas_cumprod[t_val]
                sqrt_one_minus_alpha_cur = self.sqrt_one_minus_alphas_cumprod[t_val]
                # Corrected eps consistent with clipped x_0_pred
                eps_corrected = (x_t - sqrt_alpha_cur * x_0_pred) / sqrt_one_minus_alpha_cur.clamp(min=1e-8)

                sqrt_alpha_next = self.sqrt_alphas_cumprod[t_next]
                sqrt_one_minus_alpha_next = self.sqrt_one_minus_alphas_cumprod[t_next]
                # x_{t-1} = sqrt(alpha_bar_{t-1}) * x_0_pred + sqrt(1-alpha_bar_{t-1}) * eps_corrected
                x_t = sqrt_alpha_next * x_0_pred + sqrt_one_minus_alpha_next * eps_corrected
            else:
                x_t = x_0_pred

        return self.decode_x0(x_t)
