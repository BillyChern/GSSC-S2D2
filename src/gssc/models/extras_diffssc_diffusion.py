"""
Anisotropic VP Gaussian Diffusion for joint geometry + semantics.

Key design (from DiffSSC paper):
- Forward: y_t = y_0 + sqrt(1-alpha_bar_t) * W * epsilon
  (additive perturbation, NOT interpolation)
- Anisotropic noise: sigma_p=1.0 (spatial), sigma_s=0.2 (semantic)
- Loss: MSE + lambda_p * spatial_reg + lambda_s * semantic_reg
- Sampling: DPM-Solver++ (sde-dpmsolver++, order=2, 50 steps)
- CFG: uncond_prob=0.1, uncond_w=6.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .diffssc_utils import points_to_sparse_tensor, semantic_encode, semantic_decode, probs_to_semantic_logits


class PointCloudGaussianDiffusion(nn.Module):
    """Anisotropic VP Gaussian diffusion for point cloud completion.

    Args:
        model: SpconvUNetDiff instance
        partial_enc: SpconvGlobalEnc instance
        beta_start: starting beta
        beta_end: ending beta
        t_steps: number of diffusion timesteps
        s_steps: number of sampling steps
        sigma_p: spatial noise scale
        sigma_s: semantic noise scale
        lambda_p: spatial regularization weight
        lambda_s: semantic regularization weight
        uncond_prob: probability of unconditional training (CFG)
        uncond_w: classifier-free guidance weight
        resolution: quantization resolution for sparse tensors
        num_classes: number of semantic classes
    """
    def __init__(
        self,
        model,
        partial_enc,
        beta_start=3.5e-5,
        beta_end=0.007,
        beta_schedule='linear',
        t_steps=1000,
        s_steps=50,
        sigma_p=1.0,
        sigma_s=0.2,
        lambda_p=5.0,
        lambda_s=4.0,
        uncond_prob=0.1,
        uncond_w=6.0,
        resolution=0.05,
        num_classes=20,
    ):
        super().__init__()
        self.model = model
        self.partial_enc = partial_enc
        self.t_steps = t_steps
        self.s_steps = s_steps
        self.sigma_p = sigma_p
        self.sigma_s = sigma_s
        self.lambda_p = lambda_p
        self.lambda_s = lambda_s
        self.uncond_prob = uncond_prob
        self.uncond_w = uncond_w
        self.resolution = resolution
        self.num_classes = num_classes

        # Beta schedule
        if beta_schedule == 'cosine':
            steps = np.arange(t_steps + 1)
            alpha_bar = np.cos(((steps / t_steps) + 0.008) / 1.008 * np.pi / 2) ** 2
            alpha_bar = alpha_bar / alpha_bar[0]
            betas = 1 - (alpha_bar[1:] / alpha_bar[:-1])
            betas = np.clip(betas, 0, 0.999)
        else:
            betas = np.linspace(beta_start, beta_end, t_steps)
        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)

        self.register_buffer('betas', torch.tensor(betas, dtype=torch.float32))
        self.register_buffer('alphas', torch.tensor(alphas, dtype=torch.float32))
        self.register_buffer('alphas_cumprod', torch.tensor(alphas_cumprod, dtype=torch.float32))
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(torch.tensor(alphas_cumprod, dtype=torch.float32)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod',
                             torch.sqrt(1.0 - torch.tensor(alphas_cumprod, dtype=torch.float32)))

        # Compute sampling timestep sequence (s_steps evenly spaced from T-1 to 0)
        self._sampling_timesteps = torch.linspace(t_steps - 1, 0, s_steps).long()

    def _ddim_step_additive(self, x_t, eps_pred, t_cur, t_next):
        """Single DDIM step for additive noise model.

        Forward: x_t = x_0 + sqrt(1-ᾱ_t) * eps
        x_0 prediction: x_0 = x_t - sqrt(1-ᾱ_t) * eps_pred
        DDIM step: x_{t-1} = x_0_pred + sqrt(1-ᾱ_{t-1}) * eps_pred

        Args:
            x_t: [B, N, F] current noisy sample
            eps_pred: [B, N, F] predicted noise
            t_cur: current timestep (scalar)
            t_next: next timestep (scalar, or None for final step)

        Returns:
            x_{t-1}: [B, N, F] denoised sample
        """
        sqrt_1m_a_t = self.sqrt_one_minus_alphas_cumprod[t_cur]
        x_0_pred = x_t - sqrt_1m_a_t * eps_pred

        if t_next is not None and t_next > 0:
            sqrt_1m_a_next = self.sqrt_one_minus_alphas_cumprod[t_next]
            return x_0_pred + sqrt_1m_a_next * eps_pred
        return x_0_pred

    def anisotropic_noise(self, shape, device):
        """Generate anisotropic noise: stronger for xyz, weaker for semantics.

        Args:
            shape: (B, N, F) where F = 3 + num_classes
            device: torch device

        Returns:
            noise: [B, N, F] with different scales for spatial/semantic channels
        """
        noise = torch.randn(shape, device=device)
        noise[:, :, :3] *= self.sigma_p   # spatial channels
        noise[:, :, 3:] *= self.sigma_s   # semantic channels
        return noise

    def q_sample(self, x_0, t, noise_scaled):
        """Forward diffusion: additive perturbation.

        y_t = y_0 + sqrt(1-alpha_bar_t) * W * epsilon
        Note: W is already applied to noise_scaled (via anisotropic_noise).

        Args:
            x_0: [B, N, F] clean data
            t: [B] timestep indices
            noise_scaled: [B, N, F] already scaled anisotropic noise

        Returns:
            x_t: [B, N, F] noisy data
        """
        sqrt_1m_a = self.sqrt_one_minus_alphas_cumprod[t][:, None, None]
        return x_0 + sqrt_1m_a * noise_scaled

    def _build_sparse(self, point_feats_list, coord_min=None):
        """Build sparse tensor from list of [N_i, F] point features."""
        return points_to_sparse_tensor(point_feats_list, resolution=self.resolution,
                                       coord_min=coord_min)

    def _compute_shared_coord_min(self, *point_cloud_lists):
        """Compute shared coord_min from multiple point clouds.

        Ensures all sparse tensors built with this coord_min share the same
        coordinate frame, which is critical for correct 1-NN matching.
        """
        all_xyz = []
        for pc in point_cloud_lists:
            for b in range(pc.shape[0]):
                all_xyz.append(pc[b, :, :3])
        all_xyz = torch.cat(all_xyz, dim=0)
        return torch.round(all_xyz / self.resolution).int().min(dim=0).values

    def _encode_partial(self, pcd_part, mean, std, coord_min=None):
        """Encode partial point cloud to sparse features.

        Args:
            pcd_part: [B, N_part, F] partial point cloud features
            mean: [B, 3] spatial mean
            std: [B, 3] spatial std
            coord_min: optional shared coordinate offset

        Returns:
            part_sparse: spconv.SparseConvTensor with encoded features
        """
        part_list = [pcd_part[b] for b in range(pcd_part.shape[0])]
        part_sparse, _, _ = self._build_sparse(part_list, coord_min=coord_min)
        return self.partial_enc(part_sparse)

    def _encode_uncond(self, pcd_part, coord_min=None):
        """Build unconditional partial encoding for CFG.

        Keeps xyz coordinates (preserving spatial voxel structure for
        numerically stable encoder forward pass) but zeros semantic
        features (unconditional on content).
        """
        uncond_list = []
        for b in range(pcd_part.shape[0]):
            pt = pcd_part[b].clone()
            pt[:, 3:] = 0.0  # Zero semantic channels, keep xyz
            uncond_list.append(pt)
        uncond_sparse, _, _ = self._build_sparse(uncond_list, coord_min=coord_min)
        return self.partial_enc(uncond_sparse)

    def classfree_forward(self, full_sparse, part_encoded, uncond_encoded, t,
                          bev=None, coord_min=None):
        """Classifier-free guidance forward.

        eps = eps_uncond + w * (eps_cond - eps_uncond)
        BEV conditioning is always present (no CFG dropout on BEV).

        Args:
            full_sparse: spconv.SparseConvTensor of noisy full cloud
            part_encoded: encoded partial features (conditional)
            uncond_encoded: encoded zero features (unconditional)
            t: [B] timesteps
            bev: [B, 256, 256] optional BEV labels (S30)
            coord_min: [3] int tensor for BEV lookup (S30)

        Returns:
            guided_noise: [N_voxels, F] noise prediction with CFG
        """
        eps_cond = self.model(full_sparse, part_encoded, t, bev=bev, coord_min=coord_min)
        eps_uncond = self.model(full_sparse, uncond_encoded, t, bev=bev, coord_min=coord_min)
        return eps_uncond + self.uncond_w * (eps_cond - eps_uncond)

    def training_step(self, batch):
        """Single training step.

        Args:
            batch: dict with 'pcd_full' [B, N, F], 'pcd_part' [B, N_part, F],
                   'mean' [B, 3], 'std' [B, 3]

        Returns:
            loss: scalar tensor
            metrics: dict of loss components
        """
        pcd_full = batch['pcd_full']   # [B, N_full, 23]
        pcd_part = batch['pcd_part']   # [B, N_part, 23]
        bev = batch.get('bev')         # [B, 256, 256] or None (S30)
        B = pcd_full.shape[0]
        device = pcd_full.device

        # Sample noise (anisotropic)
        noise = self.anisotropic_noise(pcd_full.shape, device)

        # Sample timestep
        t = torch.randint(0, self.t_steps, (B,), device=device)

        # Forward diffusion: y_t = y_0 + sqrt(1-alpha_bar_t) * noise
        x_t = self.q_sample(pcd_full, t, noise)

        # Compute shared coord_min from union of noisy full + clean partial
        # so both sparse tensors share the same coordinate frame for 1-NN matching
        shared_coord_min = self._compute_shared_coord_min(x_t, pcd_part)

        # Build sparse tensor for noisy full cloud
        x_t_list = [x_t[b] for b in range(B)]
        full_sparse, coord_min, inverse_maps = self._build_sparse(x_t_list,
                                                                   coord_min=shared_coord_min)

        # Encode partial cloud (with CFG: sometimes use zeros)
        if torch.rand(1).item() > self.uncond_prob or B == 1:
            part_encoded = self._encode_partial(pcd_part, batch.get('mean'), batch.get('std'),
                                                coord_min=shared_coord_min)
        else:
            part_encoded = self._encode_uncond(pcd_part, coord_min=shared_coord_min)

        # Predict noise (pass BEV + coord_min for S30)
        eps_pred_voxel = self.model(full_sparse, part_encoded, t,
                                    bev=bev, coord_min=coord_min)  # [N_voxels, 23]

        # Map voxel predictions back to points for loss computation
        # Since voxels are averaged, we need noise target per voxel too
        noise_flat = torch.cat([noise[b] for b in range(B)], dim=0)  # [N_total, 23]
        inverse_all = torch.cat(inverse_maps, dim=0)  # [N_total]

        # Average noise per voxel (to match averaged features)
        n_voxels = eps_pred_voxel.shape[0]
        noise_target = torch.zeros(n_voxels, noise_flat.shape[1], dtype=noise_flat.dtype, device=device)
        counts = torch.zeros(n_voxels, dtype=torch.float32, device=device)
        noise_target.scatter_add_(0, inverse_all.unsqueeze(1).expand_as(noise_flat), noise_flat)
        counts.scatter_add_(0, inverse_all, torch.ones(len(inverse_all), device=device))
        counts = counts.clamp(min=1.0)
        noise_target = noise_target / counts.unsqueeze(1)

        # MSE loss
        loss_mse = F.mse_loss(eps_pred_voxel, noise_target)

        # Spatial regularization (channels 0-2): target std = sigma_p
        eps_spatial = eps_pred_voxel[:, :3]
        loss_p = self.lambda_p * (eps_spatial.mean() ** 2 + (eps_spatial.std() - self.sigma_p) ** 2)

        # Semantic regularization (channels 3-22): target std = sigma_s
        eps_semantic = eps_pred_voxel[:, 3:]
        loss_s = self.lambda_s * (eps_semantic.mean() ** 2 + (eps_semantic.std() - self.sigma_s) ** 2)

        loss = loss_mse + loss_p + loss_s

        metrics = {
            'loss_mse': loss_mse.item(),
            'loss_p': loss_p.item(),
            'loss_s': loss_s.item(),
            'loss': loss.item(),
            'eps_mean': eps_pred_voxel.mean().item(),
            'eps_std': eps_pred_voxel.std().item(),
        }

        return loss, metrics

    @torch.no_grad()
    def sample(self, batch, num_samples=1):
        """Generate completed point cloud via additive-noise DDIM sampling.

        Uses SDEdit initialization: repeat partial 10x + noise.
        Uses custom DDIM loop correct for additive forward process:
            x_t = x_0 + sqrt(1-ᾱ_t) * eps  (no sqrt(ᾱ) scaling on x_0)

        Args:
            batch: dict with 'pcd_part' [B, N_part, 23], 'mean', 'std'
            num_samples: unused (kept for API compat)

        Returns:
            completed: [B, N_full, 23] completed point clouds
        """
        was_training = self.training
        self.eval()  # Required for BatchNorm1d with small voxel counts

        pcd_part = batch['pcd_part']   # [B, N_part, 23]
        bev = batch.get('bev')         # [B, 256, 256] or None
        B = pcd_part.shape[0]
        device = pcd_part.device

        # SDEdit init: filter partial to eval grid, repeat to N_full, add noise
        # 1. Convert pcd_part semantics from LSK3DNet probs [0,1] to
        #    semantic_encode logit scale [-0.263, +5.0] to match training x_0
        # 2. Filter to SSC eval grid (forward hemisphere) to concentrate
        #    points where evaluation happens — matches training distribution
        EVAL_RANGE = (0, -25.6, -2, 51.2, 25.6, 4.4)
        n_full = pcd_part.shape[1] * 10  # target size (10x partial)
        x_init_parts = []
        for b in range(B):
            pts = pcd_part[b].clone()  # [N_part, 23]
            pts[:, 3:] = probs_to_semantic_logits(pts[:, 3:],
                                                   num_classes=self.num_classes)
            # Filter to eval grid
            mask = ((pts[:, 0] >= EVAL_RANGE[0]) & (pts[:, 0] <= EVAL_RANGE[3]) &
                    (pts[:, 1] >= EVAL_RANGE[1]) & (pts[:, 1] <= EVAL_RANGE[4]) &
                    (pts[:, 2] >= EVAL_RANGE[2]) & (pts[:, 2] <= EVAL_RANGE[5]))
            pts_fwd = pts[mask]
            if len(pts_fwd) < 100:
                pts_fwd = pts  # Fallback if too few forward points
            # Repeat to reach n_full
            repeats = (n_full + len(pts_fwd) - 1) // len(pts_fwd)
            x_init_parts.append(pts_fwd.repeat(repeats, 1)[:n_full])
        x_init = torch.stack(x_init_parts)  # [B, n_full, 23]

        noise_init = self.anisotropic_noise(x_init.shape, device)
        sqrt_1m_a_T = self.sqrt_one_minus_alphas_cumprod[self._sampling_timesteps[0]]
        x_t = x_init + sqrt_1m_a_T * noise_init

        # Compute shared coord_min from clean partial (fixed reference frame)
        shared_coord_min = self._compute_shared_coord_min(x_init, pcd_part)

        # Encode partial cloud (conditional + unconditional for CFG)
        part_encoded = self._encode_partial(pcd_part, batch.get('mean'), batch.get('std'),
                                            coord_min=shared_coord_min)
        uncond_encoded = self._encode_uncond(pcd_part, coord_min=shared_coord_min)

        timesteps = self._sampling_timesteps.to(device)

        # Additive-noise DDIM loop
        for step_idx in range(len(timesteps)):
            t_cur = timesteps[step_idx].item()
            t_next = timesteps[step_idx + 1].item() if step_idx + 1 < len(timesteps) else None
            t = torch.full((B,), t_cur, device=device, dtype=torch.long)

            # Build sparse tensor for current x_t
            x_t_list = [x_t[b] for b in range(B)]
            full_sparse, coord_offset, inverse_maps = self._build_sparse(x_t_list,
                                                                          coord_min=shared_coord_min)

            # CFG forward
            eps_pred = self.classfree_forward(full_sparse, part_encoded, uncond_encoded, t,
                                              bev=bev, coord_min=coord_offset)

            # Map voxel predictions back to per-point
            noise_per_point = []
            for b in range(B):
                inv_map = inverse_maps[b]
                noise_per_point.append(eps_pred[inv_map])

            eps_pred_full = torch.stack(noise_per_point, dim=0)  # [B, N_full, 23]

            # Additive-noise DDIM step (correct: no 1/sqrt(ᾱ) division)
            x_t = self._ddim_step_additive(x_t, eps_pred_full, t_cur, t_next)

            torch.cuda.empty_cache()

        if was_training:
            self.train()
        return x_t
