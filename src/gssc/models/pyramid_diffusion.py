"""
Pyramid Discrete Diffusion for SemanticKITTI - Exact implementation from original paper.
Adapted for SemanticKITTI voxel dimensions: s1(32×32×4) → s2(64×64×8) → s4(256×256×32)

DERIVED FROM THIRD-PARTY CODE. Ancestry of this file, measured line-by-line and
recorded in THIRD_PARTY_NOTICES.md (sections 1-3) at the repository root:

  * pyramid-discrete-diffusion (models/conditional_diffusion/con_diffusion.py)
    https://github.com/yuhengliu02/pyramid-discrete-diffusion
    MIT License, Copyright (c) 2023 Yuheng Liu           -- 184 matched lines
  * ehoogeboom/multinomial_diffusion (diffusion_utils/diffusion_multinomial.py)
    NO upstream license is published; see THIRD_PARTY_NOTICES.md section 2
                                                          -- 165 matched lines
  * lucidrains/denoising-diffusion-pytorch
    MIT License, Copyright (c) 2020 Phil Wang             -- credited by the
    upstream con_diffusion.py header

Most of the shared content is the multinomial-diffusion core common to all
three; the PDD-specific residue is the auxiliary-loss weighting. The MIT
permission notices these copyright lines require are reproduced in full in
THIRD_PARTY_NOTICES.md, which ships in the wheel at
``gssc/THIRD_PARTY_NOTICES.md``.
"""
from inspect import isfunction

import numpy as np
import torch
import torch.nn.functional as F

eps = 1e-8

# Core multinomial diffusion utilities (copied from pyramid project)
def sum_except_batch(x, num_dims=1):
    return x.reshape(*x.shape[:num_dims], -1).sum(-1)

def log_1_min_a(a):
    return torch.log(1 - a.exp() + 1e-40)

def log_add_exp(a, b):
    maximum = torch.max(a, b)
    return maximum + torch.log(torch.exp(a - maximum) + torch.exp(b - maximum))

def exists(x):
    return x is not None

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

def log_categorical(log_x_start, log_prob):
    return (log_x_start.exp() * log_prob).sum(dim=1)

def index_to_log_onehot(x, num_classes):
    assert x.max().item() < num_classes, f'Error: {x.max().item()} >= {num_classes}'
    x_onehot = F.one_hot(x, num_classes)
    permute_order = (0, -1) + tuple(range(1, len(x.size())))
    x_onehot = x_onehot.permute(permute_order)
    log_x = torch.log(x_onehot.float().clamp(min=1e-30))
    return log_x

def log_onehot_to_index(log_x):
    return log_x.argmax(1)

def cosine_beta_schedule(timesteps, s=0.008):
    """Cosine schedule as proposed in https://openreview.net/forum?id=-NEXDKk8gZ"""
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    alphas = (alphas_cumprod[1:] / alphas_cumprod[:-1])
    alphas = np.clip(alphas, a_min=0.001, a_max=1.)
    alphas = np.sqrt(alphas)
    return alphas


class PyramidDiscreteDiffusion(torch.nn.Module):
    """
    Pyramid Discrete Diffusion - exact implementation from original paper.
    Adapted for SemanticKITTI with proper multinomial diffusion.
    """

    def __init__(self, args, denoise_model, num_classes, num_timesteps=100,
                 multi_criterion=None, auxiliary_loss_weight=0.0005,
                 adaptive_auxiliary_loss=True, recon_loss=False):
        super().__init__()

        self.num_classes = num_classes
        self.num_timesteps = num_timesteps
        self._denoise_fn = denoise_model

        # Auxiliary loss parameters (from original)
        self.multi_criterion = multi_criterion
        self.auxiliary_loss_weight = auxiliary_loss_weight
        self.adaptive_auxiliary_loss = adaptive_auxiliary_loss
        self.recon_loss = recon_loss

        # Multinomial diffusion schedule (exact from original)
        alphas = cosine_beta_schedule(self.num_timesteps)
        alphas = torch.tensor(alphas.astype('float64'))

        log_alpha = torch.log(alphas)
        log_cumprod_alpha = torch.cumsum(log_alpha, dim=0)

        log_1_min_alpha = log_1_min_a(log_alpha)
        log_1_min_cumprod_alpha = log_1_min_a(log_cumprod_alpha)

        # Verify multinomial properties
        assert log_add_exp(log_alpha, log_1_min_alpha).abs().sum().item() < 1.e-5
        assert log_add_exp(log_cumprod_alpha, log_1_min_cumprod_alpha).abs().sum().item() < 1e-5
        assert (torch.cumsum(log_alpha, dim=0) - log_cumprod_alpha).abs().sum().item() < 1.e-5

        # Register buffers
        self.register_buffer('log_alpha', log_alpha.float())
        self.register_buffer('log_1_min_alpha', log_1_min_alpha.float())
        self.register_buffer('log_cumprod_alpha', log_cumprod_alpha.float())
        self.register_buffer('log_1_min_cumprod_alpha', log_1_min_cumprod_alpha.float())

        # For importance sampling
        self.register_buffer('Lt_history', torch.zeros(self.num_timesteps))
        self.register_buffer('Lt_count', torch.zeros(self.num_timesteps))

    def multinomial_kl(self, log_prob1, log_prob2):
        kl = (log_prob1.exp() * (log_prob1 - log_prob2)).sum(dim=1)
        return kl

    def q_pred_one_timestep(self, log_x_t, t):
        log_alpha_t = extract(self.log_alpha, t, log_x_t.shape)
        log_1_min_alpha_t = extract(self.log_1_min_alpha, t, log_x_t.shape)

        log_probs = log_add_exp(
            log_x_t + log_alpha_t,
            log_1_min_alpha_t - np.log(self.num_classes)
        )
        return log_probs

    def q_pred(self, log_x_start, t):
        log_cumprod_alpha_t = extract(self.log_cumprod_alpha, t, log_x_start.shape)
        log_1_min_cumprod_alpha = extract(self.log_1_min_cumprod_alpha, t, log_x_start.shape)

        log_probs = log_add_exp(
            log_x_start + log_cumprod_alpha_t,
            log_1_min_cumprod_alpha - np.log(self.num_classes)
        )
        return log_probs

    def predict_start(self, log_x_t, t, cond=None):
        x_t = log_onehot_to_index(log_x_t)
        out = self._denoise_fn(x_t, cond, t)
        log_pred = F.log_softmax(out, dim=1)
        return log_pred

    def q_posterior(self, log_x_start, log_x_t, t):
        t_minus_1 = t - 1
        t_minus_1 = torch.where(t_minus_1 < 0, torch.zeros_like(t_minus_1), t_minus_1)
        log_EV_qxtmin_x0 = self.q_pred(log_x_start, t_minus_1)

        num_axes = (1,) * (len(log_x_start.size()) - 1)
        t_broadcast = t.view(-1, *num_axes) * torch.ones_like(log_x_start)
        log_EV_qxtmin_x0 = torch.where(t_broadcast == 0, log_x_start, log_EV_qxtmin_x0)

        unnormed_logprobs = log_EV_qxtmin_x0 + self.q_pred_one_timestep(log_x_t, t)
        log_EV_xtmin_given_xt_given_xstart = unnormed_logprobs - torch.logsumexp(unnormed_logprobs, dim=1, keepdim=True)

        return log_EV_xtmin_given_xt_given_xstart

    def p_pred(self, log_x, t, cond=None):
        log_x0_recon = self.predict_start(log_x, t, cond)
        log_model_pred = self.q_posterior(log_x_start=log_x0_recon, log_x_t=log_x, t=t)
        return log_model_pred, log_x0_recon

    def log_sample_categorical(self, logits):
        uniform = torch.rand_like(logits)
        gumbel_noise = -torch.log(-torch.log(uniform + 1e-30) + 1e-30)
        sample = (gumbel_noise + logits).argmax(dim=1)
        log_sample = index_to_log_onehot(sample, self.num_classes)
        return log_sample

    def q_sample(self, log_x_start, t):
        log_EV_qxt_x0 = self.q_pred(log_x_start, t)
        log_sample = self.log_sample_categorical(log_EV_qxt_x0)
        return log_sample

    def sample_time(self, b, device, method='uniform'):
        if method == 'importance':
            if not (self.Lt_count > 10).all():
                return self.sample_time(b, device, method='uniform')

            Lt_sqrt = torch.sqrt(self.Lt_history + 1e-10) + 0.0001
            Lt_sqrt[0] = Lt_sqrt[1]
            pt_all = Lt_sqrt / Lt_sqrt.sum()

            t = torch.multinomial(pt_all, num_samples=b, replacement=True)
            pt = pt_all.gather(dim=0, index=t)
            return t, pt

        elif method == 'uniform':
            t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
            pt = torch.ones_like(t).float() / self.num_timesteps
            return t, pt
        else:
            raise ValueError(f"Unknown sampling method: {method}")

    def kl_prior(self, log_x_start):
        b = log_x_start.size(0)
        device = log_x_start.device
        ones = torch.ones(b, device=device).long()

        log_qxT_prob = self.q_pred(log_x_start, t=(self.num_timesteps - 1) * ones)
        log_half_prob = -torch.log(self.num_classes * torch.ones_like(log_qxT_prob))

        kl_prior = self.multinomial_kl(log_qxT_prob, log_half_prob)
        return sum_except_batch(kl_prior)

    def forward(self, x, cond=None):
        """
        Training forward pass with proper multinomial diffusion loss.

        Args:
            x: Ground truth voxels [B, H, W, D]
            cond: Conditional input (previous pyramid stage) [B, H, W, D]
        """
        b, device = x.size(0), x.device
        self.shape = x.size()[1:]  # Store shape as instance variable (matches original)
        t, pt = self.sample_time(b, device, 'importance')

        # Convert to log one-hot
        log_x_start = index_to_log_onehot(x, self.num_classes)

        # Forward diffusion
        log_x_t = self.q_sample(log_x_start, t)

        # Get true and predicted posteriors
        log_true_prob = self.q_posterior(log_x_start=log_x_start, log_x_t=log_x_t, t=t)
        log_model_prob, log_x0_recon = self.p_pred(log_x=log_x_t, t=t, cond=cond)

        # Compute losses
        kl = self.multinomial_kl(log_true_prob, log_model_prob)
        kl = sum_except_batch(kl)

        decoder_nll = -log_categorical(log_x_start, log_model_prob)
        decoder_nll = sum_except_batch(decoder_nll)

        mask = (t == torch.zeros_like(t)).float()
        kl_loss = mask * decoder_nll + (1. - mask) * kl

        # Update loss history for importance sampling
        if self.training:
            Lt2 = kl_loss.pow(2)
            Lt2_prev = self.Lt_history.gather(dim=0, index=t)
            new_Lt_history = (0.1 * Lt2 + 0.9 * Lt2_prev).detach()
            self.Lt_history.scatter_(dim=0, index=t, src=new_Lt_history)
            self.Lt_count.scatter_add_(dim=0, index=t, src=torch.ones_like(Lt2))

        # Prior loss
        kl_prior = self.kl_prior(log_x_start)

        # Main loss
        loss = kl_loss / pt + kl_prior

        # Auxiliary loss (from original con_diffusion.py:247-260)
        # Compute KL on reconstructed x0 (excluding last class dimension)
        kl_aux = self.multinomial_kl(log_x_start[:, :-1, :, :, :], log_x0_recon[:, :-1, :, :, :])
        kl_aux = sum_except_batch(kl_aux)

        # Optionally add multi-criterion loss (Lovasz)
        if self.recon_loss and self.multi_criterion is not None:
            kl_aux += self.multi_criterion(log_x0_recon.exp(), x)

        # Auxiliary loss with same masking
        kl_aux_loss = mask * decoder_nll + (1. - mask) * kl_aux

        # Adaptive weighting based on timestep
        if self.adaptive_auxiliary_loss:
            addition_loss_weight = (1 - t.float() / self.num_timesteps) + 1.0
        else:
            addition_loss_weight = 1.0

        # Add weighted auxiliary loss
        aux_loss = addition_loss_weight * self.auxiliary_loss_weight * kl_aux_loss / pt
        loss += aux_loss

        # Original normalization: sum() / (H * W) not total voxels
        # For s1 (32,32,4): divide by 32*32=1024, not 4096
        # This matches original code at con_diffusion.py:261
        loss = -loss.sum() / (self.shape[0] * self.shape[1])

        return -loss  # Negate to match original (loss is already negative)

    def p_sample(self, log_x, t, cond=None):
        """Single denoising step."""
        log_model_pred, log_x0_recon = self.p_pred(log_x, t, cond)

        if t[0] == 0:
            return log_x0_recon
        else:
            log_sample = self.log_sample_categorical(log_model_pred)
            return log_sample

    def p_sample_loop(self, shape, cond=None, device='cuda'):
        """Full sampling loop for generation."""
        # Start from uniform noise
        log_x = torch.log(torch.ones(shape[0], self.num_classes, *shape[1:], device=device) / self.num_classes)

        for i in reversed(range(self.num_timesteps)):
            t = torch.full((shape[0],), i, device=device, dtype=torch.long)
            log_x = self.p_sample(log_x, t, cond)

        return log_onehot_to_index(log_x)