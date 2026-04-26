"""
Multinomial Diffusion for 2D BEV Semantic Prediction

Adapted from scene_completion/multinomial_diffusion_3d.py for 2D BEV maps.
This is the FIXED version applying Phase 3's successful techniques:

Key differences from original BEV diffusion (d3pm.py):
1. KL posterior loss instead of CE (proper variational bound)
2. Stronger noise schedule (beta_max=0.1 instead of 0.02)
3. Focal loss (gamma=2.0) for hard example mining
4. Class-balanced weights (beta=0.999) for rare class handling
5. Occupied pixel weighting (non-empty pixels get higher weight)
6. Observation weighting (LiDAR-observed areas get higher weight)
7. Lovász loss for direct IoU optimization

Shapes:
- 3D SSC: [B, K, H, W, D] -> 2D BEV: [B, K, H, W]
"""

import math

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


class MultinomialDiffusion2D(nn.Module):
    """
    Multinomial Diffusion for 2D BEV semantic maps.

    Adapted from MultinomialDiffusion3DV2 with all enhancements.
    The only difference is shape: [B, K, H, W] instead of [B, K, H, W, D].

    Args:
        num_classes: Number of semantic classes (K)
        num_timesteps: Number of diffusion steps (T)
        beta_min: Minimum noise level
        beta_max: Maximum noise level (0.1 for strong noise)
        focal_gamma: Focal loss gamma (2.0 for hard mining)
        class_0_weight: Weight for empty/road class
        occupied_weight: Weight multiplier for non-empty pixels
        lovasz_weight: Weight for Lovász loss component
        obs_weight_factor: Extra weight for LiDAR-observed pixels
    """

    def __init__(
        self,
        num_classes: int = 20,
        num_timesteps: int = 100,
        beta_min: float = 0.0001,
        beta_max: float = 0.1,  # 5× stronger than original 0.02!
        focal_gamma: float = 2.0,
        class_0_weight: float = 0.02,  # Very low weight for empty class
        occupied_weight: float = 10.0,  # Weight for non-empty pixels
        lovasz_weight: float = 0.3,  # Weight for Lovász loss
        obs_weight_factor: float = 2.0,  # Extra weight for LiDAR-observed
        class_weights: torch.Tensor = None,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.num_timesteps = num_timesteps
        self.focal_gamma = focal_gamma
        self.class_0_weight = class_0_weight
        self.occupied_weight = occupied_weight
        self.lovasz_weight = lovasz_weight
        self.obs_weight_factor = obs_weight_factor

        # Register class weights
        if class_weights is not None:
            self.register_buffer('class_weights', class_weights)
        else:
            weights = self._compute_class_balanced_weights()
            self.register_buffer('class_weights', weights)

        # Create STRONGER beta schedule
        betas = torch.linspace(beta_min, beta_max, num_timesteps, dtype=torch.float64)

        # Compute α values
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        # Log for debugging
        print("[MultinomialDiffusion2D] Noise schedule:")
        print(f"  beta: {beta_min} → {beta_max}")
        mid = num_timesteps // 2
        last = num_timesteps - 1
        print(f"  alpha_cumprod[0] = {alphas_cumprod[0]:.4f}")
        print(f"  alpha_cumprod[{mid}] = {alphas_cumprod[mid]:.4f}")
        print(f"  alpha_cumprod[{last}] = {alphas_cumprod[last]:.6f}")
        print(f"  At t={last}: P(correct) = {alphas_cumprod[last] + (1-alphas_cumprod[last])/num_classes:.4f}")

        # Convert to log space
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
        Compute Class-Balanced weights for SemanticKITTI BEV.

        BEV class frequencies differ from 3D (projected from above).
        """
        # BEV frequencies (from analysis of BEV maps)
        # Note: BEV has higher occupied ratio (~40% vs 3D's 5%)
        frequencies = [
            0.6000,   # 0: empty (largest)
            0.0180,   # 1: car
            0.0005,   # 2: bicycle
            0.0005,   # 3: motorcycle
            0.0010,   # 4: truck
            0.0015,   # 5: other-vehicle
            0.0020,   # 6: person
            0.0005,   # 7: bicyclist
            0.0005,   # 8: motorcyclist
            0.1200,   # 9: road (second largest)
            0.0050,   # 10: parking
            0.0600,   # 11: sidewalk
            0.0010,   # 12: other-ground
            0.0300,   # 13: building
            0.0020,   # 14: fence
            0.0700,   # 15: vegetation
            0.0020,   # 16: trunk
            0.0400,   # 17: terrain
            0.0030,   # 18: pole
            0.0025,   # 19: traffic-sign
        ]

        total_samples = 2_000_000
        sample_counts = [int(f * total_samples) for f in frequencies]

        raw_weights = []
        for n_c in sample_counts:
            if n_c == 0:
                raw_weights.append(0.0)
            else:
                raw_weights.append((1 - beta) / (1 - beta ** n_c))

        # Normalize
        nonzero = [w for i, w in enumerate(raw_weights) if w > 0 and i > 0]
        if nonzero:
            mean_weight = sum(nonzero) / len(nonzero)
            normalized = [w / mean_weight if w > 0 else 0.0 for w in raw_weights]
        else:
            normalized = raw_weights

        # Clip and set class 0 weight very low
        clipped = [min(w, max_weight) for w in normalized]
        clipped[0] = self.class_0_weight

        return torch.tensor(clipped, dtype=torch.float32)

    def q_probs(self, x_0: torch.Tensor, t: torch.Tensor, x_scpnet: torch.Tensor = None) -> torch.Tensor:
        """
        Compute q(x_t | x_0) - the forward diffusion distribution.

        Standard: q(x_t | x_0) = Cat(x_t | α_t * x_0 + (1 - α_t) / K)
        Cold:     q(x_t | x_0) = Cat(x_t | α_t * x_0 + (1 - α_t) * x_scpnet)

        Args:
            x_0: One-hot encoded clean data [B, K, H, W]
            t: Timesteps [B]
            x_scpnet: Optional SCPNet prediction one-hot [B, K, H, W] for structured-source forward

        Returns:
            Probabilities for x_t [B, K, H, W]
        """
        alpha_t = extract(self.alphas_cumprod, t, x_0.shape)  # [B, 1, 1, 1]
        if x_scpnet is not None:
            probs = alpha_t * x_0 + (1.0 - alpha_t) * x_scpnet  # Cold diffusion
        else:
            probs = alpha_t * x_0 + (1.0 - alpha_t) / self.num_classes  # Standard
        return probs

    def q_sample(self, x_0: torch.Tensor, t: torch.Tensor, x_scpnet: torch.Tensor = None) -> torch.Tensor:
        """
        Sample from q(x_t | x_0) - the forward diffusion process.

        Args:
            x_0: One-hot encoded clean data [B, K, H, W]
            t: Timesteps [B]
            x_scpnet: Optional SCPNet prediction one-hot [B, K, H, W] for structured-source forward

        Returns:
            Sampled x_t as one-hot [B, K, H, W]
        """
        probs = self.q_probs(x_0, t, x_scpnet=x_scpnet)  # [B, K, H, W]

        # Reshape for sampling: [B*H*W, K]
        shape = probs.shape
        K = shape[1]
        probs_flat = probs.permute(0, 2, 3, 1).reshape(-1, K)  # [B*H*W, K]

        # Sample from categorical
        samples = torch.multinomial(probs_flat, num_samples=1).squeeze(-1)  # [B*H*W]

        # Convert to one-hot and reshape
        x_t = F.one_hot(samples, num_classes=K).float()  # [B*H*W, K]
        x_t = x_t.reshape(shape[0], shape[2], shape[3], K)
        x_t = x_t.permute(0, 3, 1, 2)  # [B, K, H, W]

        return x_t

    def q_posterior_logits(
        self,
        x_0: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor,
        x_scpnet: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Compute log posterior q(x_{t-1} | x_t, x_0) numerically stable.

        Uses Bayes' rule:
        q(x_{t-1}=k | x_t, x_0) ∝ q(x_t | x_{t-1}=k) * q(x_{t-1}=k | x_0)

        For structured-source mode, uses x_scpnet instead of 1/K for the noise distribution.

        Args:
            x_0: Clean data probabilities [B, K, H, W] - can be one-hot or soft
            x_t: Noisy data (one-hot) [B, K, H, W]
            t: Timesteps [B]
            x_scpnet: Optional SCPNet prediction for structured-source forward [B, K, H, W]

        Returns:
            Log posterior probabilities (unnormalized) [B, K, H, W]
        """
        K = self.num_classes

        alpha_t = extract(self.alphas, t, x_0.shape)  # [B, 1, 1, 1]
        alpha_bar_t = extract(self.alphas_cumprod, t, x_0.shape)
        alpha_bar_t_minus_1 = extract(self.alphas_cumprod_prev, t, x_0.shape)

        noise = x_scpnet if x_scpnet is not None else torch.ones_like(x_0) / K

        # log q(x_t | x_{t-1}=k) — cold: uses x_scpnet instead of 1/K
        log_q_xt_given_xtm1 = torch.log(
            alpha_t * x_t + (1.0 - alpha_t) * noise + 1e-10
        )  # [B, K, H, W]

        # log q(x_{t-1}=k | x_0) — cold: uses x_scpnet instead of 1/K
        log_q_xtm1_given_x0 = torch.log(
            alpha_bar_t_minus_1 * x_0 + (1.0 - alpha_bar_t_minus_1) * noise + 1e-10
        )  # [B, K, H, W]

        # log q(x_{t-1}=k | x_t, x_0) ∝ log sum
        log_posterior_unnorm = log_q_xt_given_xtm1 + log_q_xtm1_given_x0

        return log_posterior_unnorm

    def q_posterior(
        self,
        x_0: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor,
        x_scpnet: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Compute normalized posterior q(x_{t-1} | x_t, x_0).

        Args:
            x_0: Clean data probabilities [B, K, H, W]
            x_t: Noisy data (one-hot) [B, K, H, W]
            t: Timesteps [B]
            x_scpnet: Optional SCPNet prediction for structured-source forward [B, K, H, W]

        Returns:
            Posterior probabilities [B, K, H, W]
        """
        log_posterior_unnorm = self.q_posterior_logits(x_0, x_t, t, x_scpnet=x_scpnet)
        log_posterior = log_posterior_unnorm - torch.logsumexp(
            log_posterior_unnorm, dim=1, keepdim=True
        )
        posterior = torch.exp(log_posterior)
        return posterior

    def _lovasz_grad(self, gt_sorted: torch.Tensor) -> torch.Tensor:
        """Compute gradient of Lovász extension."""
        gts = gt_sorted.sum()
        intersection = gts - gt_sorted.float().cumsum(0)
        union = gts + (1 - gt_sorted).float().cumsum(0)
        jaccard = 1. - intersection / union
        if len(jaccard) > 1:
            jaccard[1:] = jaccard[1:] - jaccard[:-1]
        return jaccard

    def _lovasz_softmax_flat(self, probas: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Multi-class Lovász-Softmax loss."""
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
        lidar_features: torch.Tensor,
        lidar_obs: torch.Tensor | None = None,
        addp_weighting: bool = False,
        x_scpnet: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Compute enhanced training loss with PROPER KL base + all enhancements.

        Args:
            model: Neural network predicting x_0 logits
            x_0: Clean BEV (class indices) [B, H, W]
            t: Timesteps [B]
            lidar_features: LiDAR conditioning features [B, C, H, W]
            lidar_obs: Optional LiDAR observation mask [B, H, W] for obs weighting
            addp_weighting: If True, weight noisy timesteps (high t) more heavily.
            x_scpnet: Optional SCPNet BEV one-hot [B, K, H, W] for structured-source forward.
                     When provided, uses structured-source forward process instead of uniform.

        Returns:
            Dictionary with 'loss' and metrics
        """
        B = x_0.shape[0]
        device = x_0.device

        # Convert x_0 to one-hot [B, K, H, W]
        x_0_onehot = F.one_hot(x_0.long(), num_classes=self.num_classes).float()
        x_0_onehot = x_0_onehot.permute(0, 3, 1, 2)  # [B, K, H, W]

        # Sample x_t from q(x_t | x_0) — structured-source forward uses x_scpnet
        x_t = self.q_sample(x_0_onehot, t, x_scpnet=x_scpnet)  # [B, K, H, W]

        # Get model prediction
        x_0_logits = model(x_t, t, lidar_features)  # [B, K, H, W]
        x_0_pred_probs = F.softmax(x_0_logits, dim=1)  # [B, K, H, W]

        # === PROPER KL LOSS ===
        # L_{t-1} = KL(q(x_{t-1}|x_t, x_0) || p(x_{t-1}|x_t))

        # True posterior: q(x_{t-1}|x_t, x_0) — structured-source forward uses x_scpnet
        q_posterior_true = self.q_posterior(x_0_onehot, x_t, t, x_scpnet=x_scpnet)

        # Predicted posterior: p(x_{t-1}|x_t) = q(x_{t-1}|x_t, x̂_0)
        p_posterior_pred = self.q_posterior(x_0_pred_probs, x_t, t, x_scpnet=x_scpnet)

        # KL(q || p) per pixel
        kl_per_class = q_posterior_true * (
            torch.log(q_posterior_true + 1e-10) - torch.log(p_posterior_pred + 1e-10)
        )  # [B, K, H, W]
        kl_per_pixel = kl_per_class.sum(dim=1)  # [B, H, W]

        # Flatten
        kl_flat = kl_per_pixel.reshape(B, -1)  # [B, N]
        x_0_flat = x_0.reshape(B, -1).long()  # [B, N]

        # === FOCAL MODULATION ===
        probs_flat = x_0_pred_probs.reshape(B, self.num_classes, -1)  # [B, K, N]
        p_t = probs_flat.gather(1, x_0_flat.unsqueeze(1)).squeeze(1)  # [B, N]
        focal_weight = (1 - p_t) ** self.focal_gamma

        focal_kl = focal_weight * kl_flat  # [B, N]

        # === CLASS-BALANCED WEIGHTING ===
        class_weight_per_pixel = self.class_weights.to(device)[x_0_flat]  # [B, N]
        weighted_kl = focal_kl * class_weight_per_pixel  # [B, N]

        # === OCCUPIED PIXEL WEIGHTING ===
        occupied_mask = (x_0_flat > 0).float()  # [B, N]
        pixel_weights = 1.0 + (self.occupied_weight - 1.0) * occupied_mask  # [B, N]

        # === OBSERVATION WEIGHTING ===
        if lidar_obs is not None and self.obs_weight_factor > 0:
            obs_flat = lidar_obs.reshape(B, -1)  # [B, N]
            obs_weights = 1.0 + self.obs_weight_factor * obs_flat
            pixel_weights = pixel_weights * obs_weights

        # === ADDP TIMESTEP WEIGHTING ===
        # Key insight from ADDP/DDP (ICLR 2025): For PERCEPTION tasks,
        # noisy timesteps (high t) matter MORE than clean timesteps (low t).
        # This is OPPOSITE to image generation!
        # Weight: t=0 (clean) -> 1.0, t=T-1 (noisy) -> 2.0
        if addp_weighting:
            timestep_weights = 1.0 + (t.float() / self.num_timesteps)  # [B]
            # Expand to match pixel weights: [B] -> [B, 1] for broadcasting
            timestep_weights = timestep_weights.view(B, 1)
            pixel_weights = pixel_weights * timestep_weights

        final_weighted_kl = (weighted_kl * pixel_weights).mean()

        # === LOVÁSZ LOSS ===
        if self.lovasz_weight > 0:
            probs_for_lovasz = probs_flat.permute(0, 2, 1).reshape(-1, self.num_classes)  # [B*N, K]
            labels_for_lovasz = x_0_flat.reshape(-1)  # [B*N]
            lovasz_loss = self._lovasz_softmax_flat(probs_for_lovasz, labels_for_lovasz)
            total_loss = final_weighted_kl + self.lovasz_weight * lovasz_loss
        else:
            lovasz_loss = torch.tensor(0.0, device=device)
            total_loss = final_weighted_kl

        # Compute metrics
        x_0_pred = x_0_logits.argmax(dim=1)  # [B, H, W]
        accuracy = (x_0_pred == x_0).float().mean()

        # Accuracy on occupied pixels only
        occupied_pixels = (x_0 > 0)
        if occupied_pixels.sum() > 0:
            occupied_accuracy = ((x_0_pred == x_0) & occupied_pixels).float().sum() / occupied_pixels.float().sum()
        else:
            occupied_accuracy = torch.tensor(0.0, device=device)

        return {
            'loss': total_loss,
            'kl_loss': final_weighted_kl,
            'lovasz_loss': lovasz_loss,
            'accuracy': accuracy,
            'occupied_accuracy': occupied_accuracy,
        }

    @torch.no_grad()
    def p_sample(
        self,
        model: nn.Module,
        x_t: torch.Tensor,
        t: torch.Tensor,
        lidar_features: torch.Tensor,
        soft: bool = False,
    ) -> torch.Tensor:
        """
        Sample x_{t-1} from p(x_{t-1} | x_t).

        Args:
            model: Neural network
            x_t: Current noisy sample (one-hot or probs) [B, K, H, W]
            t: Current timestep [B]
            lidar_features: LiDAR conditioning [B, C, H, W]
            soft: If True, return soft probabilities instead of hard one-hot samples.
                  This preserves information and avoids accumulating sampling errors.

        Returns:
            x_{t-1} as one-hot [B, K, H, W] (if soft=False) or probs [B, K, H, W] (if soft=True)
        """
        # Get model prediction
        x_0_logits = model(x_t, t, lidar_features)
        x_0_pred_probs = F.softmax(x_0_logits, dim=1)  # [B, K, H, W]

        # Use x_0_pred directly as sampling distribution
        probs = x_0_pred_probs

        # Soft sampling: keep probabilities, only argmax at t=0
        if soft:
            if (t == 0).all():
                # At t=0, convert to one-hot via argmax
                samples = probs.argmax(dim=1)  # [B, H, W]
                x_t_minus_1 = F.one_hot(samples, num_classes=self.num_classes).float()
                x_t_minus_1 = x_t_minus_1.permute(0, 3, 1, 2)  # [B, K, H, W]
            else:
                # Return soft probabilities - preserves full distribution
                x_t_minus_1 = probs
            return x_t_minus_1

        # Hard sampling: sample from categorical (original behavior)
        shape = probs.shape
        K = shape[1]
        probs_flat = probs.permute(0, 2, 3, 1).reshape(-1, K)  # [B*H*W, K]

        # For t=0, use argmax; for t>0, sample
        if (t == 0).all():
            samples = probs_flat.argmax(dim=-1)
        else:
            probs_flat = probs_flat.clamp(min=1e-10)
            probs_flat = probs_flat / probs_flat.sum(dim=-1, keepdim=True)
            samples = torch.multinomial(probs_flat, num_samples=1).squeeze(-1)

        # Convert to one-hot
        x_t_minus_1 = F.one_hot(samples, num_classes=K).float()
        x_t_minus_1 = x_t_minus_1.reshape(shape[0], shape[2], shape[3], K)
        x_t_minus_1 = x_t_minus_1.permute(0, 3, 1, 2)  # [B, K, H, W]

        return x_t_minus_1

    @torch.no_grad()
    def p_sample_cfg(
        self,
        model: nn.Module,
        x_t: torch.Tensor,
        t: torch.Tensor,
        lidar_features: torch.Tensor,
        guidance_scale: float = 1.5,
        soft: bool = True,
    ) -> torch.Tensor:
        """
        Sample x_{t-1} from p(x_{t-1} | x_t) with Classifier-Free Guidance.

        CFG formula: logits = logits_uncond + guidance_scale * (logits_cond - logits_uncond)

        Args:
            model: Neural network (must have been trained with cond_drop_prob > 0)
            x_t: Current noisy sample (one-hot or probs) [B, K, H, W]
            t: Current timestep [B]
            lidar_features: LiDAR conditioning [B, C, H, W]
            guidance_scale: CFG scale (1.0 = no guidance, >1.0 = stronger conditioning)
            soft: If True, return soft probabilities instead of hard samples

        Returns:
            x_{t-1} as one-hot [B, K, H, W] (if soft=False) or probs [B, K, H, W] (if soft=True)
        """
        # Conditional prediction (with LiDAR features)
        logits_cond = model(x_t, t, lidar_features)

        if guidance_scale > 1.0:
            # Unconditional prediction (zeroed LiDAR features)
            logits_uncond = model(x_t, t, torch.zeros_like(lidar_features))
            # CFG formula
            logits = logits_uncond + guidance_scale * (logits_cond - logits_uncond)
        else:
            logits = logits_cond

        probs = F.softmax(logits, dim=1)  # [B, K, H, W]

        # Soft sampling: keep probabilities, only argmax at t=0
        if soft:
            if (t == 0).all():
                samples = probs.argmax(dim=1)  # [B, H, W]
                x_t_minus_1 = F.one_hot(samples, num_classes=self.num_classes).float()
                x_t_minus_1 = x_t_minus_1.permute(0, 3, 1, 2)  # [B, K, H, W]
            else:
                x_t_minus_1 = probs
            return x_t_minus_1

        # Hard sampling
        shape = probs.shape
        K = shape[1]
        probs_flat = probs.permute(0, 2, 3, 1).reshape(-1, K)

        if (t == 0).all():
            samples = probs_flat.argmax(dim=-1)
        else:
            probs_flat = probs_flat.clamp(min=1e-10)
            probs_flat = probs_flat / probs_flat.sum(dim=-1, keepdim=True)
            samples = torch.multinomial(probs_flat, num_samples=1).squeeze(-1)

        x_t_minus_1 = F.one_hot(samples, num_classes=K).float()
        x_t_minus_1 = x_t_minus_1.reshape(shape[0], shape[2], shape[3], K)
        x_t_minus_1 = x_t_minus_1.permute(0, 3, 1, 2)

        return x_t_minus_1

    @torch.no_grad()
    def sample(
        self,
        model: nn.Module,
        lidar_features: torch.Tensor,
        shape: tuple[int, int, int],  # (B, H, W)
        device: torch.device,
        show_progress: bool = True,
        soft: bool = True,
        num_steps: int | None = None,
    ) -> torch.Tensor:
        """
        Generate samples via reverse diffusion.

        Args:
            model: Neural network
            lidar_features: LiDAR conditioning [B, C, H, W]
            shape: Output shape (B, H, W)
            device: Device
            show_progress: Show progress bar
            soft: If True (default), use soft probability updates instead of hard
                  multinomial sampling. This preserves information across steps and
                  significantly improves performance (25% vs 3% mIoU).
            num_steps: Number of denoising steps. If None, use all timesteps.
                      For soft sampling, 10 steps is usually sufficient.

        Returns:
            Generated BEV (class indices) [B, H, W]
        """
        B, H, W = shape
        # Only call eval() if model is a nn.Module (not a function wrapper)
        if isinstance(model, nn.Module):
            model.eval()

        # Start from uniform noise - ALWAYS do hard initial sampling
        # (model was trained on one-hot inputs, not soft probabilities)
        x_t = torch.ones(B, self.num_classes, H, W, device=device) / self.num_classes
        probs_flat = x_t.permute(0, 2, 3, 1).reshape(-1, self.num_classes)
        samples = torch.multinomial(probs_flat, num_samples=1).squeeze(-1)
        x_t = F.one_hot(samples, num_classes=self.num_classes).float()
        x_t = x_t.reshape(B, H, W, self.num_classes)
        x_t = x_t.permute(0, 3, 1, 2).to(device)  # [B, K, H, W]

        # Determine timesteps to use
        if num_steps is not None and num_steps < self.num_timesteps:
            # Use evenly spaced timesteps for faster sampling
            # Match training validation: [90, 80, 70, 60, 50, 40, 30, 20, 10, 0] for T=100, steps=10
            step_size = self.num_timesteps // num_steps
            # Start from (T-10) and go down in steps of step_size, ending at 0
            timesteps = list(range(self.num_timesteps - step_size, -1, -step_size))
            # e.g., for T=100, steps=10: [90, 80, 70, 60, 50, 40, 30, 20, 10, 0]
        else:
            # Use all timesteps
            timesteps = list(range(self.num_timesteps))[::-1]

        if show_progress:
            timesteps = tqdm(timesteps, desc="Sampling")

        for t in timesteps:
            t_batch = torch.full((B,), t, device=device, dtype=torch.long)
            x_t = self.p_sample(model, x_t, t_batch, lidar_features, soft=soft)

        # Convert to class indices
        x_0 = x_t.argmax(dim=1)  # [B, H, W]

        return x_0

    @torch.no_grad()
    def sample_algo2(
        self,
        model: nn.Module,
        lidar_features: torch.Tensor,
        scpnet_bev: torch.Tensor,
        n_steps: int = 100,
        show_progress: bool = False,
    ) -> torch.Tensor:
        """S2D2 correction sampling for BEV (specialising the non-noise correction sampler of Cold Diffusion (Bansal et al., 2022) to our linear simplex interpolant).

        x_{t-1} = x_t + (α_{t-1} - α_t)·(x̂_0 - x_scpnet)
        Monotonically improves with more steps, no posterior bottleneck.

        Args:
            model: BEV denoising model
            lidar_features: LiDAR conditioning [B, C, H, W]
            scpnet_bev: SCPNet BEV predictions (class indices) [B, H, W]
            n_steps: Number of sampling steps
            show_progress: Show progress bar
        Returns:
            Refined BEV (class indices) [B, H, W]
        """
        import numpy as np
        K = self.num_classes
        device = lidar_features.device
        alphas = self.alphas_cumprod.to(device)

        # Prepare SCPNet BEV as one-hot
        scp_oh = F.one_hot(scpnet_bev.long(), K).float().permute(0, 3, 1, 2).to(device)
        x_t = scp_oh.clone()

        # Build timestep schedule
        timesteps = list(range(99, -1, -1)) if n_steps >= 100 else \
                    list(np.linspace(99, 0, n_steps, dtype=int))

        if show_progress:
            timesteps = tqdm(timesteps, desc="S2D2 BEV")

        for t in timesteps:
            B = x_t.shape[0]
            t_batch = torch.full((B,), t, device=device, dtype=torch.long)

            x_0_logits = model(x_t, t_batch, lidar_features)
            x_0_pred = F.softmax(x_0_logits, dim=1)

            if t > 0:
                alpha_t = alphas[t]
                alpha_prev = alphas[t - 1]
                delta_alpha = alpha_prev - alpha_t
                correction = x_0_pred - scp_oh
                x_t = x_t + delta_alpha * correction
                x_t = x_t.clamp(min=0.0)
                x_t = x_t / (x_t.sum(dim=1, keepdim=True) + 1e-10)
            else:
                x_t = x_0_pred

        return x_t.argmax(dim=1)

    @torch.no_grad()
    def sample_cfg(
        self,
        model: nn.Module,
        lidar_features: torch.Tensor,
        shape: tuple[int, int, int],  # (B, H, W)
        device: torch.device,
        guidance_scale: float = 1.5,
        show_progress: bool = True,
        soft: bool = True,
        num_steps: int | None = None,
    ) -> torch.Tensor:
        """
        Generate samples with Classifier-Free Guidance.

        Args:
            model: Neural network (must have been trained with cond_drop_prob > 0)
            lidar_features: LiDAR conditioning [B, C, H, W]
            shape: Output shape (B, H, W)
            device: Device
            guidance_scale: CFG scale (1.0 = no guidance, >1.0 = stronger conditioning)
            show_progress: Show progress bar
            soft: Use soft probability updates
            num_steps: Number of denoising steps (None = all timesteps)

        Returns:
            Generated BEV (class indices) [B, H, W]
        """
        B, H, W = shape
        if isinstance(model, nn.Module):
            model.eval()

        # Start from uniform noise
        x_t = torch.ones(B, self.num_classes, H, W, device=device) / self.num_classes
        probs_flat = x_t.permute(0, 2, 3, 1).reshape(-1, self.num_classes)
        samples = torch.multinomial(probs_flat, num_samples=1).squeeze(-1)
        x_t = F.one_hot(samples, num_classes=self.num_classes).float()
        x_t = x_t.reshape(B, H, W, self.num_classes)
        x_t = x_t.permute(0, 3, 1, 2).to(device)

        # Determine timesteps
        if num_steps is not None and num_steps < self.num_timesteps:
            step_size = self.num_timesteps // num_steps
            timesteps = list(range(self.num_timesteps - step_size, -1, -step_size))
        else:
            timesteps = list(range(self.num_timesteps))[::-1]

        if show_progress:
            timesteps = tqdm(timesteps, desc="Sampling (CFG)")

        for t in timesteps:
            t_batch = torch.full((B,), t, device=device, dtype=torch.long)
            x_t = self.p_sample_cfg(model, x_t, t_batch, lidar_features,
                                     guidance_scale=guidance_scale, soft=soft)

        x_0 = x_t.argmax(dim=1)
        return x_0

    @torch.no_grad()
    def sample_ensemble(
        self,
        model: nn.Module,
        lidar_features: torch.Tensor,
        shape: tuple[int, int, int],  # (B, H, W)
        device: torch.device,
        num_samples: int = 5,
        guidance_scale: float = 1.5,
        show_progress: bool = False,
        soft: bool = True,
        num_steps: int | None = None,
    ) -> torch.Tensor:
        """
        Generate samples with test-time ensembling (multiple passes + voting).

        Runs the diffusion process multiple times with different random seeds
        and returns the majority vote at each pixel.

        Args:
            model: Neural network
            lidar_features: LiDAR conditioning [B, C, H, W]
            shape: Output shape (B, H, W)
            device: Device
            num_samples: Number of diffusion passes for ensemble
            guidance_scale: CFG scale
            show_progress: Show progress bar (for each sample)
            soft: Use soft probability updates
            num_steps: Number of denoising steps

        Returns:
            Generated BEV (class indices) [B, H, W]
        """
        B, H, W = shape
        predictions = []

        # Save RNG state to restore after ensemble (avoid corrupting training RNG)
        rng_state = torch.random.get_rng_state()
        cuda_rng_state = torch.cuda.get_rng_state() if device.type == 'cuda' else None

        for seed_offset in range(num_samples):
            # Set different random seed for each pass
            torch.manual_seed(42 + seed_offset)

            pred = self.sample_cfg(
                model, lidar_features, shape, device,
                guidance_scale=guidance_scale,
                show_progress=show_progress,
                soft=soft,
                num_steps=num_steps,
            )
            predictions.append(pred)

        # Restore RNG state
        torch.random.set_rng_state(rng_state)
        if cuda_rng_state is not None:
            torch.cuda.set_rng_state(cuda_rng_state)

        # Stack and vote across samples
        stacked = torch.stack(predictions, dim=0)  # [N, B, H, W]
        final_pred, _ = torch.mode(stacked, dim=0)

        return final_pred

    def q_sample_from_coarse(
        self,
        coarse_pred: torch.Tensor,
        t_start: int,
    ) -> torch.Tensor:
        """
        Add noise to a coarse prediction (SegRefiner-inspired).

        Instead of starting from pure noise, start from a coarse prediction
        with added multinomial noise. This is the key insight from SegRefiner:
        the diffusion process refines an existing prediction rather than
        generating from scratch.

        Args:
            coarse_pred: Coarse BEV prediction [B, H, W] (class indices)
            t_start: Starting timestep (0 = clean, T-1 = most noisy)

        Returns:
            Noisy version of coarse prediction [B, K, H, W] (one-hot)
        """
        B, H, W = coarse_pred.shape
        device = coarse_pred.device

        # Convert to one-hot
        coarse_onehot = F.one_hot(coarse_pred.long(), self.num_classes).float()
        coarse_onehot = coarse_onehot.permute(0, 3, 1, 2)  # [B, K, H, W]

        # Sample noisy version at t_start
        t = torch.full((B,), t_start, device=device, dtype=torch.long)
        x_t = self.q_sample(coarse_onehot, t)

        return x_t

    @torch.no_grad()
    def sample_segrefiner(
        self,
        model: nn.Module,
        coarse_pred: torch.Tensor,
        lidar_features: torch.Tensor,
        coarse_logits: torch.Tensor,
        t_start: int | None = None,
        guidance_scale: float = 1.0,
        show_progress: bool = False,
    ) -> torch.Tensor:
        """
        SegRefiner-inspired sampling with optional Classifier-Free Guidance.

        Instead of starting from pure noise, start from a noisy version of the
        coarse prediction and denoise to refine it.

        Args:
            model: SegRefiner UNet model
            coarse_pred: Coarse BEV prediction [B, H, W] (class indices)
            lidar_features: LiDAR encoder features [B, C, H, W]
            coarse_logits: Coarse prediction logits [B, K, H, W]
            t_start: Starting timestep (default: num_timesteps // 2)
            guidance_scale: CFG scale (1.0 = no guidance, >1 = stronger conditioning)
            show_progress: Show progress bar

        Returns:
            Refined BEV prediction [B, H, W] (class indices)
        """
        B, H, W = coarse_pred.shape
        device = coarse_pred.device
        if isinstance(model, nn.Module):
            model.eval()

        if t_start is None:
            t_start = self.num_timesteps // 2

        # Start from noisy coarse prediction
        x_t = self.q_sample_from_coarse(coarse_pred, t_start)  # [B, K, H, W]

        # Denoise from t_start down to 0
        timesteps = list(range(t_start, -1, -1))
        if show_progress:
            timesteps = tqdm(timesteps, desc="SegRefiner sampling")

        for t in timesteps:
            t_batch = torch.full((B,), t, device=device, dtype=torch.long)

            # Conditional prediction (with LiDAR)
            logits_cond = model(x_t, t_batch, lidar_features, coarse_logits)

            # CFG: Unconditional prediction (zero LiDAR)
            if guidance_scale > 1.0:
                logits_uncond = model(
                    x_t, t_batch,
                    torch.zeros_like(lidar_features),  # Dropped LiDAR
                    coarse_logits,  # Keep coarse conditioning
                    force_drop_cond=True,
                )
                # CFG formula: pred = uncond + w * (cond - uncond)
                logits = logits_uncond + guidance_scale * (logits_cond - logits_uncond)
            else:
                logits = logits_cond

            # Sample x_{t-1}
            if t > 0:
                probs = F.softmax(logits, dim=1)
                # Multinomial sampling
                probs_flat = probs.permute(0, 2, 3, 1).reshape(-1, self.num_classes)
                probs_flat = probs_flat.clamp(min=1e-10)
                probs_flat = probs_flat / probs_flat.sum(dim=-1, keepdim=True)
                samples = torch.multinomial(probs_flat, num_samples=1).squeeze(-1)
                x_t = F.one_hot(samples, num_classes=self.num_classes).float()
                x_t = x_t.reshape(B, H, W, self.num_classes)
                x_t = x_t.permute(0, 3, 1, 2)  # [B, K, H, W]
            else:
                # At t=0, use argmax (deterministic)
                x_t = F.one_hot(logits.argmax(dim=1), self.num_classes).float()
                x_t = x_t.permute(0, 3, 1, 2)

        # Convert to class indices
        return x_t.argmax(dim=1)  # [B, H, W]


def create_segrefiner_diffusion(
    num_classes: int = 20,
    num_timesteps: int = 6,
    beta_start: float = 0.8,
    beta_end: float = 0.0,
    **kwargs,
) -> 'SegRefinerDiffusion':
    """
    Create a SegRefiner-inspired diffusion model.

    SegRefiner uses a different noise schedule:
    - Only 6 timesteps (not 100 or 1000)
    - Linear beta from 0.8 to 0.0 (high noise to clean)

    Args:
        num_classes: Number of semantic classes
        num_timesteps: Number of diffusion steps (default: 6)
        beta_start: Starting beta (high noise, default: 0.8)
        beta_end: Ending beta (clean, default: 0.0)
        **kwargs: Additional arguments for diffusion

    Returns:
        SegRefinerDiffusion instance
    """
    return SegRefinerDiffusion(
        num_classes=num_classes,
        num_timesteps=num_timesteps,
        beta_start=beta_start,
        beta_end=beta_end,
        **kwargs,
    )


class SegRefinerDiffusion(nn.Module):
    """
    SegRefiner-inspired discrete diffusion for multi-class semantic refinement.

    Key differences from standard MultinomialDiffusion2D:
    1. Uses discrete transitions between GT and COARSE (not uniform noise!)
    2. Linear beta schedule from high (0.8) to low (0.0)
    3. Only 6 timesteps by default
    4. Designed for refinement (coarse → refined) not generation

    The forward process corrupts GT by replacing pixels with coarse predictions:
        x_t = transition_map * x_start + (1 - transition_map) * x_coarse
        where transition_map ~ Bernoulli(alpha_bar_t)

    This is fundamentally different from multinomial diffusion which adds
    uniform noise. Here we specifically transition between GT and coarse.

    Based on SegRefiner: Wang et al., NeurIPS 2023
    https://github.com/MengyuWang826/SegRefiner
    Code: https://github.com/MengyuWang826/SegRefiner/blob/main/mmdet/models/detectors/segrefiner_base.py
    """

    def __init__(
        self,
        num_classes: int = 20,
        num_timesteps: int = 6,
        beta_start: float = 0.8,
        beta_end: float = 0.0,
        focal_gamma: float = 2.0,
        lovasz_weight: float = 0.3,
        obs_weight_factor: float = 0.5,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.num_timesteps = num_timesteps
        self.focal_gamma = focal_gamma
        self.lovasz_weight = lovasz_weight
        self.obs_weight_factor = obs_weight_factor

        # SegRefiner-inspired schedule: betas_cumprod linearly from beta_start to beta_end
        # betas_cumprod[t] = probability of keeping GT at timestep t
        # At t=0: betas_cumprod[0] = beta_start (high, e.g., 0.8 = 80% GT)
        # At t=T-1: betas_cumprod[T-1] = beta_end (low, e.g., 0.0 = 0% GT)
        betas_cumprod = torch.linspace(beta_start, beta_end, num_timesteps, dtype=torch.float64)

        # Compute betas from cumulative product (for consistency with other code)
        # betas_cumprod[t] = prod_{s=0}^{t} (1 - beta[s])
        # So beta[t] = 1 - betas_cumprod[t] / betas_cumprod[t-1]
        betas_cumprod_prev = F.pad(betas_cumprod[:-1], (1, 0), value=1.0)
        betas = 1.0 - betas_cumprod / betas_cumprod_prev.clamp(min=1e-10)
        betas = betas.clamp(min=0.0, max=1.0)

        alphas = 1.0 - betas
        alphas_cumprod = betas_cumprod  # This IS the probability of keeping GT

        # Log for debugging
        print(f"[SegRefinerDiffusion] Created with {num_timesteps} timesteps")
        print(f"  P(keep GT): {beta_start} → {beta_end}")
        print(f"  alphas_cumprod (P(keep GT)): {alphas_cumprod.numpy()}")
        print(f"  At t=0: P(keep GT) = {alphas_cumprod[0]:.4f}")
        print(f"  At t={num_timesteps-1}: P(keep GT) = {alphas_cumprod[-1]:.4f}")

        # Register buffers
        self.register_buffer('betas', betas.float())
        self.register_buffer('alphas', alphas.float())
        self.register_buffer('alphas_cumprod', alphas_cumprod.float())
        self.register_buffer('alphas_cumprod_prev', betas_cumprod_prev.float())

        # Class-balanced weights
        weights = self._compute_class_weights()
        self.register_buffer('class_weights', weights)

    def _compute_class_weights(self) -> torch.Tensor:
        """Compute class-balanced weights for SemanticKITTI BEV."""
        # Same frequencies as MultinomialDiffusion2D
        frequencies = [
            0.6000, 0.0180, 0.0005, 0.0005, 0.0010,
            0.0015, 0.0020, 0.0005, 0.0005, 0.1200,
            0.0050, 0.0600, 0.0010, 0.0300, 0.0020,
            0.0700, 0.0020, 0.0400, 0.0030, 0.0025,
        ]
        weights = [1.0 / (f + 1e-6) for f in frequencies]
        # Normalize
        mean_w = sum(weights[1:]) / (len(weights) - 1)
        weights = [w / mean_w for w in weights]
        weights[0] = 0.02  # Very low weight for empty class
        return torch.tensor(weights, dtype=torch.float32)

    def q_sample(
        self,
        x_start: torch.Tensor,
        x_coarse: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Sample from q(x_t | x_start, x_coarse) - SegRefiner forward process.

        Unlike standard diffusion which adds noise, SegRefiner transitions
        between GT (x_start) and coarse prediction (x_coarse):

            transition_map ~ Bernoulli(alpha_bar_t)
            x_t = transition_map * x_start + (1 - transition_map) * x_coarse

        At t=0 (high alpha): mostly keep GT
        At t=T-1 (low alpha): mostly use coarse

        Args:
            x_start: Ground truth one-hot [B, K, H, W]
            x_coarse: Coarse prediction one-hot [B, K, H, W]
            t: Timesteps [B]

        Returns:
            Noisy sample x_t [B, K, H, W] (one-hot)
        """
        # Get probability of keeping GT at this timestep
        alpha_bar_t = extract(self.alphas_cumprod, t, x_start.shape)  # [B, 1, 1, 1]

        # Sample transition map: which pixels keep GT vs use coarse
        noise = torch.rand_like(x_start[:, :1, :, :])  # [B, 1, H, W]
        transition_map = (noise < alpha_bar_t).float()  # [B, 1, H, W]

        # Apply transition: GT where transition_map=1, coarse where transition_map=0
        x_t = transition_map * x_start + (1.0 - transition_map) * x_coarse

        return x_t

    def q_sample_training(
        self,
        x_0: torch.Tensor,
        x_coarse: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Sample for training: corrupt GT with coarse predictions.

        Args:
            x_0: Ground truth one-hot [B, K, H, W]
            x_coarse: Coarse prediction one-hot [B, K, H, W]
            t: Timesteps [B]

        Returns:
            Noisy sample x_t [B, K, H, W]
        """
        return self.q_sample(x_0, x_coarse, t)

    def q_sample_from_coarse(
        self,
        coarse_pred: torch.Tensor,
        t_start: int,
    ) -> torch.Tensor:
        """
        Start sampling from coarse prediction (inference).

        Following SegRefiner: At inference, we start directly from the coarse
        prediction (x = x_last in their code). No noise is added - the coarse
        prediction IS the noisy starting point that we will refine.

        See: https://github.com/MengyuWang826/SegRefiner/blob/main/mmdet/models/detectors/segrefiner_base.py
        Line: "x = x_last" in p_sample_loop

        Args:
            coarse_pred: Coarse prediction [B, H, W] (class indices)
            t_start: Starting timestep (unused, kept for API compatibility)

        Returns:
            Coarse prediction as one-hot [B, K, H, W]
        """
        # Convert to one-hot - this IS our starting point for refinement
        coarse_onehot = F.one_hot(coarse_pred.long(), self.num_classes).float()
        coarse_onehot = coarse_onehot.permute(0, 3, 1, 2)  # [B, K, H, W]

        return coarse_onehot

    def q_posterior(
        self,
        x_0: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Compute normalized posterior q(x_{t-1} | x_t, x_0)."""
        K = self.num_classes
        alpha_t = extract(self.alphas, t, x_0.shape)
        alpha_bar_t_minus_1 = extract(self.alphas_cumprod_prev, t, x_0.shape)

        log_q_xt_given_xtm1 = torch.log(
            alpha_t * x_t + (1.0 - alpha_t) / K + 1e-10
        )
        log_q_xtm1_given_x0 = torch.log(
            alpha_bar_t_minus_1 * x_0 + (1.0 - alpha_bar_t_minus_1) / K + 1e-10
        )
        log_posterior_unnorm = log_q_xt_given_xtm1 + log_q_xtm1_given_x0
        log_posterior = log_posterior_unnorm - torch.logsumexp(
            log_posterior_unnorm, dim=1, keepdim=True
        )
        return torch.exp(log_posterior)

    def _lovasz_grad(self, gt_sorted: torch.Tensor) -> torch.Tensor:
        """Compute gradient of Lovász extension."""
        gts = gt_sorted.sum()
        intersection = gts - gt_sorted.float().cumsum(0)
        union = gts + (1 - gt_sorted).float().cumsum(0)
        jaccard = 1. - intersection / union
        if len(jaccard) > 1:
            jaccard[1:] = jaccard[1:] - jaccard[:-1]
        return jaccard

    def _lovasz_softmax_flat(self, probas: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Multi-class Lovász-Softmax loss."""
        C = probas.size(1)
        losses = []
        for c in range(1, C):
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
        lidar_features: torch.Tensor,
        coarse_logits: torch.Tensor,
        lidar_obs: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Compute SegRefiner training loss.

        The model learns to predict the clean GT from noisy versions,
        conditioned on both LiDAR features and coarse predictions.

        Key insight from SegRefiner: During training, we corrupt GT by
        randomly replacing pixels with the COARSE prediction (not uniform noise).
        This teaches the model to refine coarse predictions to match GT.

        Args:
            model: SegRefiner UNet
            x_0: Clean BEV (class indices) [B, H, W]
            t: Timesteps [B]
            lidar_features: LiDAR conditioning [B, C, H, W]
            coarse_logits: Coarse prediction logits [B, K, H, W]
            lidar_obs: Optional observation mask [B, H, W]

        Returns:
            Dictionary with 'loss' and metrics
        """
        B = x_0.shape[0]
        device = x_0.device

        # Convert x_0 (GT) to one-hot
        x_0_onehot = F.one_hot(x_0.long(), num_classes=self.num_classes).float()
        x_0_onehot = x_0_onehot.permute(0, 3, 1, 2)  # [B, K, H, W]

        # Convert coarse logits to one-hot (take argmax then one-hot)
        coarse_pred = coarse_logits.argmax(dim=1)  # [B, H, W]
        coarse_onehot = F.one_hot(coarse_pred.long(), num_classes=self.num_classes).float()
        coarse_onehot = coarse_onehot.permute(0, 3, 1, 2)  # [B, K, H, W]

        # Sample x_t from q(x_t | x_0, x_coarse) - SegRefiner forward process
        # This corrupts GT by replacing some pixels with coarse predictions
        x_t = self.q_sample(x_0_onehot, coarse_onehot, t)

        # Get model prediction (conditioned on LiDAR and coarse)
        x_0_logits = model(x_t, t, lidar_features, coarse_logits)
        x_0_pred_probs = F.softmax(x_0_logits, dim=1)

        # KL loss: KL(q(x_{t-1}|x_t, x_0) || p(x_{t-1}|x_t))
        q_posterior_true = self.q_posterior(x_0_onehot, x_t, t)
        p_posterior_pred = self.q_posterior(x_0_pred_probs, x_t, t)

        kl_per_class = q_posterior_true * (
            torch.log(q_posterior_true + 1e-10) - torch.log(p_posterior_pred + 1e-10)
        )
        kl_per_pixel = kl_per_class.sum(dim=1)

        # Flatten
        kl_flat = kl_per_pixel.reshape(B, -1)
        x_0_flat = x_0.reshape(B, -1).long()

        # Focal modulation
        probs_flat = x_0_pred_probs.reshape(B, self.num_classes, -1)
        p_t = probs_flat.gather(1, x_0_flat.unsqueeze(1)).squeeze(1)
        focal_weight = (1 - p_t) ** self.focal_gamma
        focal_kl = focal_weight * kl_flat

        # Class-balanced weighting
        class_weight_per_pixel = self.class_weights.to(device)[x_0_flat]
        weighted_kl = focal_kl * class_weight_per_pixel

        # Occupied pixel weighting
        occupied_mask = (x_0_flat > 0).float()
        pixel_weights = 1.0 + 9.0 * occupied_mask

        # Observation weighting
        if lidar_obs is not None and self.obs_weight_factor > 0:
            obs_flat = lidar_obs.reshape(B, -1)
            obs_weights = 1.0 + self.obs_weight_factor * obs_flat
            pixel_weights = pixel_weights * obs_weights

        final_kl_loss = (weighted_kl * pixel_weights).mean()

        # Lovász loss
        if self.lovasz_weight > 0:
            probs_for_lovasz = probs_flat.permute(0, 2, 1).reshape(-1, self.num_classes)
            labels_for_lovasz = x_0_flat.reshape(-1)
            lovasz_loss = self._lovasz_softmax_flat(probs_for_lovasz, labels_for_lovasz)
            total_loss = final_kl_loss + self.lovasz_weight * lovasz_loss
        else:
            lovasz_loss = torch.tensor(0.0, device=device)
            total_loss = final_kl_loss

        # Metrics
        x_0_pred = x_0_logits.argmax(dim=1)
        accuracy = (x_0_pred == x_0).float().mean()

        return {
            'loss': total_loss,
            'kl_loss': final_kl_loss,
            'lovasz_loss': lovasz_loss,
            'accuracy': accuracy,
        }

    @torch.no_grad()
    def sample(
        self,
        model: nn.Module,
        coarse_pred: torch.Tensor,
        lidar_features: torch.Tensor,
        coarse_logits: torch.Tensor,
        t_start: int | None = None,
        guidance_scale: float = 1.0,
        show_progress: bool = False,
    ) -> torch.Tensor:
        """
        SegRefiner sampling with CFG.

        Follows the SegRefiner paper's approach:
        - Start from coarse prediction (no noise)
        - Iteratively refine by transitioning between prediction and coarse
        - Use confidence-based transition (accumulating fine probability)

        See: https://github.com/MengyuWang826/SegRefiner/blob/main/mmdet/models/detectors/segrefiner_base.py

        Args:
            model: SegRefiner UNet
            coarse_pred: Coarse prediction [B, H, W]
            lidar_features: LiDAR features [B, C, H, W]
            coarse_logits: Coarse logits [B, K, H, W]
            t_start: Starting timestep (default: num_timesteps - 1)
            guidance_scale: CFG scale
            show_progress: Show progress bar

        Returns:
            Refined prediction [B, H, W]
        """
        B, H, W = coarse_pred.shape
        device = coarse_pred.device
        if isinstance(model, nn.Module):
            model.eval()

        if t_start is None:
            t_start = self.num_timesteps - 1

        # Start from coarse prediction (SegRefiner: x = x_last)
        x_t = self.q_sample_from_coarse(coarse_pred, t_start)  # [B, K, H, W]

        # Keep coarse as one-hot for transition
        coarse_onehot = x_t.clone()

        # Cumulative fine probability (SegRefiner's cur_fine_probs)
        cur_fine_probs = torch.zeros(B, 1, H, W, device=device)

        timesteps = list(range(t_start, -1, -1))
        if show_progress:
            timesteps = tqdm(timesteps, desc="SegRefiner")

        for t in timesteps:
            t_batch = torch.full((B,), t, device=device, dtype=torch.long)

            # Conditional prediction (predicts x_0)
            logits_cond = model(x_t, t_batch, lidar_features, coarse_logits)

            # CFG
            if guidance_scale > 1.0:
                logits_uncond = model(
                    x_t, t_batch,
                    torch.zeros_like(lidar_features),
                    coarse_logits,
                    force_drop_cond=True,
                )
                logits = logits_uncond + guidance_scale * (logits_cond - logits_uncond)
            else:
                logits = logits_cond

            # Get predicted x_0 as probabilities and one-hot
            probs = F.softmax(logits, dim=1)  # [B, K, H, W]
            pred_x0 = F.one_hot(probs.argmax(dim=1), self.num_classes).float()
            pred_x0 = pred_x0.permute(0, 3, 1, 2)  # [B, K, H, W]

            # SegRefiner-inspired transition (adapted for multi-class)
            # Original binary: x_start_fine_probs = 2 * |sigmoid(logits) - 0.5|
            # This maps uncertain (0.5) -> 0, certain (0 or 1) -> 1
            #
            # For multi-class softmax with K classes:
            # - Uniform distribution: max_prob = 1/K
            # - Certain prediction: max_prob = 1.0
            # We normalize to [0, 1]: (max_prob - 1/K) / (1 - 1/K)
            max_prob = probs.max(dim=1, keepdim=True)[0]  # [B, 1, H, W]
            uniform_prob = 1.0 / self.num_classes
            x_start_fine_probs = (max_prob - uniform_prob) / (1.0 - uniform_prob)
            x_start_fine_probs = x_start_fine_probs.clamp(min=0.0, max=1.0)

            # Compute transition probability
            # SegRefiner: p_c_to_f = conf * (alpha_prev - alpha) / (1 - conf * alpha)
            alpha_t = self.alphas_cumprod[t].item()
            alpha_t_prev = self.alphas_cumprod[t - 1].item() if t > 0 else 1.0

            denominator = (1.0 - x_start_fine_probs * alpha_t).clamp(min=1e-6)
            p_c_to_f = x_start_fine_probs * (alpha_t_prev - alpha_t) / denominator

            # Update cumulative fine probability
            cur_fine_probs = cur_fine_probs + (1.0 - cur_fine_probs) * p_c_to_f

            # Create transition map based on accumulated confidence
            # At final step (t=0), this determines which pixels use prediction vs coarse
            sample_noise = torch.rand_like(cur_fine_probs)
            fine_map = (sample_noise < cur_fine_probs).float()  # [B, 1, H, W]

            # Transition: use prediction where confident, coarse otherwise
            # This applies at ALL timesteps including t=0 (fixes SegRefiner bug)
            x_t = pred_x0 * fine_map + coarse_onehot * (1.0 - fine_map)

        return x_t.argmax(dim=1)

    @torch.no_grad()
    def sample_entropy_gated(
        self,
        model: nn.Module,
        coarse_pred: torch.Tensor,
        lidar_features: torch.Tensor,
        coarse_logits: torch.Tensor,
        guidance_scale: float = 1.0,
        temperature: float = 1.5,
        confidence_threshold: float = 0.7,
        show_progress: bool = False,
    ) -> torch.Tensor:
        """
        OPTIMAL sampling with entropy-based confidence gating.

        Key improvements over standard SegRefiner:
        1. Entropy-based confidence (not max_prob) - principled uncertainty
        2. Temperature scaling - reduces overconfidence
        3. Deterministic threshold - no stochastic noise
        4. Only refine where model disagrees with coarse

        Args:
            model: SegRefiner UNet
            coarse_pred: Coarse prediction [B, H, W]
            lidar_features: LiDAR features [B, C, H, W]
            coarse_logits: Coarse logits [B, K, H, W]
            guidance_scale: CFG scale (1.0 = no guidance)
            temperature: Softmax temperature (>1 reduces overconfidence)
            confidence_threshold: Only refine if confidence > threshold
            show_progress: Show progress bar

        Returns:
            Refined prediction [B, H, W]
        """

        B, H, W = coarse_pred.shape
        device = coarse_pred.device
        if isinstance(model, nn.Module):
            model.eval()

        t_start = self.num_timesteps - 1

        # Start from coarse prediction
        x_t = self.q_sample_from_coarse(coarse_pred, t_start)  # [B, K, H, W]
        coarse_onehot = x_t.clone()

        # Accumulate confidence through diffusion steps
        accumulated_confidence = torch.zeros(B, 1, H, W, device=device)
        accumulated_logits = torch.zeros(B, self.num_classes, H, W, device=device)

        timesteps = list(range(t_start, -1, -1))
        if show_progress:
            timesteps = tqdm(timesteps, desc="Entropy-Gated Sampling")

        for t in timesteps:
            t_batch = torch.full((B,), t, device=device, dtype=torch.long)

            # Get model prediction
            logits_cond = model(x_t, t_batch, lidar_features, coarse_logits)

            # CFG
            if guidance_scale > 1.0:
                logits_uncond = model(
                    x_t, t_batch,
                    torch.zeros_like(lidar_features),
                    coarse_logits,
                    force_drop_cond=True,
                )
                logits = logits_uncond + guidance_scale * (logits_cond - logits_uncond)
            else:
                logits = logits_cond

            # Accumulate logits (ensemble effect)
            accumulated_logits = accumulated_logits + logits

            # Temperature-scaled softmax for calibrated probabilities
            probs = F.softmax(logits / temperature, dim=1)  # [B, K, H, W]

            # ENTROPY-BASED CONFIDENCE (key improvement)
            # H(p) = -Σ p_i * log(p_i)
            entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=1, keepdim=True)  # [B, 1, H, W]
            max_entropy = math.log(self.num_classes)  # log(20) ≈ 3.0

            # Confidence: 1 when certain (entropy=0), 0 when uniform (entropy=max)
            step_confidence = 1.0 - entropy / max_entropy
            step_confidence = step_confidence.clamp(min=0.0, max=1.0)

            # Update accumulated confidence (exponential moving average)
            alpha = 1.0 / (t_start - t + 1)  # More weight to later (cleaner) steps
            accumulated_confidence = (1 - alpha) * accumulated_confidence + alpha * step_confidence

            # Get prediction for this step
            pred_x0 = F.one_hot(probs.argmax(dim=1), self.num_classes).float()
            pred_x0 = pred_x0.permute(0, 3, 1, 2)  # [B, K, H, W]

            # Update x_t for next iteration (use accumulated confidence)
            if t > 0:
                conf_map = (accumulated_confidence > confidence_threshold).float()
                x_t = pred_x0 * conf_map + coarse_onehot * (1.0 - conf_map)

        # FINAL OUTPUT: Entropy-gated selective refinement
        # Use averaged logits for final prediction
        final_probs = F.softmax(accumulated_logits / temperature, dim=1)
        final_pred = final_probs.argmax(dim=1)  # [B, H, W]

        # Compute final entropy-based confidence
        final_entropy = -(final_probs * torch.log(final_probs + 1e-10)).sum(dim=1)
        final_confidence = 1.0 - final_entropy / max_entropy

        # SELECTIVE REFINEMENT: Only change pixels where:
        # 1. Model is confident (entropy-based)
        # 2. Model disagrees with coarse (otherwise why change?)
        confident_mask = (final_confidence > confidence_threshold)  # [B, H, W]
        disagree_mask = (final_pred != coarse_pred)  # [B, H, W]
        refine_mask = confident_mask & disagree_mask

        # Output: Use model prediction only where confident AND disagrees
        output = torch.where(refine_mask, final_pred, coarse_pred)

        return output


if __name__ == "__main__":
    # Test
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    diffusion = MultinomialDiffusion2D(
        num_classes=20,
        num_timesteps=100,
        beta_max=0.1,
    ).to(device)

    print(f"Diffusion created with {diffusion.num_timesteps} timesteps")

    # Test forward diffusion
    B, H, W = 2, 256, 256
    x_0 = torch.randint(0, 20, (B, H, W), device=device)
    x_0_onehot = F.one_hot(x_0.long(), num_classes=20).float()
    x_0_onehot = x_0_onehot.permute(0, 3, 1, 2)  # [B, K, H, W]

    t = torch.randint(0, 100, (B,), device=device)

    print(f"Input shape: {x_0.shape}")
    print(f"One-hot shape: {x_0_onehot.shape}")

    x_t = diffusion.q_sample(x_0_onehot, t)
    print(f"Sampled x_t shape: {x_t.shape}")

    assert x_t.sum(dim=1).allclose(torch.ones_like(x_t[:, 0])), "x_t should be one-hot"
    print("Forward diffusion test passed!")
