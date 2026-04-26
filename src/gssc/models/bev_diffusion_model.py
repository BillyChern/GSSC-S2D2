"""
BEV Diffusion Model

Combines D3PM, LiDAR encoder, and BEV U-Net into a complete model
for BEV semantic segmentation via discrete diffusion.

Architecture:
    LiDAR Voxels (256×256×32) → Sparse 3D U-Net → BEV Features (256×256×C)
                                                       ↓
    Noisy BEV Labels → 2D U-Net Denoiser → Clean BEV Labels (256×256×20)
                            ↑
                     (conditioning + self-conditioning)
"""


import torch
import torch.nn as nn
import torch.nn.functional as F

from gssc.models.bev_d3pm import D3PM
from gssc.models.bev_lidar_encoder import LightweightLiDAREncoder, SparseLiDAREncoder
from gssc.models.bev_unet import BEVUNet, LightweightBEVUNet


class BEVDiffusionModel(nn.Module):
    """
    Complete BEV Diffusion Model for semantic segmentation.

    Combines:
        - Sparse 3D U-Net for LiDAR → BEV feature extraction
        - D3PM discrete diffusion process
        - 2D U-Net denoiser with self-conditioning

    Args:
        num_classes: Number of semantic classes (20 for SemanticKITTI)
        num_timesteps: Number of diffusion timesteps
        lidar_channels: Output channels from LiDAR encoder
        unet_base_channels: Base channels for U-Net
        use_self_conditioning: Whether to use self-conditioning
        use_cross_attention: Whether to use cross-attention for conditioning
        diffusion_type: 'absorbing' or 'uniform'
        schedule_type: 'linear' or 'cosine'
        class_weights: Optional class weights for imbalanced data
        lightweight: Use lightweight models for fast experiments
    """

    def __init__(
        self,
        num_classes: int = 20,
        num_timesteps: int = 1000,
        lidar_channels: int = 128,
        unet_base_channels: int = 128,
        use_self_conditioning: bool = True,
        use_cross_attention: bool = True,
        diffusion_type: str = 'absorbing',
        schedule_type: str = 'cosine',
        class_weights: torch.Tensor | None = None,
        aux_loss_weight: float = 0.001,
        lightweight: bool = False,
        input_shape: tuple[int, int, int] = (256, 256, 32),
    ):
        super().__init__()

        self.num_classes = num_classes
        self.num_timesteps = num_timesteps
        self.use_self_conditioning = use_self_conditioning
        self.lightweight = lightweight

        # D3PM diffusion process
        self.diffusion = D3PM(
            num_classes=num_classes,
            num_timesteps=num_timesteps,
            transition_type=diffusion_type,
            schedule_type=schedule_type,
            aux_loss_weight=aux_loss_weight,
            class_weights=class_weights,
        )

        # LiDAR encoder
        if lightweight:
            self.lidar_encoder = LightweightLiDAREncoder(
                in_channels=1,
                out_channels=lidar_channels,
                input_shape=input_shape,
            )
        else:
            self.lidar_encoder = SparseLiDAREncoder(
                in_channels=1,
                base_channels=32,
                out_channels=lidar_channels,
                height_pool='max',
                input_shape=input_shape,
            )

        # BEV U-Net denoiser
        if lightweight:
            self.denoiser = LightweightBEVUNet(
                num_classes=num_classes,
                base_channels=64,
                cond_channels=lidar_channels,
                use_self_conditioning=use_self_conditioning,
            )
        else:
            self.denoiser = BEVUNet(
                num_classes=num_classes,
                base_channels=unet_base_channels,
                channel_mults=(1, 2, 4, 8),
                num_res_blocks=2,
                attention_resolutions=(32, 16),
                cond_channels=lidar_channels,
                use_cross_attention=use_cross_attention,
                use_self_conditioning=use_self_conditioning,
            )

    def encode_lidar(self, lidar_voxels: torch.Tensor) -> torch.Tensor:
        """
        Encode LiDAR voxels to BEV features.

        Args:
            lidar_voxels: [B, 1, X, Y, Z] or [B, X, Y, Z] sparse voxel grid

        Returns:
            BEV features [B, C, H, W]
        """
        if lidar_voxels.dim() == 4:
            lidar_voxels = lidar_voxels.unsqueeze(1)
        return self.lidar_encoder(lidar_voxels)

    def forward(
        self,
        noisy_labels: torch.Tensor,
        timesteps: torch.Tensor,
        lidar_voxels: torch.Tensor,
        prev_pred: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Forward pass for training.

        Args:
            noisy_labels: [B, H, W] noisy semantic labels
            timesteps: [B] diffusion timesteps
            lidar_voxels: [B, 1, X, Y, Z] sparse LiDAR voxels
            prev_pred: [B, H, W] previous prediction for self-conditioning

        Returns:
            logits: [B, num_classes, H, W]
            aux_logits: [B, num_classes, H, W] or None
        """
        # Encode LiDAR to BEV features
        bev_condition = self.encode_lidar(lidar_voxels)

        # Denoise
        return self.denoiser(noisy_labels, timesteps, bev_condition, prev_pred)

    def compute_loss(
        self,
        lidar_voxels: torch.Tensor,
        bev_labels: torch.Tensor,
        timesteps: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Compute training loss.

        Args:
            lidar_voxels: [B, 1, X, Y, Z] sparse LiDAR input
            bev_labels: [B, H, W] ground truth BEV semantic labels
            timesteps: [B] optional timesteps (randomly sampled if None)

        Returns:
            Dictionary with loss components
        """
        batch_size = bev_labels.shape[0]
        device = bev_labels.device

        # Sample timesteps if not provided
        if timesteps is None:
            timesteps = torch.randint(0, self.num_timesteps, (batch_size,), device=device)

        # Encode LiDAR
        bev_condition = self.encode_lidar(lidar_voxels)

        # Forward diffusion: add noise to labels
        noisy_labels = self.diffusion.q_sample(bev_labels, timesteps)

        # Self-conditioning: 50% of time use previous prediction
        prev_pred = None
        if self.use_self_conditioning and self.training and torch.rand(1).item() < 0.5:
            with torch.no_grad():
                logits, _ = self.denoiser(noisy_labels, timesteps, bev_condition, prev_pred=None)
                prev_pred = logits.argmax(dim=1)

        # Predict clean labels
        logits, aux_logits = self.denoiser(noisy_labels, timesteps, bev_condition, prev_pred)

        # Compute losses
        # Main loss: cross-entropy
        ce_loss = F.cross_entropy(
            logits.view(-1, self.num_classes),
            bev_labels.view(-1),
            weight=self.diffusion.class_weights if hasattr(self.diffusion, 'class_weights') else None,
        )

        # Weight by noise level (harder timesteps = more weight)
        if self.diffusion.transition_type == 'absorbing':
            mask_prob = self.diffusion.mask_prob[timesteps].view(batch_size, 1, 1)
            weights = mask_prob.expand_as(bev_labels).reshape(-1)
            ce_loss_weighted = (F.cross_entropy(
                logits.view(-1, self.num_classes),
                bev_labels.view(-1),
                weight=self.diffusion.class_weights if hasattr(self.diffusion, 'class_weights') else None,
                reduction='none'
            ) * weights).mean()
        else:
            ce_loss_weighted = ce_loss

        # Auxiliary loss
        aux_loss = torch.tensor(0.0, device=device)
        if aux_logits is not None:
            aux_loss = F.cross_entropy(
                aux_logits.view(-1, self.num_classes),
                bev_labels.view(-1),
                weight=self.diffusion.class_weights if hasattr(self.diffusion, 'class_weights') else None,
            )

        # Total loss
        total_loss = ce_loss_weighted + self.diffusion.aux_loss_weight * aux_loss

        return {
            'loss': total_loss,
            'ce_loss': ce_loss,
            'ce_loss_weighted': ce_loss_weighted,
            'aux_loss': aux_loss,
        }

    @torch.no_grad()
    def sample(
        self,
        lidar_voxels: torch.Tensor,
        num_steps: int | None = None,
        temperature: float = 1.0,
        use_self_conditioning: bool = True,
        guidance_scale: float = 1.0,
    ) -> torch.Tensor:
        """
        Generate BEV semantic map from LiDAR input.

        Args:
            lidar_voxels: [B, 1, X, Y, Z] sparse LiDAR input
            num_steps: Number of denoising steps (default: num_timesteps)
            temperature: Sampling temperature
            use_self_conditioning: Whether to use self-conditioning
            guidance_scale: Classifier-free guidance scale (1.0 = no guidance)

        Returns:
            BEV semantic labels [B, H, W]
        """
        if num_steps is None:
            num_steps = self.num_timesteps

        # Encode LiDAR
        bev_condition = self.encode_lidar(lidar_voxels)

        batch_size = bev_condition.shape[0]
        H, W = bev_condition.shape[2], bev_condition.shape[3]
        device = bev_condition.device

        # Start from fully masked (absorbing) or random (uniform)
        if self.diffusion.transition_type == 'absorbing':
            x_t = torch.full((batch_size, H, W), self.diffusion.mask_token, device=device, dtype=torch.long)
        else:
            x_t = torch.randint(0, self.num_classes, (batch_size, H, W), device=device)

        # Timesteps for sampling
        if num_steps < self.num_timesteps:
            timesteps = torch.linspace(
                self.num_timesteps - 1, 0, num_steps, device=device
            ).long()
        else:
            timesteps = torch.arange(self.num_timesteps - 1, -1, -1, device=device)

        prev_pred = None

        for t_idx, t in enumerate(timesteps):
            t_batch = t.expand(batch_size)

            # Model prediction
            logits, _ = self.denoiser(x_t, t_batch, bev_condition, prev_pred, return_aux=False)

            # Classifier-free guidance (if scale > 1)
            if guidance_scale > 1.0:
                # Unconditional prediction (zero condition)
                uncond = torch.zeros_like(bev_condition)
                logits_uncond, _ = self.denoiser(x_t, t_batch, uncond, prev_pred, return_aux=False)
                logits = logits_uncond + guidance_scale * (logits - logits_uncond)

            # Apply temperature
            logits = logits / temperature

            # Sample from predicted distribution
            probs = F.softmax(logits, dim=1)
            gumbel = -torch.log(-torch.log(torch.rand_like(probs) + 1e-8) + 1e-8)
            pred = (torch.log(probs + 1e-8) + gumbel).argmax(dim=1)

            # For absorbing: only update masked positions
            if self.diffusion.transition_type == 'absorbing':
                if t_idx < len(timesteps) - 1:
                    t_next = timesteps[t_idx + 1]
                    unmask_prob = self.diffusion.mask_prob[t] - self.diffusion.mask_prob[t_next]
                else:
                    unmask_prob = self.diffusion.mask_prob[t]

                unmask_mask = (x_t == self.diffusion.mask_token) & (torch.rand_like(x_t.float()) < unmask_prob)
                x_t = torch.where(unmask_mask, pred, x_t)
            else:
                x_t = pred

            # Self-conditioning
            if use_self_conditioning:
                prev_pred = pred

        # Final prediction for any remaining masked tokens
        if self.diffusion.transition_type == 'absorbing':
            mask = x_t == self.diffusion.mask_token
            if mask.any():
                logits, _ = self.denoiser(
                    x_t,
                    torch.zeros(batch_size, device=device, dtype=torch.long),
                    bev_condition,
                    prev_pred,
                    return_aux=False
                )
                final_pred = logits.argmax(dim=1)
                x_t = torch.where(mask, final_pred, x_t)

        return x_t

    def get_num_params(self) -> dict[str, int]:
        """Get parameter counts for each component."""
        return {
            'lidar_encoder': sum(p.numel() for p in self.lidar_encoder.parameters()),
            'denoiser': sum(p.numel() for p in self.denoiser.parameters()),
            'total': sum(p.numel() for p in self.parameters()),
        }


def create_bev_diffusion_model(
    config: str = 'base',
    num_classes: int = 20,
    class_weights: torch.Tensor | None = None,
    **kwargs
) -> BEVDiffusionModel:
    """
    Factory function to create BEV diffusion models with different configurations.

    Args:
        config: Model configuration ('lightweight', 'base', 'large')
        num_classes: Number of semantic classes
        class_weights: Optional class weights
        **kwargs: Additional arguments passed to model

    Returns:
        BEVDiffusionModel instance
    """
    configs = {
        'lightweight': {
            'num_timesteps': 500,
            'lidar_channels': 64,
            'unet_base_channels': 64,
            'lightweight': True,
        },
        'base': {
            'num_timesteps': 1000,
            'lidar_channels': 128,
            'unet_base_channels': 128,
            'lightweight': False,
        },
        'large': {
            'num_timesteps': 1000,
            'lidar_channels': 256,
            'unet_base_channels': 256,
            'lightweight': False,
        },
    }

    model_config = configs.get(config, configs['base'])
    model_config.update(kwargs)

    return BEVDiffusionModel(
        num_classes=num_classes,
        class_weights=class_weights,
        **model_config
    )
