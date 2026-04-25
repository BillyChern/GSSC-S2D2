"""
BEV Diffusion Model V2 with All Phase 3 Enhancements

This is the FIXED version of BEV diffusion applying Phase 3's successful techniques:
1. KL posterior loss (not CE-based ELBO)
2. Stronger noise schedule (beta_max=0.1)
3. Focal loss (gamma=2.0)
4. Class-balanced weights (beta=0.999)
5. Occupied pixel weighting
6. Observation weighting
7. Lovász loss
8. Proper mIoU evaluation

Combines:
- SparseLiDAREncoder: 3D sparse LiDAR → 2D BEV features
- ModularBEVUNet: 2D diffusion UNet with time conditioning
- MultinomialDiffusion2D: Proper KL-based diffusion

Reference: Phase 3 scene_completion achieved 49.01% mIoU using these techniques.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict

from gssc.models.bev_multinomial_diffusion_2d import MultinomialDiffusion2D
from gssc.models.bev_lidar_encoder import SparseLiDAREncoder
from gssc.models.bev_unet_v2 import ModularBEVUNet, get_timestep_embedding
from gssc.models.bev_conditioning import ConditioningType, MultiScaleConditioningFusion


class BEVDiffusionUNetWrapper(nn.Module):
    """
    Wrapper around ModularBEVUNet for multinomial diffusion.

    Converts between:
    - Diffusion interface: x_t one-hot [B, K, H, W], output logits [B, K, H, W]
    - ModularBEVUNet interface: x class indices [B, H, W], output logits [B, K, H, W]
    """

    def __init__(
        self,
        num_classes: int = 20,
        input_resolution: int = 256,
        base_channels: int = 64,
        cond_channels: int = 64,
        conditioning_type: ConditioningType = ConditioningType.SUM,
        **kwargs,
    ):
        super().__init__()

        self.num_classes = num_classes

        # Use the tested ModularBEVUNet
        self.unet = ModularBEVUNet(
            num_classes=num_classes,
            input_resolution=input_resolution,
            base_channels=base_channels,
            cond_channels=cond_channels,
            use_self_conditioning=False,  # We handle conditioning differently
            conditioning_type=conditioning_type,
        )

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        lidar_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x_t: Noisy BEV (one-hot) [B, K, H, W]
            t: Timesteps [B]
            lidar_features: LiDAR conditioning [B, C, H, W]

        Returns:
            x_0 logits [B, K, H, W]
        """
        # Convert one-hot to class indices for ModularBEVUNet
        x_t_indices = x_t.argmax(dim=1)  # [B, H, W]

        # ModularBEVUNet forward
        logits, _ = self.unet(x_t_indices, t, lidar_features, prev_pred=None, return_aux=False)

        return logits


# Keep simple helper modules for potential future use
class ResBlockWithTime(nn.Module):
    """Residual block with time embedding."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.norm1 = nn.GroupNorm(min(8, in_channels), in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.time_proj = nn.Linear(time_emb_dim, out_channels)
        self.norm2 = nn.GroupNorm(min(8, out_channels), out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)

        # Add time embedding
        h = h + self.time_proj(t_emb)[:, :, None, None]

        h = self.norm2(h)
        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)

        return self.skip(x) + h


class SelfAttention(nn.Module):
    """Self-attention for spatial features."""

    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.norm = nn.GroupNorm(min(8, channels), channels)
        self.attn = nn.MultiheadAttention(channels, num_heads, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h = self.norm(x)
        h = h.view(B, C, H * W).permute(0, 2, 1)  # [B, H*W, C]
        h, _ = self.attn(h, h, h)
        h = h.permute(0, 2, 1).view(B, C, H, W)
        return x + h


class BEVDiffusionV2(nn.Module):
    """
    Complete BEV Diffusion Model V2 with all Phase 3 enhancements.

    Pipeline:
    1. SparseLiDAREncoder: 3D voxels → 2D BEV features
    2. MultinomialDiffusion2D: Proper KL-based diffusion
    3. BEVDiffusionUNet: Denoise x_t → x_0

    Args:
        num_classes: Number of semantic classes
        num_timesteps: Diffusion timesteps
        beta_max: Maximum noise level (0.1 for strong noise)
        focal_gamma: Focal loss gamma
        lovasz_weight: Lovász loss weight
        lidar_out_channels: LiDAR encoder output channels
        unet_base_channels: UNet base channels
    """

    def __init__(
        self,
        num_classes: int = 20,
        num_timesteps: int = 100,
        beta_min: float = 0.0001,
        beta_max: float = 0.1,
        focal_gamma: float = 2.0,
        class_0_weight: float = 0.02,
        occupied_weight: float = 10.0,
        lovasz_weight: float = 0.3,
        obs_weight_factor: float = 2.0,
        lidar_in_channels: int = 1,
        lidar_out_channels: int = 64,
        unet_base_channels: int = 64,
        input_shape: Tuple[int, int, int] = (256, 256, 32),
        # New: conditioning options
        conditioning_type: str = "sum",  # "sum", "gated_sum", "gated_film"
        use_multiscale_fusion: bool = False,
        fusion_type: str = "bifpn",  # "weighted", "bifpn", "concat", "attention"
    ):
        super().__init__()

        self.num_classes = num_classes
        self.num_timesteps = num_timesteps
        self.input_shape = input_shape
        self.use_multiscale_fusion = use_multiscale_fusion

        # Parse conditioning type
        cond_type = ConditioningType(conditioning_type)

        # 1. LiDAR Encoder: 3D sparse → 2D BEV features
        self.lidar_encoder = SparseLiDAREncoder(
            in_channels=lidar_in_channels,
            base_channels=32,
            out_channels=lidar_out_channels,
            height_pool='max',
            input_shape=input_shape,
        )

        # 2. Optional: Multi-scale conditioning fusion
        if use_multiscale_fusion:
            # Channels from forward_multiscale(): scale_256=64, scale_64=128, scale_32=256
            # These are based on base_channels=32 → 32*2=64, 32*4=128, 32*8=256
            self.multiscale_fusion = MultiScaleConditioningFusion(
                in_channels_256=lidar_out_channels,  # 64
                in_channels_64=32 * 4,   # 128
                in_channels_32=32 * 8,   # 256
                out_channels=lidar_out_channels,
                fusion_type=fusion_type,
            )
            print(f"  - MultiScale fusion: {fusion_type}")

        # 3. Diffusion process with all enhancements
        self.diffusion = MultinomialDiffusion2D(
            num_classes=num_classes,
            num_timesteps=num_timesteps,
            beta_min=beta_min,
            beta_max=beta_max,
            focal_gamma=focal_gamma,
            class_0_weight=class_0_weight,
            occupied_weight=occupied_weight,
            lovasz_weight=lovasz_weight,
            obs_weight_factor=obs_weight_factor,
        )

        # 4. Denoising UNet (uses tested ModularBEVUNet internally)
        self.unet = BEVDiffusionUNetWrapper(
            num_classes=num_classes,
            input_resolution=input_shape[0],  # 256
            base_channels=unet_base_channels,
            cond_channels=lidar_out_channels,
            conditioning_type=cond_type,
        )

        print(f"[BEVDiffusionV2] Model initialized:")
        print(f"  - LiDAR encoder: {input_shape} → {lidar_out_channels}ch BEV")
        print(f"  - Conditioning: {conditioning_type}" + (f" + multiscale({fusion_type})" if use_multiscale_fusion else ""))
        print(f"  - Diffusion: T={num_timesteps}, beta_max={beta_max}")
        print(f"  - UNet: base={unet_base_channels}ch")

    def forward(
        self,
        lidar_voxels: torch.Tensor,
        bev_gt: torch.Tensor,
        t: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Training forward pass.

        Args:
            lidar_voxels: 3D LiDAR voxels [B, H, W, D] or sparse format
            bev_gt: Ground truth BEV (class indices) [B, H, W]
            t: Optional timesteps [B]

        Returns:
            Dictionary with loss and metrics
        """
        B = bev_gt.shape[0]
        device = bev_gt.device

        # 1. Create observation mask from LiDAR BEFORE unsqueeze (for obs weighting)
        # LiDAR-observed pixels get higher weight
        if lidar_voxels.dim() == 4:  # [B, H, W, D]
            lidar_obs = lidar_voxels.any(dim=-1).float()  # [B, H, W]
        elif lidar_voxels.dim() == 5:  # [B, 1, H, W, D]
            lidar_obs = lidar_voxels[:, 0].any(dim=-1).float()  # [B, H, W]
        else:
            lidar_obs = None

        # 2. Encode LiDAR to BEV features
        # LiDAR encoder expects [B, 1, X, Y, Z] but input is [B, H, W, D]
        if lidar_voxels.dim() == 4:
            lidar_voxels = lidar_voxels.unsqueeze(1)  # [B, H, W, D] -> [B, 1, H, W, D]

        if self.use_multiscale_fusion:
            # Get multi-scale features and fuse them
            multiscale_features = self.lidar_encoder.forward_multiscale(lidar_voxels)
            lidar_features = self.multiscale_fusion(multiscale_features, target_size=(self.input_shape[0], self.input_shape[1]))
        else:
            lidar_features = self.lidar_encoder(lidar_voxels)  # [B, C, H, W]

        # 3. Sample timesteps if not provided
        if t is None:
            t = torch.randint(0, self.num_timesteps, (B,), device=device)

        # 4. Compute diffusion loss using the UNet as model
        # The UNet predicts x_0 given x_t, t, and lidar_features
        losses = self.diffusion.training_losses(
            model=self._unet_wrapper,
            x_0=bev_gt,
            t=t,
            lidar_features=lidar_features,
            lidar_obs=lidar_obs,
        )

        return losses

    def _unet_wrapper(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        lidar_features: torch.Tensor,
    ) -> torch.Tensor:
        """Wrapper to pass LiDAR features to UNet."""
        return self.unet(x_t, t, lidar_features)

    @torch.no_grad()
    def sample(
        self,
        lidar_voxels: torch.Tensor,
        show_progress: bool = True,
        soft: bool = True,
        num_steps: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Generate BEV prediction via reverse diffusion.

        Args:
            lidar_voxels: 3D LiDAR voxels
            show_progress: Show progress bar
            soft: If True (default), use soft probability updates instead of hard
                  multinomial sampling. This preserves information and significantly
                  improves performance (25% vs 3% mIoU).
            num_steps: Number of denoising steps. If None, use all timesteps.
                      For soft sampling, 10 steps is usually sufficient.

        Returns:
            Predicted BEV (class indices) [B, H, W]
        """
        device = lidar_voxels.device

        # Encode LiDAR
        # LiDAR encoder expects [B, 1, X, Y, Z] but input is [B, H, W, D]
        if lidar_voxels.dim() == 4:
            lidar_voxels = lidar_voxels.unsqueeze(1)

        # NOTE: Training validation uses simple encoder forward, NOT multiscale fusion.
        # Even if use_multiscale_fusion is True, we use the simple forward for consistency.
        # The multiscale fusion path was defined but not used during training validation.
        lidar_features = self.lidar_encoder(lidar_voxels)  # [B, C, H, W]
        B, C, H, W = lidar_features.shape

        # Sample via diffusion
        bev_pred = self.diffusion.sample(
            model=self._unet_wrapper,
            lidar_features=lidar_features,
            shape=(B, H, W),
            device=device,
            show_progress=show_progress,
            soft=soft,
            num_steps=num_steps,
        )

        return bev_pred

    def get_num_params(self) -> int:
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters())


def create_bev_diffusion_v2(
    resolution: int = 256,
    model_size: str = 'base',
    **kwargs,
) -> BEVDiffusionV2:
    """
    Factory function to create BEV Diffusion V2 model.

    Args:
        resolution: Input/output resolution (256)
        model_size: 'base' or 'large'
        **kwargs: Additional arguments

    Returns:
        BEVDiffusionV2 model
    """
    if model_size == 'base':
        unet_base = 64
        lidar_out = 64
    elif model_size == 'large':
        unet_base = 96
        lidar_out = 96
    else:
        raise ValueError(f"Unknown model_size: {model_size}")

    # Merge defaults with kwargs (kwargs takes precedence)
    defaults = {
        'num_classes': 20,
        'num_timesteps': 100,
        'beta_min': 0.0001,
        'beta_max': 0.1,  # Strong noise!
        'focal_gamma': 2.0,
        'class_0_weight': 0.02,
        'occupied_weight': 10.0,
        'lovasz_weight': 0.3,
        'obs_weight_factor': 2.0,
        'lidar_out_channels': lidar_out,
        'unet_base_channels': unet_base,
        'input_shape': (resolution, resolution, 32),
        # New: conditioning options
        'conditioning_type': 'sum',
        'use_multiscale_fusion': False,
        'fusion_type': 'bifpn',
    }
    # Override defaults with any provided kwargs
    defaults.update(kwargs)

    return BEVDiffusionV2(**defaults)


if __name__ == "__main__":
    # Test
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = create_bev_diffusion_v2(resolution=256, model_size='base').to(device)
    print(f"Total parameters: {model.get_num_params():,}")

    # Test forward
    B = 2
    lidar = torch.randint(0, 2, (B, 256, 256, 32), device=device).float()
    bev_gt = torch.randint(0, 20, (B, 256, 256), device=device)

    losses = model(lidar, bev_gt)
    print(f"Loss: {losses['loss'].item():.4f}")
    print(f"KL loss: {losses['kl_loss'].item():.4f}")
    print(f"Lovász loss: {losses['lovasz_loss'].item():.4f}")
    print(f"Accuracy: {losses['accuracy'].item():.4f}")

    # Test sampling
    bev_pred = model.sample(lidar, show_progress=False)
    print(f"Predicted BEV shape: {bev_pred.shape}")
