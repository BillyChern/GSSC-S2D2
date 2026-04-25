"""
3D U-Net for Scene Completion with Multinomial Diffusion

Based on the architecture from Figure 4-4, adapted for SemanticKITTI resolution.

Key differences from CarlaSC (128×128×8):
- SemanticKITTI uses 256×256×32 voxels
- Need more downsampling levels (4 instead of 3)
- Larger memory footprint

Architecture:
- 3D U-Net with residual blocks
- BEV conditioning: expanded along Z axis and embedded
- LiDAR conditioning: sparse binary voxels embedded
- Time conditioning: FiLM (scale + shift) after GroupNorm
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List


def timestep_embedding(timesteps: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    """
    Create sinusoidal timestep embeddings.

    Args:
        timesteps: 1D tensor of timesteps [B]
        dim: embedding dimension
        max_period: controls the minimum frequency

    Returns:
        Tensor of shape [B, dim]
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, device=timesteps.device) / half
    )
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class ResidualBlock3D(nn.Module):
    """
    3D Residual Block with FiLM time conditioning and additive BEV/LiDAR conditioning.

    Architecture (from Figure 4-4):
    1. GroupNorm → SiLU → 3D Conv
    2. + BEV embedding (via 3D conv)
    3. + LiDAR embedding (via 3D conv)
    4. GroupNorm → FiLM(t) → SiLU → 3D Conv
    5. Residual connection
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_dim: int,
        cond_channels: int,
        num_groups: int = 8,
    ):
        super().__init__()

        # First half
        self.norm1 = nn.GroupNorm(min(num_groups, in_channels), in_channels)
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)

        # Conditioning projections
        self.bev_proj = nn.Conv3d(cond_channels, out_channels, kernel_size=3, padding=1)
        self.lidar_proj = nn.Conv3d(cond_channels, out_channels, kernel_size=3, padding=1)

        # Second half with FiLM
        self.norm2 = nn.GroupNorm(min(num_groups, out_channels), out_channels)
        self.time_fc = nn.Linear(time_emb_dim, out_channels * 2)  # w and b for FiLM
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)

        # Skip connection
        if in_channels != out_channels:
            self.skip = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        else:
            self.skip = nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        t_emb: torch.Tensor,
        bev_emb: torch.Tensor,
        lidar_emb: torch.Tensor,
    ) -> torch.Tensor:
        h = x

        # First conv
        h = self.norm1(h)
        h = F.silu(h)
        h = self.conv1(h)

        # Add conditioning
        h = h + self.bev_proj(bev_emb) + self.lidar_proj(lidar_emb)

        # Second conv with FiLM
        h = self.norm2(h)

        # FiLM: w * h + b
        wb = self.time_fc(t_emb)  # [B, out_channels * 2]
        w, b = wb.chunk(2, dim=-1)  # [B, out_channels] each
        w = w[:, :, None, None, None]  # [B, C, 1, 1, 1]
        b = b[:, :, None, None, None]
        h = w * h + b

        h = F.silu(h)
        h = self.conv2(h)

        # Residual
        return h + self.skip(x)


class SceneCompletionUNet(nn.Module):
    """
    3D U-Net for Scene Completion on SemanticKITTI (256×256×32).

    Takes noisy voxel grid x_t and predicts clean x_0.
    Conditioned on:
    - BEV semantic map (expanded along Z)
    - Sparse LiDAR binary voxels
    - Timestep t

    Architecture for 256×256×32:
    - Level 0: 256×256×32, 32 channels
    - Level 1: 128×128×16, 64 channels (down by 2)
    - Level 2: 64×64×8, 128 channels (down by 2)
    - Level 3: 32×32×4, 256 channels (down by 2)
    - Bottleneck: 16×16×2, 256 channels (down by 2)

    Training config (from paper, scaled for SemanticKITTI):
    - Batch size: 4 (reduced due to larger resolution)
    - Learning rate: 1e-4
    - Diffusion timesteps: 100
    - EMA decay: 0.9999
    """

    def __init__(
        self,
        num_classes: int = 20,
        base_channels: int = 32,
        channel_mult: Tuple[int, ...] = (1, 2, 4, 8),  # 4 levels for 256→16
        time_emb_dim: int = 128,
        dropout: float = 0.0,
        num_groups: int = 8,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.base_channels = base_channels

        # Channel progression: 32 → 64 → 128 → 256
        channels = [base_channels * m for m in channel_mult]

        # Shared embedding for voxels and BEV (from notes: "BEV and voxels share embedding layer")
        self.voxel_embed = nn.Conv3d(num_classes, base_channels, kernel_size=3, padding=1)

        # LiDAR embedding: 1 → base_channels
        self.lidar_embed = nn.Conv3d(1, base_channels, kernel_size=3, padding=1)

        # Time embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(base_channels, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        # ============ Encoder ============
        # Level 0: 256×256×32
        self.enc0 = ResidualBlock3D(channels[0], channels[0], time_emb_dim, base_channels, num_groups)
        self.down0 = nn.Conv3d(channels[0], channels[1], kernel_size=3, stride=2, padding=1)

        # Level 1: 128×128×16
        self.enc1 = ResidualBlock3D(channels[1], channels[1], time_emb_dim, base_channels, num_groups)
        self.down1 = nn.Conv3d(channels[1], channels[2], kernel_size=3, stride=2, padding=1)

        # Level 2: 64×64×8
        self.enc2 = ResidualBlock3D(channels[2], channels[2], time_emb_dim, base_channels, num_groups)
        self.down2 = nn.Conv3d(channels[2], channels[3], kernel_size=3, stride=2, padding=1)

        # Level 3: 32×32×4
        self.enc3 = ResidualBlock3D(channels[3], channels[3], time_emb_dim, base_channels, num_groups)
        self.down3 = nn.Conv3d(channels[3], channels[3], kernel_size=3, stride=2, padding=1)

        # ============ Bottleneck ============
        # 16×16×2
        self.mid0 = ResidualBlock3D(channels[3], channels[3], time_emb_dim, base_channels, num_groups)
        self.mid1 = ResidualBlock3D(channels[3], channels[3], time_emb_dim, base_channels, num_groups)

        # Optional: PaSCo's SPCDense3Dv2 for dense hallucination at bottleneck
        # This helps complete occluded regions that sparse convs can't reach
        self.use_dense3d_bottleneck = False
        self.dense3d_bottleneck = None

        # S1/S2: Optional lifted features embedding (from 2D→3D lifting)
        # Will be initialized when enable_lifted_features() is called
        self.use_lifted_features = False
        self.lifted_embed = None

        # ============ Decoder ============
        # Level 3: 32×32×4
        self.up3 = nn.ConvTranspose3d(channels[3], channels[3], kernel_size=4, stride=2, padding=1)
        self.dec3 = ResidualBlock3D(channels[3] * 2, channels[3], time_emb_dim, base_channels, num_groups)

        # Level 2: 64×64×8
        self.up2 = nn.ConvTranspose3d(channels[3], channels[2], kernel_size=4, stride=2, padding=1)
        self.dec2 = ResidualBlock3D(channels[2] * 2, channels[2], time_emb_dim, base_channels, num_groups)

        # Level 1: 128×128×16
        self.up1 = nn.ConvTranspose3d(channels[2], channels[1], kernel_size=4, stride=2, padding=1)
        self.dec1 = ResidualBlock3D(channels[1] * 2, channels[1], time_emb_dim, base_channels, num_groups)

        # Level 0: 256×256×32
        self.up0 = nn.ConvTranspose3d(channels[1], channels[0], kernel_size=4, stride=2, padding=1)
        self.dec0 = ResidualBlock3D(channels[0] * 2, channels[0], time_emb_dim, base_channels, num_groups)

        # Output
        self.out_conv = nn.Sequential(
            nn.GroupNorm(num_groups, channels[0]),
            nn.SiLU(),
            nn.Conv3d(channels[0], num_classes, kernel_size=3, padding=1),
        )

    def enable_dense3d_bottleneck(self, dropout: float = 0.1):
        """
        Enable PaSCo's SPCDense3Dv2 at the bottleneck for dense hallucination.

        This helps complete occluded regions that sparse convolutions can't reach.
        Reference: PaSCo (CVPR 2024) - <paco-reference>/pasco/models/layers.py

        Args:
            dropout: Dropout probability for dense3d (PaSCo default: 0.1)
        """
        from gssc.models.dense_3d_cnn import SPCDense3Dv2

        bottleneck_channels = self.base_channels * 8  # 256 for base=32

        self.dense3d_bottleneck = nn.Sequential(
            SPCDense3Dv2(init_size=bottleneck_channels),
            nn.Dropout3d(dropout),
        )
        self.use_dense3d_bottleneck = True

        # Move to same device as model
        device = next(self.parameters()).device
        self.dense3d_bottleneck = self.dense3d_bottleneck.to(device)

        print(f"[SceneCompletionUNet] Enabled SPCDense3Dv2 bottleneck (channels={bottleneck_channels}, dropout={dropout})")

    def enable_lifted_features(self, feature_dim: int = 64):
        """
        Enable S1/S2 lifted features conditioning.

        This allows the model to receive additional 3D features from the 2D→3D
        lifting module (S2) and apply CFG dropout (S1) on these features.

        Reference: S3CNet (CoRL 2020) - 2D→3D Lifting section

        Args:
            feature_dim: Dimension of lifted features (from LiftingModule)
        """
        # Project lifted features to base_channels for addition to embeddings
        self.lifted_embed = nn.Conv3d(feature_dim, self.base_channels, kernel_size=1)
        self.use_lifted_features = True

        # Move to same device as model
        device = next(self.parameters()).device
        self.lifted_embed = self.lifted_embed.to(device)

        print(f"[SceneCompletionUNet] Enabled lifted features conditioning (feature_dim={feature_dim})")

    def forward(
        self,
        x_t: torch.Tensor,  # [B, num_classes, H, W, D] one-hot noisy voxels
        t: torch.Tensor,    # [B] timesteps
        bev: torch.Tensor,  # [B, num_classes, H, W] BEV semantic map (one-hot)
        lidar: torch.Tensor,  # [B, 1, H, W, D] sparse LiDAR binary voxels
        lifted_features: Optional[torch.Tensor] = None,  # S1/S2: [B, feat_dim, H, W, D] lifted 3D features
    ) -> torch.Tensor:
        """
        Forward pass: predict clean voxels x_0 from noisy x_t.

        Args:
            x_t: Noisy voxel grid, one-hot encoded [B, num_classes, H, W, D]
            t: Timesteps [B]
            bev: BEV semantic map, one-hot encoded [B, num_classes, H, W]
            lidar: Sparse LiDAR binary voxels [B, 1, H, W, D]
            lifted_features: Optional S1/S2 lifted 3D features from 2D→3D lifting
                            [B, feature_dim, H, W, D]. Used for CFG (S1) - can be
                            zeroed during training for classifier-free guidance.

        Returns:
            x_0_pred: Predicted logits [B, num_classes, H, W, D]
        """
        B, C, H, W, D = x_t.shape

        # Expand BEV along Z: [B, C, H, W] → [B, C, H, W, D]
        bev_3d = bev.unsqueeze(-1).expand(-1, -1, -1, -1, D)

        # Embed inputs (shared embedding for voxels and BEV)
        x = self.voxel_embed(x_t)  # [B, 32, H, W, D]
        bev_emb = self.voxel_embed(bev_3d)  # [B, 32, H, W, D] - shared embedding!
        lidar_emb = self.lidar_embed(lidar)  # [B, 32, H, W, D]

        # S1/S2: Add lifted features conditioning if enabled and provided
        # This enables CFG - when lifted_features is zeros, model learns unconditional
        if self.use_lifted_features and lifted_features is not None and self.lifted_embed is not None:
            lifted_emb = self.lifted_embed(lifted_features)  # [B, 32, H, W, D]
            # Add to the main signal (similar to how bev_emb and lidar_emb are added)
            x = x + lifted_emb

        # Time embedding
        t_emb = timestep_embedding(t, self.base_channels)
        t_emb = self.time_mlp(t_emb)  # [B, time_emb_dim]

        # ============ Encoder ============
        # Level 0: 256×256×32
        e0 = self.enc0(x, t_emb, bev_emb, lidar_emb)
        x = self.down0(e0)
        bev_emb = F.avg_pool3d(bev_emb, 2)
        lidar_emb = F.max_pool3d(lidar_emb, 2)

        # Level 1: 128×128×16
        e1 = self.enc1(x, t_emb, bev_emb, lidar_emb)
        x = self.down1(e1)
        bev_emb = F.avg_pool3d(bev_emb, 2)
        lidar_emb = F.max_pool3d(lidar_emb, 2)

        # Level 2: 64×64×8
        e2 = self.enc2(x, t_emb, bev_emb, lidar_emb)
        x = self.down2(e2)
        bev_emb = F.avg_pool3d(bev_emb, 2)
        lidar_emb = F.max_pool3d(lidar_emb, 2)

        # Level 3: 32×32×4
        e3 = self.enc3(x, t_emb, bev_emb, lidar_emb)
        x = self.down3(e3)
        bev_emb = F.avg_pool3d(bev_emb, 2)
        lidar_emb = F.max_pool3d(lidar_emb, 2)

        # ============ Bottleneck ============
        # 16×16×2
        x = self.mid0(x, t_emb, bev_emb, lidar_emb)
        x = self.mid1(x, t_emb, bev_emb, lidar_emb)

        # Optional: PaSCo's SPCDense3Dv2 for dense hallucination
        # This allows completing occluded regions that sparse convs can't reach
        if self.use_dense3d_bottleneck and self.dense3d_bottleneck is not None:
            x = self.dense3d_bottleneck(x)

        # ============ Decoder ============
        # Upsample conditioning back
        bev_emb = F.interpolate(bev_emb, scale_factor=2, mode='trilinear', align_corners=False)
        lidar_emb = F.interpolate(lidar_emb, scale_factor=2, mode='nearest')

        # Level 3: 32×32×4
        x = self.up3(x)
        x = torch.cat([x, e3], dim=1)
        x = self.dec3(x, t_emb, bev_emb, lidar_emb)

        bev_emb = F.interpolate(bev_emb, scale_factor=2, mode='trilinear', align_corners=False)
        lidar_emb = F.interpolate(lidar_emb, scale_factor=2, mode='nearest')

        # Level 2: 64×64×8
        x = self.up2(x)
        x = torch.cat([x, e2], dim=1)
        x = self.dec2(x, t_emb, bev_emb, lidar_emb)

        bev_emb = F.interpolate(bev_emb, scale_factor=2, mode='trilinear', align_corners=False)
        lidar_emb = F.interpolate(lidar_emb, scale_factor=2, mode='nearest')

        # Level 1: 128×128×16
        x = self.up1(x)
        x = torch.cat([x, e1], dim=1)
        x = self.dec1(x, t_emb, bev_emb, lidar_emb)

        bev_emb = F.interpolate(bev_emb, scale_factor=2, mode='trilinear', align_corners=False)
        lidar_emb = F.interpolate(lidar_emb, scale_factor=2, mode='nearest')

        # Level 0: 256×256×32
        x = self.up0(x)
        x = torch.cat([x, e0], dim=1)
        x = self.dec0(x, t_emb, bev_emb, lidar_emb)

        # Output
        x = self.out_conv(x)

        return x

    def forward_features(
        self,
        x_t: torch.Tensor,  # [B, num_classes, H, W, D] one-hot noisy voxels
        t: torch.Tensor,    # [B] timesteps
        bev: torch.Tensor,  # [B, num_classes, H, W] BEV semantic map (one-hot)
        lidar: torch.Tensor,  # [B, 1, H, W, D] sparse LiDAR binary voxels
        lifted_features: Optional[torch.Tensor] = None,  # S1/S2: [B, feat_dim, H, W, D] lifted 3D features
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass that returns both predictions and bottleneck features.
        Used for DSKD (Dense-to-Sparse Knowledge Distillation).

        Args:
            x_t: Noisy voxel grid, one-hot encoded [B, num_classes, H, W, D]
            t: Timesteps [B]
            bev: BEV semantic map, one-hot encoded [B, num_classes, H, W]
            lidar: Sparse LiDAR binary voxels [B, 1, H, W, D]
            lifted_features: Optional S1/S2 lifted 3D features [B, feature_dim, H, W, D]

        Returns:
            x_0_pred: Predicted logits [B, num_classes, H, W, D]
            bottleneck_features: Features at bottleneck [B, C, 16, 16, 2]
        """
        B, C, H, W, D = x_t.shape

        # Expand BEV along Z: [B, C, H, W] → [B, C, H, W, D]
        bev_3d = bev.unsqueeze(-1).expand(-1, -1, -1, -1, D)

        # Embed inputs (shared embedding for voxels and BEV)
        x = self.voxel_embed(x_t)  # [B, 32, H, W, D]
        bev_emb = self.voxel_embed(bev_3d)  # [B, 32, H, W, D] - shared embedding!
        lidar_emb = self.lidar_embed(lidar)  # [B, 32, H, W, D]

        # S1/S2: Add lifted features conditioning if enabled and provided
        if self.use_lifted_features and lifted_features is not None and self.lifted_embed is not None:
            lifted_emb = self.lifted_embed(lifted_features)  # [B, 32, H, W, D]
            x = x + lifted_emb

        # Time embedding
        t_emb = timestep_embedding(t, self.base_channels)
        t_emb = self.time_mlp(t_emb)  # [B, time_emb_dim]

        # ============ Encoder ============
        e0 = self.enc0(x, t_emb, bev_emb, lidar_emb)
        x = self.down0(e0)
        bev_emb = F.avg_pool3d(bev_emb, 2)
        lidar_emb = F.max_pool3d(lidar_emb, 2)

        e1 = self.enc1(x, t_emb, bev_emb, lidar_emb)
        x = self.down1(e1)
        bev_emb = F.avg_pool3d(bev_emb, 2)
        lidar_emb = F.max_pool3d(lidar_emb, 2)

        e2 = self.enc2(x, t_emb, bev_emb, lidar_emb)
        x = self.down2(e2)
        bev_emb = F.avg_pool3d(bev_emb, 2)
        lidar_emb = F.max_pool3d(lidar_emb, 2)

        e3 = self.enc3(x, t_emb, bev_emb, lidar_emb)
        x = self.down3(e3)
        bev_emb = F.avg_pool3d(bev_emb, 2)
        lidar_emb = F.max_pool3d(lidar_emb, 2)

        # ============ Bottleneck ============
        x = self.mid0(x, t_emb, bev_emb, lidar_emb)
        x = self.mid1(x, t_emb, bev_emb, lidar_emb)

        # Save bottleneck features for DSKD (before dense3d transform)
        bottleneck_features = x.clone()  # [B, 256, 16, 16, 2]

        # Optional: PaSCo's SPCDense3Dv2 for dense hallucination
        if self.use_dense3d_bottleneck and self.dense3d_bottleneck is not None:
            x = self.dense3d_bottleneck(x)

        # ============ Decoder ============
        bev_emb = F.interpolate(bev_emb, scale_factor=2, mode='trilinear', align_corners=False)
        lidar_emb = F.interpolate(lidar_emb, scale_factor=2, mode='nearest')

        x = self.up3(x)
        x = torch.cat([x, e3], dim=1)
        x = self.dec3(x, t_emb, bev_emb, lidar_emb)

        bev_emb = F.interpolate(bev_emb, scale_factor=2, mode='trilinear', align_corners=False)
        lidar_emb = F.interpolate(lidar_emb, scale_factor=2, mode='nearest')

        x = self.up2(x)
        x = torch.cat([x, e2], dim=1)
        x = self.dec2(x, t_emb, bev_emb, lidar_emb)

        bev_emb = F.interpolate(bev_emb, scale_factor=2, mode='trilinear', align_corners=False)
        lidar_emb = F.interpolate(lidar_emb, scale_factor=2, mode='nearest')

        x = self.up1(x)
        x = torch.cat([x, e1], dim=1)
        x = self.dec1(x, t_emb, bev_emb, lidar_emb)

        bev_emb = F.interpolate(bev_emb, scale_factor=2, mode='trilinear', align_corners=False)
        lidar_emb = F.interpolate(lidar_emb, scale_factor=2, mode='nearest')

        x = self.up0(x)
        x = torch.cat([x, e0], dim=1)
        x = self.dec0(x, t_emb, bev_emb, lidar_emb)

        # Output
        x = self.out_conv(x)

        return x, bottleneck_features


class SceneCompletionUNetLite(nn.Module):
    """
    Lighter version of 3D U-Net for faster experimentation.

    Uses fewer channels: 16 → 32 → 64 → 128 instead of 32 → 64 → 128 → 256
    Still handles 256×256×32 but with ~4x fewer parameters.
    """

    def __init__(
        self,
        num_classes: int = 20,
        base_channels: int = 16,  # Half of full model
        channel_mult: Tuple[int, ...] = (1, 2, 4, 8),
        time_emb_dim: int = 64,  # Smaller time embedding
        num_groups: int = 4,  # Fewer groups
    ):
        super().__init__()

        self.num_classes = num_classes
        self.base_channels = base_channels

        channels = [base_channels * m for m in channel_mult]  # [16, 32, 64, 128]

        # Embeddings
        self.voxel_embed = nn.Conv3d(num_classes, base_channels, kernel_size=3, padding=1)
        self.lidar_embed = nn.Conv3d(1, base_channels, kernel_size=3, padding=1)

        self.time_mlp = nn.Sequential(
            nn.Linear(base_channels, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        # Encoder
        self.enc0 = ResidualBlock3D(channels[0], channels[0], time_emb_dim, base_channels, num_groups)
        self.down0 = nn.Conv3d(channels[0], channels[1], kernel_size=3, stride=2, padding=1)

        self.enc1 = ResidualBlock3D(channels[1], channels[1], time_emb_dim, base_channels, num_groups)
        self.down1 = nn.Conv3d(channels[1], channels[2], kernel_size=3, stride=2, padding=1)

        self.enc2 = ResidualBlock3D(channels[2], channels[2], time_emb_dim, base_channels, num_groups)
        self.down2 = nn.Conv3d(channels[2], channels[3], kernel_size=3, stride=2, padding=1)

        self.enc3 = ResidualBlock3D(channels[3], channels[3], time_emb_dim, base_channels, num_groups)
        self.down3 = nn.Conv3d(channels[3], channels[3], kernel_size=3, stride=2, padding=1)

        # Bottleneck
        self.mid0 = ResidualBlock3D(channels[3], channels[3], time_emb_dim, base_channels, num_groups)
        self.mid1 = ResidualBlock3D(channels[3], channels[3], time_emb_dim, base_channels, num_groups)

        # Decoder
        self.up3 = nn.ConvTranspose3d(channels[3], channels[3], kernel_size=4, stride=2, padding=1)
        self.dec3 = ResidualBlock3D(channels[3] * 2, channels[3], time_emb_dim, base_channels, num_groups)

        self.up2 = nn.ConvTranspose3d(channels[3], channels[2], kernel_size=4, stride=2, padding=1)
        self.dec2 = ResidualBlock3D(channels[2] * 2, channels[2], time_emb_dim, base_channels, num_groups)

        self.up1 = nn.ConvTranspose3d(channels[2], channels[1], kernel_size=4, stride=2, padding=1)
        self.dec1 = ResidualBlock3D(channels[1] * 2, channels[1], time_emb_dim, base_channels, num_groups)

        self.up0 = nn.ConvTranspose3d(channels[1], channels[0], kernel_size=4, stride=2, padding=1)
        self.dec0 = ResidualBlock3D(channels[0] * 2, channels[0], time_emb_dim, base_channels, num_groups)

        # Output
        self.out_conv = nn.Sequential(
            nn.GroupNorm(num_groups, channels[0]),
            nn.SiLU(),
            nn.Conv3d(channels[0], num_classes, kernel_size=3, padding=1),
        )

    def forward(self, x_t, t, bev, lidar):
        """Same interface as SceneCompletionUNet."""
        B, C, H, W, D = x_t.shape

        # Expand BEV along Z
        bev_3d = bev.unsqueeze(-1).expand(-1, -1, -1, -1, D)

        # Embed
        x = self.voxel_embed(x_t)
        bev_emb = self.voxel_embed(bev_3d)
        lidar_emb = self.lidar_embed(lidar)

        t_emb = timestep_embedding(t, self.base_channels)
        t_emb = self.time_mlp(t_emb)

        # Encoder
        e0 = self.enc0(x, t_emb, bev_emb, lidar_emb)
        x = self.down0(e0)
        bev_emb = F.avg_pool3d(bev_emb, 2)
        lidar_emb = F.max_pool3d(lidar_emb, 2)

        e1 = self.enc1(x, t_emb, bev_emb, lidar_emb)
        x = self.down1(e1)
        bev_emb = F.avg_pool3d(bev_emb, 2)
        lidar_emb = F.max_pool3d(lidar_emb, 2)

        e2 = self.enc2(x, t_emb, bev_emb, lidar_emb)
        x = self.down2(e2)
        bev_emb = F.avg_pool3d(bev_emb, 2)
        lidar_emb = F.max_pool3d(lidar_emb, 2)

        e3 = self.enc3(x, t_emb, bev_emb, lidar_emb)
        x = self.down3(e3)
        bev_emb = F.avg_pool3d(bev_emb, 2)
        lidar_emb = F.max_pool3d(lidar_emb, 2)

        # Bottleneck
        x = self.mid0(x, t_emb, bev_emb, lidar_emb)
        x = self.mid1(x, t_emb, bev_emb, lidar_emb)

        # Decoder
        bev_emb = F.interpolate(bev_emb, scale_factor=2, mode='trilinear', align_corners=False)
        lidar_emb = F.interpolate(lidar_emb, scale_factor=2, mode='nearest')
        x = self.up3(x)
        x = torch.cat([x, e3], dim=1)
        x = self.dec3(x, t_emb, bev_emb, lidar_emb)

        bev_emb = F.interpolate(bev_emb, scale_factor=2, mode='trilinear', align_corners=False)
        lidar_emb = F.interpolate(lidar_emb, scale_factor=2, mode='nearest')
        x = self.up2(x)
        x = torch.cat([x, e2], dim=1)
        x = self.dec2(x, t_emb, bev_emb, lidar_emb)

        bev_emb = F.interpolate(bev_emb, scale_factor=2, mode='trilinear', align_corners=False)
        lidar_emb = F.interpolate(lidar_emb, scale_factor=2, mode='nearest')
        x = self.up1(x)
        x = torch.cat([x, e1], dim=1)
        x = self.dec1(x, t_emb, bev_emb, lidar_emb)

        bev_emb = F.interpolate(bev_emb, scale_factor=2, mode='trilinear', align_corners=False)
        lidar_emb = F.interpolate(lidar_emb, scale_factor=2, mode='nearest')
        x = self.up0(x)
        x = torch.cat([x, e0], dim=1)
        x = self.dec0(x, t_emb, bev_emb, lidar_emb)

        return self.out_conv(x)

    def forward_features(self, x_t, t, bev, lidar):
        """Forward pass returning both predictions and bottleneck features for DSKD."""
        B, C, H, W, D = x_t.shape

        # Expand BEV along Z
        bev_3d = bev.unsqueeze(-1).expand(-1, -1, -1, -1, D)

        # Embed
        x = self.voxel_embed(x_t)
        bev_emb = self.voxel_embed(bev_3d)
        lidar_emb = self.lidar_embed(lidar)

        t_emb = timestep_embedding(t, self.base_channels)
        t_emb = self.time_mlp(t_emb)

        # Encoder
        e0 = self.enc0(x, t_emb, bev_emb, lidar_emb)
        x = self.down0(e0)
        bev_emb = F.avg_pool3d(bev_emb, 2)
        lidar_emb = F.max_pool3d(lidar_emb, 2)

        e1 = self.enc1(x, t_emb, bev_emb, lidar_emb)
        x = self.down1(e1)
        bev_emb = F.avg_pool3d(bev_emb, 2)
        lidar_emb = F.max_pool3d(lidar_emb, 2)

        e2 = self.enc2(x, t_emb, bev_emb, lidar_emb)
        x = self.down2(e2)
        bev_emb = F.avg_pool3d(bev_emb, 2)
        lidar_emb = F.max_pool3d(lidar_emb, 2)

        e3 = self.enc3(x, t_emb, bev_emb, lidar_emb)
        x = self.down3(e3)
        bev_emb = F.avg_pool3d(bev_emb, 2)
        lidar_emb = F.max_pool3d(lidar_emb, 2)

        # Bottleneck
        x = self.mid0(x, t_emb, bev_emb, lidar_emb)
        x = self.mid1(x, t_emb, bev_emb, lidar_emb)

        # Save bottleneck features for DSKD
        bottleneck_features = x.clone()

        # Decoder
        bev_emb = F.interpolate(bev_emb, scale_factor=2, mode='trilinear', align_corners=False)
        lidar_emb = F.interpolate(lidar_emb, scale_factor=2, mode='nearest')
        x = self.up3(x)
        x = torch.cat([x, e3], dim=1)
        x = self.dec3(x, t_emb, bev_emb, lidar_emb)

        bev_emb = F.interpolate(bev_emb, scale_factor=2, mode='trilinear', align_corners=False)
        lidar_emb = F.interpolate(lidar_emb, scale_factor=2, mode='nearest')
        x = self.up2(x)
        x = torch.cat([x, e2], dim=1)
        x = self.dec2(x, t_emb, bev_emb, lidar_emb)

        bev_emb = F.interpolate(bev_emb, scale_factor=2, mode='trilinear', align_corners=False)
        lidar_emb = F.interpolate(lidar_emb, scale_factor=2, mode='nearest')
        x = self.up1(x)
        x = torch.cat([x, e1], dim=1)
        x = self.dec1(x, t_emb, bev_emb, lidar_emb)

        bev_emb = F.interpolate(bev_emb, scale_factor=2, mode='trilinear', align_corners=False)
        lidar_emb = F.interpolate(lidar_emb, scale_factor=2, mode='nearest')
        x = self.up0(x)
        x = torch.cat([x, e0], dim=1)
        x = self.dec0(x, t_emb, bev_emb, lidar_emb)

        return self.out_conv(x), bottleneck_features


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test the models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing on {device}")

    # Test full model with small batch
    print("\n=== Testing SceneCompletionUNet (Full) ===")
    model = SceneCompletionUNet(
        num_classes=20,
        base_channels=32,
    ).to(device)
    print(f"Parameters: {count_parameters(model):,}")

    # Small test to check architecture
    B = 1
    x_t = torch.randn(B, 20, 256, 256, 32, device=device)
    t = torch.randint(0, 100, (B,), device=device)
    bev = torch.randn(B, 20, 256, 256, device=device)
    lidar = torch.randn(B, 1, 256, 256, 32, device=device)

    print(f"Input shape: {x_t.shape}")
    print("Running forward pass...")

    with torch.no_grad():
        try:
            out = model(x_t, t, bev, lidar)
            print(f"Output shape: {out.shape}")
            print("Full model test passed!")
        except RuntimeError as e:
            print(f"OOM expected for full resolution: {e}")

    # Test lite model
    print("\n=== Testing SceneCompletionUNetLite ===")
    model_lite = SceneCompletionUNetLite(
        num_classes=20,
        base_channels=16,
    ).to(device)
    print(f"Parameters: {count_parameters(model_lite):,}")

    with torch.no_grad():
        try:
            out = model_lite(x_t, t, bev, lidar)
            print(f"Output shape: {out.shape}")
            print("Lite model test passed!")
        except RuntimeError as e:
            print(f"Error: {e}")
