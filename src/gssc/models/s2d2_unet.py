"""
3D U-Net for Scene Completion with Sparse LiDAR Encoder (exp_2)

This variant uses spconv for efficient LiDAR feature extraction,
which is more suitable for SemanticKITTI's ~1% sparse occupancy.

Key differences from scene_unet.py (exp_1):
- LiDAR processed with SparseLiDAREncoder (spconv) instead of dense Conv3d
- Multi-scale LiDAR features injected at each U-Net level
- More memory efficient for sparse inputs

Architecture:
- SparseLiDAREncoder: extracts multi-scale features from sparse 3D voxels
- 3D U-Net: denoises voxels conditioned on BEV + multi-scale LiDAR features
"""

import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .sparse_lidar_encoder import SparseLiDAREncoder, SparseLiDAREncoderLite

logger = logging.getLogger(__name__)


def timestep_embedding(timesteps: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    """Create sinusoidal timestep embeddings."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, device=timesteps.device) / half
    )
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class ResidualBlock3DSparse(nn.Module):
    """
    3D Residual Block with FiLM time conditioning.

    BEV and LiDAR conditioning are added externally (multi-scale).
    This block only handles the voxel features and time embedding.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_dim: int,
        cond_channels: int,  # Channels for BEV/LiDAR conditioning
        num_groups: int = 8,
        no_bev: bool = False,
        lidar_film_mode: bool = False,
    ):
        super().__init__()
        self.no_bev = no_bev
        self.lidar_film_mode = lidar_film_mode

        # First conv
        self.norm1 = nn.GroupNorm(min(num_groups, in_channels), in_channels)
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)

        # Conditioning projections
        if not no_bev:
            self.bev_proj = nn.Conv3d(cond_channels, out_channels, kernel_size=3, padding=1)

        if lidar_film_mode:
            # S27: Multiplicative FiLM conditioning (DiffSSC-inspired)
            # Uses 1×1 conv (fewer params than 3×3). At init, output ≈ 0 → gate ≈ 1 (identity).
            self.lidar_film = nn.Conv3d(cond_channels, out_channels, kernel_size=1)
        else:
            # Default: Additive conditioning
            self.lidar_proj = nn.Conv3d(cond_channels, out_channels, kernel_size=3, padding=1)

        # Second conv with FiLM
        self.norm2 = nn.GroupNorm(min(num_groups, out_channels), out_channels)
        self.time_fc = nn.Linear(time_emb_dim, out_channels * 2)
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
        bev_emb: torch.Tensor | None,
        lidar_emb: torch.Tensor,
    ) -> torch.Tensor:
        h = x

        # First conv
        h = self.norm1(h)
        h = F.silu(h)
        h = self.conv1(h)

        # Add BEV conditioning (always additive)
        if not self.no_bev and bev_emb is not None:
            h = h + self.bev_proj(bev_emb)

        # LiDAR conditioning
        if self.lidar_film_mode:
            # S27: Multiplicative — h * (1 + film(lidar)). Identity at init.
            h = h * (1 + self.lidar_film(lidar_emb))
        else:
            # Default: Additive — h + proj(lidar)
            h = h + self.lidar_proj(lidar_emb)

        # Second conv with FiLM
        h = self.norm2(h)
        wb = self.time_fc(t_emb)
        w, b = wb.chunk(2, dim=-1)
        w = w[:, :, None, None, None]
        b = b[:, :, None, None, None]
        h = w * h + b

        h = F.silu(h)
        h = self.conv2(h)

        return h + self.skip(x)


class SceneCompletionUNetSparse(nn.Module):
    """
    3D U-Net with Sparse LiDAR Encoder (exp_2).

    Uses spconv to efficiently extract multi-scale LiDAR features,
    which are then used to condition each level of the U-Net.

    Architecture for SemanticKITTI (256×256×32):
    - Level 0: 256×256×32, 32 channels + LiDAR level0
    - Level 1: 128×128×16, 64 channels + LiDAR level1
    - Level 2: 64×64×8, 128 channels + LiDAR level2
    - Level 3: 32×32×4, 256 channels + LiDAR level3
    - Bottleneck: 16×16×2, 256 channels + LiDAR level4
    """

    def __init__(
        self,
        num_classes: int = 20,
        base_channels: int = 32,
        channel_mult: tuple[int, ...] = (1, 2, 4, 8),
        time_emb_dim: int = 128,
        lidar_base_channels: int = 16,
        lidar_out_channels: int = 32,
        num_groups: int = 8,
        lidar_in_channels: int = 1,
        no_bev: bool = False,
        densify_nn: bool = False,
        add_distance_gate: bool = False,
        lidar_film_mode: bool = False,
        geom_bev_channels: int = 0,
        ssc_cond_channels: int = 0,
        ssc_multiscale: bool = False,
        obs_mask_channel: bool = False,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.base_channels = base_channels
        self.obs_mask_channel = obs_mask_channel
        self.no_bev = no_bev

        channels = [base_channels * m for m in channel_mult]

        # Sparse LiDAR encoder
        self.lidar_encoder = SparseLiDAREncoder(
            in_channels=lidar_in_channels,
            base_channels=lidar_base_channels,
            out_channels=lidar_out_channels,
            densify_nn=densify_nn,
            add_distance_gate=add_distance_gate,
        )

        # Voxel and BEV embedding
        voxel_in_ch = num_classes + (1 if obs_mask_channel else 0)
        self.voxel_embed = nn.Conv3d(voxel_in_ch, base_channels, kernel_size=3, padding=1)

        # BEV embedding (also used for SCPNet 3D in no_bev mode)
        if not no_bev or ssc_cond_channels > 0:
            self.bev_embed = nn.Conv3d(num_classes, lidar_out_channels, kernel_size=3, padding=1)

        # S46: TSDF BEV geometric conditioning (injected once at input level)
        self.geom_bev_embed = None
        if geom_bev_channels > 0:
            self.geom_bev_embed = nn.Conv3d(geom_bev_channels, base_channels, kernel_size=3, padding=1)

        # SCPNet SSC prediction conditioning (for refinement, injected at input level)
        self.ssc_embed = None
        self.ssc_multiscale_embed = None
        if ssc_cond_channels > 0:
            self.ssc_embed = nn.Conv3d(ssc_cond_channels, base_channels, kernel_size=3, padding=1)
            # B1c: Multi-scale SCPNet 3D conditioning (summed with bev_emb at every level)
            if ssc_multiscale:
                self.ssc_multiscale_embed = nn.Conv3d(ssc_cond_channels, lidar_out_channels, kernel_size=3, padding=1)

        # Time embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(base_channels, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        # ============ Encoder ============
        self.enc0 = ResidualBlock3DSparse(channels[0], channels[0], time_emb_dim, lidar_out_channels, num_groups, no_bev=no_bev, lidar_film_mode=lidar_film_mode)
        self.down0 = nn.Conv3d(channels[0], channels[1], kernel_size=3, stride=2, padding=1)

        self.enc1 = ResidualBlock3DSparse(channels[1], channels[1], time_emb_dim, lidar_out_channels, num_groups, no_bev=no_bev, lidar_film_mode=lidar_film_mode)
        self.down1 = nn.Conv3d(channels[1], channels[2], kernel_size=3, stride=2, padding=1)

        self.enc2 = ResidualBlock3DSparse(channels[2], channels[2], time_emb_dim, lidar_out_channels, num_groups, no_bev=no_bev, lidar_film_mode=lidar_film_mode)
        self.down2 = nn.Conv3d(channels[2], channels[3], kernel_size=3, stride=2, padding=1)

        self.enc3 = ResidualBlock3DSparse(channels[3], channels[3], time_emb_dim, lidar_out_channels, num_groups, no_bev=no_bev, lidar_film_mode=lidar_film_mode)
        self.down3 = nn.Conv3d(channels[3], channels[3], kernel_size=3, stride=2, padding=1)

        # ============ Bottleneck ============
        self.mid0 = ResidualBlock3DSparse(channels[3], channels[3], time_emb_dim, lidar_out_channels, num_groups, no_bev=no_bev, lidar_film_mode=lidar_film_mode)
        self.mid1 = ResidualBlock3DSparse(channels[3], channels[3], time_emb_dim, lidar_out_channels, num_groups, no_bev=no_bev, lidar_film_mode=lidar_film_mode)

        # ============ Decoder ============
        self.up3 = nn.ConvTranspose3d(channels[3], channels[3], kernel_size=4, stride=2, padding=1)
        self.dec3 = ResidualBlock3DSparse(channels[3] * 2, channels[3], time_emb_dim, lidar_out_channels, num_groups, no_bev=no_bev, lidar_film_mode=lidar_film_mode)

        self.up2 = nn.ConvTranspose3d(channels[3], channels[2], kernel_size=4, stride=2, padding=1)
        self.dec2 = ResidualBlock3DSparse(channels[2] * 2, channels[2], time_emb_dim, lidar_out_channels, num_groups, no_bev=no_bev, lidar_film_mode=lidar_film_mode)

        self.up1 = nn.ConvTranspose3d(channels[2], channels[1], kernel_size=4, stride=2, padding=1)
        self.dec1 = ResidualBlock3DSparse(channels[1] * 2, channels[1], time_emb_dim, lidar_out_channels, num_groups, no_bev=no_bev, lidar_film_mode=lidar_film_mode)

        self.up0 = nn.ConvTranspose3d(channels[1], channels[0], kernel_size=4, stride=2, padding=1)
        self.dec0 = ResidualBlock3DSparse(channels[0] * 2, channels[0], time_emb_dim, lidar_out_channels, num_groups, no_bev=no_bev, lidar_film_mode=lidar_film_mode)

        # Output
        self.out_conv = nn.Sequential(
            nn.GroupNorm(num_groups, channels[0]),
            nn.SiLU(),
            nn.Conv3d(channels[0], num_classes, kernel_size=3, padding=1),
        )

        # Lifted features support (for CFG)
        self.use_lifted_features = False
        self.lifted_embed = None

        # Dense 3D bottleneck support (for S5)
        self.use_dense3d_bottleneck = False
        self.dense3d_bottleneck = None

    def enable_dense3d_bottleneck(self, dropout: float = 0.1) -> None:
        """
        Enable PaSCo's SPCDense3Dv2 at the bottleneck for dense hallucination.

        This helps complete occluded regions that sparse convolutions can't reach.
        Reference: PaSCo (CVPR 2024) - reference/PaSCo/pasco/models/layers.py

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

        logger.debug(
            "SceneCompletionUNetSparse: SPCDense3Dv2 bottleneck enabled "
            "(channels=%d, dropout=%g)", bottleneck_channels, dropout,
        )

    def enable_lifted_features(self, feature_dim: int = 64) -> None:
        """Enable lifted features conditioning for CFG."""
        self.use_lifted_features = True
        self.lifted_embed = nn.Conv3d(feature_dim, self.base_channels, kernel_size=3, padding=1)
        # Move to same device as model
        if hasattr(self, 'voxel_embed'):
            device = next(self.voxel_embed.parameters()).device
            self.lifted_embed = self.lifted_embed.to(device)

    def forward(
        self,
        x_t: torch.Tensor,  # [B, num_classes, H, W, D] one-hot noisy voxels
        t: torch.Tensor,    # [B] timesteps
        bev: torch.Tensor,  # [B, num_classes, H, W] BEV semantic map (one-hot)
        lidar: torch.Tensor,  # [B, 1, H, W, D] sparse LiDAR binary voxels
        lifted_features: torch.Tensor | None = None,  # [B, feature_dim, H, W, D] for CFG
        nn_indices: torch.Tensor | None = None,  # [B, 3, H, W, D] precomputed NN indices
        geom_bev: torch.Tensor | None = None,  # [B, 4, H, W] TSDF BEV features
        ssc_pred: torch.Tensor | None = None,  # [B, num_classes, H, W, D] SCPNet one-hot prediction
        obs_mask: torch.Tensor | None = None,  # [B, 1, H, W, D] observation mask
    ) -> torch.Tensor:
        # Concatenate observation mask with x_t if enabled
        if self.obs_mask_channel and obs_mask is not None:
            x_t = torch.cat([x_t, obs_mask], dim=1)  # [B, num_classes+1, H, W, D]

        B, C, H, W, D = x_t.shape

        # Extract multi-scale LiDAR features with sparse encoder
        lidar_feats = self.lidar_encoder(lidar, nn_indices=nn_indices)  # Dict with level0-level4

        # Multi-scale semantic conditioning via bev_embed pathway
        # If no_bev: use ssc_pred (3D) if available, else skip
        # If bev available: use BEV (2D→3D), ssc_pred goes through ssc_embed instead
        if self.no_bev:
            if ssc_pred is not None:
                # no_bev mode: route SCPNet 3D pred through bev_embed (multi-scale)
                cond_3d = ssc_pred  # [B, 20, H, W, D] already 3D
                bev_emb0 = self.bev_embed(cond_3d)
                bev_emb1 = F.avg_pool3d(bev_emb0, 2)
                bev_emb2 = F.avg_pool3d(bev_emb1, 2)
                bev_emb3 = F.avg_pool3d(bev_emb2, 2)
                bev_emb4 = F.avg_pool3d(bev_emb3, 2)
            else:
                bev_emb0 = bev_emb1 = bev_emb2 = bev_emb3 = bev_emb4 = None
        else:
            # BEV mode: expand BEV to 3D for multi-scale conditioning
            bev_3d = bev.unsqueeze(-1).expand(-1, -1, -1, -1, D)
            bev_emb0 = self.bev_embed(bev_3d)
            bev_emb1 = F.avg_pool3d(bev_emb0, 2)
            bev_emb2 = F.avg_pool3d(bev_emb1, 2)
            bev_emb3 = F.avg_pool3d(bev_emb2, 2)
            bev_emb4 = F.avg_pool3d(bev_emb3, 2)

        # B1c: Multi-scale SCPNet 3D conditioning (summed with bev_emb at every level)
        if self.ssc_multiscale_embed is not None and ssc_pred is not None:
            ssc_ms0 = self.ssc_multiscale_embed(ssc_pred)
            ssc_ms1 = F.avg_pool3d(ssc_ms0, 2)
            ssc_ms2 = F.avg_pool3d(ssc_ms1, 2)
            ssc_ms3 = F.avg_pool3d(ssc_ms2, 2)
            ssc_ms4 = F.avg_pool3d(ssc_ms3, 2)
            if bev_emb0 is not None:
                bev_emb0 = bev_emb0 + ssc_ms0
                bev_emb1 = bev_emb1 + ssc_ms1
                bev_emb2 = bev_emb2 + ssc_ms2
                bev_emb3 = bev_emb3 + ssc_ms3
                bev_emb4 = bev_emb4 + ssc_ms4
            else:
                bev_emb0, bev_emb1, bev_emb2 = ssc_ms0, ssc_ms1, ssc_ms2
                bev_emb3, bev_emb4 = ssc_ms3, ssc_ms4

        # Embed voxels
        x = self.voxel_embed(x_t)

        # S46: Add TSDF BEV geometric conditioning (injected once at input level)
        if self.geom_bev_embed is not None and geom_bev is not None:
            geom_3d = geom_bev.unsqueeze(-1).expand(-1, -1, -1, -1, D)  # [B, 4, H, W, D]
            x = x + self.geom_bev_embed(geom_3d)

        # SCPNet SSC prediction conditioning (injected once at input level)
        if self.ssc_embed is not None and ssc_pred is not None:
            x = x + self.ssc_embed(ssc_pred)

        # Add lifted features if provided (for CFG)
        if self.use_lifted_features and lifted_features is not None and self.lifted_embed is not None:
            x = x + self.lifted_embed(lifted_features)

        # Time embedding
        t_emb = timestep_embedding(t, self.base_channels)
        t_emb = self.time_mlp(t_emb)

        # ============ Encoder ============
        e0 = self.enc0(x, t_emb, bev_emb0, lidar_feats['level0'])
        x = self.down0(e0)

        e1 = self.enc1(x, t_emb, bev_emb1, lidar_feats['level1'])
        x = self.down1(e1)

        e2 = self.enc2(x, t_emb, bev_emb2, lidar_feats['level2'])
        x = self.down2(e2)

        e3 = self.enc3(x, t_emb, bev_emb3, lidar_feats['level3'])
        x = self.down3(e3)

        # ============ Bottleneck ============
        x = self.mid0(x, t_emb, bev_emb4, lidar_feats['level4'])
        x = self.mid1(x, t_emb, bev_emb4, lidar_feats['level4'])

        # Apply dense 3D bottleneck if enabled (S5)
        if self.use_dense3d_bottleneck and self.dense3d_bottleneck is not None:
            x = self.dense3d_bottleneck(x)

        # ============ Decoder ============
        x = self.up3(x)
        x = torch.cat([x, e3], dim=1)
        x = self.dec3(x, t_emb, bev_emb3, lidar_feats['level3'])

        x = self.up2(x)
        x = torch.cat([x, e2], dim=1)
        x = self.dec2(x, t_emb, bev_emb2, lidar_feats['level2'])

        x = self.up1(x)
        x = torch.cat([x, e1], dim=1)
        x = self.dec1(x, t_emb, bev_emb1, lidar_feats['level1'])

        x = self.up0(x)
        x = torch.cat([x, e0], dim=1)
        x = self.dec0(x, t_emb, bev_emb0, lidar_feats['level0'])

        return self.out_conv(x)

    def forward_features(
        self,
        x_t: torch.Tensor,  # [B, num_classes, H, W, D] one-hot noisy voxels
        t: torch.Tensor,    # [B] timesteps
        bev: torch.Tensor,  # [B, num_classes, H, W] BEV semantic map (one-hot)
        lidar: torch.Tensor,  # [B, 1, H, W, D] sparse LiDAR binary voxels
        lifted_features: torch.Tensor | None = None,  # [B, feature_dim, H, W, D] for CFG
        nn_indices: torch.Tensor | None = None,  # [B, 3, H, W, D] precomputed NN indices
        geom_bev: torch.Tensor | None = None,  # [B, 4, H, W] TSDF BEV features
        ssc_pred: torch.Tensor | None = None,  # [B, num_classes, H, W, D] SCPNet one-hot prediction
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass that returns both predictions and decoder features (full resolution).
        Used for DSKD (Dense-to-Sparse Knowledge Distillation).

        Returns full-resolution decoder features (completion features) for rich
        spatial detail in pairwise similarity matching.

        Returns:
            predictions: [B, num_classes, H, W, D] logits
            decoder_features: [B, base_channels, H, W, D] features at full resolution for DSKD
        """
        B, C, H, W, D = x_t.shape

        # Extract multi-scale LiDAR features with sparse encoder
        lidar_feats = self.lidar_encoder(lidar, nn_indices=nn_indices)  # Dict with level0-level4

        # Multi-scale semantic conditioning via bev_embed pathway
        # If no_bev: use ssc_pred (3D) if available, else skip
        # If bev available: use BEV (2D→3D), ssc_pred goes through ssc_embed instead
        if self.no_bev:
            if ssc_pred is not None:
                # no_bev mode: route SCPNet 3D pred through bev_embed (multi-scale)
                cond_3d = ssc_pred  # [B, 20, H, W, D] already 3D
                bev_emb0 = self.bev_embed(cond_3d)
                bev_emb1 = F.avg_pool3d(bev_emb0, 2)
                bev_emb2 = F.avg_pool3d(bev_emb1, 2)
                bev_emb3 = F.avg_pool3d(bev_emb2, 2)
                bev_emb4 = F.avg_pool3d(bev_emb3, 2)
            else:
                bev_emb0 = bev_emb1 = bev_emb2 = bev_emb3 = bev_emb4 = None
        else:
            # BEV mode: expand BEV to 3D for multi-scale conditioning
            bev_3d = bev.unsqueeze(-1).expand(-1, -1, -1, -1, D)
            bev_emb0 = self.bev_embed(bev_3d)
            bev_emb1 = F.avg_pool3d(bev_emb0, 2)
            bev_emb2 = F.avg_pool3d(bev_emb1, 2)
            bev_emb3 = F.avg_pool3d(bev_emb2, 2)
            bev_emb4 = F.avg_pool3d(bev_emb3, 2)

        # B1c: Multi-scale SCPNet 3D conditioning (summed with bev_emb at every level)
        if self.ssc_multiscale_embed is not None and ssc_pred is not None:
            ssc_ms0 = self.ssc_multiscale_embed(ssc_pred)
            ssc_ms1 = F.avg_pool3d(ssc_ms0, 2)
            ssc_ms2 = F.avg_pool3d(ssc_ms1, 2)
            ssc_ms3 = F.avg_pool3d(ssc_ms2, 2)
            ssc_ms4 = F.avg_pool3d(ssc_ms3, 2)
            if bev_emb0 is not None:
                bev_emb0 = bev_emb0 + ssc_ms0
                bev_emb1 = bev_emb1 + ssc_ms1
                bev_emb2 = bev_emb2 + ssc_ms2
                bev_emb3 = bev_emb3 + ssc_ms3
                bev_emb4 = bev_emb4 + ssc_ms4
            else:
                bev_emb0, bev_emb1, bev_emb2 = ssc_ms0, ssc_ms1, ssc_ms2
                bev_emb3, bev_emb4 = ssc_ms3, ssc_ms4

        # Embed voxels
        x = self.voxel_embed(x_t)

        # S46: Add TSDF BEV geometric conditioning (injected once at input level)
        if self.geom_bev_embed is not None and geom_bev is not None:
            geom_3d = geom_bev.unsqueeze(-1).expand(-1, -1, -1, -1, D)  # [B, 4, H, W, D]
            x = x + self.geom_bev_embed(geom_3d)

        # SCPNet SSC prediction conditioning (injected once at input level)
        if self.ssc_embed is not None and ssc_pred is not None:
            x = x + self.ssc_embed(ssc_pred)

        # Add lifted features if provided (for CFG)
        if self.use_lifted_features and lifted_features is not None and self.lifted_embed is not None:
            x = x + self.lifted_embed(lifted_features)

        # Time embedding
        t_emb = timestep_embedding(t, self.base_channels)
        t_emb = self.time_mlp(t_emb)

        # ============ Encoder ============
        e0 = self.enc0(x, t_emb, bev_emb0, lidar_feats['level0'])
        x = self.down0(e0)

        e1 = self.enc1(x, t_emb, bev_emb1, lidar_feats['level1'])
        x = self.down1(e1)

        e2 = self.enc2(x, t_emb, bev_emb2, lidar_feats['level2'])
        x = self.down2(e2)

        e3 = self.enc3(x, t_emb, bev_emb3, lidar_feats['level3'])
        x = self.down3(e3)

        # ============ Bottleneck ============
        x = self.mid0(x, t_emb, bev_emb4, lidar_feats['level4'])
        x = self.mid1(x, t_emb, bev_emb4, lidar_feats['level4'])

        # Apply dense 3D bottleneck if enabled (S5)
        if self.use_dense3d_bottleneck and self.dense3d_bottleneck is not None:
            x = self.dense3d_bottleneck(x)

        # ============ Decoder ============
        x = self.up3(x)
        x = torch.cat([x, e3], dim=1)
        x = self.dec3(x, t_emb, bev_emb3, lidar_feats['level3'])

        x = self.up2(x)
        x = torch.cat([x, e2], dim=1)
        x = self.dec2(x, t_emb, bev_emb2, lidar_feats['level2'])

        x = self.up1(x)
        x = torch.cat([x, e1], dim=1)
        x = self.dec1(x, t_emb, bev_emb1, lidar_feats['level1'])

        x = self.up0(x)
        x = torch.cat([x, e0], dim=1)
        x = self.dec0(x, t_emb, bev_emb0, lidar_feats['level0'])

        # DSKD: Return decoder features (full resolution) instead of bottleneck
        # This matches SCPNet's "completion features" - rich spatial detail for distillation
        # Shape: [B, base_channels, 256, 256, 32] - same as input resolution
        decoder_features = x.clone()

        return self.out_conv(x), decoder_features


class SceneCompletionUNetSparseLite(nn.Module):
    """
    Lighter version with smaller channels.

    Channels: 16 → 32 → 64 → 128 (instead of 32 → 64 → 128 → 256)
    """

    def __init__(
        self,
        num_classes: int = 20,
        base_channels: int = 16,
        channel_mult: tuple[int, ...] = (1, 2, 4, 8),
        time_emb_dim: int = 64,
        lidar_base_channels: int = 8,
        lidar_out_channels: int = 16,
        num_groups: int = 4,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.base_channels = base_channels

        channels = [base_channels * m for m in channel_mult]

        # Sparse LiDAR encoder (lite version)
        self.lidar_encoder = SparseLiDAREncoderLite(
            in_channels=1,
            base_channels=lidar_base_channels,
            out_channels=lidar_out_channels,
        )

        # Embeddings
        self.voxel_embed = nn.Conv3d(num_classes, base_channels, kernel_size=3, padding=1)
        self.bev_embed = nn.Conv3d(num_classes, lidar_out_channels, kernel_size=3, padding=1)

        self.time_mlp = nn.Sequential(
            nn.Linear(base_channels, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        # Encoder
        self.enc0 = ResidualBlock3DSparse(channels[0], channels[0], time_emb_dim, lidar_out_channels, num_groups)
        self.down0 = nn.Conv3d(channels[0], channels[1], kernel_size=3, stride=2, padding=1)

        self.enc1 = ResidualBlock3DSparse(channels[1], channels[1], time_emb_dim, lidar_out_channels, num_groups)
        self.down1 = nn.Conv3d(channels[1], channels[2], kernel_size=3, stride=2, padding=1)

        self.enc2 = ResidualBlock3DSparse(channels[2], channels[2], time_emb_dim, lidar_out_channels, num_groups)
        self.down2 = nn.Conv3d(channels[2], channels[3], kernel_size=3, stride=2, padding=1)

        self.enc3 = ResidualBlock3DSparse(channels[3], channels[3], time_emb_dim, lidar_out_channels, num_groups)
        self.down3 = nn.Conv3d(channels[3], channels[3], kernel_size=3, stride=2, padding=1)

        # Bottleneck
        self.mid0 = ResidualBlock3DSparse(channels[3], channels[3], time_emb_dim, lidar_out_channels, num_groups)
        self.mid1 = ResidualBlock3DSparse(channels[3], channels[3], time_emb_dim, lidar_out_channels, num_groups)

        # Decoder
        self.up3 = nn.ConvTranspose3d(channels[3], channels[3], kernel_size=4, stride=2, padding=1)
        self.dec3 = ResidualBlock3DSparse(channels[3] * 2, channels[3], time_emb_dim, lidar_out_channels, num_groups)

        self.up2 = nn.ConvTranspose3d(channels[3], channels[2], kernel_size=4, stride=2, padding=1)
        self.dec2 = ResidualBlock3DSparse(channels[2] * 2, channels[2], time_emb_dim, lidar_out_channels, num_groups)

        self.up1 = nn.ConvTranspose3d(channels[2], channels[1], kernel_size=4, stride=2, padding=1)
        self.dec1 = ResidualBlock3DSparse(channels[1] * 2, channels[1], time_emb_dim, lidar_out_channels, num_groups)

        self.up0 = nn.ConvTranspose3d(channels[1], channels[0], kernel_size=4, stride=2, padding=1)
        self.dec0 = ResidualBlock3DSparse(channels[0] * 2, channels[0], time_emb_dim, lidar_out_channels, num_groups)

        self.out_conv = nn.Sequential(
            nn.GroupNorm(num_groups, channels[0]),
            nn.SiLU(),
            nn.Conv3d(channels[0], num_classes, kernel_size=3, padding=1),
        )

        # Lifted features support (for CFG)
        self.use_lifted_features = False
        self.lifted_embed = None

        # Dense 3D bottleneck support (for S5)
        self.use_dense3d_bottleneck = False
        self.dense3d_bottleneck = None

    def enable_dense3d_bottleneck(self, dropout: float = 0.1) -> None:
        """Enable PaSCo's SPCDense3Dv2 at the bottleneck."""
        from gssc.models.dense_3d_cnn import SPCDense3Dv2

        bottleneck_channels = self.base_channels * 8  # 128 for base=16

        self.dense3d_bottleneck = nn.Sequential(
            SPCDense3Dv2(init_size=bottleneck_channels),
            nn.Dropout3d(dropout),
        )
        self.use_dense3d_bottleneck = True

        device = next(self.parameters()).device
        self.dense3d_bottleneck = self.dense3d_bottleneck.to(device)
        logger.debug(
            "SceneCompletionUNetSparseLite: SPCDense3Dv2 bottleneck enabled "
            "(channels=%d, dropout=%g)", bottleneck_channels, dropout,
        )

    def enable_lifted_features(self, feature_dim: int = 64) -> None:
        """Enable lifted features conditioning for CFG."""
        self.use_lifted_features = True
        self.lifted_embed = nn.Conv3d(feature_dim, self.base_channels, kernel_size=3, padding=1)
        # Move to same device as model
        if hasattr(self, 'voxel_embed'):
            device = next(self.voxel_embed.parameters()).device
            self.lifted_embed = self.lifted_embed.to(device)

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        bev: torch.Tensor,
        lidar: torch.Tensor,
        lifted_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Same interface as SceneCompletionUNetSparse."""
        B, C, H, W, D = x_t.shape

        lidar_feats = self.lidar_encoder(lidar)

        bev_3d = bev.unsqueeze(-1).expand(-1, -1, -1, -1, D)
        bev_emb0 = self.bev_embed(bev_3d)

        x = self.voxel_embed(x_t)

        # Add lifted features if provided (for CFG)
        if self.use_lifted_features and lifted_features is not None and self.lifted_embed is not None:
            x = x + self.lifted_embed(lifted_features)

        t_emb = timestep_embedding(t, self.base_channels)
        t_emb = self.time_mlp(t_emb)

        # Encoder
        e0 = self.enc0(x, t_emb, bev_emb0, lidar_feats['level0'])
        x = self.down0(e0)
        bev_emb1 = F.avg_pool3d(bev_emb0, 2)

        e1 = self.enc1(x, t_emb, bev_emb1, lidar_feats['level1'])
        x = self.down1(e1)
        bev_emb2 = F.avg_pool3d(bev_emb1, 2)

        e2 = self.enc2(x, t_emb, bev_emb2, lidar_feats['level2'])
        x = self.down2(e2)
        bev_emb3 = F.avg_pool3d(bev_emb2, 2)

        e3 = self.enc3(x, t_emb, bev_emb3, lidar_feats['level3'])
        x = self.down3(e3)
        bev_emb4 = F.avg_pool3d(bev_emb3, 2)

        # Bottleneck
        x = self.mid0(x, t_emb, bev_emb4, lidar_feats['level4'])
        x = self.mid1(x, t_emb, bev_emb4, lidar_feats['level4'])

        # Apply dense 3D bottleneck if enabled (S5)
        if self.use_dense3d_bottleneck and self.dense3d_bottleneck is not None:
            x = self.dense3d_bottleneck(x)

        # Decoder
        x = self.up3(x)
        x = torch.cat([x, e3], dim=1)
        x = self.dec3(x, t_emb, bev_emb3, lidar_feats['level3'])

        x = self.up2(x)
        x = torch.cat([x, e2], dim=1)
        x = self.dec2(x, t_emb, bev_emb2, lidar_feats['level2'])

        x = self.up1(x)
        x = torch.cat([x, e1], dim=1)
        x = self.dec1(x, t_emb, bev_emb1, lidar_feats['level1'])

        x = self.up0(x)
        x = torch.cat([x, e0], dim=1)
        x = self.dec0(x, t_emb, bev_emb0, lidar_feats['level0'])

        return self.out_conv(x)

    def forward_features(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        bev: torch.Tensor,
        lidar: torch.Tensor,
        lifted_features: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning both predictions and decoder features (full resolution) for DSKD."""
        B, C, H, W, D = x_t.shape

        lidar_feats = self.lidar_encoder(lidar)

        bev_3d = bev.unsqueeze(-1).expand(-1, -1, -1, -1, D)
        bev_emb0 = self.bev_embed(bev_3d)

        x = self.voxel_embed(x_t)

        if self.use_lifted_features and lifted_features is not None and self.lifted_embed is not None:
            x = x + self.lifted_embed(lifted_features)

        t_emb = timestep_embedding(t, self.base_channels)
        t_emb = self.time_mlp(t_emb)

        # Encoder
        e0 = self.enc0(x, t_emb, bev_emb0, lidar_feats['level0'])
        x = self.down0(e0)
        bev_emb1 = F.avg_pool3d(bev_emb0, 2)

        e1 = self.enc1(x, t_emb, bev_emb1, lidar_feats['level1'])
        x = self.down1(e1)
        bev_emb2 = F.avg_pool3d(bev_emb1, 2)

        e2 = self.enc2(x, t_emb, bev_emb2, lidar_feats['level2'])
        x = self.down2(e2)
        bev_emb3 = F.avg_pool3d(bev_emb2, 2)

        e3 = self.enc3(x, t_emb, bev_emb3, lidar_feats['level3'])
        x = self.down3(e3)
        bev_emb4 = F.avg_pool3d(bev_emb3, 2)

        # Bottleneck
        x = self.mid0(x, t_emb, bev_emb4, lidar_feats['level4'])
        x = self.mid1(x, t_emb, bev_emb4, lidar_feats['level4'])

        if self.use_dense3d_bottleneck and self.dense3d_bottleneck is not None:
            x = self.dense3d_bottleneck(x)

        # Decoder
        x = self.up3(x)
        x = torch.cat([x, e3], dim=1)
        x = self.dec3(x, t_emb, bev_emb3, lidar_feats['level3'])

        x = self.up2(x)
        x = torch.cat([x, e2], dim=1)
        x = self.dec2(x, t_emb, bev_emb2, lidar_feats['level2'])

        x = self.up1(x)
        x = torch.cat([x, e1], dim=1)
        x = self.dec1(x, t_emb, bev_emb1, lidar_feats['level1'])

        x = self.up0(x)
        x = torch.cat([x, e0], dim=1)
        x = self.dec0(x, t_emb, bev_emb0, lidar_feats['level0'])

        # DSKD: Return decoder features (full resolution) instead of bottleneck
        decoder_features = x.clone()

        return self.out_conv(x), decoder_features


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    print("=== Phase 3 exp_2: Sparse LiDAR Encoder ===\n")

    # Parameter counts
    model = SceneCompletionUNetSparse(num_classes=20, base_channels=32)
    print("SceneCompletionUNetSparse (Full):")
    print(f"  Parameters: {count_parameters(model):,}")
    print("  U-Net channels: 32 → 64 → 128 → 256")
    print("  LiDAR encoder: 16 → 32 → 64 → 128")

    model_lite = SceneCompletionUNetSparseLite(num_classes=20, base_channels=16)
    print("\nSceneCompletionUNetSparseLite:")
    print(f"  Parameters: {count_parameters(model_lite):,}")
    print("  U-Net channels: 16 → 32 → 64 → 128")
    print("  LiDAR encoder: 8 → 16 → 32 → 64")

    print("\n=== Comparison with exp_1 (Dense Conv3d) ===")
    from .scene_unet import SceneCompletionUNet, SceneCompletionUNetLite
    exp1_full = SceneCompletionUNet(num_classes=20, base_channels=32)
    exp1_lite = SceneCompletionUNetLite(num_classes=20, base_channels=16)
    print(f"exp_1 Full: {count_parameters(exp1_full):,} params")
    print(f"exp_2 Full: {count_parameters(model):,} params")
    print(f"exp_1 Lite: {count_parameters(exp1_lite):,} params")
    print(f"exp_2 Lite: {count_parameters(model_lite):,} params")
