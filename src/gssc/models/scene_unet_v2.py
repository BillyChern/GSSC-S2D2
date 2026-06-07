"""
3D U-Net V2 for Scene Completion: FiLM Conditioning + Multi-Scale Auxiliary BEV

DEPRECATED RESEARCH VARIANT — NOT the released architecture.
The S2D2 denoiser described in the paper is a DENSE ``Conv3d`` 3D U-Net with
additive conditioning + AdaGN, implemented in ``s2d2_unet.py``
(``SceneCompletionUNetSparse``; the ``sparse_full`` model_type). This V2 module is
an exploratory FiLM-conditioning prototype kept only for research reference; it is
not used by any shipped config and does not correspond to the paper's method.

Key differences from scene_unet_sparse.py (V1):
- FiLM conditioning (scale + shift) instead of additive dual-projection
  Inspired by DiffSSC's multiplicative conditioning (minkunet.py:420-497)
- No BEV input — BEV serves only as auxiliary loss (JS3C-Net style)
- Multi-scale auxiliary BEV head using proven BEV decoder architecture (26.27% mIoU)
- Compatible with MultinomialDiffusion3D via V2ModelWrapper

Architecture:
- SparseLiDAREncoder(20ch): processes LSK3DNet soft probs in 3D
- FiLM conditioning at all 10 UNet levels (4 enc + 2 mid + 4 dec)
- Multi-scale aux BEV: HeightCompress(4 levels) + BEVDecoder(FPN) + SegHead
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from gssc.models.bev_sparse_bev_net import BEVDecoder, SegmentationHead

from .sparse_lidar_encoder import SparseLiDAREncoder


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


class ResidualBlock3DV2(nn.Module):
    """3D Residual Block with FiLM conditioning for both time and spatial features.

    Key difference from ResidualBlock3DSparse:
    - Single FiLM-style conditioning (scale + shift) instead of dual additive projections
    - Inspired by DiffSSC's multiplicative conditioning (minkunet.py:420-497)
    - Time conditioning remains FiLM as before (scene_unet_sparse.py:95-99)
    """

    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int,
                 cond_channels: int, num_groups: int = 8):
        super().__init__()
        # First conv
        self.norm1 = nn.GroupNorm(min(num_groups, in_channels), in_channels)
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)

        # FiLM conditioning: produces scale and shift from spatial condition
        self.cond_film = nn.Conv3d(cond_channels, out_channels * 2, kernel_size=3, padding=1)

        # Second conv with FiLM time conditioning
        self.norm2 = nn.GroupNorm(min(num_groups, out_channels), out_channels)
        self.time_fc = nn.Linear(time_emb_dim, out_channels * 2)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)

        # Skip connection
        self.skip = nn.Conv3d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor,
                cond_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C_in, H, W, D] voxel features
            t_emb: [B, time_emb_dim] timestep embedding
            cond_emb: [B, cond_channels, H, W, D] multi-scale LSK3DNet features
        """
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)

        # FiLM spatial conditioning (DiffSSC-inspired multiplicative)
        scale_shift = self.cond_film(cond_emb)
        scale, shift = scale_shift.chunk(2, dim=1)
        h = h * (1 + scale) + shift  # FiLM: centered around identity

        # FiLM time conditioning (unchanged from existing code)
        h = self.norm2(h)
        wb = self.time_fc(t_emb)
        w, b = wb.chunk(2, dim=-1)
        h = w[:, :, None, None, None] * h + b[:, :, None, None, None]

        h = F.silu(h)
        h = self.conv2(h)

        return h + self.skip(x)


class DenseHeightCompression(nn.Module):
    """Height compression for dense 3D tensors.

    Same logic as HeightCompression in sparse_bev_net.py:140-173,
    but operates on dense [B, C, H, W, D] tensors instead of spconv.SparseConvTensor.
    Dual max+mean pooling over Z-axis, then project to output channels.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels * 2, out_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, dense_3d: torch.Tensor) -> torch.Tensor:
        """
        Args:
            dense_3d: [B, C, H, W, D] dense 3D features from UNet encoder
        Returns:
            bev_2d: [B, out_channels, H, W] compressed BEV features
        """
        max_pool = dense_3d.max(dim=-1)[0]   # [B, C, H, W]
        mean_pool = dense_3d.mean(dim=-1)     # [B, C, H, W]
        combined = torch.cat([max_pool, mean_pool], dim=1)  # [B, 2C, H, W]
        out = self.conv(combined)
        out = self.bn(out)
        out = F.relu(out)
        return out


class MultiScaleAuxBEVHead(nn.Module):
    """Training-only multi-scale auxiliary BEV prediction head.

    Reuses the PROVEN architecture from BEV supervised model (26.27% mIoU):
    - DenseHeightCompression at ALL 4 encoder levels (dual max+mean pool)
    - BEVDecoder: FPN-style 2D decoder with skip connections
    - SegmentationHead: 64->64->20

    SSC UNet encoder features [e0..e3] have IDENTICAL dimensions to BEV supervised
    encoder features [f1..f4], so the same BEVDecoder + SegHead plugs directly.
    ~2.5M extra params (training only, discarded at inference).
    """

    def __init__(self, encoder_channels: list[int] = None, num_classes: int = 20):
        super().__init__()
        if encoder_channels is None:
            encoder_channels = [32, 64, 128, 256]

        decoder_channels = [256, 128, 64]

        # Height compression at all 4 levels
        self.height_compress = nn.ModuleList([
            DenseHeightCompression(encoder_channels[i], encoder_channels[i])
            for i in range(4)
        ])

        # FPN decoder (same as BEV supervised model)
        self.decoder = BEVDecoder(encoder_channels, decoder_channels)

        # Segmentation head (same as BEV supervised model)
        self.seg_head = SegmentationHead(decoder_channels[-1], num_classes, dropout=0.1)

    def forward(self, encoder_features: list[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            encoder_features: [e0, e1, e2, e3] from SSC UNet encoder
        Returns:
            bev_logits: [B, num_classes, H, W]
        """
        bev_features = [
            self.height_compress[i](encoder_features[i])
            for i in range(4)
        ]
        decoded = self.decoder(bev_features)
        return self.seg_head(decoded)


class SceneCompletionUNetV2(nn.Module):
    """SSC UNet V2: FiLM conditioning + multi-scale auxiliary BEV head.

    Key differences from SceneCompletionUNetSparse:
    - No BEV input at all (no bev_embed, no bev_proj, no Z-expansion)
    - Single FiLM conditioning from SparseLiDAREncoder(20ch) features
    - Multi-scale auxiliary BEV head using proven BEV decoder architecture
    - Compatible with MultinomialDiffusion3D via V2ModelWrapper
    """

    def __init__(
        self,
        num_classes: int = 20,
        base_channels: int = 32,
        channel_mult: tuple[int, ...] = (1, 2, 4, 8),
        time_emb_dim: int = 128,
        lidar_in_channels: int = 20,
        lidar_base_channels: int = 16,
        lidar_out_channels: int = 32,
        num_groups: int = 8,
        aux_bev: bool = True,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.base_channels = base_channels
        self.aux_bev_enabled = aux_bev

        channels = [base_channels * m for m in channel_mult]
        cond_ch = lidar_out_channels

        # Sparse LiDAR encoder (reused from sparse_lidar_encoder.py)
        self.lidar_encoder = SparseLiDAREncoder(
            in_channels=lidar_in_channels,
            base_channels=lidar_base_channels,
            out_channels=lidar_out_channels,
        )

        # Voxel embedding (for diffusion x_t only)
        self.voxel_embed = nn.Conv3d(num_classes, base_channels, 3, padding=1)

        # Time embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(base_channels, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        # Encoder
        self.enc0 = ResidualBlock3DV2(channels[0], channels[0], time_emb_dim, cond_ch, num_groups)
        self.down0 = nn.Conv3d(channels[0], channels[1], 3, stride=2, padding=1)

        self.enc1 = ResidualBlock3DV2(channels[1], channels[1], time_emb_dim, cond_ch, num_groups)
        self.down1 = nn.Conv3d(channels[1], channels[2], 3, stride=2, padding=1)

        self.enc2 = ResidualBlock3DV2(channels[2], channels[2], time_emb_dim, cond_ch, num_groups)
        self.down2 = nn.Conv3d(channels[2], channels[3], 3, stride=2, padding=1)

        self.enc3 = ResidualBlock3DV2(channels[3], channels[3], time_emb_dim, cond_ch, num_groups)
        self.down3 = nn.Conv3d(channels[3], channels[3], 3, stride=2, padding=1)

        # Bottleneck
        self.mid0 = ResidualBlock3DV2(channels[3], channels[3], time_emb_dim, cond_ch, num_groups)
        self.mid1 = ResidualBlock3DV2(channels[3], channels[3], time_emb_dim, cond_ch, num_groups)

        # Decoder
        self.up3 = nn.ConvTranspose3d(channels[3], channels[3], 4, stride=2, padding=1)
        self.dec3 = ResidualBlock3DV2(channels[3] * 2, channels[3], time_emb_dim, cond_ch, num_groups)

        self.up2 = nn.ConvTranspose3d(channels[3], channels[2], 4, stride=2, padding=1)
        self.dec2 = ResidualBlock3DV2(channels[2] * 2, channels[2], time_emb_dim, cond_ch, num_groups)

        self.up1 = nn.ConvTranspose3d(channels[2], channels[1], 4, stride=2, padding=1)
        self.dec1 = ResidualBlock3DV2(channels[1] * 2, channels[1], time_emb_dim, cond_ch, num_groups)

        self.up0 = nn.ConvTranspose3d(channels[1], channels[0], 4, stride=2, padding=1)
        self.dec0 = ResidualBlock3DV2(channels[0] * 2, channels[0], time_emb_dim, cond_ch, num_groups)

        # Output
        self.out_conv = nn.Sequential(
            nn.GroupNorm(num_groups, channels[0]),
            nn.SiLU(),
            nn.Conv3d(channels[0], num_classes, 3, padding=1),
        )

        # Multi-scale auxiliary BEV head (training only)
        if aux_bev:
            self.aux_bev_head = MultiScaleAuxBEVHead(
                encoder_channels=channels,
                num_classes=num_classes,
            )

    def forward(self, x_t: torch.Tensor, t: torch.Tensor,
                lidar: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Args:
            x_t: [B, 20, H, W, D] one-hot noisy voxels
            t: [B] timesteps
            lidar: [B, 20, H, W, D] LSK3DNet soft probs
        Returns:
            ssc_logits: [B, 20, H, W, D] class logits
            aux_bev: [B, 20, H_bev, W_bev] BEV logits or None (inference)
        """
        # Extract multi-scale 3D features via SparseLiDAREncoder
        lidar_feats = self.lidar_encoder(lidar)

        # Embed noisy voxels
        x = self.voxel_embed(x_t)

        # Time embedding
        t_emb = timestep_embedding(t, self.base_channels)
        t_emb = self.time_mlp(t_emb)

        # Encoder with FiLM conditioning
        e0 = self.enc0(x, t_emb, lidar_feats['level0'])
        x = self.down0(e0)

        e1 = self.enc1(x, t_emb, lidar_feats['level1'])
        x = self.down1(e1)

        e2 = self.enc2(x, t_emb, lidar_feats['level2'])
        x = self.down2(e2)

        e3 = self.enc3(x, t_emb, lidar_feats['level3'])
        x = self.down3(e3)

        # Bottleneck
        x = self.mid0(x, t_emb, lidar_feats['level4'])
        x = self.mid1(x, t_emb, lidar_feats['level4'])

        # Multi-scale auxiliary BEV (training only)
        aux_bev = None
        if self.training and self.aux_bev_enabled:
            aux_bev = self.aux_bev_head([e0, e1, e2, e3])

        # Decoder with FiLM conditioning
        x = self.up3(x)
        x = torch.cat([x, e3], dim=1)
        x = self.dec3(x, t_emb, lidar_feats['level3'])

        x = self.up2(x)
        x = torch.cat([x, e2], dim=1)
        x = self.dec2(x, t_emb, lidar_feats['level2'])

        x = self.up1(x)
        x = torch.cat([x, e1], dim=1)
        x = self.dec1(x, t_emb, lidar_feats['level1'])

        x = self.up0(x)
        x = torch.cat([x, e0], dim=1)
        x = self.dec0(x, t_emb, lidar_feats['level0'])

        return self.out_conv(x), aux_bev


class V2ModelWrapper(nn.Module):
    """Wraps SceneCompletionUNetV2 to match MultinomialDiffusion3D's expected interface.

    MultinomialDiffusion3D calls: model(x_t, t, bev, lidar, lifted_features=None)
    V2 UNet expects:             model(x_t, t, lidar)
    V2 UNet returns:             (ssc_logits, aux_bev)
    Wrapper returns:             ssc_logits (stores aux_bev as attribute)
    """

    def __init__(self, model: SceneCompletionUNetV2):
        super().__init__()
        self.model = model
        self.last_aux_bev = None

    def forward(self, x_t: torch.Tensor, t: torch.Tensor,
                bev: torch.Tensor, lidar: torch.Tensor,
                lifted_features: torch.Tensor | None = None) -> torch.Tensor:
        # bev and lifted_features are IGNORED
        ssc_logits, aux_bev = self.model(x_t, t, lidar)
        self.last_aux_bev = aux_bev
        return ssc_logits
