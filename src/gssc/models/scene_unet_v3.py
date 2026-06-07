"""
3D U-Net V3 for Scene Completion: Internal BEV + Sparse 3D FiLM

DEPRECATED RESEARCH VARIANT — NOT the released architecture.
The S2D2 denoiser described in the paper is a DENSE ``Conv3d`` 3D U-Net with
additive conditioning + AdaGN, implemented in ``s2d2_unet.py``
(``SceneCompletionUNetSparse``; the ``sparse_full`` model_type). This V3 module is
an exploratory FiLM / internal-BEV prototype kept only for research reference; it
is not used by any shipped config and does not correspond to the paper's method.

Key differences from V2 (scene_unet_v2.py):
- V2 FAILED because SparseLiDAREncoder outputs are ~99% zeros at level0.
  FiLM conditioning becomes identity at zero voxels → no spatial guidance
  at 60-70% of XY columns → model collapses (S8: 0.67% mIoU, S6: 0.88%)
- V3 adds InternalBEVCompletion: height-compress sparse 3D features to 2D,
  run 2D completion (dilated convs), Z-expand back to 3D → DENSE conditioning
- Dual conditioning at every block:
  1. Sparse FiLM: h * (1 + scale) + shift from LSK3DNet features (high quality at obs voxels)
  2. Dense additive: h + proj(dense_cond) from internal BEV (spatial layout everywhere)

V3.1 improvements (Feb 16, 2026):
- CoarseToFine3DFusion: replaces InternalBEVCompletion. Trilinear upsample
  level3 (32×32×4, ~90% dense) + level4 (16×16×2, ~99% dense) to full resolution.
  PRESERVES height info (4+2 Z bins → 32 via interpolation) unlike InternalBEV
  which produces Z-invariant features (identical at z=1..30).
- CFG (Classifier-Free Guidance): 10% conditioning dropout during training,
  guidance_scale amplification at inference. Critical for diffusion quality.

Architecture:
- SparseLiDAREncoder(20ch): processes LSK3DNet soft probs in 3D
- CoarseToFine3DFusion: trilinear upsample dense lower levels → preserves 3D structure
- InternalBEVCompletion: sparse 3D → dense 3D (V3.0 legacy, height-loss bug)
- ResidualBlock3DV3: dual conditioning (sparse FiLM + dense additive)
- Multi-scale aux BEV: HeightCompress(4 levels) + BEVDecoder(FPN) + SegHead
"""


import torch
import torch.nn as nn
import torch.nn.functional as F

from .scene_unet_v2 import (
    MultiScaleAuxBEVHead,
    timestep_embedding,
)
from .sparse_lidar_encoder import SparseLiDAREncoder


class InternalBEVCompletion(nn.Module):
    """Generates dense 3D conditioning from sparse 3D features.

    Pipeline: sparse 3D [B,C,H,W,D] → height compress (max+mean) → 2D [B,C',H,W]
              → 2D completion (dilated convs, fill ~70% empty columns) → Z-expand
              → 3D [B,C',H,W,D] with learned Z-dependent features

    This replaces external BEV Z-expansion with a LEARNED internal version.
    No external BEV input needed → no train/test distribution gap.

    Receptive field: 5 Conv2d layers (3 dilation=1, 2 dilation=2) → RF ~15×15.
    At 256×256 with ~30-40% column occupancy, each output sees ~67-90 observed columns.
    """

    def __init__(self, in_channels: int = 32, bev_channels: int = 32,
                 num_conv_layers: int = 5):
        super().__init__()
        # Height compression: 3D → 2D (dual max+mean pool, proven in BEV supervised)
        self.height_compress = nn.Sequential(
            nn.Conv2d(in_channels * 2, bev_channels, 1, bias=False),
            nn.BatchNorm2d(bev_channels),
            nn.ReLU(inplace=True),
        )
        # 2D completion: fill empty columns with dilated convs for large receptive field
        layers = []
        for i in range(num_conv_layers):
            dilation = 2 if i >= num_conv_layers - 2 else 1  # Last 2 layers dilated
            layers.extend([
                nn.Conv2d(bev_channels, bev_channels, 3,
                         padding=dilation, dilation=dilation, bias=False),
                nn.BatchNorm2d(bev_channels),
                nn.ReLU(inplace=True),
            ])
        self.complete = nn.Sequential(*layers)
        # Z-expansion: learn height-dependent features (like bev_embed Conv3d in V1)
        self.to_3d = nn.Sequential(
            nn.Conv3d(bev_channels, bev_channels, 3, padding=1, bias=False),
            nn.GroupNorm(8, bev_channels),
            nn.SiLU(),
        )

    def forward(self, sparse_3d_level0: torch.Tensor) -> torch.Tensor:
        """
        Args:
            sparse_3d_level0: [B, C, H, W, D] — mostly zeros (~99%)
        Returns:
            dense_3d: [B, bev_channels, H, W, D] — dense everywhere
        """
        # Height compress (same as DenseHeightCompression)
        max_pool = sparse_3d_level0.max(dim=-1)[0]   # [B, C, H, W]
        mean_pool = sparse_3d_level0.mean(dim=-1)     # [B, C, H, W]
        bev_2d = self.height_compress(
            torch.cat([max_pool, mean_pool], dim=1)   # [B, 2C, H, W]
        )  # [B, bev_ch, H, W]
        # 2D completion with residual
        bev_2d = bev_2d + self.complete(bev_2d)       # [B, bev_ch, H, W]
        # Z-expand + learn height-dependent features
        D = sparse_3d_level0.shape[-1]
        bev_3d = bev_2d.unsqueeze(-1).expand(-1, -1, -1, -1, D)
        return self.to_3d(bev_3d.contiguous())         # [B, bev_ch, H, W, D]


class CoarseToFine3DFusion(nn.Module):
    """Upsample dense lower-level features to full resolution.

    Uses level3 (32×32×4, ~90% dense) and level4 (16×16×2, ~99% dense)
    from SparseLiDAREncoder. Trilinear upsampling preserves height structure
    (4+2 Z bins → 32 via interpolation).

    Key advantage over InternalBEVCompletion:
    - Preserves height info: level3 has 4 Z bins, level4 has 2 Z bins
    - Nearly dense at coarse levels (no 2D completion network needed)
    - Simpler, fewer params, no Z-invariance bug
    """

    def __init__(self, in_channels: int = 32, out_channels: int = 32):
        super().__init__()
        self.fuse = nn.Sequential(
            nn.Conv3d(in_channels * 2, out_channels, 1, bias=False),
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
            nn.Conv3d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
        )

    def forward(self, lidar_feats: dict) -> torch.Tensor:
        """
        Args:
            lidar_feats: dict from SparseLiDAREncoder with 'level0'..'level4'
        Returns:
            dense_3d: [B, out_ch, H, W, D] — dense with height-dependent features
        """
        target_size = lidar_feats['level0'].shape[2:]  # (H, W, D)
        # Trilinear upsample: preserves height structure from 4/2 Z bins
        up3 = F.interpolate(lidar_feats['level3'], size=target_size,
                           mode='trilinear', align_corners=False)
        up4 = F.interpolate(lidar_feats['level4'], size=target_size,
                           mode='trilinear', align_corners=False)
        return self.fuse(torch.cat([up3, up4], dim=1))


class ResidualBlock3DV3(nn.Module):
    """3D Residual Block with dual conditioning:
    1. Sparse FiLM: h * (1 + scale) + shift from sparse LSK3DNet features
       - High-quality semantic modulation at observed voxels
       - Identity at unobserved voxels (harmless)
    2. Dense additive: h + proj(dense_cond) from internal BEV completion
       - Spatial layout guidance at ALL voxels
       - Prevents collapse to trivial solutions (S6/S8 failure mode)

    With fuse_time_cond=True (DiffSSC-style, default for S13/S14):
    - Concatenates sparse condition + timestep BEFORE producing gate weight
    - gate = MLP(cat(cond, time)) → per-voxel, per-timestep multiplicative gate
    - Mirrors DiffSSC minkunet.py:425-431 exactly
    - Uses LeakyReLU(0.1) matching DiffSSC
    """

    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int,
                 sparse_cond_channels: int, dense_cond_channels: int,
                 num_groups: int = 8, no_sparse_film: bool = False,
                 fuse_time_cond: bool = False):
        super().__init__()
        self.no_sparse_film = no_sparse_film
        self.fuse_time_cond = fuse_time_cond

        # First conv
        self.norm1 = nn.GroupNorm(min(num_groups, in_channels), in_channels)
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)

        if fuse_time_cond and not no_sparse_film:
            # DiffSSC-style: concat sparse_condition + time → per-voxel gate
            mid_ch = min(out_channels, 64)
            self.cond_proj = nn.Conv3d(sparse_cond_channels, mid_ch, 3, padding=1)
            self.time_proj = nn.Sequential(
                nn.Linear(time_emb_dim, mid_ch),
                nn.LeakyReLU(0.1),
                nn.Linear(mid_ch, mid_ch),
            )
            self.fused_gate = nn.Sequential(
                nn.Linear(mid_ch * 2, mid_ch),
                nn.LeakyReLU(0.1),
                nn.Linear(mid_ch, out_channels),
            )
            # Dense additive (our addition — DiffSSC doesn't need this)
            self.dense_proj = nn.Conv3d(dense_cond_channels, out_channels, kernel_size=1)
            # Second conv (no separate time_fc — time is already fused)
            self.norm2 = nn.GroupNorm(min(num_groups, out_channels), out_channels)
            self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        else:
            # S11/S12 backward compat: separate FiLM + time FiLM
            if not no_sparse_film:
                self.cond_film = nn.Conv3d(sparse_cond_channels, out_channels * 2,
                                           kernel_size=3, padding=1)
            self.dense_proj = nn.Conv3d(dense_cond_channels, out_channels, kernel_size=1)
            self.norm2 = nn.GroupNorm(min(num_groups, out_channels), out_channels)
            self.time_fc = nn.Linear(time_emb_dim, out_channels * 2)
            self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)

        # Skip connection
        self.skip = nn.Conv3d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor,
                sparse_cond: torch.Tensor, dense_cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C_in, H, W, D] voxel features
            t_emb: [B, time_emb_dim] timestep embedding
            sparse_cond: [B, sparse_cond_ch, H, W, D] from SparseLiDAREncoder (~99% zeros at level0)
            dense_cond: [B, dense_cond_ch, H, W, D] from InternalBEVCompletion (dense everywhere)
        """
        h = F.silu(self.norm1(x))
        h = self.conv1(h)

        if self.fuse_time_cond and not self.no_sparse_film:
            # STEP 1: Fused sparse+time gate (DiffSSC core mechanism)
            cond_feat = self.cond_proj(sparse_cond)  # [B, mid, H, W, D]
            time_feat = self.time_proj(t_emb)  # [B, mid]
            time_feat = time_feat[:, :, None, None, None].expand_as(cond_feat)
            # Concat along channel dim, apply per-voxel MLP via permute trick
            fused = torch.cat([cond_feat, time_feat], dim=1)  # [B, 2*mid, H, W, D]
            gate = self.fused_gate(fused.permute(0, 2, 3, 4, 1)).permute(0, 4, 1, 2, 3)
            h = h * gate  # multiplicative (DiffSSC x*w)

            # STEP 2: Dense additive (spatial layout everywhere)
            h = h + self.dense_proj(dense_cond)

            # STEP 3: Second conv (no separate time FiLM — already fused)
            h = F.silu(self.norm2(h))
            h = self.conv2(h)
        else:
            # S11/S12 path: separate sparse FiLM → dense add → time FiLM
            if not self.no_sparse_film:
                scale_shift = self.cond_film(sparse_cond)
                scale, shift = scale_shift.chunk(2, dim=1)
                h = h * (1 + scale) + shift

            h = h + self.dense_proj(dense_cond)

            h = self.norm2(h)
            wb = self.time_fc(t_emb)
            w, b = wb.chunk(2, dim=-1)
            h = w[:, :, None, None, None] * h + b[:, :, None, None, None]
            h = F.silu(h)
            h = self.conv2(h)

        return h + self.skip(x)


class SceneCompletionUNetV3(nn.Module):
    """SSC UNet V3: Internal BEV Completion + Sparse 3D FiLM.

    Key differences from V2:
    - V2 only had sparse FiLM conditioning → 99% of voxels get identity → collapse
    - V3 adds InternalBEVCompletion: generates DENSE conditioning from sparse features
    - Dual conditioning: sparse FiLM (accurate at obs) + dense additive (layout everywhere)

    Key differences from V1 (scene_unet_sparse.py):
    - V1 uses EXTERNAL BEV as cascade input → train/test distribution gap
    - V3 generates internal BEV from LSK3DNet features → no gap
    - V3 uses FiLM for sparse cond (V1 uses additive for both BEV and LiDAR)
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
        dense_bev_channels: int = 32,
        bev_completion_layers: int = 5,
        num_groups: int = 8,
        aux_bev: bool = True,
        no_sparse_film: bool = False,
        dense_mode: str = 'internal_bev',
        cfg_drop_prob: float = 0.0,
        in_channels: int | None = None,
        out_channels: int | None = None,
        fuse_time_cond: bool = False,
        cond_3d_channels: int = 0,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.base_channels = base_channels
        self.aux_bev_enabled = aux_bev
        self.dense_mode = dense_mode
        self.cfg_drop_prob = cfg_drop_prob
        self.dense_bev_channels = dense_bev_channels
        self.cond_3d_channels = cond_3d_channels

        # in_channels defaults to num_classes (20 for discrete, 21 for continuous, 22 for factored)
        if in_channels is None:
            in_channels = num_classes
        if out_channels is None:
            out_channels = in_channels

        channels = [base_channels * m for m in channel_mult]
        cond_ch = lidar_out_channels  # Sparse conditioning channels

        # Sparse LiDAR encoder (reused from V1/V2)
        self.lidar_encoder = SparseLiDAREncoder(
            in_channels=lidar_in_channels,
            base_channels=lidar_base_channels,
            out_channels=lidar_out_channels,
        )

        # Dense conditioning: choose between InternalBEV (V3.0) and CoarseToFine (V3.1)
        if dense_mode == 'internal_bev':
            self.internal_bev = InternalBEVCompletion(
                in_channels=lidar_out_channels,
                bev_channels=dense_bev_channels,
                num_conv_layers=bev_completion_layers,
            )
        elif dense_mode == 'coarse2fine':
            self.coarse_to_fine = CoarseToFine3DFusion(
                in_channels=lidar_out_channels,
                out_channels=dense_bev_channels,
            )
        else:
            raise ValueError(f"Unknown dense_mode: {dense_mode}")

        # Voxel embedding: processes [x_t, cond_3d] if cond_3d_channels > 0
        self.voxel_embed = nn.Conv3d(in_channels + cond_3d_channels, base_channels, 3, padding=1)

        # Time embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(base_channels, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        # Encoder with ResidualBlock3DV3 (dual conditioning)
        self.enc0 = ResidualBlock3DV3(channels[0], channels[0], time_emb_dim,
                                       cond_ch, dense_bev_channels, num_groups, no_sparse_film, fuse_time_cond)
        self.down0 = nn.Conv3d(channels[0], channels[1], 3, stride=2, padding=1)

        self.enc1 = ResidualBlock3DV3(channels[1], channels[1], time_emb_dim,
                                       cond_ch, dense_bev_channels, num_groups, no_sparse_film, fuse_time_cond)
        self.down1 = nn.Conv3d(channels[1], channels[2], 3, stride=2, padding=1)

        self.enc2 = ResidualBlock3DV3(channels[2], channels[2], time_emb_dim,
                                       cond_ch, dense_bev_channels, num_groups, no_sparse_film, fuse_time_cond)
        self.down2 = nn.Conv3d(channels[2], channels[3], 3, stride=2, padding=1)

        self.enc3 = ResidualBlock3DV3(channels[3], channels[3], time_emb_dim,
                                       cond_ch, dense_bev_channels, num_groups, no_sparse_film, fuse_time_cond)
        self.down3 = nn.Conv3d(channels[3], channels[3], 3, stride=2, padding=1)

        # Bottleneck
        self.mid0 = ResidualBlock3DV3(channels[3], channels[3], time_emb_dim,
                                       cond_ch, dense_bev_channels, num_groups, no_sparse_film, fuse_time_cond)
        self.mid1 = ResidualBlock3DV3(channels[3], channels[3], time_emb_dim,
                                       cond_ch, dense_bev_channels, num_groups, no_sparse_film, fuse_time_cond)

        # Decoder
        self.up3 = nn.ConvTranspose3d(channels[3], channels[3], 4, stride=2, padding=1)
        self.dec3 = ResidualBlock3DV3(channels[3] * 2, channels[3], time_emb_dim,
                                       cond_ch, dense_bev_channels, num_groups, no_sparse_film, fuse_time_cond)

        self.up2 = nn.ConvTranspose3d(channels[3], channels[2], 4, stride=2, padding=1)
        self.dec2 = ResidualBlock3DV3(channels[2] * 2, channels[2], time_emb_dim,
                                       cond_ch, dense_bev_channels, num_groups, no_sparse_film, fuse_time_cond)

        self.up1 = nn.ConvTranspose3d(channels[2], channels[1], 4, stride=2, padding=1)
        self.dec1 = ResidualBlock3DV3(channels[1] * 2, channels[1], time_emb_dim,
                                       cond_ch, dense_bev_channels, num_groups, no_sparse_film, fuse_time_cond)

        self.up0 = nn.ConvTranspose3d(channels[1], channels[0], 4, stride=2, padding=1)
        self.dec0 = ResidualBlock3DV3(channels[0] * 2, channels[0], time_emb_dim,
                                       cond_ch, dense_bev_channels, num_groups, no_sparse_film, fuse_time_cond)

        # Output
        self.out_conv = nn.Sequential(
            nn.GroupNorm(num_groups, channels[0]),
            nn.SiLU(),
            nn.Conv3d(channels[0], out_channels, 3, padding=1),
        )

        # Multi-scale auxiliary BEV head (training only, from V2)
        if aux_bev:
            self.aux_bev_head = MultiScaleAuxBEVHead(
                encoder_channels=channels,
                num_classes=num_classes,
            )

    def forward(self, x_t: torch.Tensor, t: torch.Tensor,
                lidar: torch.Tensor, cond_3d: torch.Tensor | None = None,
                **kwargs) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Args:
            x_t: [B, 20, H, W, D] one-hot noisy voxels
            t: [B] timesteps
            lidar: [B, 20, H, W, D] LSK3DNet soft probs
            cond_3d: [B, 20, H, W, D] optional 3D conditioning (e.g. LSK3DNet probs)
        Returns:
            ssc_logits: [B, 20, H, W, D] class logits
            aux_bev: [B, 20, H_bev, W_bev] BEV logits or None (inference)
        """
        force_uncond = kwargs.get('force_uncond', False)
        cfg_dropped = False

        if force_uncond:
            # CFG unconditional pass: bypass sparse encoder (crashes on empty input)
            # Create zero features at all levels directly
            B = x_t.shape[0]
            H, W, D = x_t.shape[2], x_t.shape[3], x_t.shape[4]
            cond_ch = self.lidar_encoder.out_channels
            lidar_feats = {
                'level0': torch.zeros(B, cond_ch, H, W, D, device=x_t.device),
                'level1': torch.zeros(B, cond_ch, H//2, W//2, D//2, device=x_t.device),
                'level2': torch.zeros(B, cond_ch, H//4, W//4, D//4, device=x_t.device),
                'level3': torch.zeros(B, cond_ch, H//8, W//8, D//8, device=x_t.device),
                'level4': torch.zeros(B, cond_ch, H//16, W//16, D//16, device=x_t.device),
            }
            dense_3d_full = torch.zeros(B, self.dense_bev_channels, H, W, D, device=x_t.device)
        else:
            # 1. Extract sparse multi-scale features
            lidar_feats = self.lidar_encoder(lidar)

            # 2. Generate DENSE conditioning
            if self.dense_mode == 'coarse2fine':
                dense_3d_full = self.coarse_to_fine(lidar_feats)
            else:
                dense_3d_full = self.internal_bev(lidar_feats['level0'])

            # 3. CFG: randomly drop ALL conditioning during training
            cfg_dropped = False
            if self.training and self.cfg_drop_prob > 0:
                if torch.rand(1).item() < self.cfg_drop_prob:
                    lidar_feats = {k: torch.zeros_like(v) for k, v in lidar_feats.items()}
                    dense_3d_full = torch.zeros_like(dense_3d_full)
                    cfg_dropped = True

        # 4. Pre-compute dense levels via avg_pool3d at each spatial scale
        dense_levels = {
            'level0': dense_3d_full,
            'level1': F.avg_pool3d(dense_3d_full, 2),
            'level2': F.avg_pool3d(dense_3d_full, 4),
            'level3': F.avg_pool3d(dense_3d_full, 8),
            'level4': F.avg_pool3d(dense_3d_full, 16),
        }

        # 5. Prepare model input: [x_t, cond_3d] if cond_3d conditioning enabled
        if self.cond_3d_channels > 0:
            B_c, _, H_c, W_c, D_c = x_t.shape
            if cond_3d is None or force_uncond or cfg_dropped:
                cond_3d_input = torch.zeros(B_c, self.cond_3d_channels, H_c, W_c, D_c, device=x_t.device)
            else:
                cond_3d_input = cond_3d
            model_input = torch.cat([x_t, cond_3d_input], dim=1)
        else:
            model_input = x_t

        # 6. Embed noisy voxels (+ conditioning channels)
        x = self.voxel_embed(model_input)

        # 6. Time embedding
        t_emb = timestep_embedding(t, self.base_channels)
        t_emb = self.time_mlp(t_emb)

        # 7. Encoder with dual conditioning (sparse FiLM + dense additive)
        e0 = self.enc0(x, t_emb, lidar_feats['level0'], dense_levels['level0'])
        x = self.down0(e0)

        e1 = self.enc1(x, t_emb, lidar_feats['level1'], dense_levels['level1'])
        x = self.down1(e1)

        e2 = self.enc2(x, t_emb, lidar_feats['level2'], dense_levels['level2'])
        x = self.down2(e2)

        e3 = self.enc3(x, t_emb, lidar_feats['level3'], dense_levels['level3'])
        x = self.down3(e3)

        # 8. Bottleneck
        x = self.mid0(x, t_emb, lidar_feats['level4'], dense_levels['level4'])
        x = self.mid1(x, t_emb, lidar_feats['level4'], dense_levels['level4'])

        # 9. Multi-scale auxiliary BEV (training only)
        aux_bev = None
        if self.training and self.aux_bev_enabled:
            aux_bev = self.aux_bev_head([e0, e1, e2, e3])

        # 10. Decoder with dual conditioning (matching levels)
        x = self.up3(x)
        x = torch.cat([x, e3], dim=1)
        x = self.dec3(x, t_emb, lidar_feats['level3'], dense_levels['level3'])

        x = self.up2(x)
        x = torch.cat([x, e2], dim=1)
        x = self.dec2(x, t_emb, lidar_feats['level2'], dense_levels['level2'])

        x = self.up1(x)
        x = torch.cat([x, e1], dim=1)
        x = self.dec1(x, t_emb, lidar_feats['level1'], dense_levels['level1'])

        x = self.up0(x)
        x = torch.cat([x, e0], dim=1)
        x = self.dec0(x, t_emb, lidar_feats['level0'], dense_levels['level0'])

        return self.out_conv(x), aux_bev


class V3ModelWrapper(nn.Module):
    """Wraps SceneCompletionUNetV3 to match MultinomialDiffusion3D's expected interface.

    MultinomialDiffusion3D calls: model(x_t, t, bev, lidar, lifted_features=None)
    V3 UNet expects:             model(x_t, t, lidar)
    V3 UNet returns:             (ssc_logits, aux_bev)
    Wrapper returns:             ssc_logits (stores aux_bev as attribute)

    Used by: S9-S12 (discrete diffusion on V3/V3.1 UNet)
    """

    def __init__(self, model: SceneCompletionUNetV3):
        super().__init__()
        self.model = model
        self.last_aux_bev = None

    def forward(self, x_t: torch.Tensor, t: torch.Tensor,
                bev: torch.Tensor, lidar: torch.Tensor,
                lifted_features: torch.Tensor | None = None,
                **kwargs) -> torch.Tensor:
        # bev and lifted_features are IGNORED — V3 generates its own internal BEV
        ssc_logits, aux_bev = self.model(x_t, t, lidar, **kwargs)
        self.last_aux_bev = aux_bev
        return ssc_logits


class V4ModelWrapper(nn.Module):
    """Wraps SceneCompletionUNetV3 for GaussianDiffusion3D / FactoredDiffusion3D.

    Same as V3ModelWrapper but named V4 to distinguish the factored/continuous
    diffusion experiments (S13/S14). Interface is identical — both diffusion types
    call model(x_t, t, bev, lidar, lifted_features=None) and the wrapper ignores
    bev and lifted_features.

    Used by: S13 (factored discrete), S14 (continuous Gaussian), S17+ (VE logit)
    """

    def __init__(self, model: SceneCompletionUNetV3):
        super().__init__()
        self.model = model
        self.last_aux_bev = None

    def forward(self, x_t: torch.Tensor, t: torch.Tensor,
                bev: torch.Tensor, lidar: torch.Tensor,
                lifted_features: torch.Tensor | None = None,
                cond_3d: torch.Tensor | None = None,
                **kwargs) -> torch.Tensor:
        ssc_logits, aux_bev = self.model(x_t, t, lidar, cond_3d=cond_3d, **kwargs)
        self.last_aux_bev = aux_bev
        return ssc_logits
