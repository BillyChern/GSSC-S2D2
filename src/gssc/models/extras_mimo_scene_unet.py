"""
MIMO-Aware 3D U-Net for Scene Completion (PaSCo-style)

This implements the TRUE PaSCo MIMO architecture:
1. Channel-concatenated inputs: [B, N*C, H, W, D]
2. SHARED encoder and decoder backbone
3. N SEPARATE completion heads (one per subnet)
4. Single forward pass produces N outputs

Reference: PaSCo (CVPR 2024)
- decoder_v3.py lines 130-136: N separate completion heads
- net_panoptic_sparse.py line 127: in_channels = f * n_infers
- augmenter.py merge(): Channel concatenation

Architecture:
- Input: [B, N*C, H, W, D] channel-concatenated LiDAR from N samples
- BEV: [B, N, H, W] stacked BEV maps from N samples
- Encoder: SHARED, processes merged channels
- Decoder: SHARED backbone + N SEPARATE completion heads
- Output: List of N logits tensors, one per head
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .sparse_lidar_encoder import SparseLiDAREncoder, SparseLiDAREncoderLite


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


class ResidualBlock3DMIMO(nn.Module):
    """
    3D Residual Block with FiLM time conditioning for MIMO.

    Same as ResidualBlock3DSparse but handles N*C input channels.
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

        # First conv
        self.norm1 = nn.GroupNorm(min(num_groups, in_channels), in_channels)
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)

        # Conditioning projections
        self.bev_proj = nn.Conv3d(cond_channels, out_channels, kernel_size=3, padding=1)
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
        wb = self.time_fc(t_emb)
        w, b = wb.chunk(2, dim=-1)
        w = w[:, :, None, None, None]
        b = b[:, :, None, None, None]
        h = w * h + b

        h = F.silu(h)
        h = self.conv2(h)

        return h + self.skip(x)


class MIMOSceneCompletionUNet(nn.Module):
    """
    PaSCo-style MIMO 3D U-Net for Scene Completion.

    This implements the TRUE PaSCo architecture with:
    1. Channel-concatenated inputs (N samples merged along channel dim)
    2. SHARED encoder and decoder backbone
    3. N SEPARATE completion heads

    From PaSCo decoder_v3.py lines 130-136:
    ```python
    self.completion_heads = nn.ModuleDict()
    for i_head in range(n_heads):
        self.completion_heads[str(i_head)] = nn.Sequential(
            ME.MinkowskiConvolution(out_channels, compl_head_dim, kernel_size=1, ...),
        )
    ```

    Architecture for SemanticKITTI (256×256×32) with n_subnets=3:
    - Input LiDAR: [B, 3*1, H, W, D] = [B, 3, H, W, D]
    - Input BEV: [B, 3, H, W] (stacked)
    - Encoder: processes merged channels
    - Decoder: shared backbone
    - Heads: 3 separate Conv3d(decoder_channels → num_classes)
    - Output: List of 3 tensors [B, num_classes, H, W, D]
    """

    def __init__(
        self,
        num_classes: int = 20,
        n_subnets: int = 3,  # PaSCo paper default
        base_channels: int = 32,
        channel_mult: tuple[int, ...] = (1, 2, 4, 8),
        time_emb_dim: int = 128,
        lidar_base_channels: int = 16,
        lidar_out_channels: int = 32,
        num_groups: int = 8,
        use_waffleiron: bool = True,  # B6: WaffleIron pretrained features (COMPULSORY like PaSCo)
        waffleiron_dim: int = 64,  # WaffleIron output dimension
    ):
        super().__init__()

        self.num_classes = num_classes
        self.n_subnets = n_subnets
        self.base_channels = base_channels
        self.use_waffleiron = use_waffleiron

        channels = [base_channels * m for m in channel_mult]

        # Sparse LiDAR encoder - processes each subnet's LiDAR independently
        # then merges the multi-scale features
        # Input: [B, N, H, W, D] where N=n_subnets (each channel is a binary voxel grid)
        # Each subnet's LiDAR is processed through the same encoder, features are summed
        self.lidar_encoder = SparseLiDAREncoder(
            in_channels=1,  # Each subnet's LiDAR is single-channel
            base_channels=lidar_base_channels,
            out_channels=lidar_out_channels,
        )

        # Voxel embedding - takes N*num_classes channels (one-hot from each subnet)
        self.voxel_embed = nn.Conv3d(
            num_classes * n_subnets,  # N copies of one-hot
            base_channels,
            kernel_size=3,
            padding=1
        )

        # BEV embedding - takes N*num_classes channels
        self.bev_embed = nn.Conv3d(
            num_classes * n_subnets,
            lidar_out_channels,
            kernel_size=3,
            padding=1
        )

        # Time embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(base_channels, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        # B6: WaffleIron pretrained feature embedding (PaSCo-style) - COMPULSORY
        # WaffleIron features: [B*N, 64, H, W] -> project to lidar_out_channels
        # Following PaSCo: WaffleIron semantic features are a critical component
        self.waffleiron_embed = nn.Sequential(
            nn.Conv2d(waffleiron_dim * n_subnets, lidar_out_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(lidar_out_channels * 2),
            nn.SiLU(),
            nn.Conv2d(lidar_out_channels * 2, lidar_out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(lidar_out_channels),
            nn.SiLU(),
        )

        # ============ SHARED Encoder ============
        self.enc0 = ResidualBlock3DMIMO(channels[0], channels[0], time_emb_dim, lidar_out_channels, num_groups)
        self.down0 = nn.Conv3d(channels[0], channels[1], kernel_size=3, stride=2, padding=1)

        self.enc1 = ResidualBlock3DMIMO(channels[1], channels[1], time_emb_dim, lidar_out_channels, num_groups)
        self.down1 = nn.Conv3d(channels[1], channels[2], kernel_size=3, stride=2, padding=1)

        self.enc2 = ResidualBlock3DMIMO(channels[2], channels[2], time_emb_dim, lidar_out_channels, num_groups)
        self.down2 = nn.Conv3d(channels[2], channels[3], kernel_size=3, stride=2, padding=1)

        self.enc3 = ResidualBlock3DMIMO(channels[3], channels[3], time_emb_dim, lidar_out_channels, num_groups)
        self.down3 = nn.Conv3d(channels[3], channels[3], kernel_size=3, stride=2, padding=1)

        # ============ SHARED Bottleneck ============
        self.mid0 = ResidualBlock3DMIMO(channels[3], channels[3], time_emb_dim, lidar_out_channels, num_groups)
        self.mid1 = ResidualBlock3DMIMO(channels[3], channels[3], time_emb_dim, lidar_out_channels, num_groups)

        # ============ SHARED Decoder Backbone ============
        self.up3 = nn.ConvTranspose3d(channels[3], channels[3], kernel_size=4, stride=2, padding=1)
        self.dec3 = ResidualBlock3DMIMO(channels[3] * 2, channels[3], time_emb_dim, lidar_out_channels, num_groups)

        self.up2 = nn.ConvTranspose3d(channels[3], channels[2], kernel_size=4, stride=2, padding=1)
        self.dec2 = ResidualBlock3DMIMO(channels[2] * 2, channels[2], time_emb_dim, lidar_out_channels, num_groups)

        self.up1 = nn.ConvTranspose3d(channels[2], channels[1], kernel_size=4, stride=2, padding=1)
        self.dec1 = ResidualBlock3DMIMO(channels[1] * 2, channels[1], time_emb_dim, lidar_out_channels, num_groups)

        self.up0 = nn.ConvTranspose3d(channels[1], channels[0], kernel_size=4, stride=2, padding=1)
        self.dec0 = ResidualBlock3DMIMO(channels[0] * 2, channels[0], time_emb_dim, lidar_out_channels, num_groups)

        # ============ N SEPARATE Completion Heads ============
        # Following PaSCo decoder_v3.py lines 130-136
        self.completion_heads = nn.ModuleDict()
        for i_head in range(n_subnets):
            self.completion_heads[str(i_head)] = nn.Sequential(
                nn.GroupNorm(num_groups, channels[0]),
                nn.SiLU(),
                nn.Conv3d(channels[0], num_classes, kernel_size=3, padding=1),
            )

        # ============ MULTI-SCALE AUXILIARY HEADS (PaSCo-style) ============
        # Following PaSCo decoder_v3.py lines 258-283:
        # Auxiliary completion heads at scales [1, 2, 4] for multi-scale supervision
        # Scale 4 (1:4): 64×64×8 for 256×256×32 input - at dec2 output (channels[2])
        # Scale 2 (1:2): 128×128×16 for 256×256×32 input - at dec1 output (channels[1])
        # Scale 1 (1:1): 256×256×32 - main completion heads (already defined above)
        self.use_multiscale_supervision = False

        # Scale 4 (1:4) auxiliary heads - after dec2
        self.aux_heads_scale4 = nn.ModuleDict()
        for i_head in range(n_subnets):
            self.aux_heads_scale4[str(i_head)] = nn.Sequential(
                nn.GroupNorm(min(num_groups, channels[2]), channels[2]),
                nn.SiLU(),
                nn.Conv3d(channels[2], num_classes, kernel_size=3, padding=1),
            )

        # Scale 2 (1:2) auxiliary heads - after dec1
        self.aux_heads_scale2 = nn.ModuleDict()
        for i_head in range(n_subnets):
            self.aux_heads_scale2[str(i_head)] = nn.Sequential(
                nn.GroupNorm(min(num_groups, channels[1]), channels[1]),
                nn.SiLU(),
                nn.Conv3d(channels[1], num_classes, kernel_size=3, padding=1),
            )

        # Dense 3D bottleneck support (S5)
        self.use_dense3d_bottleneck = False
        self.dense3d_bottleneck = None

        # ============ PaSCo-style Feature Projection Layers ============
        # After concatenating N LiDAR features, project back to single-channel
        # This follows PaSCo's channel concatenation approach (augmenter.py)
        # Concat: [B, N*C, H, W, D] → Project: [B, C, H, W, D]
        self.lidar_proj_level0 = nn.Conv3d(n_subnets * lidar_out_channels, lidar_out_channels, kernel_size=1)
        self.lidar_proj_level1 = nn.Conv3d(n_subnets * lidar_out_channels, lidar_out_channels, kernel_size=1)
        self.lidar_proj_level2 = nn.Conv3d(n_subnets * lidar_out_channels, lidar_out_channels, kernel_size=1)
        self.lidar_proj_level3 = nn.Conv3d(n_subnets * lidar_out_channels, lidar_out_channels, kernel_size=1)
        self.lidar_proj_level4 = nn.Conv3d(n_subnets * lidar_out_channels, lidar_out_channels, kernel_size=1)

        print("[MIMOSceneCompletionUNet] Created with:")
        print(f"  - n_subnets: {n_subnets} (N separate completion heads)")
        print(f"  - LiDAR input channels: {n_subnets} (channel-concatenated)")
        print(f"  - BEV input channels: {num_classes * n_subnets}")
        print(f"  - Voxel input channels: {num_classes * n_subnets}")
        print("  - Feature merging: CHANNEL CONCATENATION + PROJECTION (PaSCo-style)")
        print(f"  - WaffleIron features: {'ENABLED' if use_waffleiron else 'DISABLED'}")

    def enable_dense3d_bottleneck(self, dropout: float = 0.1):
        """Enable PaSCo's SPCDense3Dv2 at the bottleneck."""
        from gssc.models.dense_3d_cnn import SPCDense3Dv2

        bottleneck_channels = self.base_channels * 8

        self.dense3d_bottleneck = nn.Sequential(
            SPCDense3Dv2(init_size=bottleneck_channels),
            nn.Dropout3d(dropout),
        )
        self.use_dense3d_bottleneck = True

        device = next(self.parameters()).device
        self.dense3d_bottleneck = self.dense3d_bottleneck.to(device)
        print("[MIMOSceneCompletionUNet] Enabled SPCDense3Dv2 bottleneck")

    def enable_multiscale_supervision(self):
        """
        Enable PaSCo-style multi-scale auxiliary supervision.

        When enabled, the forward pass will return auxiliary outputs at scales 4 (1:4)
        and 2 (1:2) in addition to the main 1:1 scale outputs. These auxiliary outputs
        are used for multi-scale supervision loss during training.

        Reference: PaSCo decoder_v3.py lines 258-283
        """
        self.use_multiscale_supervision = True
        print("[MIMOSceneCompletionUNet] Enabled multi-scale supervision at scales [1, 2, 4]")

    def forward(
        self,
        x_t: torch.Tensor,  # [B, N*num_classes, H, W, D] channel-concat one-hot noisy voxels
        t: torch.Tensor,    # [B] timesteps
        bev: torch.Tensor,  # [B, N, H, W] stacked BEV maps OR [B, N*C, H, W] one-hot
        lidar: torch.Tensor,  # [B, N*1, H, W, D] channel-concat binary voxels
        waffleiron: torch.Tensor = None,  # [B, N, 64, H, W] WaffleIron features (optional, uses zeros if None)
    ) -> dict[str, list[torch.Tensor]]:
        """
        Forward pass with PaSCo-style MIMO.

        SINGLE forward pass → N outputs from N completion heads.

        Args:
            x_t: [B, N*num_classes, H, W, D] concatenated one-hot noisy voxels
            t: [B] diffusion timesteps
            bev: [B, N, H, W] stacked BEV labels (will be one-hot encoded) OR
                 [B, N*num_classes, H, W] already one-hot encoded
            lidar: [B, N*1, H, W, D] concatenated binary LiDAR voxels
            waffleiron: [B, N, 64, H, W] WaffleIron pretrained BEV features (COMPULSORY)
                       Following PaSCo: WaffleIron semantic features are critical for SSC

        Returns:
            Dict with:
            - 'head_outputs': List of N tensors [B, num_classes, H, W, D]
            - 'ensemble': [B, num_classes, H, W, D] averaged output
        """
        B = x_t.shape[0]
        H, W, D = x_t.shape[2], x_t.shape[3], x_t.shape[4]

        # Handle BEV input format
        # If BEV is [B, N, H, W] (stacked labels), convert to one-hot
        if bev.dim() == 4 and bev.shape[1] == self.n_subnets:
            # Convert each BEV to one-hot and concatenate
            bev_onehot_list = []
            for i in range(self.n_subnets):
                bev_i = bev[:, i]  # [B, H, W]
                bev_onehot = F.one_hot(bev_i.long(), self.num_classes).float()  # [B, H, W, C]
                bev_onehot = bev_onehot.permute(0, 3, 1, 2)  # [B, C, H, W]
                bev_onehot_list.append(bev_onehot)
            bev = torch.cat(bev_onehot_list, dim=1)  # [B, N*C, H, W]

        # ============ PaSCo-style: CHANNEL CONCATENATION ============
        # Extract multi-scale LiDAR features from each subnet independently
        # Then CONCATENATE along channel dimension (NOT sum!)
        # lidar: [B, N, H, W, D] where N = n_subnets
        #
        # Reference: PaSCo augmenter.py merge()
        # in_feat_dense = torch.cat([...], dim=0)  # Channel concatenation!

        lidar_feats_list = []  # List of N feature dicts
        for i in range(self.n_subnets):
            lidar_i = lidar[:, i:i+1, :, :, :]  # [B, 1, H, W, D]
            feats_i = self.lidar_encoder(lidar_i)  # Dict with level0-level4
            lidar_feats_list.append(feats_i)

        # Concatenate features along channel dim, then project to original channels
        # This preserves N different LiDAR viewpoints while matching expected dims
        lidar_feats = {}
        for level in ['level0', 'level1', 'level2', 'level3', 'level4']:
            # Concatenate all N features: [B, C, H, W, D] * N → [B, N*C, H, W, D]
            concat_feats = torch.cat([feats[level] for feats in lidar_feats_list], dim=1)
            # Project back to original channels: [B, N*C, H, W, D] → [B, C, H, W, D]
            proj_layer = getattr(self, f'lidar_proj_{level}')
            lidar_feats[level] = proj_layer(concat_feats)

        # Expand BEV along Z and embed
        bev_3d = bev.unsqueeze(-1).expand(-1, -1, -1, -1, D)
        bev_emb0 = self.bev_embed(bev_3d)  # [B, lidar_out_channels, H, W, D]

        # B6: WaffleIron pretrained features (optional - uses zeros if not provided)
        # waffleiron: [B, N, 64, H, W] -> reshape to [B, N*64, H, W]
        if waffleiron is not None:
            B_wf = waffleiron.shape[0]
            # Use reshape instead of view - tensor may not be contiguous after expand() in sample_mimo
            wf_concat = waffleiron.reshape(B_wf, -1, waffleiron.shape[3], waffleiron.shape[4])  # [B, N*64, H, W]
            wf_emb = self.waffleiron_embed(wf_concat)  # [B, lidar_out_channels, H, W]
            # Expand along Z and add to BEV embedding
            wf_emb_3d = wf_emb.unsqueeze(-1).expand(-1, -1, -1, -1, D)  # [B, lidar_out_channels, H, W, D]
            bev_emb0 = bev_emb0 + wf_emb_3d

        # Embed voxels
        x = self.voxel_embed(x_t)

        # Time embedding
        t_emb = timestep_embedding(t, self.base_channels)
        t_emb = self.time_mlp(t_emb)

        # ============ SHARED Encoder ============
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

        # ============ SHARED Bottleneck ============
        x = self.mid0(x, t_emb, bev_emb4, lidar_feats['level4'])
        x = self.mid1(x, t_emb, bev_emb4, lidar_feats['level4'])

        # Apply dense 3D bottleneck if enabled (S5)
        if self.use_dense3d_bottleneck and self.dense3d_bottleneck is not None:
            x = self.dense3d_bottleneck(x)

        # ============ SHARED Decoder ============
        x = self.up3(x)
        x = torch.cat([x, e3], dim=1)
        x = self.dec3(x, t_emb, bev_emb3, lidar_feats['level3'])

        x = self.up2(x)
        x = torch.cat([x, e2], dim=1)
        x = self.dec2(x, t_emb, bev_emb2, lidar_feats['level2'])

        # ============ Scale 4 (1:4) Auxiliary Outputs ============
        # Following PaSCo decoder_v3.py: auxiliary heads at 64×64×8 resolution
        aux_outputs_scale4 = None
        if self.use_multiscale_supervision:
            aux_outputs_scale4 = []
            for i_head in range(self.n_subnets):
                aux_logits = self.aux_heads_scale4[str(i_head)](x)
                aux_outputs_scale4.append(aux_logits)

        x = self.up1(x)
        x = torch.cat([x, e1], dim=1)
        x = self.dec1(x, t_emb, bev_emb1, lidar_feats['level1'])

        # ============ Scale 2 (1:2) Auxiliary Outputs ============
        # Following PaSCo decoder_v3.py: auxiliary heads at 128×128×16 resolution
        aux_outputs_scale2 = None
        if self.use_multiscale_supervision:
            aux_outputs_scale2 = []
            for i_head in range(self.n_subnets):
                aux_logits = self.aux_heads_scale2[str(i_head)](x)
                aux_outputs_scale2.append(aux_logits)

        x = self.up0(x)
        x = torch.cat([x, e0], dim=1)
        x = self.dec0(x, t_emb, bev_emb0, lidar_feats['level0'])

        # ============ N SEPARATE Completion Heads (Scale 1, 1:1) ============
        head_outputs = []
        for i_head in range(self.n_subnets):
            logits = self.completion_heads[str(i_head)](x)  # [B, num_classes, H, W, D]
            head_outputs.append(logits)

        # Compute ensemble (mean of probabilities, then log back to logits)
        # Following PaSCo's ensemble_sem_compl()
        probs = [F.softmax(logits, dim=1) for logits in head_outputs]
        ensemble_probs = torch.stack(probs, dim=0).mean(dim=0)  # [B, C, H, W, D]
        ensemble_logits = torch.log(ensemble_probs + 1e-10)

        result = {
            'head_outputs': head_outputs,
            'ensemble': ensemble_logits,
            'probs': probs,
            'ensemble_probs': ensemble_probs,
        }

        # Add multi-scale auxiliary outputs if enabled
        if self.use_multiscale_supervision:
            result['aux_scale4'] = aux_outputs_scale4  # List of [B, C, H/4, W/4, D/4]
            result['aux_scale2'] = aux_outputs_scale2  # List of [B, C, H/2, W/2, D/2]

        return result


class MIMOSceneCompletionUNetLite(nn.Module):
    """
    Lighter version of MIMO U-Net for testing.

    Channels: 16 → 32 → 64 → 128 (instead of 32 → 64 → 128 → 256)
    """

    def __init__(
        self,
        num_classes: int = 20,
        n_subnets: int = 3,
        base_channels: int = 16,
        channel_mult: tuple[int, ...] = (1, 2, 4, 8),
        time_emb_dim: int = 64,
        lidar_base_channels: int = 8,
        lidar_out_channels: int = 16,
        num_groups: int = 4,
        use_waffleiron: bool = True,  # B6: WaffleIron pretrained features (COMPULSORY like PaSCo)
        waffleiron_dim: int = 64,  # WaffleIron output dimension
    ):
        super().__init__()

        self.num_classes = num_classes
        self.n_subnets = n_subnets
        self.base_channels = base_channels
        self.use_waffleiron = use_waffleiron

        channels = [base_channels * m for m in channel_mult]

        # Sparse LiDAR encoder - processes each subnet's LiDAR independently
        self.lidar_encoder = SparseLiDAREncoderLite(
            in_channels=1,  # Each subnet's LiDAR is single-channel
            base_channels=lidar_base_channels,
            out_channels=lidar_out_channels,
        )

        # Voxel and BEV embedding
        self.voxel_embed = nn.Conv3d(num_classes * n_subnets, base_channels, kernel_size=3, padding=1)
        self.bev_embed = nn.Conv3d(num_classes * n_subnets, lidar_out_channels, kernel_size=3, padding=1)

        self.time_mlp = nn.Sequential(
            nn.Linear(base_channels, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        # B6: WaffleIron pretrained feature embedding (PaSCo-style) - COMPULSORY
        self.waffleiron_embed = nn.Sequential(
            nn.Conv2d(waffleiron_dim * n_subnets, lidar_out_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(lidar_out_channels * 2),
            nn.SiLU(),
            nn.Conv2d(lidar_out_channels * 2, lidar_out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(lidar_out_channels),
            nn.SiLU(),
        )

        # Encoder
        self.enc0 = ResidualBlock3DMIMO(channels[0], channels[0], time_emb_dim, lidar_out_channels, num_groups)
        self.down0 = nn.Conv3d(channels[0], channels[1], kernel_size=3, stride=2, padding=1)

        self.enc1 = ResidualBlock3DMIMO(channels[1], channels[1], time_emb_dim, lidar_out_channels, num_groups)
        self.down1 = nn.Conv3d(channels[1], channels[2], kernel_size=3, stride=2, padding=1)

        self.enc2 = ResidualBlock3DMIMO(channels[2], channels[2], time_emb_dim, lidar_out_channels, num_groups)
        self.down2 = nn.Conv3d(channels[2], channels[3], kernel_size=3, stride=2, padding=1)

        self.enc3 = ResidualBlock3DMIMO(channels[3], channels[3], time_emb_dim, lidar_out_channels, num_groups)
        self.down3 = nn.Conv3d(channels[3], channels[3], kernel_size=3, stride=2, padding=1)

        # Bottleneck
        self.mid0 = ResidualBlock3DMIMO(channels[3], channels[3], time_emb_dim, lidar_out_channels, num_groups)
        self.mid1 = ResidualBlock3DMIMO(channels[3], channels[3], time_emb_dim, lidar_out_channels, num_groups)

        # Decoder
        self.up3 = nn.ConvTranspose3d(channels[3], channels[3], kernel_size=4, stride=2, padding=1)
        self.dec3 = ResidualBlock3DMIMO(channels[3] * 2, channels[3], time_emb_dim, lidar_out_channels, num_groups)

        self.up2 = nn.ConvTranspose3d(channels[3], channels[2], kernel_size=4, stride=2, padding=1)
        self.dec2 = ResidualBlock3DMIMO(channels[2] * 2, channels[2], time_emb_dim, lidar_out_channels, num_groups)

        self.up1 = nn.ConvTranspose3d(channels[2], channels[1], kernel_size=4, stride=2, padding=1)
        self.dec1 = ResidualBlock3DMIMO(channels[1] * 2, channels[1], time_emb_dim, lidar_out_channels, num_groups)

        self.up0 = nn.ConvTranspose3d(channels[1], channels[0], kernel_size=4, stride=2, padding=1)
        self.dec0 = ResidualBlock3DMIMO(channels[0] * 2, channels[0], time_emb_dim, lidar_out_channels, num_groups)

        # N separate completion heads
        self.completion_heads = nn.ModuleDict()
        for i_head in range(n_subnets):
            self.completion_heads[str(i_head)] = nn.Sequential(
                nn.GroupNorm(num_groups, channels[0]),
                nn.SiLU(),
                nn.Conv3d(channels[0], num_classes, kernel_size=3, padding=1),
            )

        self.use_dense3d_bottleneck = False
        self.dense3d_bottleneck = None

        # ============ PaSCo-style Feature Projection Layers ============
        self.lidar_proj_level0 = nn.Conv3d(n_subnets * lidar_out_channels, lidar_out_channels, kernel_size=1)
        self.lidar_proj_level1 = nn.Conv3d(n_subnets * lidar_out_channels, lidar_out_channels, kernel_size=1)
        self.lidar_proj_level2 = nn.Conv3d(n_subnets * lidar_out_channels, lidar_out_channels, kernel_size=1)
        self.lidar_proj_level3 = nn.Conv3d(n_subnets * lidar_out_channels, lidar_out_channels, kernel_size=1)
        self.lidar_proj_level4 = nn.Conv3d(n_subnets * lidar_out_channels, lidar_out_channels, kernel_size=1)

        print("[MIMOSceneCompletionUNetLite] Created with:")
        print(f"  - n_subnets: {n_subnets} (N separate completion heads)")
        print(f"  - LiDAR input channels: {n_subnets} (channel-concatenated)")
        print(f"  - BEV input channels: {num_classes * n_subnets}")
        print(f"  - Voxel input channels: {num_classes * n_subnets}")
        print("  - Feature merging: CHANNEL CONCATENATION + PROJECTION (PaSCo-style)")
        print(f"  - WaffleIron features: {'ENABLED' if use_waffleiron else 'DISABLED'}")

    def enable_dense3d_bottleneck(self, dropout: float = 0.1):
        """Enable PaSCo's SPCDense3Dv2 at the bottleneck."""
        from gssc.models.dense_3d_cnn import SPCDense3Dv2

        bottleneck_channels = self.base_channels * 8

        self.dense3d_bottleneck = nn.Sequential(
            SPCDense3Dv2(init_size=bottleneck_channels),
            nn.Dropout3d(dropout),
        )
        self.use_dense3d_bottleneck = True

        device = next(self.parameters()).device
        self.dense3d_bottleneck = self.dense3d_bottleneck.to(device)
        print("[MIMOSceneCompletionUNetLite] Enabled SPCDense3Dv2 bottleneck")

    def forward(self, x_t, t, bev, lidar, waffleiron: torch.Tensor = None):
        """Same interface as MIMOSceneCompletionUNet. WaffleIron is optional."""
        B = x_t.shape[0]
        H, W, D = x_t.shape[2], x_t.shape[3], x_t.shape[4]

        # Handle BEV input format
        if bev.dim() == 4 and bev.shape[1] == self.n_subnets:
            bev_onehot_list = []
            for i in range(self.n_subnets):
                bev_i = bev[:, i]
                bev_onehot = F.one_hot(bev_i.long(), self.num_classes).float()
                bev_onehot = bev_onehot.permute(0, 3, 1, 2)
                bev_onehot_list.append(bev_onehot)
            bev = torch.cat(bev_onehot_list, dim=1)

        # ============ PaSCo-style: CHANNEL CONCATENATION ============
        lidar_feats_list = []  # List of N feature dicts
        for i in range(self.n_subnets):
            lidar_i = lidar[:, i:i+1, :, :, :]  # [B, 1, H, W, D]
            feats_i = self.lidar_encoder(lidar_i)  # Dict with level0-level4
            lidar_feats_list.append(feats_i)

        # Concatenate features along channel dim, then project to original channels
        lidar_feats = {}
        for level in ['level0', 'level1', 'level2', 'level3', 'level4']:
            concat_feats = torch.cat([feats[level] for feats in lidar_feats_list], dim=1)
            proj_layer = getattr(self, f'lidar_proj_{level}')
            lidar_feats[level] = proj_layer(concat_feats)

        bev_3d = bev.unsqueeze(-1).expand(-1, -1, -1, -1, D)
        bev_emb0 = self.bev_embed(bev_3d)

        # B6: WaffleIron pretrained features (optional - uses zeros if not provided)
        if waffleiron is not None:
            B_wf = waffleiron.shape[0]
            # Use reshape instead of view - tensor may not be contiguous after expand() in sample_mimo
            wf_concat = waffleiron.reshape(B_wf, -1, waffleiron.shape[3], waffleiron.shape[4])
            wf_emb = self.waffleiron_embed(wf_concat)
            wf_emb_3d = wf_emb.unsqueeze(-1).expand(-1, -1, -1, -1, D)
            bev_emb0 = bev_emb0 + wf_emb_3d

        x = self.voxel_embed(x_t)

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

        # N separate completion heads
        head_outputs = []
        for i_head in range(self.n_subnets):
            logits = self.completion_heads[str(i_head)](x)
            head_outputs.append(logits)

        # Ensemble
        probs = [F.softmax(logits, dim=1) for logits in head_outputs]
        ensemble_probs = torch.stack(probs, dim=0).mean(dim=0)
        ensemble_logits = torch.log(ensemble_probs + 1e-10)

        return {
            'head_outputs': head_outputs,
            'ensemble': ensemble_logits,
            'probs': probs,
            'ensemble_probs': ensemble_probs,
        }


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    print("=== MIMO Scene Completion U-Net (PaSCo-style) ===\n")

    # Test with n_subnets=3
    model = MIMOSceneCompletionUNet(num_classes=20, n_subnets=3, base_channels=32)
    print("MIMOSceneCompletionUNet (n_subnets=3):")
    print(f"  Parameters: {count_parameters(model):,}")
    print("  Completion heads: 3 separate")

    model_lite = MIMOSceneCompletionUNetLite(num_classes=20, n_subnets=3, base_channels=16)
    print("\nMIMOSceneCompletionUNetLite (n_subnets=3):")
    print(f"  Parameters: {count_parameters(model_lite):,}")

    # Test forward pass with dummy data
    print("\n=== Testing Forward Pass ===")
    B, N, C, H, W, D = 2, 3, 20, 32, 32, 8
    x_t = torch.randn(B, N * C, H, W, D)  # Channel-concatenated one-hot
    t = torch.randint(0, 100, (B,))
    bev = torch.randint(0, 20, (B, N, H, W))  # Stacked BEV labels
    lidar = torch.rand(B, N, H, W, D)  # Channel-concatenated binary

    output = model_lite(x_t, t, bev, lidar)
    print(f"Input x_t shape: {x_t.shape}")
    print(f"Input bev shape: {bev.shape}")
    print(f"Input lidar shape: {lidar.shape}")
    print(f"Output head_outputs: {len(output['head_outputs'])} tensors of shape {output['head_outputs'][0].shape}")
    print(f"Output ensemble shape: {output['ensemble'].shape}")
    print("\nMIMO Scene Completion U-Net test passed!")
