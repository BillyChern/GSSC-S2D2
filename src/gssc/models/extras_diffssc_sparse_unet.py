"""
SpconvUNetDiff: Full UNet for denoising complete point cloud with
multiplicative gating conditioned on partial cloud features + time.

Replaces MinkUNetDiff from DiffSSC reference with spconv v2.

Architecture:
- Channel schedule: cs = [32, 32, 64, 128, 256, 256, 128, 96, 96]
- 4 encoder stages + bottleneck + 4 decoder stages = 9 gating stages
- Each stage: 1-NN match partial→full, project, combine with time, gate, convolve
- Output head: Linear(96, 40) → LeakyReLU → Linear(40, out_channels)
"""

import torch
import torch.nn as nn
import spconv.pytorch as spconv

from .diffssc_utils import match_part_to_full, expand_time_to_voxels, get_timestep_embedding


class BasicConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, ks=3, stride=1, indice_key=None):
        super().__init__()
        if stride > 1:
            self.net = spconv.SparseSequential(
                spconv.SparseConv3d(in_ch, out_ch, kernel_size=ks, stride=stride,
                                     padding=ks // 2, indice_key=indice_key),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(inplace=True),
            )
        else:
            self.net = spconv.SparseSequential(
                spconv.SubMConv3d(in_ch, out_ch, kernel_size=ks, indice_key=indice_key),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(inplace=True),
            )

    def forward(self, x):
        return self.net(x)


class BasicDeconvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, ks=2, stride=2, indice_key=None):
        super().__init__()
        self.net = spconv.SparseSequential(
            spconv.SparseInverseConv3d(in_ch, out_ch, kernel_size=ks,
                                        indice_key=indice_key),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, ks=3, indice_key=None):
        super().__init__()
        self.conv1 = spconv.SparseSequential(
            spconv.SubMConv3d(in_ch, out_ch, kernel_size=ks, indice_key=indice_key),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.conv2 = spconv.SparseSequential(
            spconv.SubMConv3d(out_ch, out_ch, kernel_size=ks, indice_key=indice_key),
            nn.BatchNorm1d(out_ch),
        )
        if in_ch != out_ch:
            self.downsample = spconv.SparseSequential(
                spconv.SubMConv3d(in_ch, out_ch, kernel_size=1,
                                   indice_key=indice_key + '_skip' if indice_key else None),
                nn.BatchNorm1d(out_ch),
            )
        else:
            self.downsample = None
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample is not None:
            identity = self.downsample(identity)
        out = out.replace_feature(self.relu(out.features + identity.features))
        return out


class GatingModule(nn.Module):
    """Per-stage gating: projects matched partial features + time embedding
    into per-voxel multiplicative weights.

    Pattern from MinkUNetDiff:
        match = 1NN_match(full_coords, part_feats)  → [N_full, part_ch]
        p = latent_proj(match)                       → [N_full, part_ch]
        t = time_proj(time_emb)                      → [B, part_ch]
        t_exp = expand(t, sparse)                    → [N_full, part_ch]
        w = latemp_proj(cat(p, t_exp))              → [N_full, gate_ch]
    """
    def __init__(self, part_ch, gate_ch, time_embed_dim):
        super().__init__()
        self.latent_proj = nn.Sequential(
            nn.Linear(part_ch, part_ch),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(part_ch, part_ch),
        )
        self.time_proj = nn.Sequential(
            nn.Linear(time_embed_dim, time_embed_dim),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(time_embed_dim, part_ch),
        )
        self.latemp_proj = nn.Sequential(
            nn.Linear(part_ch + part_ch, max(part_ch, gate_ch)),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(max(part_ch, gate_ch), gate_ch),
        )

    def forward(self, matched_feats, time_emb, sparse_tensor, bev_feats=None):
        """
        Args:
            matched_feats: [N_full, part_ch] from 1-NN matching
            time_emb: [B, embed_dim] time embedding
            sparse_tensor: spconv.SparseConvTensor for batch index expansion
            bev_feats: ignored (for API compatibility with BEVGatingModule)

        Returns:
            weights: [N_full, gate_ch] multiplicative gating weights
        """
        p = self.latent_proj(matched_feats)
        t = self.time_proj(time_emb)
        t_exp = expand_time_to_voxels(t, sparse_tensor)
        w = self.latemp_proj(torch.cat([p, t_exp], dim=-1))
        return w


class BEVGatingModule(nn.Module):
    """Per-stage gating with BEV conditioning (S30).

    Extends GatingModule by adding BEV class embeddings as a third input
    to the combined projection, enabling BEV-guided multiplicative gating.
    """
    def __init__(self, part_ch, gate_ch, time_embed_dim, bev_emb_dim=32):
        super().__init__()
        self.latent_proj = nn.Sequential(
            nn.Linear(part_ch, part_ch),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(part_ch, part_ch),
        )
        self.time_proj = nn.Sequential(
            nn.Linear(time_embed_dim, time_embed_dim),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(time_embed_dim, part_ch),
        )
        self.bev_proj = nn.Sequential(
            nn.Linear(bev_emb_dim, part_ch),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.latemp_proj = nn.Sequential(
            nn.Linear(part_ch + part_ch + part_ch, max(part_ch, gate_ch)),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(max(part_ch, gate_ch), gate_ch),
        )

    def forward(self, matched_feats, time_emb, sparse_tensor, bev_feats=None):
        p = self.latent_proj(matched_feats)
        t = self.time_proj(time_emb)
        t_exp = expand_time_to_voxels(t, sparse_tensor)
        if bev_feats is not None:
            b = self.bev_proj(bev_feats)
            w = self.latemp_proj(torch.cat([p, t_exp, b], dim=-1))
        else:
            # Fallback: zero BEV features (shouldn't happen in normal use)
            b = torch.zeros_like(p)
            w = self.latemp_proj(torch.cat([p, t_exp, b], dim=-1))
        return w


class SpconvUNetDiff(nn.Module):
    """Full UNet for denoising with multiplicative gating.

    Args:
        in_channels: input feature dimension (23 for xyz + 20 semantic)
        out_channels: output feature dimension (23 for noise prediction)
        part_ch: partial encoder output channels (256)
    """
    def __init__(self, in_channels=23, out_channels=23, part_ch=256,
                 use_bev=False, bev_emb_dim=32, num_classes=20):
        super().__init__()
        cs = [32, 32, 64, 128, 256, 256, 128, 96, 96]
        self.embed_dim = cs[-1]  # 96 — same as reference
        self.part_ch = part_ch
        self.use_bev = use_bev

        # BEV embedding (shared across all gating stages)
        if use_bev:
            self.bev_embed = nn.Embedding(num_classes, bev_emb_dim)
            GateClass = lambda p, g, t: BEVGatingModule(p, g, t, bev_emb_dim)
        else:
            GateClass = GatingModule

        # ─── Stem ──────────────────────────────────────────────
        self.stem = spconv.SparseSequential(
            spconv.SubMConv3d(in_channels, cs[0], kernel_size=3, indice_key='unet_stem0'),
            nn.BatchNorm1d(cs[0]),
            nn.ReLU(inplace=True),
            spconv.SubMConv3d(cs[0], cs[0], kernel_size=3, indice_key='unet_stem1'),
            nn.BatchNorm1d(cs[0]),
            nn.ReLU(inplace=True),
        )

        # ─── Encoder stages ───────────────────────────────────
        # Note: Separate downsample (spconv) from ResidualBlocks (nn.Sequential)
        # because spconv.SparseSequential unwraps features for non-spconv modules
        # Stage 1: gate stem output → downsample + residual blocks
        self.gate_s1 = GateClass(part_ch, cs[0], self.embed_dim)
        self.down1 = BasicConvBlock(cs[0], cs[0], ks=2, stride=2, indice_key='unet_down1')
        self.stage1_res = nn.Sequential(
            ResidualBlock(cs[0], cs[1], ks=3, indice_key='unet_s1'),
            ResidualBlock(cs[1], cs[1], ks=3, indice_key='unet_s1b'),
        )

        # Stage 2
        self.gate_s2 = GateClass(part_ch, cs[1], self.embed_dim)
        self.down2 = BasicConvBlock(cs[1], cs[1], ks=2, stride=2, indice_key='unet_down2')
        self.stage2_res = nn.Sequential(
            ResidualBlock(cs[1], cs[2], ks=3, indice_key='unet_s2'),
            ResidualBlock(cs[2], cs[2], ks=3, indice_key='unet_s2b'),
        )

        # Stage 3
        self.gate_s3 = GateClass(part_ch, cs[2], self.embed_dim)
        self.down3 = BasicConvBlock(cs[2], cs[2], ks=2, stride=2, indice_key='unet_down3')
        self.stage3_res = nn.Sequential(
            ResidualBlock(cs[2], cs[3], ks=3, indice_key='unet_s3'),
            ResidualBlock(cs[3], cs[3], ks=3, indice_key='unet_s3b'),
        )

        # Stage 4
        self.gate_s4 = GateClass(part_ch, cs[3], self.embed_dim)
        self.down4 = BasicConvBlock(cs[3], cs[3], ks=2, stride=2, indice_key='unet_down4')
        self.stage4_res = nn.Sequential(
            ResidualBlock(cs[3], cs[4], ks=3, indice_key='unet_s4'),
            ResidualBlock(cs[4], cs[4], ks=3, indice_key='unet_s4b'),
        )

        # ─── Decoder stages ──────────────────────────────────
        # Up1: gate bottleneck → upsample + skip from stage3
        self.gate_up1 = GateClass(part_ch, cs[4], self.embed_dim)
        self.up1_deconv = BasicDeconvBlock(cs[4], cs[5], ks=2, stride=2, indice_key='unet_down4')
        self.up1_res = nn.Sequential(
            ResidualBlock(cs[5] + cs[3], cs[5], ks=3, indice_key='unet_up1'),
            ResidualBlock(cs[5], cs[5], ks=3, indice_key='unet_up1b'),
        )

        # Up2: gate up1 output → upsample + skip from stage2
        self.gate_up2 = GateClass(part_ch, cs[5], self.embed_dim)
        self.up2_deconv = BasicDeconvBlock(cs[5], cs[6], ks=2, stride=2, indice_key='unet_down3')
        self.up2_res = nn.Sequential(
            ResidualBlock(cs[6] + cs[2], cs[6], ks=3, indice_key='unet_up2'),
            ResidualBlock(cs[6], cs[6], ks=3, indice_key='unet_up2b'),
        )

        # Up3: gate up2 output → upsample + skip from stage1
        self.gate_up3 = GateClass(part_ch, cs[6], self.embed_dim)
        self.up3_deconv = BasicDeconvBlock(cs[6], cs[7], ks=2, stride=2, indice_key='unet_down2')
        self.up3_res = nn.Sequential(
            ResidualBlock(cs[7] + cs[1], cs[7], ks=3, indice_key='unet_up3'),
            ResidualBlock(cs[7], cs[7], ks=3, indice_key='unet_up3b'),
        )

        # Up4: gate up3 output → upsample + skip from stem
        self.gate_up4 = GateClass(part_ch, cs[7], self.embed_dim)
        self.up4_deconv = BasicDeconvBlock(cs[7], cs[8], ks=2, stride=2, indice_key='unet_down1')
        self.up4_res = nn.Sequential(
            ResidualBlock(cs[8] + cs[0], cs[8], ks=3, indice_key='unet_up4'),
            ResidualBlock(cs[8], cs[8], ks=3, indice_key='unet_up4b'),
        )

        # ─── Output head ──────────────────────────────────────
        self.last = nn.Sequential(
            nn.Linear(cs[8], 40),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(40, out_channels),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _bev_lookup(self, sparse_tensor, bev, coord_min, stride):
        """Look up BEV class embeddings for voxels in a sparse tensor.

        Maps sparse voxel indices back to the BEV grid (256×256) using the
        coordinate offset from points_to_sparse_tensor, then embeds the classes.

        Args:
            sparse_tensor: spconv.SparseConvTensor at current resolution level
            bev: [B, 256, 256] BEV class labels (int64)
            coord_min: [3] int tensor, min coordinates before shifting to non-negative
            stride: cumulative spatial stride from stem level (1, 2, 4, 8, 16)

        Returns:
            bev_feats: [N_voxels, bev_emb_dim] embedded BEV features, or None
        """
        if bev is None or not self.use_bev:
            return None
        indices = sparse_tensor.indices  # [N, 4]: batch, x, y, z
        batch_idx = indices[:, 0].long()
        # Reverse the shift and stride to get original quantized coords:
        #   coords_int = shifted * stride + coord_min
        # Then map to BEV:
        #   bev_x = coords_int_x  (since POINT_CLOUD_RANGE x_min=0, resolution=0.2)
        #   bev_y = coords_int_y + 128  (since y_min=-25.6 → offset=128 at res=0.2)
        bev_x = (indices[:, 1].long() * stride + coord_min[0].long()).clamp(0, 255)
        bev_y = (indices[:, 2].long() * stride + coord_min[1].long() + 128).clamp(0, 255)
        bev_classes = bev[batch_idx, bev_x, bev_y]  # [N]
        return self.bev_embed(bev_classes)  # [N, bev_emb_dim]

    def _gate_and_apply(self, x_sparse, gate_module, part_sparse, time_emb, bev_feats=None):
        """Apply gating: match partial→full, compute weights, multiply features."""
        matched = match_part_to_full(x_sparse.indices, None, part_sparse)
        w = gate_module(matched, time_emb, x_sparse, bev_feats=bev_feats)
        gated_feats = x_sparse.features * w
        return x_sparse.replace_feature(gated_feats)

    @staticmethod
    def _sparse_cat(a, b):
        """Concatenate features of two sparse tensors with same coordinates."""
        cat_feats = torch.cat([a.features, b.features], dim=1)
        return a.replace_feature(cat_feats)

    def forward(self, x_sparse, part_sparse, t, bev=None, coord_min=None):
        """Forward pass.

        Args:
            x_sparse: spconv.SparseConvTensor [N_full, in_channels] noisy complete cloud
            part_sparse: spconv.SparseConvTensor [N_part_coarse, 256] encoded partial features
            t: [B] int tensor of diffusion timesteps
            bev: [B, 256, 256] optional GT BEV class labels (S30)
            coord_min: [3] int tensor from points_to_sparse_tensor (S30)

        Returns:
            noise_pred: [N_full, out_channels] predicted noise
        """
        time_emb = get_timestep_embedding(t, self.embed_dim)  # [B, 96]

        # Stem
        x0 = self.stem(x_sparse)

        # Encoder: gate → downsample → residual blocks
        bev_f = self._bev_lookup(x0, bev, coord_min, stride=1)
        x0_gated = self._gate_and_apply(x0, self.gate_s1, part_sparse, time_emb, bev_f)
        x1 = self.stage1_res(self.down1(x0_gated))

        bev_f = self._bev_lookup(x1, bev, coord_min, stride=2)
        x1_gated = self._gate_and_apply(x1, self.gate_s2, part_sparse, time_emb, bev_f)
        x2 = self.stage2_res(self.down2(x1_gated))

        bev_f = self._bev_lookup(x2, bev, coord_min, stride=4)
        x2_gated = self._gate_and_apply(x2, self.gate_s3, part_sparse, time_emb, bev_f)
        x3 = self.stage3_res(self.down3(x2_gated))

        bev_f = self._bev_lookup(x3, bev, coord_min, stride=8)
        x3_gated = self._gate_and_apply(x3, self.gate_s4, part_sparse, time_emb, bev_f)
        x4 = self.stage4_res(self.down4(x3_gated))

        # Decoder
        bev_f = self._bev_lookup(x4, bev, coord_min, stride=16)
        x4_gated = self._gate_and_apply(x4, self.gate_up1, part_sparse, time_emb, bev_f)
        y1 = self.up1_deconv(x4_gated)
        y1 = self._sparse_cat(y1, x3)
        y1 = self.up1_res(y1)

        bev_f = self._bev_lookup(y1, bev, coord_min, stride=8)
        y1_gated = self._gate_and_apply(y1, self.gate_up2, part_sparse, time_emb, bev_f)
        y2 = self.up2_deconv(y1_gated)
        y2 = self._sparse_cat(y2, x2)
        y2 = self.up2_res(y2)

        bev_f = self._bev_lookup(y2, bev, coord_min, stride=4)
        y2_gated = self._gate_and_apply(y2, self.gate_up3, part_sparse, time_emb, bev_f)
        y3 = self.up3_deconv(y2_gated)
        y3 = self._sparse_cat(y3, x1)
        y3 = self.up3_res(y3)

        bev_f = self._bev_lookup(y3, bev, coord_min, stride=2)
        y3_gated = self._gate_and_apply(y3, self.gate_up4, part_sparse, time_emb, bev_f)
        y4 = self.up4_deconv(y3_gated)
        y4 = self._sparse_cat(y4, x0)
        y4 = self.up4_res(y4)

        # Output: per-voxel noise prediction
        return self.last(y4.features)
