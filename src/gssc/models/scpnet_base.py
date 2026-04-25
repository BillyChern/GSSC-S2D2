"""
SCPNet-Style Diffusion Denoiser for SSC Refinement

Replaces our generic 3D UNet with SCPNet's architecture as the diffusion denoiser:
- Frozen Cylinder3D feature extraction (7D→32ch)
- Frozen completion ASPP subnet (fills empty voxels)
- Trainable time-conditioned segmentation subnet (Cylinder3D-based sparse UNet + FiLM)

Three-signal conditioning at completed voxel coordinates:
  1. Completed features (32ch) — WHERE things are
  2. x_t noisy label embedding (32ch) — current diffusion state
  3. SCPNet prediction embedding (32ch) — semantic anchor from SCPNet

Reference: SCPNet (Xia et al., CVPR 2023)
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import spconv.pytorch as spconv
from typing import Optional, List, Tuple, Dict


# ---------------------------------------------------------------------------
# Sparse conv helper functions (matching SCPNet's kernel conventions)
# ---------------------------------------------------------------------------

def conv3x3(in_planes, out_planes, stride=1, indice_key=None):
    return spconv.SubMConv3d(in_planes, out_planes, kernel_size=3, stride=stride,
                             padding=1, bias=False, indice_key=indice_key)

def conv1x3(in_planes, out_planes, stride=1, indice_key=None):
    """Kernel (1,3,3) — focus on YZ plane."""
    return spconv.SubMConv3d(in_planes, out_planes, kernel_size=(1, 3, 3), stride=stride,
                             padding=(0, 1, 1), bias=False, indice_key=indice_key)

def conv3x1(in_planes, out_planes, stride=1, indice_key=None):
    """Kernel (3,1,3) — focus on XZ plane."""
    return spconv.SubMConv3d(in_planes, out_planes, kernel_size=(3, 1, 3), stride=stride,
                             padding=(1, 0, 1), bias=False, indice_key=indice_key)

def conv1x1x3(in_planes, out_planes, stride=1, indice_key=None):
    return spconv.SubMConv3d(in_planes, out_planes, kernel_size=(1, 1, 3), stride=stride,
                             padding=(0, 0, 1), bias=False, indice_key=indice_key)

def conv1x3x1(in_planes, out_planes, stride=1, indice_key=None):
    return spconv.SubMConv3d(in_planes, out_planes, kernel_size=(1, 3, 1), stride=stride,
                             padding=(0, 1, 0), bias=False, indice_key=indice_key)

def conv3x1x1(in_planes, out_planes, stride=1, indice_key=None):
    return spconv.SubMConv3d(in_planes, out_planes, kernel_size=(3, 1, 1), stride=stride,
                             padding=(1, 0, 0), bias=False, indice_key=indice_key)


# ---------------------------------------------------------------------------
# Time embedding
# ---------------------------------------------------------------------------

def timestep_embedding(timesteps: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    """Sinusoidal timestep embeddings [B] → [B, dim]."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, device=timesteps.device) / half
    )
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


# ---------------------------------------------------------------------------
# FiLM helper: expand [B, C] → [N, C] for sparse tensors
# ---------------------------------------------------------------------------

def expand_to_sparse(param_bc, batch_indices):
    """Expand [B, C] tensor to [N, C] using batch indices from sparse coords."""
    return param_bc[batch_indices]


# ---------------------------------------------------------------------------
# Time-conditioned sparse conv blocks
# ---------------------------------------------------------------------------

class TimeFiLMResContextBlock(nn.Module):
    """ResContextBlock with FiLM time conditioning.

    Two-path residual with asymmetric kernels, plus additive FiLM
    after the final feature merge.
    """

    def __init__(self, in_filters, out_filters, time_dim, indice_key=None):
        super().__init__()
        # Path 1: conv1x3 → conv1x3 (v1 compat: all share conv1's (1,3,3) kernel)
        self.conv1 = conv1x3(in_filters, out_filters, indice_key=indice_key + "bef1x3")
        self.bn0 = nn.BatchNorm1d(out_filters)
        self.act1 = nn.LeakyReLU()

        self.conv1_2 = conv1x3(out_filters, out_filters, indice_key=indice_key + "bef1x3")
        self.bn0_2 = nn.BatchNorm1d(out_filters)
        self.act1_2 = nn.LeakyReLU()

        # Path 2: conv1x3 → conv1x3 (v1 compat: all share conv1's (1,3,3) kernel)
        self.conv2 = conv1x3(in_filters, out_filters, indice_key=indice_key + "bef1x3")
        self.act2 = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm1d(out_filters)

        self.conv3 = conv1x3(out_filters, out_filters, indice_key=indice_key + "bef1x3")
        self.act3 = nn.LeakyReLU()
        self.bn2 = nn.BatchNorm1d(out_filters)

        # FiLM: time_dim → scale + shift
        self.time_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, out_filters * 2),
        )
        # Initialize to identity: γ=1, β=0
        nn.init.zeros_(self.time_proj[1].weight)
        nn.init.zeros_(self.time_proj[1].bias)

        self.weight_initialization()

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, time_emb, batch_indices):
        # Path 1
        shortcut = self.conv1(x)
        shortcut = shortcut.replace_feature(self.act1(shortcut.features))
        shortcut = shortcut.replace_feature(self.bn0(shortcut.features))

        shortcut = self.conv1_2(shortcut)
        shortcut = shortcut.replace_feature(self.act1_2(shortcut.features))
        shortcut = shortcut.replace_feature(self.bn0_2(shortcut.features))

        # Path 2
        resA = self.conv2(x)
        resA = resA.replace_feature(self.act2(resA.features))
        resA = resA.replace_feature(self.bn1(resA.features))

        resA = self.conv3(resA)
        resA = resA.replace_feature(self.act3(resA.features))
        resA = resA.replace_feature(self.bn2(resA.features))

        # Merge + FiLM
        resA = resA.replace_feature(resA.features + shortcut.features)

        # Apply FiLM time conditioning
        film = self.time_proj(time_emb)  # [B, 2*C]
        gamma, beta = film.chunk(2, dim=-1)  # each [B, C]
        gamma = expand_to_sparse(gamma, batch_indices) + 1.0  # center at 1
        beta = expand_to_sparse(beta, batch_indices)
        resA = resA.replace_feature(gamma * resA.features + beta)

        return resA


class TimeFiLMResBlock(nn.Module):
    """ResBlock with FiLM time conditioning and optional pooling."""

    def __init__(self, in_filters, out_filters, dropout_rate, time_dim,
                 pooling=True, height_pooling=False, indice_key=None):
        super().__init__()
        self.pooling = pooling

        # v1 compat: all share conv1's (3,1,3) kernel via same indice_key
        self.conv1 = conv3x1(in_filters, out_filters, indice_key=indice_key + "bef3x1")
        self.act1 = nn.LeakyReLU()
        self.bn0 = nn.BatchNorm1d(out_filters)

        self.conv1_2 = conv3x1(out_filters, out_filters, indice_key=indice_key + "bef3x1")
        self.act1_2 = nn.LeakyReLU()
        self.bn0_2 = nn.BatchNorm1d(out_filters)

        self.conv2 = conv3x1(in_filters, out_filters, indice_key=indice_key + "bef3x1")
        self.act2 = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm1d(out_filters)

        self.conv3 = conv3x1(out_filters, out_filters, indice_key=indice_key + "bef3x1")
        self.act3 = nn.LeakyReLU()
        self.bn2 = nn.BatchNorm1d(out_filters)

        if pooling:
            if height_pooling:
                self.pool = spconv.SparseConv3d(out_filters, out_filters, kernel_size=3,
                                                stride=2, padding=1,
                                                indice_key=indice_key, bias=False)
            else:
                self.pool = spconv.SparseConv3d(out_filters, out_filters, kernel_size=3,
                                                stride=(2, 2, 1), padding=1,
                                                indice_key=indice_key, bias=False)

        # FiLM
        self.time_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, out_filters * 2),
        )
        nn.init.zeros_(self.time_proj[1].weight)
        nn.init.zeros_(self.time_proj[1].bias)

        self.weight_initialization()

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, time_emb, batch_indices):
        shortcut = self.conv1(x)
        shortcut = shortcut.replace_feature(self.act1(shortcut.features))
        shortcut = shortcut.replace_feature(self.bn0(shortcut.features))

        shortcut = self.conv1_2(shortcut)
        shortcut = shortcut.replace_feature(self.act1_2(shortcut.features))
        shortcut = shortcut.replace_feature(self.bn0_2(shortcut.features))

        resA = self.conv2(x)
        resA = resA.replace_feature(self.act2(resA.features))
        resA = resA.replace_feature(self.bn1(resA.features))

        resA = self.conv3(resA)
        resA = resA.replace_feature(self.act3(resA.features))
        resA = resA.replace_feature(self.bn2(resA.features))

        resA = resA.replace_feature(resA.features + shortcut.features)

        # FiLM
        film = self.time_proj(time_emb)
        gamma, beta = film.chunk(2, dim=-1)
        gamma = expand_to_sparse(gamma, batch_indices) + 1.0
        beta = expand_to_sparse(beta, batch_indices)
        resA = resA.replace_feature(gamma * resA.features + beta)

        if self.pooling:
            resB = self.pool(resA)
            return resB, resA
        else:
            return resA


class TimeFiLMUpBlock(nn.Module):
    """UpBlock with FiLM time conditioning."""

    def __init__(self, in_filters, out_filters, time_dim,
                 indice_key=None, up_key=None):
        super().__init__()
        self.trans_dilao = conv3x3(in_filters, out_filters, indice_key=indice_key + "new_up")
        self.trans_act = nn.LeakyReLU()
        self.trans_bn = nn.BatchNorm1d(out_filters)

        # v1 compat: all share conv1's (1,3,3) kernel via same indice_key
        self.conv1 = conv1x3(out_filters, out_filters, indice_key=indice_key + "1x3")
        self.act1 = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm1d(out_filters)

        self.conv2 = conv1x3(out_filters, out_filters, indice_key=indice_key + "1x3")
        self.act2 = nn.LeakyReLU()
        self.bn2 = nn.BatchNorm1d(out_filters)

        self.conv3 = conv1x3(out_filters, out_filters, indice_key=indice_key + "1x3")
        self.act3 = nn.LeakyReLU()
        self.bn3 = nn.BatchNorm1d(out_filters)

        self.up_subm = spconv.SparseInverseConv3d(out_filters, out_filters, kernel_size=3,
                                                   indice_key=up_key, bias=False)

        # FiLM
        self.time_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, out_filters * 2),
        )
        nn.init.zeros_(self.time_proj[1].weight)
        nn.init.zeros_(self.time_proj[1].bias)

        self.weight_initialization()

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, skip, time_emb, batch_indices):
        upA = self.trans_dilao(x)
        upA = upA.replace_feature(self.trans_act(upA.features))
        upA = upA.replace_feature(self.trans_bn(upA.features))

        # Upsample via inverse conv
        upA = self.up_subm(upA)
        upA = upA.replace_feature(upA.features + skip.features)

        upE = self.conv1(upA)
        upE = upE.replace_feature(self.act1(upE.features))
        upE = upE.replace_feature(self.bn1(upE.features))

        upE = self.conv2(upE)
        upE = upE.replace_feature(self.act2(upE.features))
        upE = upE.replace_feature(self.bn2(upE.features))

        upE = self.conv3(upE)
        upE = upE.replace_feature(self.act3(upE.features))
        upE = upE.replace_feature(self.bn3(upE.features))

        # FiLM
        # After upsample, batch_indices may differ — recompute from coords
        up_batch_indices = upE.indices[:, 0].long()
        film = self.time_proj(time_emb)
        gamma, beta = film.chunk(2, dim=-1)
        gamma = expand_to_sparse(gamma, up_batch_indices) + 1.0
        beta = expand_to_sparse(beta, up_batch_indices)
        upE = upE.replace_feature(gamma * upE.features + beta)

        return upE


class ReconBlock(nn.Module):
    """Sigmoid-gated per-axis attention (unchanged from SCPNet)."""

    def __init__(self, in_filters, out_filters, indice_key=None):
        super().__init__()
        # v1 compat: all share conv1's (3,1,1) kernel via same indice_key
        self.conv1 = conv3x1x1(in_filters, out_filters, indice_key=indice_key + "bef3x1x1")
        self.bn0 = nn.BatchNorm1d(out_filters)
        self.act1 = nn.Sigmoid()

        self.conv1_2 = conv3x1x1(in_filters, out_filters, indice_key=indice_key + "bef3x1x1")
        self.bn0_2 = nn.BatchNorm1d(out_filters)
        self.act1_2 = nn.Sigmoid()

        self.conv1_3 = conv3x1x1(in_filters, out_filters, indice_key=indice_key + "bef3x1x1")
        self.bn0_3 = nn.BatchNorm1d(out_filters)
        self.act1_3 = nn.Sigmoid()

    def forward(self, x):
        shortcut = self.conv1(x)
        shortcut = shortcut.replace_feature(self.bn0(shortcut.features))
        shortcut = shortcut.replace_feature(self.act1(shortcut.features))

        shortcut2 = self.conv1_2(x)
        shortcut2 = shortcut2.replace_feature(self.bn0_2(shortcut2.features))
        shortcut2 = shortcut2.replace_feature(self.act1_2(shortcut2.features))

        shortcut3 = self.conv1_3(x)
        shortcut3 = shortcut3.replace_feature(self.bn0_3(shortcut3.features))
        shortcut3 = shortcut3.replace_feature(self.act1_3(shortcut3.features))

        shortcut = shortcut.replace_feature(shortcut.features + shortcut2.features + shortcut3.features)
        shortcut = shortcut.replace_feature(shortcut.features * x.features)

        return shortcut


# ---------------------------------------------------------------------------
# Completion subnet (dense Conv3d ASPP, frozen from SCPNet)
# ---------------------------------------------------------------------------

class CompletionSubnet(nn.Module):
    """SCPNet's dense ASPP completion subnet.

    Multi-path blocks with 3×3, 5×5, 7×7 kernels.
    No BatchNorm, no bias — just Conv3d + ReLU.
    """

    def __init__(self, init_size=32):
        super().__init__()
        ch = init_size  # 32

        # MPB stage 1
        self.a_conv1 = nn.Sequential(nn.Conv3d(ch, ch, 3, 1, padding=1, bias=False), nn.ReLU())
        self.a_conv2 = nn.Sequential(nn.Conv3d(ch, ch, 3, 1, padding=1, bias=False), nn.ReLU())
        self.a_conv3 = nn.Sequential(nn.Conv3d(ch, ch, 5, 1, padding=2, bias=False), nn.ReLU())
        self.a_conv4 = nn.Sequential(nn.Conv3d(ch, ch, 7, 1, padding=3, bias=False), nn.ReLU())

        # MPB stage 2
        self.a_conv5 = nn.Sequential(nn.Conv3d(ch * 3, ch, 3, 1, padding=1, bias=False), nn.ReLU())
        self.a_conv6 = nn.Sequential(nn.Conv3d(ch * 3, ch, 5, 1, padding=2, bias=False), nn.ReLU())
        self.a_conv7 = nn.Sequential(nn.Conv3d(ch * 3, ch, 7, 1, padding=3, bias=False), nn.ReLU())

        # Channel reduction
        self.ch_conv1 = nn.Sequential(nn.Conv3d(ch * 7, ch, kernel_size=1, stride=1, bias=False), nn.ReLU())

        # Residual paths
        self.res_1 = nn.Sequential(nn.Conv3d(ch, ch, 3, 1, padding=1, bias=False), nn.ReLU())
        self.res_2 = nn.Sequential(nn.Conv3d(ch, ch, 5, 1, padding=2, bias=False), nn.ReLU())
        self.res_3 = nn.Sequential(nn.Conv3d(ch, ch, 7, 1, padding=3, bias=False), nn.ReLU())

    def forward(self, x_dense):
        """
        Args:
            x_dense: [B, init_size, 256, 256, 32] dense voxel features

        Returns:
            completed: [B, init_size, 256, 256, 32] completed features
        """
        x1 = self.a_conv1(x_dense)
        x2 = self.a_conv2(x1)
        x3 = self.a_conv3(x1)
        x4 = self.a_conv4(x1)
        t1 = torch.cat((x2, x3, x4), 1)
        x5 = self.a_conv5(t1)
        x6 = self.a_conv6(t1)
        x7 = self.a_conv7(t1)
        x = torch.cat((x1, x2, x3, x4, x5, x6, x7), 1)
        y0 = self.ch_conv1(x)
        y1 = self.res_1(x_dense)
        y2 = self.res_2(x_dense)
        y3 = self.res_3(x_dense)
        return x_dense + y0 + y1 + y2 + y3


# ---------------------------------------------------------------------------
# Cylinder feature generator (from SCPNet, uses torch_scatter)
# ---------------------------------------------------------------------------

class CylinderFeatureGenerator(nn.Module):
    """Per-point MLP + scatter_max voxel aggregation + optional compression.

    Matches SCPNet's cylinder_fea exactly.
    """

    def __init__(self, grid_size, fea_dim=7, out_pt_fea_dim=256, fea_compre=32):
        super().__init__()
        self.PPmodel = nn.Sequential(
            nn.BatchNorm1d(fea_dim),
            nn.Linear(fea_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, out_pt_fea_dim),
        )
        self.fea_compre = fea_compre
        self.grid_size = grid_size
        if fea_compre is not None:
            self.fea_compression = nn.Sequential(
                nn.Linear(out_pt_fea_dim, fea_compre),
                nn.ReLU(),
            )

    def forward(self, pt_fea: List[torch.Tensor], xy_ind: List[torch.Tensor]):
        """
        Args:
            pt_fea: list of [N_i, 7] per-point features for each batch item
            xy_ind: list of [N_i, 3] grid indices for each batch item

        Returns:
            coords: [K, 4] unique voxel coords (batch_idx, x, y, z)
            features: [K, fea_compre] voxel features
        """
        import torch_scatter

        cur_dev = pt_fea[0].device

        # Prepend batch index to coordinates
        cat_pt_ind = []
        for i_batch in range(len(xy_ind)):
            cat_pt_ind.append(F.pad(xy_ind[i_batch], (1, 0), 'constant', value=i_batch))

        cat_pt_fea = torch.cat(pt_fea, dim=0)
        cat_pt_ind = torch.cat(cat_pt_ind, dim=0)
        pt_num = cat_pt_ind.shape[0]

        # Shuffle for robustness
        shuffled_ind = torch.randperm(pt_num, device=cur_dev)
        cat_pt_fea = cat_pt_fea[shuffled_ind, :]
        cat_pt_ind = cat_pt_ind[shuffled_ind, :]

        # Find unique voxels
        unq, unq_inv, unq_cnt = torch.unique(cat_pt_ind, return_inverse=True,
                                              return_counts=True, dim=0)
        unq = unq.type(torch.int64)

        # Per-point MLP + scatter-max pooling
        processed = self.PPmodel(cat_pt_fea)
        pooled = torch_scatter.scatter_max(processed, unq_inv, dim=0)[0]

        # Optional compression
        if self.fea_compre is not None:
            pooled = self.fea_compression(pooled)

        return unq, pooled


# ---------------------------------------------------------------------------
# Dense-to-sparse conversion
# ---------------------------------------------------------------------------

def extract_nonzero_features(x):
    """Extract non-zero voxel features from dense tensor.

    Args:
        x: [B, C, D, H, W] dense tensor

    Returns:
        coords: [N, 4] int32 coordinates (batch, d, h, w)
        features: [N, C] features at non-zero positions
    """
    device = x.device
    nonzero_index = torch.sum(torch.abs(x), dim=1).nonzero()
    coords = nonzero_index.type(torch.int32).to(device)
    channels = int(x.shape[1])
    features = x.permute(0, 2, 3, 4, 1).reshape(-1, channels)
    features = features[torch.sum(torch.abs(features), dim=1).nonzero(), :]
    features = features.squeeze(1).to(device)
    coords, _, _ = torch.unique(coords, return_inverse=True, return_counts=True, dim=0)
    return coords, features


# ---------------------------------------------------------------------------
# Main model: SCPNet Diffusion Denoiser
# ---------------------------------------------------------------------------

class SCPNetDiffusionDenoiser(nn.Module):
    """
    SCPNet-style denoiser for D3PM discrete diffusion.

    Frozen: Cylinder3D features + Completion subnet
    Trainable: Time-conditioned segmentation subnet

    Forward signature matches D3PM's model call:
        model(x_t, t, bev, lidar, **model_kwargs)
    """

    def __init__(
        self,
        num_classes: int = 20,
        init_size: int = 32,
        num_input_features: int = 32,
        sparse_shape: Tuple[int, int, int] = (256, 256, 32),
        time_dim: int = 256,
        use_bev_cond: bool = False,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.init_size = init_size
        self.num_input_features = num_input_features
        self.sparse_shape = np.array(sparse_shape)
        self.use_bev_cond = use_bev_cond

        # --- Frozen modules (loaded from SCPNet) ---
        self.cylinder_fea = CylinderFeatureGenerator(
            grid_size=sparse_shape,
            fea_dim=7,
            out_pt_fea_dim=256,
            fea_compre=num_input_features,  # 32
        )
        self.completion = CompletionSubnet(init_size=init_size)

        # --- Embeddings (trainable) ---
        self.x_t_embed = nn.Embedding(num_classes, num_input_features)  # 20 → 32ch
        self.scpnet_pred_embed = nn.Embedding(num_classes, num_input_features)  # 20 → 32ch

        # Time embedding MLP
        self.time_embed = nn.Sequential(
            nn.Linear(time_dim, time_dim * 4),
            nn.SiLU(),
            nn.Linear(time_dim * 4, time_dim),
        )

        # --- Segmentation subnet (trainable, time-conditioned) ---
        # Input: completed(32) + x_t(32) + scpnet_pred(32) = 96ch
        # Optionally + BEV(32) = 128ch
        seg_input_ch = num_input_features * 3  # 96
        if use_bev_cond:
            self.bev_proj = nn.Linear(num_classes, num_input_features)  # 20 → 32
            seg_input_ch += num_input_features  # 128

        # Project conditioning signals down to init_size (32ch) so downCntx
        # matches SCPNet exactly and all pretrained weights load correctly
        self.input_proj = nn.Linear(seg_input_ch, init_size)
        # Identity init: pass through completed features (first 32ch), zero out rest
        # This ensures segmentation subnet sees same input distribution as SCPNet at init
        with torch.no_grad():
            nn.init.zeros_(self.input_proj.weight)
            nn.init.zeros_(self.input_proj.bias)
            self.input_proj.weight[:, :init_size] = torch.eye(init_size)

        # Residual from SCPNet prediction — guarantees 36.17% mIoU floor
        self.scpnet_residual_scale = nn.Parameter(torch.tensor(1.0))

        self.downCntx = TimeFiLMResContextBlock(init_size, init_size, time_dim, indice_key="pre")
        self.resBlock2 = TimeFiLMResBlock(init_size, 2 * init_size, 0.2, time_dim,
                                          height_pooling=True, indice_key="down2")
        self.resBlock3 = TimeFiLMResBlock(2 * init_size, 4 * init_size, 0.2, time_dim,
                                          height_pooling=True, indice_key="down3")
        self.resBlock4 = TimeFiLMResBlock(4 * init_size, 8 * init_size, 0.2, time_dim,
                                          pooling=True, height_pooling=False, indice_key="down4")
        self.resBlock5 = TimeFiLMResBlock(8 * init_size, 16 * init_size, 0.2, time_dim,
                                          pooling=True, height_pooling=False, indice_key="down5")

        self.upBlock0 = TimeFiLMUpBlock(16 * init_size, 16 * init_size, time_dim,
                                        indice_key="up0", up_key="down5")
        self.upBlock1 = TimeFiLMUpBlock(16 * init_size, 8 * init_size, time_dim,
                                        indice_key="up1", up_key="down4")
        self.upBlock2 = TimeFiLMUpBlock(8 * init_size, 4 * init_size, time_dim,
                                        indice_key="up2", up_key="down3")
        self.upBlock3 = TimeFiLMUpBlock(4 * init_size, 2 * init_size, time_dim,
                                        indice_key="up3", up_key="down2")

        self.ReconNet = ReconBlock(2 * init_size, 2 * init_size, indice_key="recon")

        self.logits = spconv.SubMConv3d(4 * init_size, num_classes,
                                        indice_key="logit", kernel_size=3,
                                        stride=1, padding=1, bias=True)

    def freeze_pretrained(self):
        """Freeze cylinder feature generator and completion subnet."""
        for param in self.cylinder_fea.parameters():
            param.requires_grad = False
        for param in self.completion.parameters():
            param.requires_grad = False

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        bev: Optional[torch.Tensor] = None,
        lidar: Optional[torch.Tensor] = None,
        pt_fea: Optional[List[torch.Tensor]] = None,
        grid_ind: Optional[List[torch.Tensor]] = None,
        scpnet_pred: Optional[torch.Tensor] = None,
        completed_features: Optional[torch.Tensor] = None,
        completed_coords: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Args:
            x_t: [B, K, H, W, D] noisy one-hot labels (from D3PM)
            t: [B] timesteps
            bev: [B, K, H, W] BEV labels (optional, for ablation)
            lidar: ignored (kept for interface compatibility)
            pt_fea: list of [N_i, 7] per-point features (for Cylinder3D)
            grid_ind: list of [N_i, 3] grid indices (for Cylinder3D)
            scpnet_pred: [B, H, W, D] int64 SCPNet prediction labels
            completed_features: [N, 32] pre-computed (optional, skip cylinder+completion)
            completed_coords: [N, 4] pre-computed coords (optional)

        Returns:
            logits: [B, K, H, W, D] class logits (x_0 prediction)
        """
        B = x_t.shape[0]
        device = x_t.device

        # --- Step 1: Time embedding ---
        time_emb = self.time_embed(timestep_embedding(t, 256).to(device))  # [B, time_dim]

        # --- Step 2: Get completed features (frozen) ---
        if completed_features is not None and completed_coords is not None:
            # Pre-computed path
            coords = completed_coords
            comp_feats = completed_features
        else:
            # On-the-fly: Cylinder3D → completion
            with torch.no_grad():
                cyl_coords, cyl_feats = self.cylinder_fea(pt_fea, grid_ind)
                # Create sparse tensor and convert to dense
                x_sparse = spconv.SparseConvTensor(
                    cyl_feats, cyl_coords.int(), self.sparse_shape, B
                )
                x_dense = x_sparse.dense()  # [B, 32, 256, 256, 32]

                # Completion subnet
                x_completed = self.completion(x_dense)  # [B, 32, 256, 256, 32]

                # Dense to sparse
                coords, comp_feats = extract_nonzero_features(x_completed)

        # --- Step 3: Gather conditioning at completed coords ---
        batch_idx = coords[:, 0].long()
        x_idx = coords[:, 1].long()
        y_idx = coords[:, 2].long()
        z_idx = coords[:, 3].long()

        # x_t labels at completed positions
        x_t_labels = x_t.argmax(dim=1)  # [B, H, W, D]
        x_t_at_coords = x_t_labels[batch_idx, x_idx, y_idx, z_idx]  # [N]
        x_t_emb = self.x_t_embed(x_t_at_coords)  # [N, 32]

        # SCPNet prediction at completed positions
        if scpnet_pred is not None:
            scpnet_at_coords = scpnet_pred[batch_idx, x_idx, y_idx, z_idx]  # [N]
            scpnet_emb = self.scpnet_pred_embed(scpnet_at_coords)  # [N, 32]
        else:
            # If no SCPNet prediction, use zeros (unconditional)
            scpnet_emb = torch.zeros_like(x_t_emb)

        # Concatenate: completed(32) + x_t(32) + scpnet_pred(32) = 96ch
        input_features = torch.cat([comp_feats, x_t_emb, scpnet_emb], dim=1)

        # Optional BEV conditioning
        if self.use_bev_cond and bev is not None:
            # bev: [B, K, H, W] — look up at (x, y) for each voxel
            if bev.dtype == torch.long or bev.dtype == torch.int64:
                bev_onehot = F.one_hot(bev, self.num_classes).float()  # [B, H, W, K]
                bev_onehot = bev_onehot.permute(0, 3, 1, 2)  # [B, K, H, W]
            else:
                bev_onehot = bev  # already float [B, K, H, W]
            bev_at_coords = bev_onehot[batch_idx, :, x_idx, y_idx]  # [N, K]
            bev_emb = self.bev_proj(bev_at_coords)  # [N, 32]
            input_features = torch.cat([input_features, bev_emb], dim=1)

        # --- Step 4: Project to init_size channels ---
        input_features = self.input_proj(input_features)  # [N, 96/128] → [N, 32]

        # --- Step 5: Segmentation subnet (trainable) ---
        x_seg = spconv.SparseConvTensor(input_features, coords.int(),
                                        self.sparse_shape, B)

        x = self.downCntx(x_seg, time_emb, batch_idx)
        down1c, down1b = self.resBlock2(x, time_emb, x.indices[:, 0].long())
        down2c, down2b = self.resBlock3(down1c, time_emb, down1c.indices[:, 0].long())
        down3c, down3b = self.resBlock4(down2c, time_emb, down2c.indices[:, 0].long())
        down4c, down4b = self.resBlock5(down3c, time_emb, down3c.indices[:, 0].long())

        up4e = self.upBlock0(down4c, down4b, time_emb, down4c.indices[:, 0].long())
        up3e = self.upBlock1(up4e, down3b, time_emb, up4e.indices[:, 0].long())
        up2e = self.upBlock2(up3e, down2b, time_emb, up3e.indices[:, 0].long())
        up1e = self.upBlock3(up2e, down1b, time_emb, up2e.indices[:, 0].long())

        up0e = self.ReconNet(up1e)
        up0e = up0e.replace_feature(torch.cat((up0e.features, up1e.features), 1))

        logits_sparse = self.logits(up0e)
        y = logits_sparse.dense()  # [B, K, 256, 256, 32]

        # Residual from SCPNet prediction — guarantees baseline as floor
        if scpnet_pred is not None:
            scpnet_onehot = F.one_hot(scpnet_pred.long(), self.num_classes).float()
            scpnet_logits = scpnet_onehot.permute(0, 4, 1, 2, 3) * 10.0  # [B, K, H, W, D]
            y = y + self.scpnet_residual_scale * scpnet_logits

        return y

    # Layers where kernel shape was changed for v1 compat.
    # Maps our key suffix → (v1_original_kernel, our_target_kernel)
    _KERNEL_RESHAPE = {}
    # ResContextBlock: conv1_2, conv2 from (3,1,3) to (1,3,3)
    _KERNEL_RESHAPE['downCntx.conv1_2.weight'] = ((3, 1, 3), (1, 3, 3))
    _KERNEL_RESHAPE['downCntx.conv2.weight'] = ((3, 1, 3), (1, 3, 3))
    # ResBlock: conv1_2, conv2 from (1,3,3) to (3,1,3)
    for _rb in ['resBlock2', 'resBlock3', 'resBlock4', 'resBlock5']:
        _KERNEL_RESHAPE[f'{_rb}.conv1_2.weight'] = ((1, 3, 3), (3, 1, 3))
        _KERNEL_RESHAPE[f'{_rb}.conv2.weight'] = ((1, 3, 3), (3, 1, 3))
    # UpBlock: conv2 from (3,1,3) to (1,3,3), conv3 from (3,3,3) to (1,3,3)
    for _ub in ['upBlock0', 'upBlock1', 'upBlock2', 'upBlock3']:
        _KERNEL_RESHAPE[f'{_ub}.conv2.weight'] = ((3, 1, 3), (1, 3, 3))
        _KERNEL_RESHAPE[f'{_ub}.conv3.weight'] = ((3, 3, 3), (1, 3, 3))
    # ReconBlock: conv1_2 from (1,3,1) to (3,1,1), conv1_3 from (1,1,3) to (3,1,1)
    _KERNEL_RESHAPE['ReconNet.conv1_2.weight'] = ((1, 3, 1), (3, 1, 1))
    _KERNEL_RESHAPE['ReconNet.conv1_3.weight'] = ((1, 1, 3), (3, 1, 1))

    @staticmethod
    def _reshape_kernel_weight(value, orig_kernel, target_kernel):
        """Reshape v1 weight to v2 format with kernel shape change.

        Preserves flat element ordering. Truncates if target has fewer elements
        (e.g., 3x3x3 → 1x3x3: 27 → 9).
        v1: (kD, kH, kW, in_ch, out_ch) → v2: (out_ch, kD_new, kH_new, kW_new, in_ch)
        """
        num_orig = orig_kernel[0] * orig_kernel[1] * orig_kernel[2]
        num_target = target_kernel[0] * target_kernel[1] * target_kernel[2]
        in_ch, out_ch = value.shape[-2], value.shape[-1]
        flat = value.reshape(num_orig, in_ch, out_ch)
        if num_target < num_orig:
            flat = flat[:num_target]
        reshaped = flat.reshape(target_kernel[0], target_kernel[1], target_kernel[2],
                                in_ch, out_ch)
        return reshaped.permute(4, 0, 1, 2, 3).contiguous()

    def load_scpnet_weights(self, ckpt_path: str, load_segmentation: bool = True):
        """Load SCPNet pretrained weights with spconv v1→v2 transposition.

        Maps SCPNet's Asymm_3d_spconv + cylinder_fea weights to our architecture.
        Handles kernel shape changes needed for v1→v2 compatibility.

        Args:
            ckpt_path: path to SCPNet pretrained.pth
            load_segmentation: if True, also load segmentation subnet weights.
        """
        pre_weight = torch.load(ckpt_path, map_location='cpu', weights_only=False)

        loaded = 0
        skipped = 0
        reshaped_count = 0
        my_state = self.state_dict()

        for key, value in pre_weight.items():
            new_key = self._map_scpnet_key(key)
            if new_key is None or new_key not in my_state:
                skipped += 1
                continue

            # Skip segmentation subnet weights unless explicitly requested
            if not load_segmentation:
                if not (new_key.startswith('cylinder_fea.') or new_key.startswith('completion.')):
                    skipped += 1
                    continue

            target_shape = my_state[new_key].shape

            # Check if this weight needs kernel reshape (v1 compat)
            if new_key in self._KERNEL_RESHAPE:
                orig_k, target_k = self._KERNEL_RESHAPE[new_key]
                reshaped = self._reshape_kernel_weight(value, orig_k, target_k)
                if reshaped.shape == target_shape:
                    my_state[new_key] = reshaped
                    loaded += 1
                    reshaped_count += 1
                else:
                    skipped += 1
                continue

            if value.shape == target_shape:
                my_state[new_key] = value
                loaded += 1
            elif len(value.shape) == 5 and len(target_shape) == 5:
                # spconv v1→v2 weight transposition: (D,H,W,Cin,Cout) → (Cout,D,H,W,Cin)
                transposed = value.permute(4, 0, 1, 2, 3)
                if transposed.shape == target_shape:
                    my_state[new_key] = transposed.contiguous()
                    loaded += 1
                else:
                    skipped += 1
            else:
                skipped += 1

        self.load_state_dict(my_state, strict=False)
        modules = "frozen only" if not load_segmentation else "all"
        print(f"Loaded {loaded}/{loaded + skipped} SCPNet weights ({modules})"
              f" (skipped {skipped}, reshaped {reshaped_count})")

        if load_segmentation:
            # Reset BN running stats in segmentation subnet.
            # Affine params (weight, bias) are kept from SCPNet, but
            # running_mean/running_var must be recomputed from our data
            # distribution (which differs due to input_proj combining 3 signals).
            seg_modules = [self.downCntx, self.resBlock2, self.resBlock3,
                           self.resBlock4, self.resBlock5, self.upBlock0,
                           self.upBlock1, self.upBlock2, self.upBlock3,
                           self.ReconNet]
            bn_reset_count = 0
            for mod in seg_modules:
                for m in mod.modules():
                    if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                        m.running_mean.zero_()
                        m.running_var.fill_(1.0)
                        m.num_batches_tracked.zero_()
                        bn_reset_count += 1
            print(f"Reset running stats for {bn_reset_count} BN layers in segmentation subnet")

    def _map_scpnet_key(self, key: str) -> Optional[str]:
        """Map SCPNet state_dict key to our model's key.

        SCPNet structure:
          cylinder_3d_generator.* → self.cylinder_fea.*
          cylinder_3d_spconv_seg.{completion_layers} → self.completion.*
          cylinder_3d_spconv_seg.{seg_layers} → self.{seg_layers}
        """
        # Cylinder feature generator
        if key.startswith('cylinder_3d_generator.'):
            suffix = key[len('cylinder_3d_generator.'):]
            return f'cylinder_fea.{suffix}'

        # Completion subnet layers (in SCPNet these are on Asymm_3d_spconv)
        completion_prefixes = [
            'a_conv1', 'a_conv2', 'a_conv3', 'a_conv4',
            'a_conv5', 'a_conv6', 'a_conv7',
            'ch_conv1', 'res_1', 'res_2', 'res_3',
        ]
        if key.startswith('cylinder_3d_spconv_seg.'):
            suffix = key[len('cylinder_3d_spconv_seg.'):]
            # Check if it's a completion layer
            for prefix in completion_prefixes:
                if suffix.startswith(prefix):
                    return f'completion.{suffix}'

            # Segmentation subnet layers — map directly
            # downCntx, resBlock2-5, upBlock0-3, ReconNet, logits
            # These have the same names in our model
            return suffix

        return None
