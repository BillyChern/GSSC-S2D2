"""
Exact Denoise class from pyramid discrete diffusion original codebase.
Copy of pyramid-discrete-diffusion/models/conditional_diffusion/con_denoise.py

THIRD-PARTY CODE -- NOT covered by this repository's MIT grant to the GSSC-S2D2
authors. This file is a substantial copy -- 345 matched lines, 96.6% of the
upstream file and 84.8% of this one (2026-08-22, upstream commit 964fca4) -- of:

    pyramid-discrete-diffusion
    https://github.com/yuhengliu02/pyramid-discrete-diffusion
    MIT License, Copyright (c) 2023 Yuheng Liu

MIT requires that the above copyright notice and the permission notice be
included in all copies or substantial portions of the Software. The full
permission notice is reproduced in THIRD_PARTY_NOTICES.md (section 1) at the
repository root, which also ships inside the wheel at
``gssc/THIRD_PARTY_NOTICES.md``. Do not redistribute this file without it.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import einsum


def conv3x3x3(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

def conv1x3x3(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes, out_planes, kernel_size=(1, 3, 3), stride=stride,padding=(0, 1, 1), bias=False)


def conv1x1x3(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes, out_planes, kernel_size=(1, 1, 3), stride=stride, padding=(0, 0, 1), bias=False)


def conv1x3x1(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes, out_planes, kernel_size=(1, 3, 1), stride=stride, padding=(0, 1, 0), bias=False)


def conv3x1x1(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes, out_planes, kernel_size=(3, 1, 1), stride=stride, padding=(1, 0, 0), bias=False)


def conv3x1x3(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes, out_planes, kernel_size=(3, 1, 3), stride=stride, padding=(1, 0, 1), bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=stride)


class Asymmetric_Residual_Block(nn.Module):
    def __init__(self, in_filters, out_filters, time_filters=32*4):
        super().__init__()
        if in_filters<32 :
            self.GroupNorm = nn.GroupNorm(16, in_filters)
            self.bn0 = nn.GroupNorm(16, out_filters)
            self.bn0_2 = nn.GroupNorm(16, out_filters)
            self.bn1 = nn.GroupNorm(16, out_filters)
            self.bn2 = nn.GroupNorm(16, out_filters)
        else :
            self.GroupNorm = nn.GroupNorm(32, in_filters)
            self.bn0 = nn.GroupNorm(32, out_filters)
            self.bn0_2 = nn.GroupNorm(32, out_filters)
            self.bn1 = nn.GroupNorm(32, out_filters)
            self.bn2 = nn.GroupNorm(32, out_filters)
        self.time_layers = nn.Sequential(
                            nn.SiLU(),
                            nn.Linear(time_filters, in_filters*2)
                        )

        self.conv1 = conv1x3x3(in_filters, out_filters)
        self.act1 = nn.LeakyReLU()

        self.conv1_2 = conv3x1x3(out_filters, out_filters)
        self.act1_2 = nn.LeakyReLU()

        self.conv2 = conv3x1x3(in_filters, out_filters)
        self.act2 = nn.LeakyReLU()

        self.conv3 = conv1x3x3(out_filters, out_filters)
        self.act3 = nn.LeakyReLU()


    def forward(self, x, t):
        t = self.time_layers(t)
        while len(t.shape) < len(x.shape):
            t = t[..., None]
        scale, shift = torch.chunk(t, 2, dim=1)

        x = self.GroupNorm(x) * (1 + scale) + shift

        shortcut = self.conv1(x)
        shortcut = self.act1(shortcut)
        shortcut = self.bn0(shortcut)

        shortcut = self.conv1_2(shortcut)
        shortcut = self.act1_2(shortcut)
        shortcut = self.bn0_2(shortcut)

        resA = self.conv2(x)
        resA = self.act2(resA)
        resA = self.bn1(resA)

        resA = self.conv3(resA)
        resA = self.act3(resA)
        resA = self.bn2(resA)
        resA += shortcut

        return resA

class DDCM(nn.Module):
    def __init__(self, in_filters, out_filters, kernel_size=(3, 3, 3), stride=1):
        super().__init__()
        self.conv1 = conv3x1x1(in_filters, out_filters)
        if in_filters<32 :
            self.bn0 = nn.GroupNorm(16, out_filters)
            self.bn0_2 = nn.GroupNorm(16, out_filters)
            self.bn0_3 = nn.GroupNorm(16, out_filters)
        else :
            self.bn0 = nn.GroupNorm(32, out_filters)
            self.bn0_2 = nn.GroupNorm(32, out_filters)
            self.bn0_3 = nn.GroupNorm(32, out_filters)
        self.act1 = nn.Sigmoid()

        self.conv1_2 = conv1x3x1(in_filters, out_filters)
        self.act1_2 = nn.Sigmoid()

        self.conv1_3 = conv1x1x3(in_filters, out_filters)
        self.act1_3 = nn.Sigmoid()

    def forward(self, x):
        shortcut = self.conv1(x)
        shortcut = self.bn0(shortcut)
        shortcut = self.act1(shortcut)

        shortcut2 = self.conv1_2(x)
        shortcut2 = self.bn0_2(shortcut2)
        shortcut2 = self.act1_2(shortcut2)

        shortcut3 = self.conv1_3(x)
        shortcut3 = self.bn0_3(shortcut3)
        shortcut3 = self.act1_3(shortcut3)
        shortcut = shortcut + shortcut2 + shortcut3

        shortcut = shortcut * x

        return shortcut

def l2norm(t):
    return F.normalize(t, dim = -1)

class Attention(nn.Module):
    def __init__(self, dim, heads = 4, scale = 10):
        super().__init__()
        self.scale = scale
        self.heads = heads
        self.to_qkv = conv1x1(dim, dim*3, stride=1)
        self.to_out = conv1x1(dim, dim, stride=1)

    def forward(self, x):
        b, c, h, w, Z = x.shape
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y z-> b h c (x y z)', h = self.heads), qkv)

        q, k = map(l2norm, (q, k))

        sim = einsum('b h d i, b h d j -> b h i j', q, k) * self.scale
        attn = sim.softmax(dim = -1)
        out = einsum('b h i j, b h d j -> b h i d', attn, v)
        out = rearrange(out, 'b h (x y z) d -> b (h d) x y z', x = h, y = w, z = Z)
        return self.to_out(out)

class Cross_Attention(nn.Module):
    def __init__(self, dim, heads = 4, scale = 10):
        super().__init__()
        self.scale = scale
        self.heads = heads
        self.to_q = conv1x1(dim, dim, stride=1)
        self.to_k = conv1x1(dim, dim, stride=1)
        self.to_v = conv1x1(dim, dim, stride=1)

        self.to_out = conv1x1(dim, dim, stride=1)

    def forward(self, x, cond_x):
        b, c, h, w, Z = x.shape
        q = self.to_q(x)
        k = self.to_k(cond_x)
        v = self.to_v(cond_x)

        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y z-> b h c (x y z)', h = self.heads), (q, k, v))

        q, k = map(l2norm, (q, k))

        sim = einsum('b h d i, b h d j -> b h i j', q, k) * self.scale
        attn = sim.softmax(dim = -1)
        out = einsum('b h i j, b h d j -> b h i d', attn, v)
        out = rearrange(out, 'b h (x y z) d -> b (h d) x y z', x = h, y = w, z = Z)
        return self.to_out(out)

class DownBlock(nn.Module):
    def __init__(self, in_filters, out_filters, time_filters=32*4, kernel_size=(3, 3, 3), stride=1,
                 pooling=True, height_pooling=False):
        super().__init__()
        self.pooling = pooling

        self.residual_block = Asymmetric_Residual_Block(in_filters, out_filters, time_filters=time_filters)

        if pooling:
            if height_pooling:
                self.pool = nn.Conv3d(out_filters, out_filters, kernel_size=3, stride=2,padding=1, bias=False)
            else:
                self.pool = nn.Conv3d(out_filters, out_filters, kernel_size=3, stride=(2, 2, 1),padding=1, bias=False)

    def forward(self, x, t):
        resA = self.residual_block(x, t)
        if self.pooling:
            resB = self.pool(resA)
            return resB, resA
        else:
            return resA

class UpBlock(nn.Module):
    def __init__(self, in_filters, out_filters, height_pooling, time_filters=32*4):
        super().__init__()
        if out_filters<32 :
            self.trans_bn = nn.GroupNorm(16, in_filters)
            self.bn1 = nn.GroupNorm(16, out_filters)
            self.bn2 = nn.GroupNorm(16, out_filters)
            self.bn3 = nn.GroupNorm(16, out_filters)
        else :
            self.trans_bn = nn.GroupNorm(32, in_filters)
            self.bn1 = nn.GroupNorm(32, out_filters)
            self.bn2 = nn.GroupNorm(32, out_filters)
            self.bn3 = nn.GroupNorm(32, out_filters)
        self.trans_dilao = conv3x3x3(in_filters, in_filters)
        self.trans_act = nn.LeakyReLU()
        self.time_layers = nn.Sequential(
                            nn.SiLU(),
                            nn.Linear(time_filters, in_filters*2)
                        )

        self.conv1 = conv1x3x3(in_filters, out_filters)
        self.act1 = nn.LeakyReLU()

        self.conv2 = conv3x1x3(out_filters, out_filters)
        self.act2 = nn.LeakyReLU()

        self.conv3 = conv3x3x3(out_filters, out_filters)
        self.act3 = nn.LeakyReLU()

        if height_pooling :
            self.up_subm = nn.ConvTranspose3d(in_filters, in_filters, kernel_size=3, bias=False, stride=2, padding=1, output_padding=1, dilation=1)
        else :
            self.up_subm = nn.ConvTranspose3d(in_filters, in_filters, kernel_size=(3,3,1), bias=False, stride=(2,2,1), padding=(1,1,0), output_padding=(1,1,0), dilation=1)


    def forward(self, x, residual, t):
        upA = self.trans_dilao(x)
        upA = self.trans_act(upA)

        t = self.time_layers(t)
        while len(t.shape) < len(x.shape):
            t = t[..., None]
        scale, shift = torch.chunk(t, 2, dim=1)

        upA = self.trans_bn(upA) * (1 + scale) + shift
        upA = self.up_subm(upA)
        upA += residual
        upE = self.conv1(upA)
        upE = self.act1(upE)
        upE = self.bn1(upE)

        upE = self.conv2(upE)
        upE = self.act2(upE)
        upE = self.bn2(upE)

        upE = self.conv3(upE)
        upE = self.act3(upE)
        upE = self.bn3(upE)

        return upE

def timestep_embedding(timesteps, dim, max_period=10000, repeat_only=False):
    if not repeat_only:
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=timesteps.device)
        args = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    else:
        embedding = repeat(timesteps, 'b -> b d', d=dim)
    return embedding

class Denoise(nn.Module):
    def __init__(self, args, num_class = 11, init_size=32, discrete=True):
        super().__init__()
        self.args = args
        self.discrete = discrete
        self.num_class = num_class
        self.init_size = init_size
        self.time_size = self.init_size*4

        self.time_embed = nn.Sequential(
            nn.Linear(init_size, self.time_size),
            nn.SiLU(),
            nn.Linear(self.time_size, self.time_size),
        )

        self.embedding = nn.Embedding(self.num_class, init_size)

        if self.args.next_stage=='s_3':
            context_size = init_size + 2
        else:
            context_size = init_size + 1

        self.conv_in = nn.Conv3d(context_size, init_size, kernel_size=1, stride=1)

        self.A = Asymmetric_Residual_Block(init_size, init_size, time_filters=init_size*4)

        self.downBlock1 = DownBlock(init_size, 2 * init_size, height_pooling=True, time_filters=init_size*4)
        self.downBlock2 = DownBlock(2 * init_size, 4 * init_size, height_pooling=True, time_filters=init_size*4)
        self.downBlock3 = DownBlock(4 * init_size, 8 * init_size, height_pooling=False, time_filters=init_size*4)
        self.downBlock4 = DownBlock(8 * init_size, 16 * init_size, height_pooling=False, time_filters=init_size*4)
        self.midBlock1 = Asymmetric_Residual_Block(16 * init_size, 16 * init_size, time_filters=init_size*4)
        self.attention = Attention(16 * init_size, 32)
        self.midBlock2 = Asymmetric_Residual_Block(16 * init_size, 16 * init_size, time_filters=init_size*4)

        self.upBlock4 = UpBlock(16 * init_size, 8 * init_size, height_pooling=False, time_filters=init_size*4)
        self.upBlock3 = UpBlock(8 * init_size, 4 * init_size, height_pooling=False, time_filters=init_size*4)
        self.upBlock2 = UpBlock(4 * init_size, 2 * init_size, height_pooling=True, time_filters=init_size*4)
        self.upBlock1 = UpBlock(2 * init_size, 2 * init_size, height_pooling=True, time_filters=init_size*4)

        self.DDCM = DDCM(2 * init_size, 2 * init_size)
        self.logits = nn.Conv3d(4 * init_size, self.num_class, kernel_size=3, stride=1, padding=1, bias=True)

    def forward(self, x, x_cond, t):
        x = self.embedding(x)
        x = x.permute(0, 4, 1, 2, 3)

        x = torch.cat([x, x_cond], dim=1)
        x = self.conv_in(x)

        t = self.time_embed(timestep_embedding(t, self.init_size))

        x = self.A(x, t)

        down1c, down1b = self.downBlock1(x, t)
        down2c, down2b = self.downBlock2(down1c, t)
        down3c, down3b = self.downBlock3(down2c, t)
        down4c, down4b = self.downBlock4(down3c, t)

        down4c = self.midBlock1(down4c, t)
        down4c = self.attention(down4c)
        down4c = self.midBlock2(down4c, t)

        up4 = self.upBlock4(down4c, down4b, t)
        up3 = self.upBlock3(up4, down3b, t)
        up2 = self.upBlock2(up3, down2b, t)
        up1 = self.upBlock1(up2, down1b, t)
        up0 = self.DDCM(up1)
        up = torch.cat((up1, up0), 1)
        logits = self.logits(up)

        return logits


def create_pyramid_denoiser(stage: str, num_classes: int = 20):
    """
    Factory function to create exact Denoise models for different pyramid stages.
    Uses args mock to match original interface.

    CRITICAL: init_size must match original pyramid-discrete-diffusion configs!
    Evidence: pyramid-discrete-diffusion/configs/train_s_*.yaml
    """
    stage_sizes = {
        's1': (32, 32, 4),
        's2': (64, 64, 8),
        's3': (256, 256, 32)
    }

    if stage not in stage_sizes:
        raise ValueError(f"Unknown stage: {stage}. Must be one of {list(stage_sizes.keys())}")

    # Mock args object to match original interface
    class Args:
        def __init__(self, next_stage):
            self.next_stage = next_stage

    args = Args(f's_{stage[1]}')  # 's1' -> 's_1'

    # VERIFIED: Original uses DEFAULT init_size=32 for all stages!
    # Evidence: con_diffusion.py:85 calls Denoise(args, num_class) with NO init_size parameter
    # Evidence: con_denoise.py:287 has init_size=32 as default
    # Evidence: args.init_size=8 in config is ONLY for latent VAE, not conditional diffusion
    return Denoise(
        args=args,
        num_class=num_classes,
        init_size=32,  # Correct: matches original pyramid-discrete-diffusion
        discrete=True
    )