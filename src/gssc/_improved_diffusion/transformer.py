"""
Adapted from: https://github.com/openai/openai/blob/55363aa496049423c37124b440e9e30366db3ed6/orc/orc/diffusion/vit.py
"""


import math
from collections.abc import Sequence

import torch
import torch.nn as nn

#from .checkpoint import checkpoint
from .nn import checkpoint, timestep_embedding

#from .util import timestep_embedding


def init_linear(l, stddev):
    nn.init.normal_(l.weight, std=stddev)
    if l.bias is not None:
        nn.init.constant_(l.bias, 0.0)


class MultiheadAttention(nn.Module):
    def __init__(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
        n_ctx: int,
        width: int,
        heads: int,
        init_scale: float,
    ):
        super().__init__()
        self.n_ctx = n_ctx
        self.width = width
        self.heads = heads
        self.c_qkv = nn.Linear(width, width * 3, device=device, dtype=dtype)
        self.c_proj = nn.Linear(width, width, device=device, dtype=dtype)
        self.attention = QKVMultiheadAttention(device=device, dtype=dtype, heads=heads, n_ctx=n_ctx)
        init_linear(self.c_qkv, init_scale)
        init_linear(self.c_proj, init_scale)

    def forward(self, x, attn_mask=None):
        x = self.c_qkv(x)
        x = checkpoint(self.attention, (x,attn_mask), (), True)
        x = self.c_proj(x)
        return x


class MLP(nn.Module):
    def __init__(self, *, device: torch.device, dtype: torch.dtype, width: int, init_scale: float):
        super().__init__()
        self.width = width
        self.c_fc = nn.Linear(width, width * 4, device=device, dtype=dtype)
        self.c_proj = nn.Linear(width * 4, width, device=device, dtype=dtype)
        self.gelu = nn.GELU()
        init_linear(self.c_fc, init_scale)
        init_linear(self.c_proj, init_scale)

    def forward(self, x):
        return self.c_proj(self.gelu(self.c_fc(x)))


class QKVMultiheadAttention(nn.Module):
    def __init__(self, *, device: torch.device, dtype: torch.dtype, heads: int, n_ctx: int):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.heads = heads
        self.n_ctx = n_ctx

    def forward(self, qkv, attn_mask=None):
        bs, n_ctx, width = qkv.shape
        attn_ch = width // self.heads // 3
        scale = 1 / math.sqrt(math.sqrt(attn_ch))
        qkv = qkv.view(bs, n_ctx, self.heads, -1)
        q, k, v = torch.split(qkv, attn_ch, dim=-1)
        weight = torch.einsum(
            "bthc,bshc->bhts", q * scale, k * scale
        )  # More stable with f16 than dividing afterwards
        #print(self.heads, q.shape, k.shape, v.shape, weight.shape, attn_mask.shape)
        
        wdtype = weight.dtype
        if attn_mask is not None:
            weight = weight * attn_mask.unsqueeze(1)
        weight = torch.softmax(weight.float(), dim=-1).type(wdtype)
        return torch.einsum("bhts,bshc->bthc", weight, v).reshape(bs, n_ctx, -1)


class ResidualAttentionBlock(nn.Module):
    def __init__(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
        n_ctx: int,
        width: int,
        heads: int,
        init_scale: float = 1.0,
    ):
        super().__init__()

        self.attn = MultiheadAttention(
            device=device,
            dtype=dtype,
            n_ctx=n_ctx,
            width=width,
            heads=heads,
            init_scale=init_scale,
        )
        self.ln_1 = nn.LayerNorm(width, device=device, dtype=dtype)
        self.mlp = MLP(device=device, dtype=dtype, width=width, init_scale=init_scale)
        self.ln_2 = nn.LayerNorm(width, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor, attn_mask=None):
        x = x + self.attn(self.ln_1(x), attn_mask)
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
        n_ctx: int,
        width: int,
        layers: int,
        heads: int,
        init_scale: float = 0.25,
    ):
        super().__init__()
        self.n_ctx = n_ctx
        self.width = width
        self.layers = layers
        init_scale = init_scale * math.sqrt(1.0 / width)
        self.resblocks = nn.ModuleList(
            [
                ResidualAttentionBlock(
                    device=device,
                    dtype=dtype,
                    n_ctx=n_ctx,
                    width=width,
                    heads=heads,
                    init_scale=init_scale,
                )
                for _ in range(layers)
            ]
        )

    def forward(self, x: torch.Tensor, attn_mask=None):
        for block in self.resblocks:
            x = block(x, attn_mask)
        return x


class PointDiffusionTransformer(nn.Module):
    def __init__(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
        input_channels: int = 3,
        output_channels: int = 3,
        n_ctx: int = 1024,
        width: int = 512,
        layers: int = 12,
        heads: int = 8,
        init_scale: float = 0.25,
        time_token_cond: bool = False,
        use_lidar: bool = False,
    ):
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.n_ctx = n_ctx
        self.use_lidar = use_lidar
        self.time_token_cond = time_token_cond
        self.time_embed = MLP(
            device=device, dtype=dtype, width=width, init_scale=init_scale * math.sqrt(1.0 / width)
        )
        self.ln_pre = nn.LayerNorm(width, device=device, dtype=dtype)
        self.backbone = Transformer(
            device=device,
            dtype=dtype,
            n_ctx=n_ctx + int(time_token_cond),
            width=width,
            layers=layers,
            heads=heads,
            init_scale=init_scale,
        )
        self.ln_post = nn.LayerNorm(width, device=device, dtype=dtype)
        self.input_proj = nn.Linear(input_channels, width, device=device, dtype=dtype)
        self.output_proj = nn.Linear(width, output_channels, device=device, dtype=dtype)
        if use_lidar:
            self.lidar_proj = nn.Linear(3, width, device=device, dtype=dtype)
        
        with torch.no_grad():
            self.output_proj.weight.zero_()
            self.output_proj.bias.zero_()

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor = None, attn_mask: torch.Tensor = None, x_paths=None, classifier_free=False, condition_scale=False):
        """
        :param x: an [N x C x T] tensor.
        :param t: an [N] tensor.
        :return: an [N x C' x T] tensor.
        """
        assert x.shape[-1] == self.n_ctx
        #print(x.shape)
        t_embed = self.time_embed(timestep_embedding(t, self.backbone.width))
        lidar_embed = self.lidar_proj(y)
        cond = [(t_embed, self.time_token_cond)]
        if self.use_lidar:
            cond += [(lidar_embed, True)]
        return self._forward_with_cond(x, cond, attn_mask)

    def _forward_with_cond(
        self, x: torch.Tensor, cond_as_token: list[tuple[torch.Tensor, bool]], attn_mask=None,
    ) -> torch.Tensor:
        h = self.input_proj(x.permute(0, 2, 1))  # NCL -> NLC
        for emb, as_token in cond_as_token:
            if not as_token:
                h = h + emb[:, None]
        extra_tokens = [
            (emb[:, None] if len(emb.shape) == 2 else emb)
            for emb, as_token in cond_as_token
            if as_token
        ]
        #print(extra_tokens[0].shape, h.shape)
        if len(extra_tokens):
            h = torch.cat(extra_tokens + [h], dim=1)

        h = self.ln_pre(h)
        h = self.backbone(h, attn_mask)
        h = self.ln_post(h)
        if len(extra_tokens):
            h = h[:, sum(h.shape[1] for h in extra_tokens) :]
        h = self.output_proj(h)
        return h.permute(0, 2, 1)

class UpsamplePointDiffusionTransformer(PointDiffusionTransformer):
    def __init__(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
        cond_input_channels: int | None = None,
        cond_ctx: int = 1024,
        n_ctx: int = 4096 - 1024,
        channel_scales: Sequence[float] | None = None,
        channel_biases: Sequence[float] | None = None,
        **kwargs,
    ):
        super().__init__(device=device, dtype=dtype, n_ctx=n_ctx + cond_ctx, **kwargs)
        self.n_ctx = n_ctx
        self.cond_input_channels = cond_input_channels or self.input_channels
        self.cond_point_proj = nn.Linear(
            self.cond_input_channels, self.backbone.width, device=device, dtype=dtype
        )

        self.register_buffer(
            "channel_scales",
            torch.tensor(channel_scales, dtype=dtype, device=device)
            if channel_scales is not None
            else None,
        )
        self.register_buffer(
            "channel_biases",
            torch.tensor(channel_biases, dtype=dtype, device=device)
            if channel_biases is not None
            else None,
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor, *, low_res: torch.Tensor):
        """
        :param x: an [N x C1 x T] tensor.
        :param t: an [N] tensor.
        :param low_res: an [N x C2 x T'] tensor of conditioning points.
        :return: an [N x C3 x T] tensor.
        """
        assert x.shape[-1] == self.n_ctx
        t_embed = self.time_embed(timestep_embedding(t, self.backbone.width))
        low_res_embed = self._embed_low_res(low_res)
        cond = [(t_embed, self.time_token_cond), (low_res_embed, True)]
        return self._forward_with_cond(x, cond)

    def _embed_low_res(self, x: torch.Tensor) -> torch.Tensor:
        if self.channel_scales is not None:
            x = x * self.channel_scales[None, :, None]
        if self.channel_biases is not None:
            x = x + self.channel_biases[None, :, None]
        return self.cond_point_proj(x.permute(0, 2, 1))

