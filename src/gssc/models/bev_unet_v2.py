"""
Modular 2D U-Net for the BEV second task with Configurable Conditioning.

This is a more flexible version of the BEV U-Net that supports:
- Different conditioning mechanisms (SUM, CONCAT, FiLM, BottleneckAttn, Hybrid)
- Different input resolutions (64x64 for architecture search, 256x256 for final)
- Configurable model sizes

Note: the released BEV second-task checkpoint uses ``conditioning_type='sum'``
(additive), matching the paper's additive/AdaGN conditioning story. The other
conditioning types listed above (CONCAT, FiLM, BottleneckAttn, Hybrid) are
research-time options exercised by no shipped config.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from gssc.models.bev_conditioning import (
    ConditioningType,
    IdentityConditioning,
    create_conditioning_block,
    get_num_groups,
)


def get_timestep_embedding(timesteps: torch.Tensor, embedding_dim: int) -> torch.Tensor:
    """Sinusoidal timestep embeddings."""
    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class AdaGN(nn.Module):
    """Adaptive Group Normalization with time embedding."""

    def __init__(self, num_channels: int, num_groups: int = 32, emb_dim: int = 256):
        super().__init__()
        # Find valid num_groups (must divide num_channels)
        num_groups = min(num_groups, num_channels)
        while num_channels % num_groups != 0 and num_groups > 1:
            num_groups -= 1
        self.norm = nn.GroupNorm(num_groups, num_channels, affine=False)
        self.proj = nn.Linear(emb_dim, num_channels * 2)

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        scale, shift = self.proj(emb).chunk(2, dim=-1)
        return x * (1 + scale[:, :, None, None]) + shift[:, :, None, None]


class ResBlock(nn.Module):
    """Residual block with time embedding and optional self-attention."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        emb_dim: int = 256,
        dropout: float = 0.1,
        use_attention: bool = False,
        num_heads: int = 4,
    ):
        super().__init__()

        self.use_attention = use_attention

        self.norm1 = AdaGN(in_channels, emb_dim=emb_dim)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm2 = AdaGN(out_channels, emb_dim=emb_dim)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.skip = nn.Identity()

        if use_attention:
            self.attn_norm = nn.GroupNorm(get_num_groups(out_channels), out_channels)
            self.attn = nn.MultiheadAttention(out_channels, num_heads, batch_first=True)

        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x, emb)
        h = self.act(h)
        h = self.conv1(h)
        h = self.norm2(h, emb)
        h = self.act(h)
        h = self.dropout(h)
        h = self.conv2(h)

        x = self.skip(x) + h

        if self.use_attention:
            B, C, H, W = x.shape
            h = self.attn_norm(x)
            h = h.reshape(B, C, H * W).permute(0, 2, 1)
            h, _ = self.attn(h, h, h)
            h = h.permute(0, 2, 1).reshape(B, C, H, W)
            x = x + h

        return x


class DownBlock(nn.Module):
    """Downsampling block with configurable conditioning."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        emb_dim: int = 256,
        num_blocks: int = 2,
        use_attention: bool = False,
        conditioning_type: ConditioningType = ConditioningType.SUM,
        cond_channels: int = 128,
        is_bottleneck: bool = False,
    ):
        super().__init__()

        self.blocks = nn.ModuleList()
        self.cond_blocks = nn.ModuleList()

        for i in range(num_blocks):
            ch_in = in_channels if i == 0 else out_channels
            self.blocks.append(ResBlock(ch_in, out_channels, emb_dim, use_attention=use_attention))

            # Create conditioning block
            cond_block = create_conditioning_block(
                conditioning_type, out_channels, cond_channels,
                is_bottleneck=is_bottleneck
            )
            self.cond_blocks.append(cond_block)

        self.downsample = nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1)

    def forward(
        self,
        x: torch.Tensor,
        emb: torch.Tensor,
        cond: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        for block, cond_block in zip(self.blocks, self.cond_blocks):
            x = block(x, emb)
            if cond is not None and not isinstance(cond_block, (nn.Identity, IdentityConditioning)):
                x = cond_block(x, cond)

        skip = x
        x = self.downsample(x)
        return x, skip


class UpBlock(nn.Module):
    """Upsampling block with configurable conditioning."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        skip_channels: int,
        emb_dim: int = 256,
        num_blocks: int = 2,
        use_attention: bool = False,
        conditioning_type: ConditioningType = ConditioningType.SUM,
        cond_channels: int = 128,
        is_bottleneck: bool = False,
    ):
        super().__init__()

        self.upsample = nn.ConvTranspose2d(in_channels, in_channels, 4, stride=2, padding=1)

        self.blocks = nn.ModuleList()
        self.cond_blocks = nn.ModuleList()

        for i in range(num_blocks):
            ch_in = in_channels + skip_channels if i == 0 else out_channels
            self.blocks.append(ResBlock(ch_in, out_channels, emb_dim, use_attention=use_attention))

            cond_block = create_conditioning_block(
                conditioning_type, out_channels, cond_channels,
                is_bottleneck=is_bottleneck
            )
            self.cond_blocks.append(cond_block)

    def forward(
        self,
        x: torch.Tensor,
        skip: torch.Tensor,
        emb: torch.Tensor,
        cond: torch.Tensor | None = None
    ) -> torch.Tensor:
        x = self.upsample(x)

        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)

        x = torch.cat([x, skip], dim=1)

        for block, cond_block in zip(self.blocks, self.cond_blocks):
            x = block(x, emb)
            if cond is not None and not isinstance(cond_block, (nn.Identity, IdentityConditioning)):
                x = cond_block(x, cond)

        return x


class ModularBEVUNet(nn.Module):
    """
    Modular 2D U-Net for BEV diffusion with configurable conditioning.

    Supports different:
    - Conditioning mechanisms (SUM, CONCAT, FiLM, BottleneckAttn, Hybrid)
    - Input resolutions (64, 128, 256)
    - Model sizes (lightweight, base)
    """

    def __init__(
        self,
        num_classes: int = 20,
        input_resolution: int = 64,  # 64 for architecture search, 256 for final
        base_channels: int = 64,
        channel_mults: tuple[int, ...] = (1, 2, 4, 8),
        num_res_blocks: int = 2,
        attention_resolutions: tuple[int, ...] = (16, 8),  # Relative to input
        dropout: float = 0.1,
        cond_channels: int = 64,
        conditioning_type: ConditioningType = ConditioningType.SUM,
        use_self_conditioning: bool = True,
        time_emb_dim: int = 256,
        use_input_residual: bool = False,  # Add residual from input to output
        residual_scale: float = 0.1,  # Scale for residual correction
        cond_drop_prob: float = 0.0,  # CFG: probability of dropping condition during training
    ):
        super().__init__()

        self.num_classes = num_classes
        self.cond_drop_prob = cond_drop_prob
        self.input_resolution = input_resolution
        self.use_self_conditioning = use_self_conditioning
        self.cond_channels = cond_channels
        self.conditioning_type = conditioning_type
        self.use_input_residual = use_input_residual

        # Fixed residual scale - small value forces model to rely on input
        # and only make small corrections. This prevents the model from
        # learning to ignore the input residual completely.
        if use_input_residual:
            # output = input + scale * correction
            # Smaller scale = model relies more on input
            self.register_buffer('residual_scale', torch.tensor(residual_scale))

        # Input channels
        in_channels = num_classes + 1  # +1 for MASK token
        if use_self_conditioning:
            in_channels += num_classes

        # Time embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(base_channels, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        # Input projection (always concatenate condition at input)
        self.input_proj = nn.Conv2d(in_channels + cond_channels, base_channels, 3, padding=1)

        # Build U-Net
        channels = [base_channels * m for m in channel_mults]
        self.down_blocks = nn.ModuleList()
        self.up_blocks = nn.ModuleList()

        resolution = input_resolution
        prev_ch = base_channels
        skip_channels = []

        # Determine bottleneck resolution
        bottleneck_resolution = input_resolution // (2 ** len(channel_mults))

        # Encoder
        for i, ch in enumerate(channels):
            res_after_down = resolution // 2
            use_attn = resolution in attention_resolutions
            is_bottleneck = (res_after_down <= bottleneck_resolution * 2)

            self.down_blocks.append(
                DownBlock(
                    prev_ch, ch, time_emb_dim, num_res_blocks,
                    use_attention=use_attn,
                    conditioning_type=conditioning_type,
                    cond_channels=cond_channels,
                    is_bottleneck=is_bottleneck,
                )
            )
            skip_channels.append(ch)
            prev_ch = ch
            resolution //= 2

        # Middle
        self.middle_block1 = ResBlock(prev_ch, prev_ch, time_emb_dim, use_attention=True)
        self.middle_cond = create_conditioning_block(
            conditioning_type, prev_ch, cond_channels, is_bottleneck=True
        )
        self.middle_block2 = ResBlock(prev_ch, prev_ch, time_emb_dim, use_attention=True)

        # Decoder
        num_up_blocks = len(channels) - 1
        for i in range(num_up_blocks):
            ch = channels[-(i + 2)]
            skip_ch = skip_channels[-(i + 1)]
            resolution *= 2
            use_attn = resolution in attention_resolutions
            is_bottleneck = (resolution <= bottleneck_resolution * 4)

            self.up_blocks.append(
                UpBlock(
                    prev_ch, ch, skip_ch, time_emb_dim, num_res_blocks,
                    use_attention=use_attn,
                    conditioning_type=conditioning_type,
                    cond_channels=cond_channels,
                    is_bottleneck=is_bottleneck,
                )
            )
            prev_ch = ch

        # Final upblock
        resolution *= 2
        self.final_up = UpBlock(
            prev_ch, base_channels, skip_channels[0], time_emb_dim, num_res_blocks,
            use_attention=False,
            conditioning_type=conditioning_type,
            cond_channels=cond_channels,
            is_bottleneck=False,
        )

        # Output
        self.output_norm = nn.GroupNorm(get_num_groups(base_channels), base_channels)
        self.output_proj = nn.Conv2d(base_channels, num_classes, 3, padding=1)

        # Auxiliary head
        self.aux_head = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(base_channels, num_classes, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        condition: torch.Tensor,
        prev_pred: torch.Tensor | None = None,
        return_aux: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Args:
            x: Noisy labels [B, H, W]
            t: Timesteps [B]
            condition: LiDAR BEV features [B, cond_channels, H, W]
            prev_pred: Previous prediction for self-conditioning [B, H, W]
            return_aux: Whether to return auxiliary output
        """
        B, H, W = x.shape
        device = x.device

        # CFG: Randomly drop condition during training
        if self.training and self.cond_drop_prob > 0:
            if torch.rand(1).item() < self.cond_drop_prob:
                condition = torch.zeros_like(condition)

        # One-hot encode
        x_onehot = F.one_hot(x.long(), num_classes=self.num_classes + 1).float()
        x_onehot = x_onehot.permute(0, 3, 1, 2)

        # Self-conditioning
        if self.use_self_conditioning:
            if prev_pred is not None:
                prev_onehot = F.one_hot(prev_pred.long(), num_classes=self.num_classes).float()
                prev_onehot = prev_onehot.permute(0, 3, 1, 2)
            else:
                prev_onehot = torch.zeros(B, self.num_classes, H, W, device=device)
            x_onehot = torch.cat([x_onehot, prev_onehot], dim=1)

        # Concatenate condition at input
        x_in = torch.cat([x_onehot, condition], dim=1)

        # Time embedding
        t_emb = get_timestep_embedding(t, self.time_mlp[0].in_features)
        t_emb = self.time_mlp(t_emb)

        # Input projection
        h = self.input_proj(x_in)

        # Encoder
        skips = []
        for down_block in self.down_blocks:
            h, skip = down_block(h, t_emb, condition)
            skips.append(skip)

        # Middle
        h = self.middle_block1(h, t_emb)
        if not isinstance(self.middle_cond, (nn.Identity, IdentityConditioning)):
            h = self.middle_cond(h, condition)
        h = self.middle_block2(h, t_emb)

        # Decoder
        for up_block in self.up_blocks:
            skip = skips.pop()
            h = up_block(h, skip, t_emb, condition)

        # Final up
        skip = skips.pop()
        h = self.final_up(h, skip, t_emb, condition)

        # Output
        h = self.output_norm(h)
        h = F.silu(h)
        logits = self.output_proj(h)

        # Input residual connection with fixed small scale
        # output = input + 0.1 * correction
        # This forces the model to rely on input and only make small corrections
        if self.use_input_residual:
            # x_onehot has shape [B, num_classes+1, H, W] (or more with self-cond)
            # Take only the first num_classes channels (exclude MASK token)
            input_logits = x_onehot[:, :self.num_classes, :, :].contiguous()
            # output = input + scale * correction (where logits is the correction)
            logits = (input_logits + self.residual_scale * logits).contiguous()

        # Auxiliary output
        aux_logits = None
        if return_aux and self.training:
            aux_logits = self.aux_head(h)
            if self.use_input_residual:
                aux_logits = (input_logits + self.residual_scale * aux_logits).contiguous()

        return logits, aux_logits


def create_modular_bev_unet(
    num_classes: int = 20,
    input_resolution: int = 64,
    conditioning_type: str = "sum",
    use_self_conditioning: bool = True,
    model_size: str = "base",
    cond_channels: int = 64,
    use_input_residual: bool = False,
    cond_drop_prob: float = 0.0,
) -> ModularBEVUNet:
    """
    Factory function to create ModularBEVUNet.

    Args:
        num_classes: Number of semantic classes
        input_resolution: BEV resolution (64 or 256)
        conditioning_type: One of "sum", "concat", "film", "bottleneck_attn", "hybrid"
        use_self_conditioning: Whether to use self-conditioning
        model_size: "lightweight" or "base"
        cond_channels: Condition feature channels
        use_input_residual: Add residual from input one-hot to output logits
        cond_drop_prob: CFG dropout probability (0.0 = no dropout, 0.1 = 10% dropout)
    """
    # Parse conditioning type
    cond_type = ConditioningType(conditioning_type)

    # Model size configurations
    if model_size == "lightweight":
        base_channels = 32
        channel_mults = (1, 2, 4, 4)
        num_res_blocks = 1
        attention_resolutions = (8,)  # Only at 8x8
    elif model_size == "base":
        base_channels = 64
        channel_mults = (1, 2, 4, 8)
        num_res_blocks = 2
        attention_resolutions = (16, 8)
    else:
        raise ValueError(f"Unknown model size: {model_size}")

    return ModularBEVUNet(
        num_classes=num_classes,
        input_resolution=input_resolution,
        base_channels=base_channels,
        channel_mults=channel_mults,
        num_res_blocks=num_res_blocks,
        attention_resolutions=attention_resolutions,
        cond_channels=cond_channels,
        conditioning_type=cond_type,
        use_self_conditioning=use_self_conditioning,
        use_input_residual=use_input_residual,
        cond_drop_prob=cond_drop_prob,
    )
