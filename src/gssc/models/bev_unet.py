"""
2D U-Net Backbone for BEV Diffusion

A time-conditioned U-Net for denoising BEV semantic maps.
Supports self-conditioning and classifier-free guidance.

References:
    - DDPM U-Net: https://arxiv.org/abs/2006.11239
    - DiffMap: https://arxiv.org/abs/2405.02008
    - Self-conditioning: https://arxiv.org/abs/2208.04202
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_timestep_embedding(timesteps: torch.Tensor, embedding_dim: int) -> torch.Tensor:
    """
    Sinusoidal timestep embeddings.

    Args:
        timesteps: [B] tensor of timesteps
        embedding_dim: Dimension of the embedding

    Returns:
        [B, embedding_dim] tensor of embeddings
    """
    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)

    if embedding_dim % 2 == 1:
        emb = F.pad(emb, (0, 1))

    return emb


class AdaGN(nn.Module):
    """
    Adaptive Group Normalization.

    Modulates normalization parameters based on timestep embedding.
    Similar to AdaLN used in DiT.
    """

    def __init__(self, num_channels: int, num_groups: int = 32, emb_dim: int = 256):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups, num_channels, affine=False)
        self.proj = nn.Linear(emb_dim, num_channels * 2)

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        # emb: [B, emb_dim]
        x = self.norm(x)
        scale, shift = self.proj(emb).chunk(2, dim=-1)
        scale = scale[:, :, None, None]
        shift = shift[:, :, None, None]
        return x * (1 + scale) + shift


class ResBlock(nn.Module):
    """
    Residual block with time embedding and optional self-attention.
    """

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

        # First conv
        self.norm1 = AdaGN(in_channels, emb_dim=emb_dim)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)

        # Second conv
        self.norm2 = AdaGN(out_channels, emb_dim=emb_dim)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        # Skip connection
        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.skip = nn.Identity()

        # Self-attention (optional)
        if use_attention:
            self.attn_norm = nn.GroupNorm(32, out_channels)
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
            h = h.reshape(B, C, H * W).permute(0, 2, 1)  # [B, H*W, C]
            h, _ = self.attn(h, h, h)
            h = h.permute(0, 2, 1).reshape(B, C, H, W)
            x = x + h

        return x


class CrossAttentionBlock(nn.Module):
    """
    SegDiff-style conditioning block using SUMMATION.

    Based on SegDiff (arXiv:2112.00390) which found that for pixel-space
    segmentation diffusion, summing condition features significantly
    outperforms both concatenation and cross-attention.

    Architecture:
        - Project condition to match feature channels
        - Sum with current features (residual learning)
        - Apply mixing convolutions

    This is O(n) complexity and proven effective for segmentation.

    References:
        - SegDiff: https://arxiv.org/abs/2112.00390
        - "The summation introduced as a conditioning approach outperforms
           concatenation on Vaihingen by a large margin"
    """

    def __init__(self, channels: int, cond_channels: int, num_heads: int = 4):
        super().__init__()

        self.norm = nn.GroupNorm(32, channels)
        self.cond_norm = nn.GroupNorm(min(32, cond_channels), cond_channels)

        # Project condition to match feature channels (like SegDiff's G encoder output)
        self.cond_proj = nn.Sequential(
            nn.Conv2d(cond_channels, channels, 3, padding=1),
            nn.GroupNorm(32, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

        # Mixing layers after summation (refines the combined features)
        self.mix = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(32, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W] - current features (from noisy label encoder path)
            cond: [B, C_cond, H, W] - conditioning features (from LiDAR encoder path)
        """
        # Normalize both paths
        h = self.norm(x)
        cond_h = self.cond_norm(cond)

        # Project condition to match channels
        cond_proj = self.cond_proj(cond_h)

        # SegDiff's key insight: SUM the features (not concat, not cross-attn)
        # This makes the task "learning the DPM of a residual model"
        combined = h + cond_proj

        # Mix the combined features
        out = self.mix(combined)

        # Residual connection
        return x + out


class DownBlock(nn.Module):
    """Downsampling block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        emb_dim: int = 256,
        num_blocks: int = 2,
        use_attention: bool = False,
        use_cross_attention: bool = False,
        cond_channels: int = 128,
    ):
        super().__init__()

        self.blocks = nn.ModuleList()
        self.cross_attns = nn.ModuleList() if use_cross_attention else None

        for i in range(num_blocks):
            ch_in = in_channels if i == 0 else out_channels
            self.blocks.append(ResBlock(ch_in, out_channels, emb_dim, use_attention=use_attention))

            if use_cross_attention:
                self.cross_attns.append(CrossAttentionBlock(out_channels, cond_channels))

        self.downsample = nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1)

    def forward(
        self,
        x: torch.Tensor,
        emb: torch.Tensor,
        cond: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            (downsampled output, skip connection)
        """
        for i, block in enumerate(self.blocks):
            x = block(x, emb)
            if self.cross_attns is not None and cond is not None:
                # Resize condition to match feature resolution
                cond_resized = F.interpolate(cond, size=x.shape[2:], mode='bilinear', align_corners=False)
                x = self.cross_attns[i](x, cond_resized)

        skip = x
        x = self.downsample(x)
        return x, skip


class UpBlock(nn.Module):
    """Upsampling block with skip connection."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        skip_channels: int,
        emb_dim: int = 256,
        num_blocks: int = 2,
        use_attention: bool = False,
        use_cross_attention: bool = False,
        cond_channels: int = 128,
    ):
        super().__init__()

        self.upsample = nn.ConvTranspose2d(in_channels, in_channels, 4, stride=2, padding=1)

        self.blocks = nn.ModuleList()
        self.cross_attns = nn.ModuleList() if use_cross_attention else None

        for i in range(num_blocks):
            ch_in = in_channels + skip_channels if i == 0 else out_channels
            self.blocks.append(ResBlock(ch_in, out_channels, emb_dim, use_attention=use_attention))

            if use_cross_attention:
                self.cross_attns.append(CrossAttentionBlock(out_channels, cond_channels))

    def forward(
        self,
        x: torch.Tensor,
        skip: torch.Tensor,
        emb: torch.Tensor,
        cond: torch.Tensor | None = None
    ) -> torch.Tensor:
        x = self.upsample(x)

        # Handle size mismatch
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)

        x = torch.cat([x, skip], dim=1)

        for i, block in enumerate(self.blocks):
            x = block(x, emb)
            if self.cross_attns is not None and cond is not None:
                cond_resized = F.interpolate(cond, size=x.shape[2:], mode='bilinear', align_corners=False)
                x = self.cross_attns[i](x, cond_resized)

        return x


class BEVUNet(nn.Module):
    """
    2D U-Net for BEV diffusion with self-conditioning support.

    Args:
        num_classes: Number of semantic classes (output channels)
        in_channels: Input channels (noisy labels one-hot + condition + self-cond)
        base_channels: Base channel count
        channel_mults: Channel multipliers for each resolution level
        num_res_blocks: Number of residual blocks per level
        attention_resolutions: Resolutions at which to use self-attention
        dropout: Dropout rate
        cond_channels: Conditioning feature channels (from LiDAR encoder)
        use_cross_attention: Whether to use cross-attention for conditioning
        use_self_conditioning: Whether to support self-conditioning
    """

    def __init__(
        self,
        num_classes: int = 20,
        base_channels: int = 128,
        channel_mults: tuple[int, ...] = (1, 2, 4, 8),
        num_res_blocks: int = 2,
        attention_resolutions: tuple[int, ...] = (32, 16),  # Use attention at 32x32 and 16x16
        dropout: float = 0.1,
        cond_channels: int = 128,
        use_cross_attention: bool = True,
        use_self_conditioning: bool = True,
        time_emb_dim: int = 256,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.use_self_conditioning = use_self_conditioning
        self.cond_channels = cond_channels

        # Input channels: one-hot labels + optional self-conditioning
        # For absorbing D3PM: num_classes + 1 (for MASK token)
        in_channels = num_classes + 1  # +1 for MASK token
        if use_self_conditioning:
            in_channels += num_classes  # Previous prediction (one-hot)

        # Time embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(base_channels, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        # Input projection (includes condition concatenation)
        self.input_proj = nn.Conv2d(in_channels + cond_channels, base_channels, 3, padding=1)

        # Build U-Net
        channels = [base_channels * m for m in channel_mults]
        self.down_blocks = nn.ModuleList()
        self.up_blocks = nn.ModuleList()

        # Input resolution (256 for BEV)
        resolution = 256

        # Encoder
        prev_ch = base_channels
        skip_channels = []

        for i, ch in enumerate(channels):
            use_attn = resolution in attention_resolutions
            self.down_blocks.append(
                DownBlock(
                    prev_ch, ch, time_emb_dim, num_res_blocks,
                    use_attention=use_attn,
                    use_cross_attention=use_cross_attention,
                    cond_channels=cond_channels,
                )
            )
            skip_channels.append(ch)
            prev_ch = ch
            resolution //= 2

        # Middle
        self.middle_block1 = ResBlock(prev_ch, prev_ch, time_emb_dim, use_attention=True)
        self.middle_cross_attn = CrossAttentionBlock(prev_ch, cond_channels) if use_cross_attention else None
        self.middle_block2 = ResBlock(prev_ch, prev_ch, time_emb_dim, use_attention=True)

        # Decoder
        # We have len(channels) down_blocks and len(channels) skip connections
        # We need len(channels)-1 up_blocks + 1 final_up = len(channels) total
        # Skips are popped in reverse order: last -> first
        num_up_blocks = len(channels) - 1
        for i in range(num_up_blocks):
            # Output channels decrease as we go up
            ch = channels[-(i + 2)]  # Start from second-to-last channel
            # Skip comes from corresponding down_block (popped in reverse)
            skip_ch = skip_channels[-(i + 1)]
            use_attn = resolution * 2 in attention_resolutions
            self.up_blocks.append(
                UpBlock(
                    prev_ch, ch, skip_ch, time_emb_dim, num_res_blocks,
                    use_attention=use_attn,
                    use_cross_attention=use_cross_attention,
                    cond_channels=cond_channels,
                )
            )
            prev_ch = ch
            resolution *= 2

        # Final upblock to return to original resolution
        # Uses the first skip connection (from first down_block)
        self.final_up = UpBlock(
            prev_ch, base_channels, skip_channels[0], time_emb_dim, num_res_blocks,
            use_attention=False,
            use_cross_attention=use_cross_attention,
            cond_channels=cond_channels,
        )

        # Output projection
        self.output_norm = nn.GroupNorm(32, base_channels)
        self.output_proj = nn.Conv2d(base_channels, num_classes, 3, padding=1)

        # Optional auxiliary head (for auxiliary loss)
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
        Forward pass.

        Args:
            x: Noisy labels [B, H, W] (integer labels, will be one-hot encoded)
            t: Timesteps [B]
            condition: LiDAR BEV features [B, cond_channels, H, W]
            prev_pred: Previous prediction for self-conditioning [B, H, W] (optional)
            return_aux: Whether to return auxiliary output

        Returns:
            logits: [B, num_classes, H, W]
            aux_logits: [B, num_classes, H, W] or None
        """
        B, H, W = x.shape
        device = x.device

        # One-hot encode noisy labels
        # x contains values 0 to num_classes (including MASK token = num_classes)
        x_onehot = F.one_hot(x.long(), num_classes=self.num_classes + 1).float()  # [B, H, W, C+1]
        x_onehot = x_onehot.permute(0, 3, 1, 2)  # [B, C+1, H, W]

        # Self-conditioning
        if self.use_self_conditioning:
            if prev_pred is not None:
                prev_onehot = F.one_hot(prev_pred.long(), num_classes=self.num_classes).float()
                prev_onehot = prev_onehot.permute(0, 3, 1, 2)
            else:
                prev_onehot = torch.zeros(B, self.num_classes, H, W, device=device)
            x_onehot = torch.cat([x_onehot, prev_onehot], dim=1)

        # Concatenate condition
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
        if self.middle_cross_attn is not None:
            cond_resized = F.interpolate(condition, size=h.shape[2:], mode='bilinear', align_corners=False)
            h = self.middle_cross_attn(h, cond_resized)
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

        # Auxiliary output
        aux_logits = None
        if return_aux and self.training:
            aux_logits = self.aux_head(h)

        if return_aux:
            return logits, aux_logits
        return logits


class LightweightBEVUNet(nn.Module):
    """
    Lightweight U-Net for fast experiments.

    Fewer channels and simpler architecture.
    """

    def __init__(
        self,
        num_classes: int = 20,
        base_channels: int = 64,
        cond_channels: int = 64,
        use_self_conditioning: bool = True,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.use_self_conditioning = use_self_conditioning

        in_channels = num_classes + 1  # +1 for MASK
        if use_self_conditioning:
            in_channels += num_classes

        time_dim = 128

        # Time embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(64, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels + cond_channels, base_channels, 3, padding=1),
            nn.GroupNorm(8, base_channels),
            nn.SiLU(),
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.GroupNorm(8, base_channels),
            nn.SiLU(),
        )
        self.down1 = nn.Conv2d(base_channels, base_channels, 3, stride=2, padding=1)

        self.enc2 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 2, 3, padding=1),
            nn.GroupNorm(8, base_channels * 2),
            nn.SiLU(),
            nn.Conv2d(base_channels * 2, base_channels * 2, 3, padding=1),
            nn.GroupNorm(8, base_channels * 2),
            nn.SiLU(),
        )
        self.down2 = nn.Conv2d(base_channels * 2, base_channels * 2, 3, stride=2, padding=1)

        # Middle
        self.mid = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels * 4, 3, padding=1),
            nn.GroupNorm(8, base_channels * 4),
            nn.SiLU(),
            nn.Conv2d(base_channels * 4, base_channels * 2, 3, padding=1),
            nn.GroupNorm(8, base_channels * 2),
            nn.SiLU(),
        )

        # Time injection
        self.time_proj = nn.Linear(time_dim, base_channels * 2)

        # Decoder
        self.up2 = nn.ConvTranspose2d(base_channels * 2, base_channels * 2, 4, stride=2, padding=1)
        self.dec2 = nn.Sequential(
            nn.Conv2d(base_channels * 4, base_channels * 2, 3, padding=1),
            nn.GroupNorm(8, base_channels * 2),
            nn.SiLU(),
            nn.Conv2d(base_channels * 2, base_channels, 3, padding=1),
            nn.GroupNorm(8, base_channels),
            nn.SiLU(),
        )

        self.up1 = nn.ConvTranspose2d(base_channels, base_channels, 4, stride=2, padding=1)
        self.dec1 = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels, 3, padding=1),
            nn.GroupNorm(8, base_channels),
            nn.SiLU(),
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.GroupNorm(8, base_channels),
            nn.SiLU(),
        )

        # Output
        self.output = nn.Conv2d(base_channels, num_classes, 1)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        condition: torch.Tensor,
        prev_pred: torch.Tensor | None = None,
        return_aux: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, H, W = x.shape
        device = x.device

        # One-hot
        x_onehot = F.one_hot(x.long(), num_classes=self.num_classes + 1).float()
        x_onehot = x_onehot.permute(0, 3, 1, 2)

        if self.use_self_conditioning:
            if prev_pred is not None:
                prev_onehot = F.one_hot(prev_pred.long(), num_classes=self.num_classes).float()
                prev_onehot = prev_onehot.permute(0, 3, 1, 2)
            else:
                prev_onehot = torch.zeros(B, self.num_classes, H, W, device=device)
            x_onehot = torch.cat([x_onehot, prev_onehot], dim=1)

        x_in = torch.cat([x_onehot, condition], dim=1)

        # Time
        t_emb = get_timestep_embedding(t, 64)
        t_emb = self.time_mlp(t_emb)

        # Encode
        e1 = self.enc1(x_in)
        e2 = self.enc2(self.down1(e1))
        m = self.mid(self.down2(e2))

        # Add time
        t_proj = self.time_proj(t_emb)[:, :, None, None]
        m = m + t_proj

        # Decode
        d2 = self.dec2(torch.cat([self.up2(m), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        logits = self.output(d1)

        # Always return tuple for consistency with BEVUNet
        return logits, None
