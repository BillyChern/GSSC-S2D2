"""
Modular Conditioning Mechanisms for BEV Diffusion.

This module implements different ways to incorporate LiDAR conditioning into the diffusion U-Net:

1. SUM (SegDiff-style): Add projected condition features to current features
2. CONCAT: Concatenate condition features channel-wise
3. FiLM: Feature-wise Linear Modulation (scale + shift)
4. BOTTLENECK_ATTN: Full cross-attention only at bottleneck resolution (e.g., 8x8)
5. HYBRID: Concatenation at all scales + cross-attention at bottleneck

References:
    - SegDiff: https://arxiv.org/abs/2112.00390 (SUM conditioning)
    - FiLM: https://arxiv.org/abs/1709.07871 (Feature-wise Linear Modulation)
"""

from enum import Enum

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_num_groups(channels: int, max_groups: int = 32) -> int:
    """Get valid number of groups for GroupNorm (must divide channels)."""
    num_groups = min(max_groups, channels)
    while channels % num_groups != 0 and num_groups > 1:
        num_groups -= 1
    return num_groups


class ConditioningType(Enum):
    """Conditioning mechanism types."""
    SUM = "sum"
    CONCAT = "concat"
    FILM = "film"
    BOTTLENECK_ATTN = "bottleneck_attn"
    HYBRID = "hybrid"
    # New: Gated conditioning (GSM-style from DifFUSER)
    GATED_SUM = "gated_sum"
    GATED_FILM = "gated_film"


class SumConditioningBlock(nn.Module):
    """
    SegDiff-style SUM conditioning.

    Projects condition features to match current features, then adds them.
    This makes the model learn a "residual" from the condition.

    Reference: SegDiff (arXiv:2112.00390)
    """

    def __init__(self, channels: int, cond_channels: int):
        super().__init__()

        self.cond_proj = nn.Sequential(
            nn.Conv2d(cond_channels, channels, 3, padding=1),
            nn.GroupNorm(get_num_groups(channels), channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

        self.mix = nn.Sequential(
            nn.GroupNorm(get_num_groups(channels), channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W] current features
            cond: [B, C_cond, H, W] condition features (will be resized if needed)
        """
        # Resize condition if needed
        if cond.shape[2:] != x.shape[2:]:
            cond = F.interpolate(cond, size=x.shape[2:], mode='bilinear', align_corners=False)

        # Project and add
        cond_proj = self.cond_proj(cond)
        combined = x + cond_proj

        # Mix
        out = self.mix(combined)

        return x + out


class ConcatConditioningBlock(nn.Module):
    """
    Simple concatenation conditioning.

    Concatenates condition features channel-wise, then projects back.
    Simple but increases channel count temporarily.
    """

    def __init__(self, channels: int, cond_channels: int):
        super().__init__()

        self.proj = nn.Sequential(
            nn.Conv2d(channels + cond_channels, channels, 3, padding=1),
            nn.GroupNorm(get_num_groups(channels), channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # Resize condition if needed
        if cond.shape[2:] != x.shape[2:]:
            cond = F.interpolate(cond, size=x.shape[2:], mode='bilinear', align_corners=False)

        # Concatenate and project
        combined = torch.cat([x, cond], dim=1)
        out = self.proj(combined)

        return x + out


class FiLMConditioningBlock(nn.Module):
    """
    Feature-wise Linear Modulation (FiLM) conditioning.

    Predicts scale and shift parameters from condition, applies to features.
    Efficient but limited expressiveness (global modulation).

    Reference: FiLM (arXiv:1709.07871)
    """

    def __init__(self, channels: int, cond_channels: int):
        super().__init__()

        # Pool condition to get global features
        self.pool = nn.AdaptiveAvgPool2d(1)

        # Predict scale and shift from condition
        self.film_proj = nn.Sequential(
            nn.Linear(cond_channels, channels * 2),
            nn.SiLU(),
            nn.Linear(channels * 2, channels * 2),
        )

        self.norm = nn.GroupNorm(get_num_groups(channels), channels)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # Global pool condition
        cond_global = self.pool(cond).flatten(1)  # [B, C_cond]

        # Predict scale and shift
        film_params = self.film_proj(cond_global)  # [B, C*2]
        scale, shift = film_params.chunk(2, dim=1)
        scale = scale.unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1]
        shift = shift.unsqueeze(-1).unsqueeze(-1)

        # Apply FiLM
        h = self.norm(x)
        out = h * (1 + scale) + shift

        return out


class GatedSumConditioningBlock(nn.Module):
    """
    Gated SUM conditioning (GSM-style from DifFUSER).

    Adds a learnable gate that controls how much conditioning is applied per-pixel.
    This is especially useful for sparse LiDAR where some regions have no observations.

    Formula: out = gate * (x + cond_proj) + (1 - gate) * x
           = x + gate * cond_proj

    The gate learns to weight conditioning based on local reliability.

    Reference: DifFUSER (ECCV 2024) - Gated Self-conditioned Modulated module
    """

    def __init__(self, channels: int, cond_channels: int):
        super().__init__()

        # Project condition features
        self.cond_proj = nn.Sequential(
            nn.Conv2d(cond_channels, channels, 3, padding=1),
            nn.GroupNorm(get_num_groups(channels), channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

        # Gate prediction from condition (sigmoid output)
        self.gate_proj = nn.Sequential(
            nn.Conv2d(cond_channels, channels // 2, 3, padding=1),
            nn.GroupNorm(get_num_groups(channels // 2), channels // 2),
            nn.SiLU(),
            nn.Conv2d(channels // 2, channels, 3, padding=1),
            nn.Sigmoid(),  # Output in [0, 1]
        )

        self.mix = nn.Sequential(
            nn.GroupNorm(get_num_groups(channels), channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W] current features
            cond: [B, C_cond, H, W] condition features
        """
        # Resize condition if needed
        if cond.shape[2:] != x.shape[2:]:
            cond = F.interpolate(cond, size=x.shape[2:], mode='bilinear', align_corners=False)

        # Project condition and compute gate
        cond_proj = self.cond_proj(cond)
        gate = self.gate_proj(cond)  # [B, C, H, W] in [0, 1]

        # Gated addition: apply conditioning where gate is high
        combined = x + gate * cond_proj

        # Mix
        out = self.mix(combined)

        return x + out


class GatedFiLMConditioningBlock(nn.Module):
    """
    Gated Feature-wise Linear Modulation (Gated FiLM) conditioning.

    Extends FiLM with a per-pixel gate that controls modulation strength.
    This combines the efficiency of FiLM with spatial selectivity.

    Formula: out = gate * (x * (1 + scale) + shift) + (1 - gate) * x

    Reference: DifFUSER (ECCV 2024) - GSM module combines gate, scale, shift
    """

    def __init__(self, channels: int, cond_channels: int, spatial_gate: bool = True):
        super().__init__()

        self.spatial_gate = spatial_gate

        # FiLM parameters (global, like original FiLM)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.film_proj = nn.Sequential(
            nn.Linear(cond_channels, channels * 2),
            nn.SiLU(),
            nn.Linear(channels * 2, channels * 2),
        )

        # Gate prediction (spatial - per-pixel gate from local features)
        if spatial_gate:
            self.gate_proj = nn.Sequential(
                nn.Conv2d(cond_channels, channels // 2, 3, padding=1),
                nn.GroupNorm(get_num_groups(channels // 2), channels // 2),
                nn.SiLU(),
                nn.Conv2d(channels // 2, channels, 3, padding=1),
                nn.Sigmoid(),
            )
        else:
            # Global gate (simpler)
            self.gate_proj = nn.Sequential(
                nn.Linear(cond_channels, channels),
                nn.Sigmoid(),
            )

        self.norm = nn.GroupNorm(get_num_groups(channels), channels)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        # Resize condition if needed
        if cond.shape[2:] != x.shape[2:]:
            cond = F.interpolate(cond, size=x.shape[2:], mode='bilinear', align_corners=False)

        # Global FiLM parameters (scale and shift)
        cond_global = self.pool(cond).flatten(1)  # [B, C_cond]
        film_params = self.film_proj(cond_global)  # [B, C*2]
        scale, shift = film_params.chunk(2, dim=1)
        scale = scale.unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1]
        shift = shift.unsqueeze(-1).unsqueeze(-1)

        # Spatial gate (per-pixel)
        if self.spatial_gate:
            gate = self.gate_proj(cond)  # [B, C, H, W]
        else:
            gate = self.gate_proj(cond_global).unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1]

        # Apply gated FiLM
        h = self.norm(x)
        modulated = h * (1 + scale) + shift

        # Gate blends between original and modulated
        out = gate * modulated + (1 - gate) * h

        return out


class BottleneckAttentionBlock(nn.Module):
    """
    Full cross-attention for conditioning.

    Uses multi-head attention between features and condition.
    Expensive at high resolution, so typically only used at bottleneck (8x8).

    This is the standard cross-attention used in Stable Diffusion, but we
    only use it at low resolution where it's tractable.
    """

    def __init__(self, channels: int, cond_channels: int, num_heads: int = 4):
        super().__init__()

        self.num_heads = num_heads

        # Project condition
        self.cond_proj = nn.Conv2d(cond_channels, channels, 1)

        # Attention
        self.norm = nn.GroupNorm(get_num_groups(channels), channels)
        self.cond_norm = nn.GroupNorm(get_num_groups(channels), channels)

        self.q_proj = nn.Conv2d(channels, channels, 1)
        self.k_proj = nn.Conv2d(channels, channels, 1)
        self.v_proj = nn.Conv2d(channels, channels, 1)
        self.out_proj = nn.Conv2d(channels, channels, 1)

        self.scale = (channels // num_heads) ** -0.5

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        # Resize and project condition
        if cond.shape[2:] != x.shape[2:]:
            cond = F.interpolate(cond, size=(H, W), mode='bilinear', align_corners=False)
        cond = self.cond_proj(cond)

        # Normalize
        x_norm = self.norm(x)
        cond_norm = self.cond_norm(cond)

        # Project to Q, K, V
        q = self.q_proj(x_norm)  # [B, C, H, W]
        k = self.k_proj(cond_norm)
        v = self.v_proj(cond_norm)

        # Reshape for multi-head attention
        head_dim = C // self.num_heads
        q = q.view(B, self.num_heads, head_dim, H * W).permute(0, 1, 3, 2)  # [B, heads, HW, head_dim]
        k = k.view(B, self.num_heads, head_dim, H * W).permute(0, 1, 3, 2)
        v = v.view(B, self.num_heads, head_dim, H * W).permute(0, 1, 3, 2)

        # Attention
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B, heads, HW, HW]
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)  # [B, heads, HW, head_dim]

        # Reshape back
        out = out.permute(0, 1, 3, 2).reshape(B, C, H, W)
        out = self.out_proj(out)

        return x + out


class HybridConditioningBlock(nn.Module):
    """
    Hybrid conditioning: Concatenation + Bottleneck Attention.

    Combines the simplicity of concatenation with the expressiveness of attention.
    Uses concatenation for all scales, attention only at bottleneck.
    """

    def __init__(self, channels: int, cond_channels: int, use_attention: bool = False, num_heads: int = 4):
        super().__init__()

        self.use_attention = use_attention

        # Concatenation path (always used)
        self.concat_proj = nn.Sequential(
            nn.Conv2d(channels + cond_channels, channels, 3, padding=1),
            nn.GroupNorm(get_num_groups(channels), channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

        # Attention path (only at bottleneck)
        if use_attention:
            self.attention = BottleneckAttentionBlock(channels, cond_channels, num_heads)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # Resize condition if needed
        if cond.shape[2:] != x.shape[2:]:
            cond = F.interpolate(cond, size=x.shape[2:], mode='bilinear', align_corners=False)

        # Concatenation path
        combined = torch.cat([x, cond], dim=1)
        concat_out = self.concat_proj(combined)

        # Attention path (if enabled)
        if self.use_attention:
            attn_out = self.attention(x + concat_out, cond)
            return attn_out
        else:
            return x + concat_out


def create_conditioning_block(
    conditioning_type: ConditioningType,
    channels: int,
    cond_channels: int,
    is_bottleneck: bool = False,
    num_heads: int = 4,
) -> nn.Module:
    """
    Factory function to create conditioning block based on type.

    Args:
        conditioning_type: Type of conditioning mechanism
        channels: Current feature channels
        cond_channels: Condition feature channels
        is_bottleneck: Whether this is the bottleneck layer
        num_heads: Number of attention heads (for attention-based methods)
    """
    if conditioning_type == ConditioningType.SUM:
        return SumConditioningBlock(channels, cond_channels)

    elif conditioning_type == ConditioningType.CONCAT:
        return ConcatConditioningBlock(channels, cond_channels)

    elif conditioning_type == ConditioningType.FILM:
        return FiLMConditioningBlock(channels, cond_channels)

    elif conditioning_type == ConditioningType.BOTTLENECK_ATTN:
        # Only use attention at bottleneck, identity otherwise
        if is_bottleneck:
            return BottleneckAttentionBlock(channels, cond_channels, num_heads)
        else:
            return nn.Identity()  # No conditioning at non-bottleneck scales

    elif conditioning_type == ConditioningType.HYBRID:
        # Concat at all scales, attention only at bottleneck
        return HybridConditioningBlock(channels, cond_channels, use_attention=is_bottleneck, num_heads=num_heads)

    elif conditioning_type == ConditioningType.GATED_SUM:
        # Gated SUM conditioning (GSM-style)
        return GatedSumConditioningBlock(channels, cond_channels)

    elif conditioning_type == ConditioningType.GATED_FILM:
        # Gated FiLM conditioning (GSM-style with spatial gate)
        return GatedFiLMConditioningBlock(channels, cond_channels, spatial_gate=True)

    else:
        raise ValueError(f"Unknown conditioning type: {conditioning_type}")


class IdentityConditioning(nn.Module):
    """Identity conditioning (no-op) for when conditioning is disabled at a layer."""

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        return x


class MultiScaleConditioningFusion(nn.Module):
    """
    Multi-Scale Conditioning Fusion module.

    Takes multi-scale LiDAR features and fuses them for conditioning at different
    UNet resolutions. Uses learnable weighted fusion similar to BiFPN.

    Input scales (from LiDAR encoder):
    - scale_256: [B, C1, 256, 256] - full resolution
    - scale_64: [B, C2, 64, 64] - 1/4 resolution
    - scale_32: [B, C3, 32, 32] - 1/8 resolution

    Output: Single conditioning tensor at target resolution.

    Reference: DifFUSER (ECCV 2024) - cMini-BiFPN architecture
    """

    def __init__(
        self,
        in_channels_256: int = 64,
        in_channels_64: int = 128,
        in_channels_32: int = 256,
        out_channels: int = 64,
        fusion_type: str = 'weighted',  # 'weighted', 'bifpn', 'concat', 'attention'
    ):
        super().__init__()

        self.fusion_type = fusion_type

        # Project each scale to common channel count
        self.proj_256 = nn.Sequential(
            nn.Conv2d(in_channels_256, out_channels, 1),
            nn.GroupNorm(get_num_groups(out_channels), out_channels),
            nn.SiLU(),
        )
        self.proj_64 = nn.Sequential(
            nn.Conv2d(in_channels_64, out_channels, 1),
            nn.GroupNorm(get_num_groups(out_channels), out_channels),
            nn.SiLU(),
        )
        self.proj_32 = nn.Sequential(
            nn.Conv2d(in_channels_32, out_channels, 1),
            nn.GroupNorm(get_num_groups(out_channels), out_channels),
            nn.SiLU(),
        )

        if fusion_type == 'weighted':
            # Learnable fusion weights (softmax normalized)
            self.fusion_weights = nn.Parameter(torch.ones(3))

            # Output refinement
            self.output_conv = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 3, padding=1),
                nn.GroupNorm(get_num_groups(out_channels), out_channels),
                nn.SiLU(),
            )

        elif fusion_type == 'bifpn':
            # BiFPN Fast Normalized Fusion: w_i / (sum(w_j) + eps)
            # Weights are ReLU'd to be non-negative, ~30% faster than softmax
            # Reference: EfficientDet (CVPR 2020) - https://arxiv.org/abs/1911.09070
            self.fusion_weights = nn.Parameter(torch.ones(3))
            self.eps = 1e-4  # Small epsilon for numerical stability

            # Output refinement
            self.output_conv = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 3, padding=1),
                nn.GroupNorm(get_num_groups(out_channels), out_channels),
                nn.SiLU(),
            )

        elif fusion_type == 'concat':
            # Concatenate and project
            self.output_conv = nn.Sequential(
                nn.Conv2d(out_channels * 3, out_channels, 3, padding=1),
                nn.GroupNorm(get_num_groups(out_channels), out_channels),
                nn.SiLU(),
                nn.Conv2d(out_channels, out_channels, 3, padding=1),
            )

        elif fusion_type == 'attention':
            # Attention-based fusion
            self.attn_pool = nn.AdaptiveAvgPool2d(1)
            self.attn_fc = nn.Sequential(
                nn.Linear(out_channels * 3, out_channels),
                nn.ReLU(),
                nn.Linear(out_channels, 3),
                nn.Softmax(dim=-1),
            )
            self.output_conv = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 3, padding=1),
                nn.GroupNorm(get_num_groups(out_channels), out_channels),
                nn.SiLU(),
            )

    def forward(
        self,
        multiscale_features: dict[str, torch.Tensor],
        target_size: tuple[int, int] = (256, 256),
    ) -> torch.Tensor:
        """
        Fuse multi-scale features into single conditioning tensor.

        Args:
            multiscale_features: Dict with 'scale_256', 'scale_64', 'scale_32'
            target_size: Output spatial size (H, W)

        Returns:
            Fused conditioning features [B, out_channels, H, W]
        """
        # Project each scale
        f256 = self.proj_256(multiscale_features['scale_256'])
        f64 = self.proj_64(multiscale_features['scale_64'])
        f32 = self.proj_32(multiscale_features['scale_32'])

        # Resize all to target size
        if f256.shape[2:] != target_size:
            f256 = F.interpolate(f256, size=target_size, mode='bilinear', align_corners=False)
        f64_up = F.interpolate(f64, size=target_size, mode='bilinear', align_corners=False)
        f32_up = F.interpolate(f32, size=target_size, mode='bilinear', align_corners=False)

        if self.fusion_type == 'weighted':
            # Normalized learnable weights (softmax)
            weights = F.softmax(self.fusion_weights, dim=0)
            fused = weights[0] * f256 + weights[1] * f64_up + weights[2] * f32_up
            out = self.output_conv(fused)

        elif self.fusion_type == 'bifpn':
            # BiFPN Fast Normalized Fusion: ReLU(w) / (sum(ReLU(w)) + eps)
            # ~30% faster than softmax, similar performance
            weights = F.relu(self.fusion_weights)
            weights = weights / (weights.sum() + self.eps)
            fused = weights[0] * f256 + weights[1] * f64_up + weights[2] * f32_up
            out = self.output_conv(fused)

        elif self.fusion_type == 'concat':
            # Concatenate and project
            concat = torch.cat([f256, f64_up, f32_up], dim=1)
            out = self.output_conv(concat)

        elif self.fusion_type == 'attention':
            # Attention-based fusion
            # Global context for attention weights
            g256 = self.attn_pool(f256).flatten(1)
            g64 = self.attn_pool(f64_up).flatten(1)
            g32 = self.attn_pool(f32_up).flatten(1)
            global_ctx = torch.cat([g256, g64, g32], dim=1)

            attn_weights = self.attn_fc(global_ctx)  # [B, 3]
            attn_weights = attn_weights.unsqueeze(-1).unsqueeze(-1)  # [B, 3, 1, 1]

            fused = (attn_weights[:, 0:1] * f256 +
                     attn_weights[:, 1:2] * f64_up +
                     attn_weights[:, 2:3] * f32_up)
            out = self.output_conv(fused)

        else:
            raise ValueError(f"Unknown fusion_type: {self.fusion_type}")

        return out

    def get_scale_features(
        self,
        multiscale_features: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """
        Get projected features at each scale (for multi-scale conditioning).

        Returns dict with features at native resolutions for per-level conditioning.
        """
        return {
            'scale_256': self.proj_256(multiscale_features['scale_256']),
            'scale_64': self.proj_64(multiscale_features['scale_64']),
            'scale_32': self.proj_32(multiscale_features['scale_32']),
        }
