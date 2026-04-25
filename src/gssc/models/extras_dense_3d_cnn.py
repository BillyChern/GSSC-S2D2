"""
Dense 3D CNN for Scene Completion at Coarse Resolution

Reference: PaSCo (CVPR 2024)
- Uses dense 3D convolutions at bottleneck for hallucination
- Operates at 1:8 resolution (32×32×4 for SemanticKITTI)
- Helps complete occluded regions that sparse convs can't reach

Key insight from PaSCo:
"Standard sparse convolution generates features only where there is
input, meaning that occluded regions are hallucinated only if
covered by the receptive field of observable regions."

Solution: Convert to dense at low resolution, apply 3D CNN, convert back.

Config reference: /workspace/reference/PaSCo/pasco/models/layers.py
- SPCDense3Dv2 is their dense 3D bottleneck (lines 646-726)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional


class SPCDense3Dv2(nn.Module):
    """
    Sparse-to-Dense Completion block - EXACT copy of PaSCo's implementation.

    Reference: /workspace/reference/PaSCo/pasco/models/layers.py:646-726

    Key design choices:
    1. Asymmetric kernels: (3,3,1), (5,5,3), (7,7,5) - larger XY than Z
       (outdoor scenes have more lateral extent than height variation)
    2. Multi-scale parallel branches for different receptive fields
    3. Dense residual connections from input

    Architecture:
        Input (x_dense)
            │
            ├──→ x1 = Conv(3,3,1) ──────────────────────────────┐
            │         │                                          │
            │         ├──→ x2 = Conv(3,3,1)  ─┐                 │
            │         ├──→ x3 = Conv(5,5,3)  ─┼─→ t1 = sum      │
            │         └──→ x4 = Conv(7,7,5)  ─┘      │          │
            │                                        ├─→ x5 ────┤
            │                                        ├─→ x6 ────┤
            │                                        └─→ x7 ────┤
            │                                                   │
            │    x = x1 + x2 + x3 + x4 + x5 + x6 + x7          │
            │         │                                         │
            │    y0 = 1×1 Conv (channel reduction) ─────────────┤
            │                                                   │
            ├──→ y1 = Conv(3,3,1) ──────────────────────────────┤
            ├──→ y2 = Conv(5,5,3) ──────────────────────────────┤
            └──→ y3 = Conv(7,7,5) ──────────────────────────────┤
                                                                │
            Output = x1 + y0 + y1 + y2 + y3 ←───────────────────┘
    """

    def __init__(self, init_size: int = 128):
        """
        Args:
            init_size: Number of input/output channels (C in [B, C, H, W, D])
        """
        super(SPCDense3Dv2, self).__init__()

        bias = False
        chs = [init_size, init_size, init_size, init_size]

        # First stage: initial conv + parallel multi-scale branches
        self.a_conv1 = nn.Conv3d(chs[1], chs[1], (3, 3, 1), 1, padding=(1, 1, 0), bias=bias)
        self.bn_1 = nn.BatchNorm3d(chs[1])

        self.a_conv2 = nn.Conv3d(chs[1], chs[1], (3, 3, 1), 1, padding=(1, 1, 0), bias=bias)
        self.bn_2 = nn.BatchNorm3d(chs[1])

        self.a_conv3 = nn.Conv3d(chs[1], chs[1], (5, 5, 3), 1, padding=(2, 2, 1), bias=bias)
        self.bn_3 = nn.BatchNorm3d(chs[1])

        self.a_conv4 = nn.Conv3d(chs[1], chs[1], (7, 7, 5), 1, padding=(3, 3, 2), bias=bias)
        self.bn_4 = nn.BatchNorm3d(chs[1])

        # Second stage: process aggregated features
        self.a_conv5 = nn.Conv3d(chs[1], chs[1], (3, 3, 1), 1, padding=(1, 1, 0), bias=bias)
        self.bn_5 = nn.BatchNorm3d(chs[1])

        self.a_conv6 = nn.Conv3d(chs[1], chs[1], (5, 5, 3), 1, padding=(2, 2, 1), bias=bias)
        self.bn_6 = nn.BatchNorm3d(chs[1])

        self.a_conv7 = nn.Conv3d(chs[1], chs[1], (7, 7, 5), 1, padding=(3, 3, 2), bias=bias)
        self.bn_7 = nn.BatchNorm3d(chs[1])

        # Channel reduction after aggregation
        self.ch_conv1 = nn.Conv3d(chs[1], chs[0], kernel_size=1, stride=1, bias=bias)
        self.bn_ch_conv1 = nn.BatchNorm3d(chs[0])

        # Residual branches from input (multi-scale)
        self.res_1 = nn.Conv3d(chs[0], chs[0], (3, 3, 1), 1, padding=(1, 1, 0), bias=bias)
        self.bn_res_1 = nn.BatchNorm3d(chs[0])

        self.res_2 = nn.Conv3d(chs[0], chs[0], (5, 5, 3), 1, padding=(2, 2, 1), bias=bias)
        self.bn_res_2 = nn.BatchNorm3d(chs[0])

        self.res_3 = nn.Conv3d(chs[0], chs[0], (7, 7, 5), 1, padding=(3, 3, 2), bias=bias)
        self.bn_res_3 = nn.BatchNorm3d(chs[0])

    def forward(self, x_dense: torch.Tensor) -> torch.Tensor:
        """
        Apply dense 3D completion.

        Args:
            x_dense: [B, C, H, W, D] dense features at coarse resolution

        Returns:
            [B, C, H, W, D] refined features with hallucinated regions
        """
        # First stage: x1 feeds into parallel x2, x3, x4
        x1 = F.relu(self.bn_1(self.a_conv1(x_dense)))

        x2 = F.relu(self.bn_2(self.a_conv2(x1)))
        x3 = F.relu(self.bn_3(self.a_conv3(x1)))
        x4 = F.relu(self.bn_4(self.a_conv4(x1)))

        # Aggregate multi-scale features
        t1 = x2 + x3 + x4

        # Second stage: process aggregated features
        x5 = F.relu(self.bn_5(self.a_conv5(t1)))
        x6 = F.relu(self.bn_6(self.a_conv6(t1)))
        x7 = F.relu(self.bn_7(self.a_conv7(t1)))

        # Sum all processed features
        x = x1 + x2 + x3 + x4 + x5 + x6 + x7
        y0 = F.relu(self.bn_ch_conv1(self.ch_conv1(x)))

        # Residual branches from original input
        y1 = F.relu(self.bn_res_1(self.res_1(x_dense)))
        y2 = F.relu(self.bn_res_2(self.res_2(x_dense)))
        y3 = F.relu(self.bn_res_3(self.res_3(x_dense)))

        # Final output: combine all
        output = x1 + y0 + y1 + y2 + y3

        return output


# Alias for backwards compatibility
Dense3DBottleneck = SPCDense3Dv2


class Dense3DBlock(nn.Module):
    """
    Simple dense 3D convolution block with residual connection.
    (Kept for simpler use cases, but SPCDense3Dv2 is preferred)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        padding = kernel_size // 2

        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.dropout = nn.Dropout3d(dropout)
        self.relu = nn.ReLU(inplace=True)

        # Residual connection
        self.skip = nn.Identity() if in_channels == out_channels else nn.Conv3d(in_channels, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.skip(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = out + identity
        out = self.relu(out)

        return out


class CoarseToFineDecoder(nn.Module):
    """
    Coarse-to-fine decoder with dense 3D at bottleneck.

    Architecture:
    1. Encode to coarse resolution (1:8)
    2. Apply dense 3D CNN for hallucination
    3. Decode back to full resolution

    Reference: PaSCo's UNet3DV2 with dense3d bottleneck
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 20,
        base_channels: int = 32,
        coarse_resolution: Tuple[int, int, int] = (32, 32, 4),
    ):
        """
        Args:
            in_channels: Input channels (1 for binary LiDAR)
            num_classes: Number of output classes
            base_channels: Base feature channels
            coarse_resolution: Resolution for dense processing
        """
        super().__init__()
        self.coarse_resolution = coarse_resolution

        # Encoder (down to 1:8 resolution)
        self.enc1 = nn.Sequential(
            nn.Conv3d(in_channels, base_channels, 3, padding=1),
            nn.BatchNorm3d(base_channels),
            nn.ReLU(inplace=True),
        )
        self.pool1 = nn.MaxPool3d(2)

        self.enc2 = nn.Sequential(
            nn.Conv3d(base_channels, base_channels * 2, 3, padding=1),
            nn.BatchNorm3d(base_channels * 2),
            nn.ReLU(inplace=True),
        )
        self.pool2 = nn.MaxPool3d(2)

        self.enc3 = nn.Sequential(
            nn.Conv3d(base_channels * 2, base_channels * 4, 3, padding=1),
            nn.BatchNorm3d(base_channels * 4),
            nn.ReLU(inplace=True),
        )
        self.pool3 = nn.MaxPool3d(2)

        # Dense 3D bottleneck at 1:8 resolution (SPCDense3Dv2 only needs init_size)
        self.dense_bottleneck = Dense3DBottleneck(init_size=base_channels * 4)

        # Decoder (up to full resolution)
        self.up3 = nn.ConvTranspose3d(base_channels * 4, base_channels * 2, 2, stride=2)
        self.dec3 = nn.Sequential(
            nn.Conv3d(base_channels * 4, base_channels * 2, 3, padding=1),
            nn.BatchNorm3d(base_channels * 2),
            nn.ReLU(inplace=True),
        )

        self.up2 = nn.ConvTranspose3d(base_channels * 2, base_channels, 2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv3d(base_channels * 2, base_channels, 3, padding=1),
            nn.BatchNorm3d(base_channels),
            nn.ReLU(inplace=True),
        )

        self.up1 = nn.ConvTranspose3d(base_channels, base_channels, 2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv3d(base_channels * 2, base_channels, 3, padding=1),
            nn.BatchNorm3d(base_channels),
            nn.ReLU(inplace=True),
        )

        # Output
        self.head = nn.Conv3d(base_channels, num_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 1, H, W, D] binary LiDAR voxels

        Returns:
            [B, num_classes, H, W, D] semantic predictions
        """
        # Encoder
        e1 = self.enc1(x)
        p1 = self.pool1(e1)

        e2 = self.enc2(p1)
        p2 = self.pool2(e2)

        e3 = self.enc3(p2)
        p3 = self.pool3(e3)

        # Dense 3D bottleneck (coarse resolution)
        bottleneck = self.dense_bottleneck(p3)

        # Decoder
        d3 = self.up3(bottleneck)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        # Output
        out = self.head(d1)

        return out


class DenseHallucinationModule(nn.Module):
    """
    Dense hallucination module that operates at coarse resolution.

    This module is designed to be plugged into existing sparse architectures:
    1. Takes sparse features from encoder
    2. Converts to dense at 1:8 resolution
    3. Applies dense 3D CNN for hallucination
    4. Returns hallucinated features to merge with sparse path

    Reference: PaSCo's SPCDense3Dv2
    """

    def __init__(
        self,
        in_channels: int = 128,
        hidden_channels: int = 128,
        out_channels: int = 128,
        target_resolution: Tuple[int, int, int] = (32, 32, 4),
        num_blocks: int = 3,
    ):
        """
        Args:
            in_channels: Input sparse feature channels
            hidden_channels: Dense bottleneck channels
            out_channels: Output channels
            target_resolution: Target resolution for dense processing
            num_blocks: Number of dense 3D conv blocks
        """
        super().__init__()
        self.target_resolution = target_resolution
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Input projection (if in_channels != hidden_channels)
        if in_channels != hidden_channels:
            self.input_proj = nn.Conv3d(in_channels, hidden_channels, 1)
        else:
            self.input_proj = nn.Identity()

        # Dense 3D bottleneck (SPCDense3Dv2 only needs init_size)
        self.dense_net = Dense3DBottleneck(init_size=hidden_channels)

        # Output projection (if hidden_channels != out_channels)
        if hidden_channels != out_channels:
            self.output_proj = nn.Conv3d(hidden_channels, out_channels, 1)
        else:
            self.output_proj = nn.Identity()

        # Merge with sparse features
        self.merge_conv = nn.Conv3d(out_channels * 2, out_channels, 1)

    def sparse_to_dense(
        self,
        sparse_feat: torch.Tensor,
        sparse_coords: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        """
        Convert sparse features to dense representation.

        Args:
            sparse_feat: [N, C] sparse features
            sparse_coords: [N, 4] coordinates (batch_idx, x, y, z)
            batch_size: Batch size

        Returns:
            [B, C, H, W, D] dense feature tensor
        """
        H, W, D = self.target_resolution
        C = sparse_feat.shape[1]

        # Initialize dense tensor
        dense = torch.zeros(batch_size, C, H, W, D, device=sparse_feat.device)

        # Fill in sparse features
        b = sparse_coords[:, 0].long()
        x = sparse_coords[:, 1].long().clamp(0, H - 1)
        y = sparse_coords[:, 2].long().clamp(0, W - 1)
        z = sparse_coords[:, 3].long().clamp(0, D - 1)

        dense[b, :, x, y, z] = sparse_feat

        return dense

    def dense_to_sparse(
        self,
        dense_feat: torch.Tensor,
        sparse_coords: torch.Tensor,
    ) -> torch.Tensor:
        """
        Extract features at sparse locations from dense tensor.

        Args:
            dense_feat: [B, C, H, W, D] dense features
            sparse_coords: [N, 4] coordinates (batch_idx, x, y, z)

        Returns:
            [N, C] sparse features at specified locations
        """
        H, W, D = self.target_resolution

        b = sparse_coords[:, 0].long()
        x = sparse_coords[:, 1].long().clamp(0, H - 1)
        y = sparse_coords[:, 2].long().clamp(0, W - 1)
        z = sparse_coords[:, 3].long().clamp(0, D - 1)

        return dense_feat[b, :, x, y, z]

    def forward(
        self,
        sparse_feat: torch.Tensor,
        sparse_coords: Optional[torch.Tensor] = None,
        batch_size: int = 1,
        return_dense: bool = False,
    ) -> torch.Tensor:
        """
        Apply dense hallucination to sparse features.

        Args:
            sparse_feat: If sparse_coords is None, expects [B, C, H, W, D] dense
                         If sparse_coords provided, expects [N, C] sparse
            sparse_coords: [N, 4] sparse coordinates (optional)
            batch_size: Batch size (used when converting sparse to dense)
            return_dense: If True, return dense features [B, C, H, W, D]

        Returns:
            Hallucinated features (sparse or dense based on input and return_dense)
        """
        # Handle both sparse and dense inputs
        if sparse_coords is not None:
            # Convert sparse to dense
            dense_input = self.sparse_to_dense(sparse_feat, sparse_coords, batch_size)
        else:
            # Already dense
            dense_input = sparse_feat

        # Project input channels if needed
        dense_input = self.input_proj(dense_input)

        # Apply dense 3D CNN for hallucination
        hallucinated = self.dense_net(dense_input)

        # Project output channels if needed
        hallucinated = self.output_proj(hallucinated)

        if return_dense or sparse_coords is None:
            return hallucinated
        else:
            # Convert back to sparse at original locations
            sparse_output = self.dense_to_sparse(hallucinated, sparse_coords)
            return sparse_output


class Dense3DEnhancer(nn.Module):
    """
    Enhances scene completion predictions using PaSCo's SPCDense3Dv2.

    Pipeline:
    1. Take coarse scene prediction (32×32×4)
    2. Apply SPCDense3Dv2 for multi-scale dense 3D refinement
    3. Output refined logits

    Reference: PaSCo's UNet3DV2 uses SPCDense3Dv2 + Dropout3d at bottleneck
    """

    def __init__(
        self,
        num_classes: int = 20,
        hidden_channels: int = 128,
        num_blocks: int = 3,  # Kept for API compatibility (ignored - SPCDense3Dv2 has fixed arch)
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.hidden_channels = hidden_channels

        # Note: num_blocks is ignored - SPCDense3Dv2 has a fixed multi-scale architecture
        # that is more effective than simple stacked blocks

        # Embedding for class predictions
        self.embed = nn.Embedding(num_classes, hidden_channels)

        # PaSCo's SPCDense3Dv2 with dropout (exact architecture)
        self.dense3d = nn.Sequential(
            SPCDense3Dv2(init_size=hidden_channels),
            nn.Dropout3d(dropout),
        )

        # Output head
        self.head = nn.Conv3d(hidden_channels, num_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, H, W, D] coarse scene predictions (class indices)

        Returns:
            [B, num_classes, H, W, D] refined logits
        """
        B, H, W, D = x.shape

        # Embed class predictions
        x_emb = self.embed(x)  # [B, H, W, D, C]
        x_emb = x_emb.permute(0, 4, 1, 2, 3)  # [B, C, H, W, D]

        # Apply PaSCo's SPCDense3Dv2 for multi-scale hallucination
        refined = self.dense3d(x_emb)

        # Output
        out = self.head(refined)

        return out


# Convenience functions

def create_dense_bottleneck(
    channels: int = 128,
    num_blocks: int = 3,  # Kept for API compatibility (ignored - SPCDense3Dv2 has fixed arch)
) -> Dense3DBottleneck:
    """Create a dense 3D bottleneck module (SPCDense3Dv2 only needs init_size)."""
    return Dense3DBottleneck(init_size=channels)


def create_hallucination_module(
    channels: int = 128,
    resolution: Tuple[int, int, int] = (32, 32, 4),
) -> DenseHallucinationModule:
    """Create a dense hallucination module."""
    return DenseHallucinationModule(
        in_channels=channels,
        hidden_channels=channels,
        out_channels=channels,
        target_resolution=resolution,
    )
