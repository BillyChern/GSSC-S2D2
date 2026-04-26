"""
SparseBEVNet: Direct BEV Semantic Segmentation from Sparse 3D LiDAR

Architecture based on BEVFusion/LMSCNet patterns:
- 3D Sparse Encoder (spconv) for efficient 3D feature extraction
- Height compression (Z-axis pooling) to create BEV features
- 2D FPN-style decoder with skip connections
- Segmentation head with Focal Loss

Two versions:
- Testing: 32x32x4 input -> 32x32 BEV output
- Production: 256x256x32 input -> 256x256 BEV output
"""


import spconv.pytorch as spconv
import torch
import torch.nn as nn
import torch.nn.functional as F


class SparseResBlock3D(nn.Module):
    """3D Sparse Residual Block using SubManifold Sparse Convolutions"""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = spconv.SubMConv3d(in_channels, out_channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = spconv.SubMConv3d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)

        # Shortcut if dimensions change
        if in_channels != out_channels:
            self.shortcut = spconv.SubMConv3d(in_channels, out_channels, 1, bias=False)
            self.shortcut_bn = nn.BatchNorm1d(out_channels)
        else:
            self.shortcut = None

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = out.replace_feature(self.bn1(out.features))
        out = out.replace_feature(F.relu(out.features))

        out = self.conv2(out)
        out = out.replace_feature(self.bn2(out.features))

        if self.shortcut is not None:
            identity = self.shortcut(identity)
            identity = identity.replace_feature(self.shortcut_bn(identity.features))

        out = out.replace_feature(F.relu(out.features + identity.features))
        return out


class SparseBEVEncoder(nn.Module):
    """
    3D Sparse Encoder using spconv
    Extracts multi-scale features from sparse 3D voxels
    """

    def __init__(self, in_channels: int = 1, base_channels: int = 32,
                 layer_channels: list[int] = None):
        super().__init__()

        if layer_channels is None:
            layer_channels = [base_channels, base_channels*2, base_channels*4, base_channels*8]

        self.layer_channels = layer_channels

        # Stem
        self.stem = spconv.SubMConv3d(in_channels, layer_channels[0], 3, padding=1, bias=False)
        self.stem_bn = nn.BatchNorm1d(layer_channels[0])

        # Stage 1
        self.stage1 = nn.Sequential(
            SparseResBlock3D(layer_channels[0], layer_channels[0]),
            SparseResBlock3D(layer_channels[0], layer_channels[0]),
        )

        # Down1 + Stage 2
        self.down1 = spconv.SparseConv3d(layer_channels[0], layer_channels[1], 3,
                                          stride=2, padding=1, bias=False)
        self.down1_bn = nn.BatchNorm1d(layer_channels[1])
        self.stage2 = nn.Sequential(
            SparseResBlock3D(layer_channels[1], layer_channels[1]),
            SparseResBlock3D(layer_channels[1], layer_channels[1]),
        )

        # Down2 + Stage 3
        self.down2 = spconv.SparseConv3d(layer_channels[1], layer_channels[2], 3,
                                          stride=2, padding=1, bias=False)
        self.down2_bn = nn.BatchNorm1d(layer_channels[2])
        self.stage3 = nn.Sequential(
            SparseResBlock3D(layer_channels[2], layer_channels[2]),
            SparseResBlock3D(layer_channels[2], layer_channels[2]),
        )

        # Down3 + Stage 4
        self.down3 = spconv.SparseConv3d(layer_channels[2], layer_channels[3], 3,
                                          stride=2, padding=1, bias=False)
        self.down3_bn = nn.BatchNorm1d(layer_channels[3])
        self.stage4 = nn.Sequential(
            SparseResBlock3D(layer_channels[3], layer_channels[3]),
            SparseResBlock3D(layer_channels[3], layer_channels[3]),
        )

    def forward(self, x: spconv.SparseConvTensor) -> list[spconv.SparseConvTensor]:
        """Returns multi-scale sparse features"""
        # Stem
        x = self.stem(x)
        x = x.replace_feature(self.stem_bn(x.features))
        x = x.replace_feature(F.relu(x.features))

        # Stage 1
        f1 = self.stage1(x)

        # Down1 + Stage 2
        x = self.down1(f1)
        x = x.replace_feature(self.down1_bn(x.features))
        x = x.replace_feature(F.relu(x.features))
        f2 = self.stage2(x)

        # Down2 + Stage 3
        x = self.down2(f2)
        x = x.replace_feature(self.down2_bn(x.features))
        x = x.replace_feature(F.relu(x.features))
        f3 = self.stage3(x)

        # Down3 + Stage 4
        x = self.down3(f3)
        x = x.replace_feature(self.down3_bn(x.features))
        x = x.replace_feature(F.relu(x.features))
        f4 = self.stage4(x)

        return [f1, f2, f3, f4]


class HeightCompression(nn.Module):
    """
    Compress 3D sparse features to 2D BEV by pooling over Z-axis
    Converts sparse tensor to dense, then pools
    """

    def __init__(self, in_channels: int, out_channels: int, spatial_shape: tuple[int, int, int]):
        super().__init__()
        self.spatial_shape = spatial_shape  # [X, Y, Z]
        # Combine max and mean pooling, then project
        self.conv = nn.Conv2d(in_channels * 2, out_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, sparse_tensor: spconv.SparseConvTensor) -> torch.Tensor:
        """
        Args:
            sparse_tensor: Sparse 3D features
        Returns:
            Dense 2D BEV features [B, C, H, W]
        """
        # Convert to dense [B, C, X, Y, Z]
        dense = sparse_tensor.dense()

        # Pool over Z axis (last dimension)
        max_pool = dense.max(dim=-1)[0]   # [B, C, X, Y]
        mean_pool = dense.mean(dim=-1)     # [B, C, X, Y]

        # Concatenate and project
        combined = torch.cat([max_pool, mean_pool], dim=1)  # [B, 2C, X, Y]
        out = self.conv(combined)
        out = self.bn(out)
        out = F.relu(out)

        return out


class ConvBlock2D(nn.Module):
    """Basic 2D convolution block"""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              padding=kernel_size//2, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))


class UpBlock2D(nn.Module):
    """Upsampling block with skip connection"""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv1 = ConvBlock2D(in_channels + skip_channels, out_channels)
        self.conv2 = ConvBlock2D(out_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # Handle size mismatch
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class BEVDecoder(nn.Module):
    """
    FPN-style 2D decoder with skip connections
    Takes multi-scale BEV features and produces full-resolution output
    """

    def __init__(self, encoder_channels: list[int], decoder_channels: list[int] = None):
        super().__init__()

        if decoder_channels is None:
            decoder_channels = [256, 128, 64]

        # Bottleneck
        self.bottleneck = ConvBlock2D(encoder_channels[3], decoder_channels[0])

        # Upsample blocks
        self.up1 = UpBlock2D(decoder_channels[0], encoder_channels[2], decoder_channels[0])
        self.up2 = UpBlock2D(decoder_channels[0], encoder_channels[1], decoder_channels[1])
        self.up3 = UpBlock2D(decoder_channels[1], encoder_channels[0], decoder_channels[2])

    def forward(self, bev_features: list[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            bev_features: List of BEV features from each encoder stage [f1, f2, f3, f4]
        Returns:
            Decoded features at full resolution
        """
        f1, f2, f3, f4 = bev_features

        # Bottleneck
        x = self.bottleneck(f4)

        # Upsample with skip connections
        x = self.up1(x, f3)
        x = self.up2(x, f2)
        x = self.up3(x, f1)

        return x


class SegmentationHead(nn.Module):
    """Segmentation head with dropout for regularization"""

    def __init__(self, in_channels: int, num_classes: int, dropout: float = 0.1):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(in_channels, num_classes, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


# ============================================================================
# Architecture Upgrades: Self-Attention and Multi-Scale Feature Fusion
# ============================================================================

class BEVSelfAttention(nn.Module):
    """
    Self-Attention module for BEV feature maps.

    Captures long-range spatial dependencies in BEV representation.
    Uses efficient implementation with downsampled keys/values for large feature maps.

    Reference: "Non-local Neural Networks" (CVPR 2018)
               "Self-Attention Generative Adversarial Networks" (ICML 2019)

    Args:
        in_channels: Number of input channels
        reduction: Channel reduction factor for Q, K, V projections
        spatial_downsample: Downsample factor for K, V to reduce memory
        num_heads: Number of attention heads (default 1 for simplicity)
    """

    def __init__(self, in_channels: int, reduction: int = 8,
                 spatial_downsample: int = 1, num_heads: int = 1):
        super().__init__()
        self.in_channels = in_channels
        self.reduction = reduction
        self.spatial_downsample = spatial_downsample
        self.num_heads = num_heads

        # Reduced channels for efficiency
        self.reduced_channels = max(in_channels // reduction, 1)

        # Query, Key, Value projections (1x1 convolutions)
        self.query = nn.Conv2d(in_channels, self.reduced_channels * num_heads, 1, bias=False)
        self.key = nn.Conv2d(in_channels, self.reduced_channels * num_heads, 1, bias=False)
        self.value = nn.Conv2d(in_channels, in_channels, 1, bias=False)

        # Output projection
        self.out_proj = nn.Conv2d(in_channels, in_channels, 1, bias=False)

        # Learnable scaling parameter (initialized to 0 for residual learning)
        self.gamma = nn.Parameter(torch.zeros(1))

        # Layer normalization for stability
        self.norm = nn.LayerNorm(in_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W] feature map

        Returns:
            [B, C, H, W] attention-enhanced features
        """
        B, C, H, W = x.shape

        # Query: [B, reduced_C * heads, H, W]
        q = self.query(x)

        # Key and Value with optional spatial downsampling for efficiency
        if self.spatial_downsample > 1:
            x_down = F.avg_pool2d(x, self.spatial_downsample)
            k = self.key(x_down)  # [B, reduced_C * heads, H//ds, W//ds]
            v = self.value(x_down)  # [B, C, H//ds, W//ds]
            H_kv, W_kv = H // self.spatial_downsample, W // self.spatial_downsample
        else:
            k = self.key(x)
            v = self.value(x)
            H_kv, W_kv = H, W

        # Reshape for attention computation
        # Q: [B, heads, reduced_C, H*W]
        q = q.view(B, self.num_heads, self.reduced_channels, H * W)
        # K: [B, heads, reduced_C, H_kv*W_kv]
        k = k.view(B, self.num_heads, self.reduced_channels, H_kv * W_kv)
        # V: [B, C, H_kv*W_kv]
        v = v.view(B, C, H_kv * W_kv)

        # Attention: softmax(Q^T K / sqrt(d))
        # [B, heads, H*W, H_kv*W_kv]
        attn = torch.matmul(q.transpose(-2, -1), k)
        attn = attn / (self.reduced_channels ** 0.5)
        attn = F.softmax(attn, dim=-1)

        # Sum over heads and apply to values
        # [B, C, H*W]
        if self.num_heads > 1:
            attn = attn.mean(dim=1)  # Average over heads

        out = torch.matmul(v, attn.transpose(-2, -1).squeeze(1) if self.num_heads > 1
                          else attn.squeeze(1).transpose(-2, -1))

        # Reshape back
        out = out.view(B, C, H, W)

        # Output projection
        out = self.out_proj(out)

        # Residual connection with learnable scale
        out = x + self.gamma * out

        return out


class ASPP(nn.Module):
    """
    Atrous Spatial Pyramid Pooling (ASPP) for multi-scale context.

    Captures features at multiple receptive field scales using dilated convolutions.

    Reference: "DeepLab: Semantic Image Segmentation" (ECCV 2018)

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        rates: Dilation rates for atrous convolutions
    """

    def __init__(self, in_channels: int, out_channels: int,
                 rates: list[int] = None):
        if rates is None:
            rates = [6, 12, 18]
        super().__init__()

        # 1x1 convolution branch
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        # Atrous convolution branches
        self.atrous_convs = nn.ModuleList()
        for rate in rates:
            self.atrous_convs.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=rate,
                          dilation=rate, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ))

        # Global average pooling branch (image-level features)
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        # Fusion: concatenate all branches and project
        num_branches = 1 + len(rates) + 1  # 1x1 + atrous + global
        self.fusion = nn.Sequential(
            nn.Conv2d(out_channels * num_branches, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W] feature map

        Returns:
            [B, out_C, H, W] multi-scale features
        """
        B, C, H, W = x.shape

        # 1x1 branch
        out1x1 = self.conv1x1(x)

        # Atrous branches
        atrous_outs = [conv(x) for conv in self.atrous_convs]

        # Global branch (upsample to match spatial size)
        global_out = self.global_pool(x)
        global_out = F.interpolate(global_out, size=(H, W), mode='bilinear', align_corners=True)

        # Concatenate and fuse
        concat = torch.cat([out1x1] + atrous_outs + [global_out], dim=1)
        out = self.fusion(concat)

        return out


class FeaturePyramidAttention(nn.Module):
    """
    Feature Pyramid Attention (FPA) for enhanced multi-scale fusion.

    Combines attention mechanism with feature pyramid for better spatial
    and semantic feature aggregation.

    Reference: "Pyramid Attention Network for Semantic Segmentation" (BMVC 2018)

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        # Global pooling branch
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        # Multi-scale convolution branches
        # 7x7 -> 5x5 -> 3x3 pyramid
        self.conv7x7 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.conv5x5 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.conv3x3 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        # Upsampling convolutions for pyramid fusion
        self.up3x3 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.up5x5 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.up7x7 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)

        # Middle branch (1x1 processing of input)
        self.mid = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        # Output projection
        self.out_conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W] feature map

        Returns:
            [B, out_C, H, W] attention-enhanced features
        """
        B, C, H, W = x.shape

        # Global branch
        global_feat = self.global_pool(x)
        global_feat = F.interpolate(global_feat, size=(H, W), mode='bilinear', align_corners=True)

        # Pyramid downsampling
        down7 = self.conv7x7(x)  # H/2, W/2
        down5 = self.conv5x5(down7)  # H/4, W/4
        down3 = self.conv3x3(down5)  # H/8, W/8

        # Pyramid upsampling and fusion (bottom-up)
        up3 = self.up3x3(down3)
        up3 = F.interpolate(up3, size=down5.shape[2:], mode='bilinear', align_corners=True)
        up5 = self.up5x5(down5 + up3)
        up5 = F.interpolate(up5, size=down7.shape[2:], mode='bilinear', align_corners=True)
        up7 = self.up7x7(down7 + up5)
        up7 = F.interpolate(up7, size=(H, W), mode='bilinear', align_corners=True)

        # Middle branch
        mid = self.mid(x)

        # Attention: multiply mid features by upsampled pyramid
        attn = mid * up7

        # Add global features
        out = attn + global_feat

        return self.out_conv(out)


class EnhancedBEVDecoder(nn.Module):
    """
    Enhanced BEV Decoder with self-attention and multi-scale fusion.

    Upgrades over basic BEVDecoder:
    1. Self-attention at bottleneck for long-range dependencies
    2. ASPP for multi-scale context aggregation
    3. Optional Feature Pyramid Attention
    """

    def __init__(self, encoder_channels: list[int],
                 decoder_channels: list[int] = None,
                 use_attention: bool = True,
                 use_aspp: bool = True):
        super().__init__()

        if decoder_channels is None:
            decoder_channels = [256, 128, 64]

        self.use_attention = use_attention
        self.use_aspp = use_aspp

        # Self-attention at bottleneck
        if use_attention:
            self.bottleneck_attn = BEVSelfAttention(
                encoder_channels[3],
                reduction=8,
                spatial_downsample=2  # Downsample K,V for efficiency
            )

        # ASPP for multi-scale context
        if use_aspp:
            self.aspp = ASPP(encoder_channels[3], decoder_channels[0], rates=[6, 12, 18])
            bottleneck_in = decoder_channels[0]
        else:
            bottleneck_in = encoder_channels[3]

        # Bottleneck
        self.bottleneck = ConvBlock2D(bottleneck_in, decoder_channels[0])

        # Upsample blocks
        self.up1 = UpBlock2D(decoder_channels[0], encoder_channels[2], decoder_channels[0])
        self.up2 = UpBlock2D(decoder_channels[0], encoder_channels[1], decoder_channels[1])
        self.up3 = UpBlock2D(decoder_channels[1], encoder_channels[0], decoder_channels[2])

    def forward(self, bev_features: list[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            bev_features: List of BEV features from each encoder stage [f1, f2, f3, f4]
        Returns:
            Decoded features at full resolution
        """
        f1, f2, f3, f4 = bev_features

        # Apply attention at bottleneck
        if self.use_attention:
            f4 = self.bottleneck_attn(f4)

        # Apply ASPP for multi-scale context
        if self.use_aspp:
            f4 = self.aspp(f4)

        # Bottleneck
        x = self.bottleneck(f4)

        # Upsample with skip connections
        x = self.up1(x, f3)
        x = self.up2(x, f2)
        x = self.up3(x, f1)

        return x


class SparseBEVNet(nn.Module):
    """
    Complete SparseBEVNet for direct BEV semantic segmentation

    Input: Sparse 3D binary voxels
    Output: 2D BEV semantic map
    """

    def __init__(self,
                 in_channels: int = 1,
                 num_classes: int = 20,
                 spatial_shape: tuple[int, int, int] = (32, 32, 4),
                 encoder_channels: list[int] = None,
                 decoder_channels: list[int] = None,
                 dropout: float = 0.1):
        super().__init__()

        self.spatial_shape = spatial_shape
        self.num_classes = num_classes

        # Default channel configurations
        if encoder_channels is None:
            # For testing (32x32x4): smaller channels
            if spatial_shape[0] <= 32:
                encoder_channels = [32, 64, 128, 256]
            else:
                encoder_channels = [32, 64, 128, 256]

        if decoder_channels is None:
            decoder_channels = [256, 128, 64]

        self.encoder_channels = encoder_channels
        self.decoder_channels = decoder_channels

        # 3D Sparse Encoder
        self.encoder = SparseBEVEncoder(
            in_channels=in_channels,
            base_channels=encoder_channels[0],
            layer_channels=encoder_channels
        )

        # Height compression modules for each scale
        # Compute spatial shapes at each scale
        shapes = [
            spatial_shape,
            (spatial_shape[0]//2, spatial_shape[1]//2, max(1, spatial_shape[2]//2)),
            (spatial_shape[0]//4, spatial_shape[1]//4, max(1, spatial_shape[2]//4)),
            (spatial_shape[0]//8, spatial_shape[1]//8, max(1, spatial_shape[2]//8)),
        ]

        self.height_compress = nn.ModuleList([
            HeightCompression(encoder_channels[i], encoder_channels[i], shapes[i])
            for i in range(4)
        ])

        # 2D BEV Decoder
        self.decoder = BEVDecoder(encoder_channels, decoder_channels)

        # Segmentation Head
        self.seg_head = SegmentationHead(decoder_channels[-1], num_classes, dropout)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv3d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, voxels: torch.Tensor, coords: torch.Tensor,
                batch_size: int) -> torch.Tensor:
        """
        Args:
            voxels: Voxel features [N, C]
            coords: Voxel coordinates [N, 4] (batch_idx, x, y, z)
            batch_size: Number of samples in batch
        Returns:
            BEV semantic logits [B, num_classes, H, W]
        """
        # Create sparse tensor
        sparse_input = spconv.SparseConvTensor(
            features=voxels,
            indices=coords.int(),
            spatial_shape=self.spatial_shape,
            batch_size=batch_size
        )

        # 3D Sparse Encoding
        sparse_features = self.encoder(sparse_input)

        # Height compression (3D -> 2D BEV)
        bev_features = [
            self.height_compress[i](sparse_features[i])
            for i in range(4)
        ]

        # 2D Decoding
        decoded = self.decoder(bev_features)

        # Segmentation
        logits = self.seg_head(decoded)

        return logits


class TinySparseBEVNet(SparseBEVNet):
    """
    Tiny version for testing on 32x32x4 voxels
    """

    def __init__(self, num_classes: int = 20):
        super().__init__(
            in_channels=1,
            num_classes=num_classes,
            spatial_shape=(32, 32, 4),
            encoder_channels=[16, 32, 64, 128],
            decoder_channels=[128, 64, 32],
            dropout=0.1
        )


class FullSparseBEVNet(SparseBEVNet):
    """
    Full version for production on 256x256x32 voxels
    """

    def __init__(self, num_classes: int = 20):
        super().__init__(
            in_channels=1,
            num_classes=num_classes,
            spatial_shape=(256, 256, 32),
            encoder_channels=[32, 64, 128, 256],
            decoder_channels=[256, 128, 64],
            dropout=0.1
        )


# ============================================================================
# Experiment Variants: Each adds ONE feature to the baseline
# ============================================================================

class FullSparseBEVNet_SelfAttn(nn.Module):
    """
    Experiment A1: Baseline + Self-Attention at bottleneck

    Only change: Add BEVSelfAttention after the deepest BEV features (f4)
    before the decoder bottleneck.
    """

    def __init__(self, num_classes: int = 20):
        super().__init__()

        self.spatial_shape = (256, 256, 32)
        self.num_classes = num_classes

        encoder_channels = [32, 64, 128, 256]
        decoder_channels = [256, 128, 64]

        # 3D Sparse Encoder (same as baseline)
        self.encoder = SparseBEVEncoder(
            in_channels=1,
            base_channels=encoder_channels[0],
            layer_channels=encoder_channels
        )

        # Height compression (same as baseline)
        shapes = [
            self.spatial_shape,
            (128, 128, 16),
            (64, 64, 8),
            (32, 32, 4),
        ]
        self.height_compress = nn.ModuleList([
            HeightCompression(encoder_channels[i], encoder_channels[i], shapes[i])
            for i in range(4)
        ])

        # NEW: Self-attention on bottleneck features (32x32 BEV)
        self.bottleneck_attn = BEVSelfAttention(
            in_channels=encoder_channels[3],  # 256 channels
            reduction=8,
            spatial_downsample=1,  # No downsample at 32x32 (already small)
            num_heads=1
        )

        # 2D BEV Decoder (same as baseline)
        self.decoder = BEVDecoder(encoder_channels, decoder_channels)

        # Segmentation Head (same as baseline)
        self.seg_head = SegmentationHead(decoder_channels[-1], num_classes, dropout=0.1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv3d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, voxels: torch.Tensor, coords: torch.Tensor,
                batch_size: int) -> torch.Tensor:
        # Create sparse tensor
        sparse_input = spconv.SparseConvTensor(
            features=voxels,
            indices=coords.int(),
            spatial_shape=self.spatial_shape,
            batch_size=batch_size
        )

        # 3D Sparse Encoding
        sparse_features = self.encoder(sparse_input)

        # Height compression (3D -> 2D BEV)
        bev_features = [
            self.height_compress[i](sparse_features[i])
            for i in range(4)
        ]

        # NEW: Apply self-attention on bottleneck (f4)
        bev_features[3] = self.bottleneck_attn(bev_features[3])

        # 2D Decoding
        decoded = self.decoder(bev_features)

        # Segmentation
        logits = self.seg_head(decoded)

        return logits


class FullSparseBEVNet_ASPP(nn.Module):
    """
    Experiment A2: Baseline + ASPP multi-scale fusion

    Only change: Add ASPP module after bottleneck features (f4)
    """

    def __init__(self, num_classes: int = 20):
        super().__init__()

        self.spatial_shape = (256, 256, 32)
        self.num_classes = num_classes

        encoder_channels = [32, 64, 128, 256]
        decoder_channels = [256, 128, 64]

        # 3D Sparse Encoder (same as baseline)
        self.encoder = SparseBEVEncoder(
            in_channels=1,
            base_channels=encoder_channels[0],
            layer_channels=encoder_channels
        )

        # Height compression (same as baseline)
        shapes = [
            self.spatial_shape,
            (128, 128, 16),
            (64, 64, 8),
            (32, 32, 4),
        ]
        self.height_compress = nn.ModuleList([
            HeightCompression(encoder_channels[i], encoder_channels[i], shapes[i])
            for i in range(4)
        ])

        # NEW: ASPP for multi-scale context at bottleneck
        self.aspp = ASPP(
            in_channels=encoder_channels[3],  # 256
            out_channels=encoder_channels[3],  # 256
            rates=[6, 12, 18]
        )

        # 2D BEV Decoder (same as baseline)
        self.decoder = BEVDecoder(encoder_channels, decoder_channels)

        # Segmentation Head (same as baseline)
        self.seg_head = SegmentationHead(decoder_channels[-1], num_classes, dropout=0.1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv3d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, voxels: torch.Tensor, coords: torch.Tensor,
                batch_size: int) -> torch.Tensor:
        # Create sparse tensor
        sparse_input = spconv.SparseConvTensor(
            features=voxels,
            indices=coords.int(),
            spatial_shape=self.spatial_shape,
            batch_size=batch_size
        )

        # 3D Sparse Encoding
        sparse_features = self.encoder(sparse_input)

        # Height compression (3D -> 2D BEV)
        bev_features = [
            self.height_compress[i](sparse_features[i])
            for i in range(4)
        ]

        # NEW: Apply ASPP on bottleneck (f4)
        bev_features[3] = self.aspp(bev_features[3])

        # 2D Decoding
        decoded = self.decoder(bev_features)

        # Segmentation
        logits = self.seg_head(decoded)

        return logits


class DeeperSparseBEVEncoder(nn.Module):
    """
    Experiment A4: Deeper 3D Sparse Encoder (ResNet-50 style)

    Compared to baseline SparseBEVEncoder:
    - Stage 1: 2 → 3 blocks
    - Stage 2: 2 → 4 blocks
    - Stage 3: 2 → 6 blocks (deepest, most capacity)
    - Stage 4: 2 → 3 blocks

    Total: 8 → 16 blocks (2× deeper)
    """

    def __init__(self, in_channels: int = 1, base_channels: int = 32,
                 layer_channels: list[int] = None):
        super().__init__()

        if layer_channels is None:
            layer_channels = [base_channels, base_channels*2, base_channels*4, base_channels*8]

        self.layer_channels = layer_channels

        # Stem
        self.stem = spconv.SubMConv3d(in_channels, layer_channels[0], 3, padding=1, bias=False)
        self.stem_bn = nn.BatchNorm1d(layer_channels[0])

        # Stage 1: 3 blocks (was 2)
        self.stage1 = nn.Sequential(
            SparseResBlock3D(layer_channels[0], layer_channels[0]),
            SparseResBlock3D(layer_channels[0], layer_channels[0]),
            SparseResBlock3D(layer_channels[0], layer_channels[0]),
        )

        # Down1 + Stage 2: 4 blocks (was 2)
        self.down1 = spconv.SparseConv3d(layer_channels[0], layer_channels[1], 3,
                                          stride=2, padding=1, bias=False)
        self.down1_bn = nn.BatchNorm1d(layer_channels[1])
        self.stage2 = nn.Sequential(
            SparseResBlock3D(layer_channels[1], layer_channels[1]),
            SparseResBlock3D(layer_channels[1], layer_channels[1]),
            SparseResBlock3D(layer_channels[1], layer_channels[1]),
            SparseResBlock3D(layer_channels[1], layer_channels[1]),
        )

        # Down2 + Stage 3: 6 blocks (was 2) - deepest stage
        self.down2 = spconv.SparseConv3d(layer_channels[1], layer_channels[2], 3,
                                          stride=2, padding=1, bias=False)
        self.down2_bn = nn.BatchNorm1d(layer_channels[2])
        self.stage3 = nn.Sequential(
            SparseResBlock3D(layer_channels[2], layer_channels[2]),
            SparseResBlock3D(layer_channels[2], layer_channels[2]),
            SparseResBlock3D(layer_channels[2], layer_channels[2]),
            SparseResBlock3D(layer_channels[2], layer_channels[2]),
            SparseResBlock3D(layer_channels[2], layer_channels[2]),
            SparseResBlock3D(layer_channels[2], layer_channels[2]),
        )

        # Down3 + Stage 4: 3 blocks (was 2)
        self.down3 = spconv.SparseConv3d(layer_channels[2], layer_channels[3], 3,
                                          stride=2, padding=1, bias=False)
        self.down3_bn = nn.BatchNorm1d(layer_channels[3])
        self.stage4 = nn.Sequential(
            SparseResBlock3D(layer_channels[3], layer_channels[3]),
            SparseResBlock3D(layer_channels[3], layer_channels[3]),
            SparseResBlock3D(layer_channels[3], layer_channels[3]),
        )

    def forward(self, x: spconv.SparseConvTensor) -> list[spconv.SparseConvTensor]:
        """Returns multi-scale sparse features"""
        # Stem
        x = self.stem(x)
        x = x.replace_feature(self.stem_bn(x.features))
        x = x.replace_feature(F.relu(x.features))

        # Stage 1
        f1 = self.stage1(x)

        # Down1 + Stage 2
        x = self.down1(f1)
        x = x.replace_feature(self.down1_bn(x.features))
        x = x.replace_feature(F.relu(x.features))
        f2 = self.stage2(x)

        # Down2 + Stage 3
        x = self.down2(f2)
        x = x.replace_feature(self.down2_bn(x.features))
        x = x.replace_feature(F.relu(x.features))
        f3 = self.stage3(x)

        # Down3 + Stage 4
        x = self.down3(f3)
        x = x.replace_feature(self.down3_bn(x.features))
        x = x.replace_feature(F.relu(x.features))
        f4 = self.stage4(x)

        return [f1, f2, f3, f4]


class VeryDeepSparseBEVEncoder(nn.Module):
    """
    32-block 3D Sparse Encoder (ResNet-101 style)

    Compared to DeeperSparseBEVEncoder (16 blocks):
    - Stage 1: 3 → 6 blocks
    - Stage 2: 4 → 8 blocks
    - Stage 3: 6 → 12 blocks (deepest, most capacity)
    - Stage 4: 3 → 6 blocks

    Total: 16 → 32 blocks (2× deeper than L3)

    Motivation:
    - Double descent phenomenon: very large models can generalize better
    - Per-sample learning signal is huge (~100K voxels per sample)
    - 19K samples × 100K voxels = ~2B labeled voxel predictions
    """

    def __init__(self, in_channels: int = 1, base_channels: int = 32,
                 layer_channels: list[int] = None):
        super().__init__()

        if layer_channels is None:
            layer_channels = [base_channels, base_channels*2, base_channels*4, base_channels*8]

        self.layer_channels = layer_channels

        # Stem
        self.stem = spconv.SubMConv3d(in_channels, layer_channels[0], 3, padding=1, bias=False)
        self.stem_bn = nn.BatchNorm1d(layer_channels[0])

        # Stage 1: 6 blocks (was 3 in Deeper)
        self.stage1 = nn.Sequential(*[
            SparseResBlock3D(layer_channels[0], layer_channels[0])
            for _ in range(6)
        ])

        # Down1 + Stage 2: 8 blocks (was 4 in Deeper)
        self.down1 = spconv.SparseConv3d(layer_channels[0], layer_channels[1], 3,
                                          stride=2, padding=1, bias=False)
        self.down1_bn = nn.BatchNorm1d(layer_channels[1])
        self.stage2 = nn.Sequential(*[
            SparseResBlock3D(layer_channels[1], layer_channels[1])
            for _ in range(8)
        ])

        # Down2 + Stage 3: 12 blocks (was 6 in Deeper) - deepest stage
        self.down2 = spconv.SparseConv3d(layer_channels[1], layer_channels[2], 3,
                                          stride=2, padding=1, bias=False)
        self.down2_bn = nn.BatchNorm1d(layer_channels[2])
        self.stage3 = nn.Sequential(*[
            SparseResBlock3D(layer_channels[2], layer_channels[2])
            for _ in range(12)
        ])

        # Down3 + Stage 4: 6 blocks (was 3 in Deeper)
        self.down3 = spconv.SparseConv3d(layer_channels[2], layer_channels[3], 3,
                                          stride=2, padding=1, bias=False)
        self.down3_bn = nn.BatchNorm1d(layer_channels[3])
        self.stage4 = nn.Sequential(*[
            SparseResBlock3D(layer_channels[3], layer_channels[3])
            for _ in range(6)
        ])

    def forward(self, x: spconv.SparseConvTensor) -> list[spconv.SparseConvTensor]:
        """Returns multi-scale sparse features"""
        # Stem
        x = self.stem(x)
        x = x.replace_feature(self.stem_bn(x.features))
        x = x.replace_feature(F.relu(x.features))

        # Stage 1
        f1 = self.stage1(x)

        # Down1 + Stage 2
        x = self.down1(f1)
        x = x.replace_feature(self.down1_bn(x.features))
        x = x.replace_feature(F.relu(x.features))
        f2 = self.stage2(x)

        # Down2 + Stage 3
        x = self.down2(f2)
        x = x.replace_feature(self.down2_bn(x.features))
        x = x.replace_feature(F.relu(x.features))
        f3 = self.stage3(x)

        # Down3 + Stage 4
        x = self.down3(f3)
        x = x.replace_feature(self.down3_bn(x.features))
        x = x.replace_feature(F.relu(x.features))
        f4 = self.stage4(x)

        return [f1, f2, f3, f4]


class FullSparseBEVNet_VeryDeep(nn.Module):
    """
    32-block Very Deep Sparse BEV Network (ResNet-101 style)

    Architecture:
    - Stage depths: [6, 8, 12, 6] instead of [3, 4, 6, 3]
    - Total blocks: 32 instead of 16
    - Expected params: ~38-42M

    Hypothesis:
    - Double descent: very large models find simpler interpolating solutions
    - Per-voxel learning: 19K samples × 100K voxels = massive training signal
    """

    def __init__(self, num_classes: int = 20):
        super().__init__()

        self.spatial_shape = (256, 256, 32)
        self.num_classes = num_classes

        encoder_channels = [32, 64, 128, 256]
        decoder_channels = [256, 128, 64]

        # Very Deep 3D Sparse Encoder (32 blocks)
        self.encoder = VeryDeepSparseBEVEncoder(
            in_channels=1,
            base_channels=encoder_channels[0],
            layer_channels=encoder_channels
        )

        # Height compression (same as baseline)
        shapes = [
            self.spatial_shape,
            (128, 128, 16),
            (64, 64, 8),
            (32, 32, 4),
        ]
        self.height_compress = nn.ModuleList([
            HeightCompression(encoder_channels[i], encoder_channels[i], shapes[i])
            for i in range(4)
        ])

        # 2D BEV Decoder (same as baseline)
        self.decoder = BEVDecoder(encoder_channels, decoder_channels)

        # Segmentation Head (same as baseline)
        self.seg_head = SegmentationHead(decoder_channels[-1], num_classes, dropout=0.1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv3d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, voxels: torch.Tensor, coords: torch.Tensor,
                batch_size: int) -> torch.Tensor:
        # Create sparse tensor
        sparse_input = spconv.SparseConvTensor(
            features=voxels,
            indices=coords.int(),
            spatial_shape=self.spatial_shape,
            batch_size=batch_size
        )

        # 3D Sparse Encoding (very deep - 32 blocks)
        sparse_features = self.encoder(sparse_input)

        # Height compression (3D -> 2D BEV)
        bev_features = [
            self.height_compress[i](sparse_features[i])
            for i in range(4)
        ]

        # 2D Decoding
        decoded = self.decoder(bev_features)

        # Segmentation
        logits = self.seg_head(decoded)

        return logits


class FullSparseBEVNet_Deeper(nn.Module):
    """
    Experiment A4: Baseline + Deeper Encoder (ResNet-50 style)

    Only change: Replace SparseBEVEncoder with DeeperSparseBEVEncoder
    - Stage depths: [3, 4, 6, 3] instead of [2, 2, 2, 2]
    - Total blocks: 16 instead of 8

    Expected: +2-3% mIoU, ~1.5× training time
    """

    def __init__(self, num_classes: int = 20, in_channels: int = 1, tsdf_bev_channels: int = 0):
        super().__init__()

        self.spatial_shape = (256, 256, 32)
        self.num_classes = num_classes

        encoder_channels = [32, 64, 128, 256]
        decoder_channels = [256, 128, 64]

        # NEW: Deeper 3D Sparse Encoder
        self.encoder = DeeperSparseBEVEncoder(
            in_channels=in_channels,
            base_channels=encoder_channels[0],
            layer_channels=encoder_channels
        )

        # Height compression (same as baseline)
        shapes = [
            self.spatial_shape,
            (128, 128, 16),
            (64, 64, 8),
            (32, 32, 4),
        ]
        self.height_compress = nn.ModuleList([
            HeightCompression(encoder_channels[i], encoder_channels[i], shapes[i])
            for i in range(4)
        ])

        # V7: TSDF BEV conditioning (dense 2D → additive injection at f1 level)
        # Zero-init BN so initial TSDF contribution is zero (smooth start for fine-tuning)
        self.tsdf_proj = None
        if tsdf_bev_channels > 0:
            self.tsdf_proj = nn.Sequential(
                nn.Conv2d(tsdf_bev_channels, encoder_channels[0], 1, bias=False),
                nn.BatchNorm2d(encoder_channels[0]),
                nn.ReLU(inplace=True),
            )

        # 2D BEV Decoder (same as baseline)
        self.decoder = BEVDecoder(encoder_channels, decoder_channels)

        # Segmentation Head (same as baseline)
        self.seg_head = SegmentationHead(decoder_channels[-1], num_classes, dropout=0.1)

        self._init_weights()

        # Zero-init TSDF BN AFTER _init_weights so initial TSDF contribution is zero
        # (smooth start for fine-tuning from non-TSDF checkpoints)
        if self.tsdf_proj is not None:
            nn.init.constant_(self.tsdf_proj[1].weight, 0)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv3d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, voxels: torch.Tensor, coords: torch.Tensor,
                batch_size: int, tsdf_bev: torch.Tensor = None) -> torch.Tensor:
        # Create sparse tensor
        sparse_input = spconv.SparseConvTensor(
            features=voxels,
            indices=coords.int(),
            spatial_shape=self.spatial_shape,
            batch_size=batch_size
        )

        # 3D Sparse Encoding (deeper)
        sparse_features = self.encoder(sparse_input)

        # Height compression (3D -> 2D BEV)
        bev_features = [
            self.height_compress[i](sparse_features[i])
            for i in range(4)
        ]

        # V7: TSDF BEV injection at f1 (highest resolution)
        if self.tsdf_proj is not None and tsdf_bev is not None:
            bev_features[0] = bev_features[0] + self.tsdf_proj(tsdf_bev)

        # 2D Decoding
        decoded = self.decoder(bev_features)

        # Segmentation
        logits = self.seg_head(decoded)

        return logits

    def forward_features(self, voxels: torch.Tensor, coords: torch.Tensor,
                         batch_size: int, tsdf_bev: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass that returns both features and logits.

        Used by B2 (Auxiliary SSC) to get intermediate BEV features for the SSC head.

        Args:
            voxels: Voxel features [N, C]
            coords: Voxel coordinates [N, 4] (batch_idx, x, y, z)
            batch_size: Number of samples in batch
            tsdf_bev: Optional [B, 4, H, W] TSDF BEV features

        Returns:
            features: [B, C, H, W] decoded BEV features before segmentation head
            logits: [B, num_classes, H, W] segmentation logits
        """
        # Create sparse tensor
        sparse_input = spconv.SparseConvTensor(
            features=voxels,
            indices=coords.int(),
            spatial_shape=self.spatial_shape,
            batch_size=batch_size
        )

        # 3D Sparse Encoding (deeper)
        sparse_features = self.encoder(sparse_input)

        # Height compression (3D -> 2D BEV)
        bev_features = [
            self.height_compress[i](sparse_features[i])
            for i in range(4)
        ]

        # V7: TSDF BEV injection at f1 (highest resolution)
        if self.tsdf_proj is not None and tsdf_bev is not None:
            bev_features[0] = bev_features[0] + self.tsdf_proj(tsdf_bev)

        # 2D Decoding
        decoded = self.decoder(bev_features)

        # Segmentation
        logits = self.seg_head(decoded)

        return decoded, logits


# SemanticKITTI class weights using α_c = 1 / log(1 + f_c)
# where f_c is the class frequency aggregated by official learning_map
#
# Official learning map (from semantic-kitti.yaml):
# 0=unlabeled, 1=car, 2=bicycle, 3=motorcycle, 4=truck, 5=other-vehicle,
# 6=person, 7=bicyclist, 8=motorcyclist, 9=road, 10=parking, 11=sidewalk,
# 12=other-ground, 13=building, 14=fence, 15=vegetation, 16=trunk, 17=terrain,
# 18=pole, 19=traffic-sign
#
# Class frequencies (aggregated from official content field via learning_map):
# 0: unlabeled      - 0.0315 (3.15%) - includes outlier, other-structure, other-object
# 1: car            - 0.0426 (4.26%) - includes moving-car
# 2: bicycle        - 0.000166 (0.017%)
# 3: motorcycle     - 0.000436 (0.044%) - includes moving-motorcyclist
# 4: truck          - 0.00217 (0.22%) - includes moving-truck
# 5: other-vehicle  - 0.00181 (0.18%) - includes bus, on-rails, moving variants
# 6: person         - 0.000338 (0.034%) - includes moving-person
# 7: bicyclist      - 0.000127 (0.013%) - includes moving-bicyclist
# 8: motorcyclist   - 0.0000055 (0.0006%)
# 9: road           - 0.1988 (19.88%) - includes lane-marking
# 10: parking       - 0.0147 (1.47%)
# 11: sidewalk      - 0.1439 (14.39%)
# 12: other-ground  - 0.0039 (0.39%)
# 13: building      - 0.1327 (13.27%)
# 14: fence         - 0.0724 (7.24%)
# 15: vegetation    - 0.2668 (26.68%)
# 16: trunk         - 0.00604 (0.60%)
# 17: terrain       - 0.0781 (7.81%)
# 18: pole          - 0.00286 (0.29%)
# 19: traffic-sign  - 0.000616 (0.06%)

import math

# Official aggregated frequencies (from semantic-kitti.yaml via learning_map)
_FREQUENCIES = [
    0.031502,   # 0: unlabeled
    0.042608,   # 1: car
    0.000166,   # 2: bicycle
    0.000436,   # 3: motorcycle (0.000398 + 0.0000375 moving)
    0.002165,   # 4: truck
    0.001807,   # 5: other-vehicle
    0.000338,   # 6: person
    0.000127,   # 7: bicyclist
    0.0000055,  # 8: motorcyclist
    0.198796,   # 9: road
    0.014717,   # 10: parking
    0.143923,   # 11: sidewalk
    0.003905,   # 12: other-ground
    0.132686,   # 13: building
    0.072359,   # 14: fence
    0.266815,   # 15: vegetation
    0.006035,   # 16: trunk
    0.078142,   # 17: terrain
    0.002855,   # 18: pole
    0.000616,   # 19: traffic-sign
]

# Assume ~2M total samples for converting frequencies to counts
_TOTAL_SAMPLES = 2_000_000
_SAMPLE_COUNTS = [int(f * _TOTAL_SAMPLES) for f in _FREQUENCIES]


def compute_class_balanced_weights(beta: float = 0.999, max_weight: float = 20.0) -> torch.Tensor:
    """
    Compute Class-Balanced (CB) weights.

    Formula: α_c = (1 - β) / (1 - β^n_c)

    Reference: "Class-Balanced Loss Based on Effective Number of Samples" (CVPR 2019)

    Args:
        beta: Controls effective number of samples.
              β=0.99 -> mild reweighting (~10x ratio)
              β=0.999 -> moderate reweighting (~100x ratio)  [recommended]
              β=0.9999 -> strong reweighting (~1000x ratio)
        max_weight: Clip weights to this maximum value for stability

    Returns:
        Normalized class weights tensor [num_classes]
    """
    raw_weights = []
    for n_c in _SAMPLE_COUNTS:
        if n_c == 0:
            raw_weights.append(0.0)
        else:
            # CB formula: (1 - β) / (1 - β^n_c)
            raw_weights.append((1 - beta) / (1 - beta ** n_c))

    # Normalize so mean of non-zero weights = 1.0
    nonzero = [w for w in raw_weights if w > 0]
    mean_weight = sum(nonzero) / len(nonzero) if nonzero else 1.0
    normalized = [w / mean_weight if w > 0 else 0.0 for w in raw_weights]

    # Clip to max_weight for stability
    clipped = [min(w, max_weight) for w in normalized]

    return torch.tensor(clipped, dtype=torch.float32)


def compute_inverse_log_weights() -> torch.Tensor:
    """
    Compute 1/log(1+f) weights (original formula).

    Warning: Creates extreme ratios (~43000x) for SemanticKITTI.
    Consider using compute_class_balanced_weights() instead.
    """
    raw_weights = []
    for f in _FREQUENCIES:
        if f <= 0:
            raw_weights.append(0.0)
        else:
            raw_weights.append(1.0 / math.log(1.0 + f))

    # Normalize
    nonzero = [w for w in raw_weights if w > 0]
    mean_weight = sum(nonzero) / len(nonzero) if nonzero else 1.0
    normalized = [w / mean_weight if w > 0 else 0.0 for w in raw_weights]

    return torch.tensor(normalized, dtype=torch.float32)


# Default: Use Class-Balanced weights with β=0.999 (recommended)
SEMANTICKITTI_CLASS_WEIGHTS = compute_class_balanced_weights(beta=0.999, max_weight=20.0)


# Loss functions
class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Args:
        alpha: If float and >= 0, uniform alpha for all classes.
               If float and < 0, no alpha weighting.
               If tensor, per-class weights.
        gamma: Focal modulation factor
        class_weights: Optional per-class weights tensor [num_classes]
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0,
                 ignore_index: int = 0, reduction: str = 'mean',
                 class_weights: torch.Tensor = None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.register_buffer('class_weights', class_weights)

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: [B, C, H, W] logits
            targets: [B, H, W] class indices
        """
        # Use class weights in CE if provided
        if self.class_weights is not None:
            ce_loss = F.cross_entropy(inputs, targets, weight=self.class_weights,
                                       reduction='none', ignore_index=self.ignore_index)
        else:
            ce_loss = F.cross_entropy(inputs, targets, reduction='none',
                                       ignore_index=self.ignore_index)

        # Get probabilities
        p = torch.softmax(inputs, dim=1)
        p_t = p.gather(1, targets.unsqueeze(1)).squeeze(1)

        # Focal weight
        focal_weight = (1 - p_t) ** self.gamma

        # Apply alpha (scalar)
        if isinstance(self.alpha, (int, float)) and self.alpha >= 0:
            focal_loss = self.alpha * focal_weight * ce_loss
        else:
            focal_loss = focal_weight * ce_loss

        # Mask ignored
        mask = (targets != self.ignore_index).float()
        focal_loss = focal_loss * mask

        if self.reduction == 'mean':
            return focal_loss.sum() / (mask.sum() + 1e-8)
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class LovaszSoftmax(nn.Module):
    """
    Lovász-Softmax loss for semantic segmentation
    Directly optimizes IoU (the evaluation metric)

    Reference: "The Lovász-Softmax loss: A tractable surrogate for
               the optimization of the intersection-over-union measure"
    """

    def __init__(self, ignore_index: int = 0, per_class: bool = True):
        super().__init__()
        self.ignore_index = ignore_index
        self.per_class = per_class

    def lovasz_grad(self, gt_sorted: torch.Tensor) -> torch.Tensor:
        """Compute gradient of the Lovász extension w.r.t sorted errors"""
        gts = gt_sorted.sum()
        intersection = gts - gt_sorted.float().cumsum(0)
        union = gts + (1 - gt_sorted).float().cumsum(0)
        jaccard = 1. - intersection / union
        if len(jaccard) > 1:
            jaccard[1:] = jaccard[1:] - jaccard[:-1]
        return jaccard

    def lovasz_softmax_flat(self, probas: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Multi-class Lovász-Softmax loss
        probas: [P, C] class probabilities
        labels: [P] ground truth labels
        """
        C = probas.size(1)
        losses = []
        for c in range(C):
            if c == self.ignore_index:
                continue
            fg = (labels == c).float()
            if fg.sum() == 0:
                continue
            errors = (fg - probas[:, c]).abs()
            errors_sorted, perm = torch.sort(errors, descending=True)
            fg_sorted = fg[perm]
            grad = self.lovasz_grad(fg_sorted)
            losses.append((errors_sorted * grad).sum())

        if len(losses) == 0:
            return probas.sum() * 0.0  # Return 0 with gradient
        return torch.stack(losses).mean()

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: [B, C, H, W] logits
            targets: [B, H, W] class indices
        """
        probas = F.softmax(inputs, dim=1)

        # Flatten
        B, C, H, W = probas.shape
        probas = probas.permute(0, 2, 3, 1).contiguous().view(-1, C)
        targets = targets.view(-1)

        # Remove ignored
        valid = targets != self.ignore_index
        probas = probas[valid]
        targets = targets[valid]

        if len(targets) == 0:
            return inputs.sum() * 0.0

        return self.lovasz_softmax_flat(probas, targets)


class SceneClassAffinityLoss(nn.Module):
    """
    Scene-Class Affinity Loss (SCAL) from MonoScene (Cao & de Charette, CVPR 2022)

    Optimizes differentiable versions of Precision, Recall, and Specificity
    per class, then macro-averages over classes.

    This forces equal attention to ALL classes regardless of frequency,
    which helps with long-tail distributions.

    Reference: https://openaccess.thecvf.com/content/CVPR2022/papers/Cao_MonoScene_Monocular_3D_Semantic_Scene_Completion_CVPR_2022_paper.pdf
    """

    def __init__(self, num_classes: int = 20, ignore_index: int = 0, eps: float = 1e-7):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.eps = eps

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: [B, C, H, W] logits
            targets: [B, H, W] class indices

        Returns:
            SCAL loss (scalar)
        """
        # Get probabilities
        probas = F.softmax(inputs, dim=1)  # [B, C, H, W]

        B, C, H, W = probas.shape

        # Flatten spatial dimensions
        probas = probas.permute(0, 2, 3, 1).contiguous().view(-1, C)  # [B*H*W, C]
        targets = targets.view(-1)  # [B*H*W]

        # Create mask for valid pixels (not ignore_index)
        valid_mask = targets != self.ignore_index

        # Filter to valid pixels only
        probas = probas[valid_mask]  # [N_valid, C]
        targets = targets[valid_mask]  # [N_valid]

        if len(targets) == 0:
            return inputs.sum() * 0.0

        # One-hot encode targets
        targets_onehot = F.one_hot(targets, num_classes=C).float()  # [N_valid, C]

        # Compute P, R, S for each class
        losses = []

        for c in range(C):
            if c == self.ignore_index:
                continue

            p_c = probas[:, c]  # Predicted prob for class c
            g_c = targets_onehot[:, c]  # Ground truth indicator for class c

            # Skip if class not present in batch (avoid division issues)
            if g_c.sum() == 0:
                continue

            # Precision: P_c = log( sum(p_c * g_c) / sum(p_c) )
            # "How precise am I on class c?"
            precision_num = (p_c * g_c).sum() + self.eps
            precision_den = p_c.sum() + self.eps
            P_c = torch.log(precision_num / precision_den)

            # Recall: R_c = log( sum(p_c * g_c) / sum(g_c) )
            # "How much recall do I have on class c?"
            recall_num = (p_c * g_c).sum() + self.eps
            recall_den = g_c.sum() + self.eps
            R_c = torch.log(recall_num / recall_den)

            # Specificity: S_c = log( sum((1-p_c) * (1-g_c)) / sum(1-g_c) )
            # "Am I correctly NOT predicting class c where it isn't present?"
            spec_num = ((1 - p_c) * (1 - g_c)).sum() + self.eps
            spec_den = (1 - g_c).sum() + self.eps
            S_c = torch.log(spec_num / spec_den)

            # Combined loss for this class
            losses.append(P_c + R_c + S_c)

        if len(losses) == 0:
            return inputs.sum() * 0.0

        # Macro-average over classes (negative because we maximize P, R, S)
        scal_loss = -torch.stack(losses).mean()

        return scal_loss


class ObservationWeightedFocalLoss(nn.Module):
    """
    Observation-Weighted Focal Loss for LiDAR semantic segmentation.

    Weights loss by observation density: pixels with more LiDAR observations
    are weighted higher since we have more confidence in their labels.

    This helps the model:
    1. Focus on well-observed regions (high confidence supervision)
    2. Avoid overfitting to sparsely observed regions (noisy supervision)
    3. Learn to complete unobserved regions from observed patterns

    Args:
        obs_weight_factor: Scale factor for observation weighting (0 = no weighting)
        focal_alpha: Focal loss alpha parameter
        focal_gamma: Focal loss gamma parameter
        ignore_index: Class index to ignore
        class_weights: Optional per-class weights
    """

    def __init__(self, obs_weight_factor: float = 0.5,
                 focal_alpha: float = -1, focal_gamma: float = 2.0,
                 ignore_index: int = 0,
                 class_weights: torch.Tensor = None):
        super().__init__()
        self.obs_weight_factor = obs_weight_factor
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.ignore_index = ignore_index
        self.register_buffer('class_weights', class_weights)

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor,
                obs_density: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            inputs: [B, C, H, W] logits
            targets: [B, H, W] class indices
            obs_density: [B, H, W] observation density map (sum of LiDAR observations per BEV pixel)
                        If None, falls back to standard Focal Loss

        Returns:
            Weighted loss scalar
        """
        # Compute standard cross entropy
        if self.class_weights is not None:
            ce_loss = F.cross_entropy(inputs, targets, weight=self.class_weights,
                                       reduction='none', ignore_index=self.ignore_index)
        else:
            ce_loss = F.cross_entropy(inputs, targets, reduction='none',
                                       ignore_index=self.ignore_index)

        # Get probabilities for focal modulation
        p = torch.softmax(inputs, dim=1)
        p_t = p.gather(1, targets.unsqueeze(1)).squeeze(1)

        # Focal weight: (1 - p_t)^gamma
        focal_weight = (1 - p_t) ** self.focal_gamma

        # Apply focal alpha
        if isinstance(self.focal_alpha, (int, float)) and self.focal_alpha >= 0:
            focal_loss = self.focal_alpha * focal_weight * ce_loss
        else:
            focal_loss = focal_weight * ce_loss

        # Create valid pixel mask
        mask = (targets != self.ignore_index).float()

        # Apply observation-based weighting if provided
        if obs_density is not None and self.obs_weight_factor > 0:
            # Normalize observation density to [0, 1]
            # Use log scale to handle high variance in observation counts
            obs_norm = torch.log1p(obs_density.float())
            obs_norm = obs_norm / (obs_norm.max() + 1e-8)

            # Create weight: base_weight + obs_weight_factor * normalized_obs
            # This ensures all pixels contribute, but observed ones contribute more
            obs_weight = 1.0 + self.obs_weight_factor * obs_norm

            focal_loss = focal_loss * obs_weight

        # Apply mask and reduce
        focal_loss = focal_loss * mask

        return focal_loss.sum() / (mask.sum() + 1e-8)


class BoundaryAwareLoss(nn.Module):
    """
    Boundary-Aware Loss for improved segmentation boundaries.

    Applies higher weight to pixels near class boundaries, helping the model
    learn precise boundaries between different semantic classes.

    Args:
        boundary_weight: Additional weight for boundary pixels
        boundary_dilation: Dilation kernel size for boundary detection
        ignore_index: Class index to ignore
    """

    def __init__(self, boundary_weight: float = 2.0,
                 boundary_dilation: int = 3,
                 ignore_index: int = 0):
        super().__init__()
        self.boundary_weight = boundary_weight
        self.boundary_dilation = boundary_dilation
        self.ignore_index = ignore_index

        # Create Laplacian kernel for edge detection
        laplacian = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32)
        self.register_buffer('laplacian', laplacian.view(1, 1, 3, 3))

    def detect_boundaries(self, targets: torch.Tensor) -> torch.Tensor:
        """
        Detect boundaries between different classes using Laplacian edge detection.

        Args:
            targets: [B, H, W] class indices

        Returns:
            [B, H, W] boundary mask (1 for boundary, 0 for non-boundary)
        """
        B, H, W = targets.shape

        # Convert to float and add channel dimension
        targets_float = targets.float().unsqueeze(1)  # [B, 1, H, W]

        # Apply Laplacian to detect changes (ensure kernel is on same device)
        laplacian = self.laplacian.to(targets_float.device)
        edges = F.conv2d(targets_float, laplacian, padding=1)

        # Threshold: any non-zero gradient indicates boundary
        boundary = (edges.abs() > 0.5).float().squeeze(1)  # [B, H, W]

        # Dilate boundaries slightly for larger boundary region
        if self.boundary_dilation > 1:
            kernel_size = self.boundary_dilation
            boundary = F.max_pool2d(
                boundary.unsqueeze(1),
                kernel_size=kernel_size,
                stride=1,
                padding=kernel_size // 2
            ).squeeze(1)

        return boundary

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: [B, C, H, W] logits
            targets: [B, H, W] class indices

        Returns:
            Boundary-weighted cross entropy loss
        """
        # Detect boundaries
        boundary_mask = self.detect_boundaries(targets)

        # Compute per-pixel cross entropy
        ce_loss = F.cross_entropy(inputs, targets, reduction='none',
                                   ignore_index=self.ignore_index)

        # Create weight map: 1.0 for non-boundary, boundary_weight for boundary
        weight_map = 1.0 + (self.boundary_weight - 1.0) * boundary_mask

        # Apply weights
        weighted_loss = ce_loss * weight_map

        # Mask and reduce
        mask = (targets != self.ignore_index).float()
        weighted_loss = weighted_loss * mask

        return weighted_loss.sum() / (mask.sum() + 1e-8)


class CombinedLoss(nn.Module):
    """
    Combined Focal + Lovász + SCAL loss

    L_total = L_CB-Focal + λ_lovasz * L_Lovász + λ_scal * L_SCAL

    - CB-Focal: Per-voxel supervision with class balancing
    - Lovász: IoU surrogate for direct metric optimization
    - SCAL: Scene-level P/R/S optimization with macro-average over classes
    """

    def __init__(self, focal_alpha: float = -1, focal_gamma: float = 2,
                 lovasz_weight: float = 1.0, scal_weight: float = 0.0,
                 ignore_index: int = 0, num_classes: int = 20,
                 class_weights: torch.Tensor = None):
        super().__init__()
        self.focal = FocalLoss(alpha=focal_alpha, gamma=focal_gamma,
                               ignore_index=ignore_index, class_weights=class_weights)
        self.lovasz = LovaszSoftmax(ignore_index=ignore_index)
        self.scal = SceneClassAffinityLoss(num_classes=num_classes, ignore_index=ignore_index)
        self.lovasz_weight = lovasz_weight
        self.scal_weight = scal_weight

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        focal_loss = self.focal(inputs, targets)
        total_loss = focal_loss

        if self.lovasz_weight > 0:
            lovasz_loss = self.lovasz(inputs, targets)
            total_loss = total_loss + self.lovasz_weight * lovasz_loss

        if self.scal_weight > 0:
            scal_loss = self.scal(inputs, targets)
            total_loss = total_loss + self.scal_weight * scal_loss

        return total_loss


class CombinedLossV2(nn.Module):
    """
    Enhanced Combined Loss with observation weighting and boundary awareness.

    L_total = L_obs_focal + λ_lovasz * L_Lovász + λ_scal * L_SCAL + λ_boundary * L_boundary

    Features:
    - Observation-weighted Focal loss for LiDAR density awareness
    - Lovász loss for direct IoU optimization
    - SCAL for scene-level class balancing
    - Boundary-aware loss for crisp segmentation edges
    """

    def __init__(self,
                 focal_alpha: float = -1, focal_gamma: float = 2,
                 obs_weight_factor: float = 0.5,
                 lovasz_weight: float = 0.5,
                 scal_weight: float = 0.0,
                 boundary_weight: float = 0.0,
                 boundary_dilation: int = 3,
                 ignore_index: int = 0,
                 num_classes: int = 20,
                 class_weights: torch.Tensor = None):
        super().__init__()

        # Primary loss: observation-weighted focal
        self.obs_focal = ObservationWeightedFocalLoss(
            obs_weight_factor=obs_weight_factor,
            focal_alpha=focal_alpha,
            focal_gamma=focal_gamma,
            ignore_index=ignore_index,
            class_weights=class_weights
        )

        # Auxiliary losses
        self.lovasz = LovaszSoftmax(ignore_index=ignore_index)
        self.scal = SceneClassAffinityLoss(num_classes=num_classes, ignore_index=ignore_index)
        self.boundary_loss = BoundaryAwareLoss(
            boundary_weight=2.0,
            boundary_dilation=boundary_dilation,
            ignore_index=ignore_index
        )

        self.lovasz_weight = lovasz_weight
        self.scal_weight = scal_weight
        self.boundary_weight = boundary_weight

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor,
                obs_density: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            inputs: [B, C, H, W] logits
            targets: [B, H, W] class indices
            obs_density: [B, H, W] optional observation density map
        """
        # Primary loss
        total_loss = self.obs_focal(inputs, targets, obs_density)

        # Lovász loss
        if self.lovasz_weight > 0:
            total_loss = total_loss + self.lovasz_weight * self.lovasz(inputs, targets)

        # SCAL loss
        if self.scal_weight > 0:
            total_loss = total_loss + self.scal_weight * self.scal(inputs, targets)

        # Boundary loss
        if self.boundary_weight > 0:
            total_loss = total_loss + self.boundary_weight * self.boundary_loss(inputs, targets)

        return total_loss


class ProgressiveLoss(nn.Module):
    """
    Progressive Loss Schedule for curriculum learning.

    Gradually introduces loss components over training:
    - Phase 1 (epochs 0-29): Only Focal loss (learn basic classification)
    - Phase 2 (epochs 30-59): Add Lovász loss with linear warmup (improve IoU)
    - Phase 3 (epochs 60+): Full Lovász weight

    This tests ONE change from baseline: progressive loss scheduling.
    The final loss formula is the same as baseline (Focal + 0.5*Lovász),
    but the weights change over time.

    Formula:
        epoch < 30:  L = L_focal
        epoch 30-59: L = L_focal + (epoch-30)/30 * 0.5 * L_lovász  (linear warmup)
        epoch >= 60: L = L_focal + 0.5 * L_lovász  (same as baseline)
    """

    def __init__(self,
                 focal_alpha: float = -1,
                 focal_gamma: float = 2,
                 lovasz_weight: float = 0.5,
                 phase1_epochs: int = 30,
                 phase2_epochs: int = 30,
                 ignore_index: int = 0,
                 num_classes: int = 20,
                 class_weights: torch.Tensor = None):
        super().__init__()

        self.focal = FocalLoss(
            alpha=focal_alpha,
            gamma=focal_gamma,
            ignore_index=ignore_index,
            class_weights=class_weights
        )
        self.lovasz = LovaszSoftmax(ignore_index=ignore_index)

        self.lovasz_weight = lovasz_weight
        self.phase1_epochs = phase1_epochs
        self.phase2_epochs = phase2_epochs
        self.current_epoch = 0

    def set_epoch(self, epoch: int):
        """Update current epoch for progressive scheduling"""
        self.current_epoch = epoch

    def get_lovasz_weight(self) -> float:
        """Get current Lovász weight based on epoch"""
        if self.current_epoch < self.phase1_epochs:
            # Phase 1: Only focal loss
            return 0.0
        elif self.current_epoch < self.phase1_epochs + self.phase2_epochs:
            # Phase 2: Linear warmup of Lovász
            progress = (self.current_epoch - self.phase1_epochs) / self.phase2_epochs
            return progress * self.lovasz_weight
        else:
            # Phase 3: Full Lovász weight
            return self.lovasz_weight

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor,
                obs_density: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            inputs: [B, C, H, W] logits
            targets: [B, H, W] class indices
            obs_density: ignored (for API compatibility)
        """
        # Primary loss: always Focal
        total_loss = self.focal(inputs, targets)

        # Progressive Lovász
        current_lovasz_weight = self.get_lovasz_weight()
        if current_lovasz_weight > 0:
            total_loss = total_loss + current_lovasz_weight * self.lovasz(inputs, targets)

        return total_loss


def test_model():
    """Test the model with random data"""
    print("Testing TinySparseBEVNet (32x32x4)...")

    model = TinySparseBEVNet(num_classes=20).cuda()

    # Create dummy sparse input
    batch_size = 2
    num_points = 1000

    # Random voxel features (just occupancy = 1)
    voxels = torch.ones(num_points, 1).cuda()

    # Random coordinates [batch_idx, x, y, z]
    coords = torch.zeros(num_points, 4, dtype=torch.int32).cuda()
    coords[:, 0] = torch.randint(0, batch_size, (num_points,))
    coords[:, 1] = torch.randint(0, 32, (num_points,))
    coords[:, 2] = torch.randint(0, 32, (num_points,))
    coords[:, 3] = torch.randint(0, 4, (num_points,))

    # Forward pass
    with torch.no_grad():
        output = model(voxels, coords, batch_size)

    print(f"Input: {num_points} voxels")
    print(f"Output shape: {output.shape}")
    print("Expected: [2, 20, 32, 32]")

    # Count parameters
    params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {params:,}")

    return output.shape == (2, 20, 32, 32)


if __name__ == "__main__":
    success = test_model()
    print(f"\nTest {'PASSED' if success else 'FAILED'}")
