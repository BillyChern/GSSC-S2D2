"""
SpconvGlobalEnc: Encoder-only network for partial semantic point cloud.

Replaces MinkGlobalEnc from DiffSSC reference with spconv v2.
4 downsample stages: [32, 32, 64, 128, 256] channels.
Output: SparseConvTensor at coarse resolution (16x spatial reduction), 256ch per voxel.
"""

import spconv.pytorch as spconv
import torch.nn as nn


class BasicConvBlock(nn.Module):
    """Conv3d + BN + ReLU for sparse tensors."""
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


class ResidualBlock(nn.Module):
    """Two SubMConv3d with residual connection."""
    def __init__(self, in_ch, out_ch, ks=3, stride=1, indice_key=None, down_key=None):
        super().__init__()
        if stride > 1:
            self.conv1 = spconv.SparseSequential(
                spconv.SparseConv3d(in_ch, out_ch, kernel_size=ks, stride=stride,
                                     padding=ks // 2, indice_key=down_key),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(inplace=True),
            )
        else:
            self.conv1 = spconv.SparseSequential(
                spconv.SubMConv3d(in_ch, out_ch, kernel_size=ks, indice_key=indice_key),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(inplace=True),
            )

        self.conv2 = spconv.SparseSequential(
            spconv.SubMConv3d(out_ch, out_ch, kernel_size=ks, indice_key=indice_key),
            nn.BatchNorm1d(out_ch),
        )

        if in_ch != out_ch or stride > 1:
            if stride > 1:
                self.downsample = spconv.SparseSequential(
                    spconv.SparseConv3d(in_ch, out_ch, kernel_size=1, stride=stride,
                                         indice_key=down_key + '_skip' if down_key else None),
                    nn.BatchNorm1d(out_ch),
                )
            else:
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


class SpconvGlobalEnc(nn.Module):
    """Encoder-only network for partial point cloud.

    Architecture matches MinkGlobalEnc:
    - Stem: 2x SubMConv3d
    - Stage 1-4: Downsample (stride=2) + 2x ResidualBlock each
    - Channel schedule: [32, 32, 64, 128, 256]
    - Output: 256ch sparse features at 16x reduced resolution

    Args:
        in_channels: input feature dimension (23 for xyz + 20 semantic)
    """
    def __init__(self, in_channels=23):
        super().__init__()
        cs = [32, 32, 64, 128, 256]

        # Stem: 2x SubMConv3d
        self.stem = spconv.SparseSequential(
            spconv.SubMConv3d(in_channels, cs[0], kernel_size=3, indice_key='enc_stem0'),
            nn.BatchNorm1d(cs[0]),
            nn.ReLU(inplace=True),
            spconv.SubMConv3d(cs[0], cs[0], kernel_size=3, indice_key='enc_stem1'),
            nn.BatchNorm1d(cs[0]),
            nn.ReLU(inplace=True),
        )

        # Stage 1: downsample + 2 residual blocks
        # Note: nn.Sequential (not spconv.SparseSequential) for ResidualBlock wrappers
        # because spconv.SparseSequential tries to unwrap features for non-spconv modules
        self.down1 = BasicConvBlock(cs[0], cs[0], ks=2, stride=2, indice_key='enc_down1')
        self.stage1 = nn.Sequential(
            ResidualBlock(cs[0], cs[1], ks=3, indice_key='enc_s1'),
            ResidualBlock(cs[1], cs[1], ks=3, indice_key='enc_s1b'),
        )

        # Stage 2
        self.down2 = BasicConvBlock(cs[1], cs[1], ks=2, stride=2, indice_key='enc_down2')
        self.stage2 = nn.Sequential(
            ResidualBlock(cs[1], cs[2], ks=3, indice_key='enc_s2'),
            ResidualBlock(cs[2], cs[2], ks=3, indice_key='enc_s2b'),
        )

        # Stage 3
        self.down3 = BasicConvBlock(cs[2], cs[2], ks=2, stride=2, indice_key='enc_down3')
        self.stage3 = nn.Sequential(
            ResidualBlock(cs[2], cs[3], ks=3, indice_key='enc_s3'),
            ResidualBlock(cs[3], cs[3], ks=3, indice_key='enc_s3b'),
        )

        # Stage 4
        self.down4 = BasicConvBlock(cs[3], cs[3], ks=2, stride=2, indice_key='enc_down4')
        self.stage4 = nn.Sequential(
            ResidualBlock(cs[3], cs[4], ks=3, indice_key='enc_s4'),
            ResidualBlock(cs[4], cs[4], ks=3, indice_key='enc_s4b'),
        )

        self.embed_dim = cs[4]  # 256
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """Forward pass.

        Args:
            x: spconv.SparseConvTensor [N, in_channels]

        Returns:
            spconv.SparseConvTensor [N_coarse, 256]
        """
        x0 = self.stem(x)
        x1 = self.stage1(self.down1(x0))
        x2 = self.stage2(self.down2(x1))
        x3 = self.stage3(self.down3(x2))
        x4 = self.stage4(self.down4(x3))
        return x4
