"""
Sparse 3D LiDAR Encoder for BEV Feature Extraction
(BEV second-task auxiliary encoder; genuinely sparse SubMConv3d --
NOT the dense Conv3d + additive/AdaGN S2D2 scene-completion denoiser,
which lives in s2d2_unet.py)

Uses Submanifold Sparse Convolutions (spconv) to efficiently process
sparse LiDAR voxel grids and project to BEV features.

Architecture:
    - Sparse 3D encoder (SubMConv3d, preserves sparsity) for the BEV
      secondary task -- distinct from the dense Conv3d S2D2 denoiser in
      s2d2_unet.py
    - Strided convolutions for downsampling
    - Height compression to produce BEV features

Reference: https://arxiv.org/abs/1711.10275 (Submanifold Sparse ConvNets)
"""


import spconv.pytorch as spconv
import torch
import torch.nn as nn
import torch.nn.functional as F

# Use Native algorithm to avoid JIT compilation issues on H100 (sm_90)
_ALGO = spconv.ConvAlgo.Native


class SparseBasicBlock(nn.Module):
    """Basic residual block using submanifold sparse convolutions."""

    def __init__(self, in_channels: int, out_channels: int, indice_key: str = None):
        super().__init__()

        self.conv1 = spconv.SubMConv3d(
            in_channels, out_channels, 3, padding=1,
            bias=False, indice_key=indice_key, algo=_ALGO
        )
        self.bn1 = nn.BatchNorm1d(out_channels)

        self.conv2 = spconv.SubMConv3d(
            out_channels, out_channels, 3, padding=1,
            bias=False, indice_key=indice_key, algo=_ALGO
        )
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.relu = nn.ReLU(inplace=True)

        # Skip connection if channels change
        if in_channels != out_channels:
            self.skip = spconv.SubMConv3d(
                in_channels, out_channels, 1, bias=False, indice_key=indice_key, algo=_ALGO
            )
            self.skip_bn = nn.BatchNorm1d(out_channels)
        else:
            self.skip = None

    def forward(self, x: spconv.SparseConvTensor) -> spconv.SparseConvTensor:
        identity = x

        out = self.conv1(x)
        out = out.replace_feature(self.bn1(out.features))
        out = out.replace_feature(self.relu(out.features))

        out = self.conv2(out)
        out = out.replace_feature(self.bn2(out.features))

        if self.skip is not None:
            identity = self.skip(identity)
            identity = identity.replace_feature(self.skip_bn(identity.features))

        out = out.replace_feature(self.relu(out.features + identity.features))
        return out


class SparseDownBlock(nn.Module):
    """Downsampling block with strided sparse convolution."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: tuple[int, int, int] = (2, 2, 2),
        indice_key: str = None
    ):
        super().__init__()

        self.conv = spconv.SparseConv3d(
            in_channels, out_channels, 3, stride=stride, padding=1,
            bias=False, indice_key=indice_key, algo=_ALGO
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: spconv.SparseConvTensor) -> spconv.SparseConvTensor:
        out = self.conv(x)
        out = out.replace_feature(self.bn(out.features))
        out = out.replace_feature(self.relu(out.features))
        return out


class SparseLiDAREncoder(nn.Module):
    """
    Sparse SubMConv3d LiDAR-to-BEV encoder for the BEV secondary task.

    This is the genuinely-sparse (spconv) auxiliary encoder used by the BEV
    refinement task — distinct from the dense ``Conv3d`` S²D² scene-completion
    denoiser. The "Sparse" here refers to this encoder only, not the denoiser
    body (see the file header).

    Takes sparse voxel grid (256×256×32) and outputs dense BEV features (256×256×C).

    Architecture:
        Input: Sparse voxels [B, 256, 256, 32] with binary occupancy

        Encoder:
            - stem: SubMConv3d 1→32
            - block1: SubMConv3d 32→32, res block
            - down1: SparseConv3d stride=(2,2,2) → 128×128×16, 64ch
            - block2: SubMConv3d 64→64, res block
            - down2: SparseConv3d stride=(2,2,2) → 64×64×8, 128ch
            - block3: SubMConv3d 128→128, res block
            - down3: SparseConv3d stride=(2,2,1) → 32×32×8, 256ch (only downsample XY)

        BEV Projection:
            - Height pooling: max/mean pool along Z axis
            - Upsample to target resolution

        Output: Dense BEV features [B, C, 256, 256]

    Args:
        in_channels: Input channels (1 for binary occupancy)
        base_channels: Base channel count (default: 32)
        out_channels: Output BEV feature channels (default: 128)
        height_pool: Height pooling method ('max', 'mean', or 'learned')
    """

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 32,
        out_channels: int = 128,
        height_pool: str = 'max',
        input_shape: tuple[int, int, int] = (256, 256, 32),
    ):
        super().__init__()

        self.in_channels = in_channels
        self.base_channels = base_channels
        self.out_channels = out_channels
        self.height_pool = height_pool
        self.input_shape = input_shape

        C = base_channels

        # Stem
        self.stem = spconv.SubMConv3d(
            in_channels, C, 3, padding=1, bias=False, indice_key='stem', algo=_ALGO
        )
        self.stem_bn = nn.BatchNorm1d(C)
        self.stem_relu = nn.ReLU(inplace=True)

        # Encoder blocks
        # Level 1: 256×256×32, 32ch
        self.block1 = SparseBasicBlock(C, C, indice_key='block1')

        # Level 2: 128×128×16, 64ch
        self.down1 = SparseDownBlock(C, C*2, stride=(2, 2, 2), indice_key='down1')
        self.block2 = SparseBasicBlock(C*2, C*2, indice_key='block2')

        # Level 3: 64×64×8, 128ch
        self.down2 = SparseDownBlock(C*2, C*4, stride=(2, 2, 2), indice_key='down2')
        self.block3 = SparseBasicBlock(C*4, C*4, indice_key='block3')

        # Level 4: 32×32×8, 256ch (only downsample XY, keep Z)
        self.down3 = SparseDownBlock(C*4, C*8, stride=(2, 2, 1), indice_key='down3')
        self.block4 = SparseBasicBlock(C*8, C*8, indice_key='block4')

        # Height compression - collapse Z dimension
        # After down3: shape is 32×32×8 with 256 channels
        # We'll convert sparse to dense and pool along Z

        # Learned height attention (optional)
        if height_pool == 'learned':
            self.height_attention = nn.Sequential(
                nn.Conv1d(C*8, C*8, 1),
                nn.ReLU(inplace=True),
                nn.Conv1d(C*8, 1, 1),
            )

        # Multi-scale BEV feature fusion
        # Upsample from 32×32 to 256×256
        self.bev_upsample = nn.Sequential(
            nn.ConvTranspose2d(C*8, C*4, 4, stride=2, padding=1),  # 32→64
            nn.BatchNorm2d(C*4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(C*4, C*2, 4, stride=2, padding=1),  # 64→128
            nn.BatchNorm2d(C*2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(C*2, C, 4, stride=2, padding=1),    # 128→256
            nn.BatchNorm2d(C),
            nn.ReLU(inplace=True),
            nn.Conv2d(C, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(
        self,
        voxels: torch.Tensor,
        coords: torch.Tensor | None = None,
        batch_size: int | None = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            voxels: Either:
                - Dense tensor [B, 1, X, Y, Z] (will be converted to sparse)
                - Sparse features [N, C] if coords provided
            coords: Voxel coordinates [N, 4] (batch_idx, x, y, z) if sparse input
            batch_size: Batch size if using sparse input

        Returns:
            BEV features [B, out_channels, H, W]
        """
        # Convert dense to sparse if needed
        if coords is None:
            # Dense input [B, 1, X, Y, Z]
            sparse_tensor = self._dense_to_sparse(voxels)
        else:
            # Sparse input
            sparse_tensor = spconv.SparseConvTensor(
                features=voxels,
                indices=coords.int(),
                spatial_shape=self.input_shape,
                batch_size=batch_size
            )

        # Encoder
        x = self.stem(sparse_tensor)
        x = x.replace_feature(self.stem_bn(x.features))
        x = x.replace_feature(self.stem_relu(x.features))

        x = self.block1(x)
        x = self.down1(x)
        x = self.block2(x)
        x = self.down2(x)
        x = self.block3(x)
        x = self.down3(x)
        x = self.block4(x)

        # Convert to dense for height pooling
        # Shape after block4: [B, C*8, 32, 32, 8]
        dense = x.dense()  # [B, C, X, Y, Z]

        # Height pooling
        if self.height_pool == 'max':
            bev = dense.max(dim=-1)[0]  # [B, C, X, Y]
        elif self.height_pool == 'mean':
            bev = dense.mean(dim=-1)
        elif self.height_pool == 'learned':
            # Learned attention over height
            B, C, X, Y, Z = dense.shape
            dense_flat = dense.permute(0, 2, 3, 1, 4).reshape(B*X*Y, C, Z)
            attn = self.height_attention(dense_flat)  # [B*X*Y, 1, Z]
            attn = F.softmax(attn, dim=-1)
            bev = (dense_flat * attn).sum(dim=-1)  # [B*X*Y, C]
            bev = bev.reshape(B, X, Y, C).permute(0, 3, 1, 2)  # [B, C, X, Y]
        else:
            raise ValueError(f"Unknown height_pool: {self.height_pool}")

        # Upsample to full resolution
        bev = self.bev_upsample(bev)  # [B, out_channels, 256, 256]

        return bev

    def forward_multiscale(
        self,
        voxels: torch.Tensor,
        coords: torch.Tensor | None = None,
        batch_size: int | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass returning multi-scale BEV features.

        Returns features at multiple resolutions for multi-scale conditioning:
        - scale_256: Full resolution [B, out_channels, 256, 256]
        - scale_64: 1/4 resolution [B, 128, 64, 64]
        - scale_32: 1/8 resolution [B, 256, 32, 32]

        This enables the UNet to condition at matching resolutions.

        Args:
            voxels: Dense tensor [B, 1, X, Y, Z] or sparse features
            coords: Voxel coordinates if sparse input
            batch_size: Batch size if using sparse input

        Returns:
            Dict with 'scale_256', 'scale_64', 'scale_32' features
        """
        # Convert dense to sparse if needed
        if coords is None:
            sparse_tensor = self._dense_to_sparse(voxels)
        else:
            sparse_tensor = spconv.SparseConvTensor(
                features=voxels,
                indices=coords.int(),
                spatial_shape=self.input_shape,
                batch_size=batch_size
            )

        # Encoder with intermediate outputs
        x = self.stem(sparse_tensor)
        x = x.replace_feature(self.stem_bn(x.features))
        x = x.replace_feature(self.stem_relu(x.features))

        x = self.block1(x)  # 256×256×32, 32ch
        x = self.down1(x)   # 128×128×16, 64ch
        x = self.block2(x)

        x = self.down2(x)   # 64×64×8, 128ch
        x = self.block3(x)
        # Extract 64×64 features
        dense_64 = x.dense()  # [B, 128, 64, 64, 8]
        bev_64 = dense_64.max(dim=-1)[0]  # [B, 128, 64, 64]

        x = self.down3(x)   # 32×32×8, 256ch
        x = self.block4(x)
        # Extract 32×32 features
        dense_32 = x.dense()  # [B, 256, 32, 32, 8]
        bev_32 = dense_32.max(dim=-1)[0]  # [B, 256, 32, 32]

        # Height pooling for full resolution path
        if self.height_pool == 'max':
            bev = dense_32.max(dim=-1)[0]
        elif self.height_pool == 'mean':
            bev = dense_32.mean(dim=-1)
        elif self.height_pool == 'learned':
            B, C, X, Y, Z = dense_32.shape
            dense_flat = dense_32.permute(0, 2, 3, 1, 4).reshape(B*X*Y, C, Z)
            attn = self.height_attention(dense_flat)
            attn = F.softmax(attn, dim=-1)
            bev = (dense_flat * attn).sum(dim=-1)
            bev = bev.reshape(B, X, Y, C).permute(0, 3, 1, 2)
        else:
            bev = dense_32.max(dim=-1)[0]

        # Upsample to full resolution
        bev_256 = self.bev_upsample(bev)  # [B, out_channels, 256, 256]

        return {
            'scale_256': bev_256,  # [B, out_channels, 256, 256]
            'scale_64': bev_64,    # [B, 128, 64, 64]
            'scale_32': bev_32,    # [B, 256, 32, 32]
        }

    def _dense_to_sparse(self, dense: torch.Tensor) -> spconv.SparseConvTensor:
        """Convert dense voxel tensor to sparse representation."""
        # dense: [B, 1, X, Y, Z]
        B = dense.shape[0]
        device = dense.device

        # Find non-zero voxels
        coords_list = []
        features_list = []

        for b in range(B):
            # Get occupied voxel indices
            occupied = dense[b, 0] > 0  # [X, Y, Z]
            coords = torch.nonzero(occupied)  # [N, 3] - (x, y, z)

            # Add batch index
            batch_idx = torch.full((coords.shape[0], 1), b, device=device, dtype=torch.int32)
            coords_with_batch = torch.cat([batch_idx, coords.int()], dim=1)  # [N, 4]

            # Features (just 1s for binary occupancy)
            feats = torch.ones(coords.shape[0], 1, device=device)

            coords_list.append(coords_with_batch)
            features_list.append(feats)

        # Concatenate all batches
        all_coords = torch.cat(coords_list, dim=0)
        all_features = torch.cat(features_list, dim=0)

        # Create sparse tensor
        sparse_tensor = spconv.SparseConvTensor(
            features=all_features,
            indices=all_coords,
            spatial_shape=self.input_shape,
            batch_size=B
        )

        return sparse_tensor


class LightweightLiDAREncoder(nn.Module):
    """
    Lightweight version of LiDAR encoder for faster experiments.

    Uses fewer channels and layers for quick iteration.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 64,
        input_shape: tuple[int, int, int] = (256, 256, 32),
    ):
        super().__init__()

        self.input_shape = input_shape

        # Simpler encoder
        self.encoder = spconv.SparseSequential(
            spconv.SubMConv3d(in_channels, 16, 3, padding=1, bias=False, indice_key='s0', algo=_ALGO),
            nn.BatchNorm1d(16),
            nn.ReLU(True),

            spconv.SparseConv3d(16, 32, 3, stride=(2,2,2), padding=1, bias=False, indice_key='d1', algo=_ALGO),
            nn.BatchNorm1d(32),
            nn.ReLU(True),

            spconv.SubMConv3d(32, 32, 3, padding=1, bias=False, indice_key='s1', algo=_ALGO),
            nn.BatchNorm1d(32),
            nn.ReLU(True),

            spconv.SparseConv3d(32, 64, 3, stride=(2,2,2), padding=1, bias=False, indice_key='d2', algo=_ALGO),
            nn.BatchNorm1d(64),
            nn.ReLU(True),

            spconv.SubMConv3d(64, 64, 3, padding=1, bias=False, indice_key='s2', algo=_ALGO),
            nn.BatchNorm1d(64),
            nn.ReLU(True),
        )

        # BEV projection (from 64×64×8 to 256×256)
        self.bev_project = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 4, stride=2, padding=1),  # 64→128
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, out_channels, 4, stride=2, padding=1),  # 128→256
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
        )

    def forward(self, voxels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            voxels: Dense voxel tensor [B, 1, X, Y, Z]
        Returns:
            BEV features [B, out_channels, 256, 256]
        """
        # Convert to sparse
        sparse = self._dense_to_sparse(voxels)

        # Encode
        sparse = self.encoder(sparse)

        # To dense and height pool
        dense = sparse.dense()  # [B, C, X, Y, Z]
        bev = dense.max(dim=-1)[0]  # [B, C, X, Y]

        # Upsample
        bev = self.bev_project(bev)

        return bev

    def _dense_to_sparse(self, dense: torch.Tensor) -> spconv.SparseConvTensor:
        """Convert dense voxel tensor to sparse representation."""
        B = dense.shape[0]
        device = dense.device

        coords_list = []
        features_list = []

        for b in range(B):
            occupied = dense[b, 0] > 0
            coords = torch.nonzero(occupied)
            batch_idx = torch.full((coords.shape[0], 1), b, device=device, dtype=torch.int32)
            coords_with_batch = torch.cat([batch_idx, coords.int()], dim=1)
            feats = torch.ones(coords.shape[0], 1, device=device)

            coords_list.append(coords_with_batch)
            features_list.append(feats)

        all_coords = torch.cat(coords_list, dim=0)
        all_features = torch.cat(features_list, dim=0)

        return spconv.SparseConvTensor(
            features=all_features,
            indices=all_coords,
            spatial_shape=self.input_shape,
            batch_size=B
        )


class TinySparseLiDAREncoder(nn.Module):
    """
    Tiny Sparse 3D Encoder for 32×32×4 voxel input.

    Designed for fast architecture search before scaling to full 256×256×32.
    Uses sparse 3D convolutions to process the voxel grid efficiently.

    Architecture:
        Input: Sparse voxels [B, 1, 32, 32, 4]

        Stem: SubMConv3d(1→16)
        Level 1: SubMConv3d(16→32), 32×32×4
        Down1: SparseConv3d stride=(2,2,1) → 16×16×4, 64ch
        Level 2: SubMConv3d(64→64), 16×16×4
        Down2: SparseConv3d stride=(2,2,2) → 8×8×2, 128ch
        Level 3: SubMConv3d(128→128), 8×8×2

        Height Pool: max over Z → 8×8
        BEV Upsample: 8×8 → 16×16 → 32×32

        Output: [B, out_channels, 32, 32]

    Args:
        in_channels: Input feature channels (1 for binary occupancy)
        out_channels: Output BEV feature channels
        height_pool: 'max', 'mean', or 'learned'
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 64,
        height_pool: str = 'max',
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.height_pool = height_pool
        self.input_shape = (32, 32, 4)  # X, Y, Z

        # Stem
        self.stem = spconv.SubMConv3d(
            in_channels, 16, 3, padding=1, bias=False, indice_key='tiny_stem', algo=_ALGO
        )
        self.stem_bn = nn.BatchNorm1d(16)

        # Level 1: 32×32×4, 32ch
        self.block1 = spconv.SubMConv3d(
            16, 32, 3, padding=1, bias=False, indice_key='tiny_block1', algo=_ALGO
        )
        self.block1_bn = nn.BatchNorm1d(32)

        # Down1: 32×32×4 → 16×16×4, 64ch
        self.down1 = spconv.SparseConv3d(
            32, 64, 3, stride=(2, 2, 1), padding=1, bias=False, indice_key='tiny_down1', algo=_ALGO
        )
        self.down1_bn = nn.BatchNorm1d(64)

        # Level 2: 16×16×4, 64ch
        self.block2 = spconv.SubMConv3d(
            64, 64, 3, padding=1, bias=False, indice_key='tiny_block2', algo=_ALGO
        )
        self.block2_bn = nn.BatchNorm1d(64)

        # Down2: 16×16×4 → 8×8×2, 128ch
        self.down2 = spconv.SparseConv3d(
            64, 128, 3, stride=(2, 2, 2), padding=1, bias=False, indice_key='tiny_down2', algo=_ALGO
        )
        self.down2_bn = nn.BatchNorm1d(128)

        # Level 3: 8×8×2, 128ch
        self.block3 = spconv.SubMConv3d(
            128, 128, 3, padding=1, bias=False, indice_key='tiny_block3', algo=_ALGO
        )
        self.block3_bn = nn.BatchNorm1d(128)

        self.relu = nn.ReLU(inplace=True)

        # Learned height attention (optional)
        if height_pool == 'learned':
            self.height_attention = nn.Sequential(
                nn.Conv1d(128, 64, 1),
                nn.ReLU(inplace=True),
                nn.Conv1d(64, 1, 1),
            )

        # BEV upsample: 8×8 → 32×32
        self.bev_upsample = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),  # 8→16
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, out_channels, 4, stride=2, padding=1),  # 16→32
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, voxels: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            voxels: Dense voxel tensor [B, 1, X, Y, Z] where X=Y=32, Z=4

        Returns:
            BEV features [B, out_channels, 32, 32]
        """
        # Convert dense to sparse
        sparse = self._dense_to_sparse(voxels)

        # Encoder
        x = self.stem(sparse)
        x = x.replace_feature(self.relu(self.stem_bn(x.features)))

        x = self.block1(x)
        x = x.replace_feature(self.relu(self.block1_bn(x.features)))

        x = self.down1(x)
        x = x.replace_feature(self.relu(self.down1_bn(x.features)))

        x = self.block2(x)
        x = x.replace_feature(self.relu(self.block2_bn(x.features)))

        x = self.down2(x)
        x = x.replace_feature(self.relu(self.down2_bn(x.features)))

        x = self.block3(x)
        x = x.replace_feature(self.relu(self.block3_bn(x.features)))

        # Convert to dense for height pooling
        # Shape after block3: [B, 128, 8, 8, 2]
        dense = x.dense()

        # Height pooling
        if self.height_pool == 'max':
            bev = dense.max(dim=-1)[0]  # [B, 128, 8, 8]
        elif self.height_pool == 'mean':
            bev = dense.mean(dim=-1)
        elif self.height_pool == 'learned':
            B, C, X, Y, Z = dense.shape
            dense_flat = dense.permute(0, 2, 3, 1, 4).reshape(B * X * Y, C, Z)
            attn = self.height_attention(dense_flat)  # [B*X*Y, 1, Z]
            attn = F.softmax(attn, dim=-1)
            bev = (dense_flat * attn).sum(dim=-1)  # [B*X*Y, C]
            bev = bev.reshape(B, X, Y, C).permute(0, 3, 1, 2)  # [B, C, X, Y]
        else:
            raise ValueError(f"Unknown height_pool: {self.height_pool}")

        # Upsample to 32×32
        bev = self.bev_upsample(bev)

        return bev

    def _dense_to_sparse(self, dense: torch.Tensor) -> spconv.SparseConvTensor:
        """Convert dense voxel tensor to sparse representation."""
        B = dense.shape[0]
        device = dense.device

        coords_list = []
        features_list = []

        for b in range(B):
            occupied = dense[b, 0] > 0  # [X, Y, Z]
            coords = torch.nonzero(occupied)  # [N, 3]
            batch_idx = torch.full((coords.shape[0], 1), b, device=device, dtype=torch.int32)
            coords_with_batch = torch.cat([batch_idx, coords.int()], dim=1)  # [N, 4]
            feats = torch.ones(coords.shape[0], 1, device=device)

            coords_list.append(coords_with_batch)
            features_list.append(feats)

        all_coords = torch.cat(coords_list, dim=0)
        all_features = torch.cat(features_list, dim=0)

        return spconv.SparseConvTensor(
            features=all_features,
            indices=all_coords,
            spatial_shape=self.input_shape,
            batch_size=B
        )


# Alias for backward compatibility
LiDAREncoder = SparseLiDAREncoder
# Note: SimpleLiDAREncoder was a 2D encoder - deprecated, use TinySparseLiDAREncoder instead
