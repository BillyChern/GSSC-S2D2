"""
Sparse LiDAR Encoder for Phase 3 Scene Completion (exp_2)

Uses spconv to efficiently process sparse 3D LiDAR voxels (~1% occupancy)
and produce multi-scale dense features for conditioning the 3D U-Net.

This is more efficient than dense Conv3d (exp_1) for SemanticKITTI where
LiDAR voxels have very low occupancy.

Architecture:
- Sparse 3D encoder with SubMConv3d (preserves sparsity)
- Strided SparseConv3d for downsampling
- Convert to dense at each scale for U-Net conditioning
"""


import spconv.pytorch as spconv
import torch
import torch.nn as nn
import torch.nn.functional as F

# Use Native algo to avoid JIT compilation issues with CUDA 12.x headers + spconv-cu118
_ALGO = spconv.ConvAlgo.Native


class SparseResBlock3D(nn.Module):
    """Sparse 3D Residual Block using SubManifold Sparse Convolutions."""

    def __init__(self, in_channels: int, out_channels: int, indice_key: str = None):
        super().__init__()
        self.conv1 = spconv.SubMConv3d(
            in_channels, out_channels, 3, padding=1, bias=False, indice_key=indice_key, algo=_ALGO
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = spconv.SubMConv3d(
            out_channels, out_channels, 3, padding=1, bias=False, indice_key=indice_key, algo=_ALGO
        )
        self.bn2 = nn.BatchNorm1d(out_channels)

        if in_channels != out_channels:
            self.shortcut = spconv.SubMConv3d(
                in_channels, out_channels, 1, bias=False, indice_key=indice_key, algo=_ALGO
            )
            self.shortcut_bn = nn.BatchNorm1d(out_channels)
        else:
            self.shortcut = None

    def forward(self, x: spconv.SparseConvTensor) -> spconv.SparseConvTensor:
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


class SparseLiDAREncoder(nn.Module):
    """
    Sparse 3D LiDAR Encoder for Scene Completion.

    Extracts multi-scale features from sparse 3D LiDAR voxels.
    Returns dense features at each scale to condition the U-Net decoder.

    For SemanticKITTI (256×256×32):
    - Level 0: 256×256×32 → base_channels features
    - Level 1: 128×128×16 → base_channels×2 features
    - Level 2: 64×64×8 → base_channels×4 features
    - Level 3: 32×32×4 → base_channels×8 features
    - Level 4: 16×16×2 → base_channels×8 features (bottleneck)
    """

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 16,
        out_channels: int = 32,  # Output channels for U-Net conditioning
        densify_nn: bool = False,
        add_distance_gate: bool = False,
    ):
        super().__init__()

        self.base_channels = base_channels
        self.out_channels = out_channels
        self.densify_nn = densify_nn
        self.add_distance_gate = add_distance_gate

        # Channel progression: 16 → 32 → 64 → 128 → 128
        ch = [base_channels, base_channels*2, base_channels*4, base_channels*8, base_channels*8]

        # Stem
        self.stem = spconv.SubMConv3d(in_channels, ch[0], 3, padding=1, bias=False, indice_key='stem', algo=_ALGO)
        self.stem_bn = nn.BatchNorm1d(ch[0])

        # Level 0: 256×256×32
        self.level0 = SparseResBlock3D(ch[0], ch[0], indice_key='l0')
        self.down0 = spconv.SparseConv3d(ch[0], ch[1], 3, stride=2, padding=1, bias=False, indice_key='d0', algo=_ALGO)
        self.down0_bn = nn.BatchNorm1d(ch[1])

        # Level 1: 128×128×16
        self.level1 = SparseResBlock3D(ch[1], ch[1], indice_key='l1')
        self.down1 = spconv.SparseConv3d(ch[1], ch[2], 3, stride=2, padding=1, bias=False, indice_key='d1', algo=_ALGO)
        self.down1_bn = nn.BatchNorm1d(ch[2])

        # Level 2: 64×64×8
        self.level2 = SparseResBlock3D(ch[2], ch[2], indice_key='l2')
        self.down2 = spconv.SparseConv3d(ch[2], ch[3], 3, stride=2, padding=1, bias=False, indice_key='d2', algo=_ALGO)
        self.down2_bn = nn.BatchNorm1d(ch[3])

        # Level 3: 32×32×4
        self.level3 = SparseResBlock3D(ch[3], ch[3], indice_key='l3')
        self.down3 = spconv.SparseConv3d(ch[3], ch[4], 3, stride=2, padding=1, bias=False, indice_key='d3', algo=_ALGO)
        self.down3_bn = nn.BatchNorm1d(ch[4])

        # Level 4 (bottleneck): 16×16×2
        self.level4 = SparseResBlock3D(ch[4], ch[4], indice_key='l4')

        # Projection layers to convert to uniform out_channels for U-Net
        self.proj0 = nn.Conv3d(ch[0], out_channels, 1)
        self.proj1 = nn.Conv3d(ch[1], out_channels, 1)
        self.proj2 = nn.Conv3d(ch[2], out_channels, 1)
        self.proj3 = nn.Conv3d(ch[3], out_channels, 1)
        self.proj4 = nn.Conv3d(ch[4], out_channels, 1)

        # Distance gate modules (S26): learned distance-dependent feature modulation
        if add_distance_gate:
            self.dist_gate0 = nn.Sequential(nn.Conv3d(1, out_channels, 1), nn.Sigmoid())
            self.dist_gate1 = nn.Sequential(nn.Conv3d(1, out_channels, 1), nn.Sigmoid())
            self.dist_gate2 = nn.Sequential(nn.Conv3d(1, out_channels, 1), nn.Sigmoid())
            self.dist_gate3 = nn.Sequential(nn.Conv3d(1, out_channels, 1), nn.Sigmoid())
            self.dist_gate4 = nn.Sequential(nn.Conv3d(1, out_channels, 1), nn.Sigmoid())

    def _nn_densify(self, dense: torch.Tensor, nn_indices=None):
        """Fill zero voxels with features from nearest occupied voxel.

        Uses scipy.ndimage.distance_transform_edt for exact NN lookup.
        This is the voxel-grid equivalent of DiffSSC's KNN matching.

        Args:
            dense: [B, C, H, W, D] feature tensor with zeros at unobserved
            nn_indices: Optional [B, 3, H, W, D] int16 precomputed NN indices
        Returns:
            dense: [B, C, H, W, D] fully dense features
            dist_map: [B, 1, H, W, D] distance to nearest observation (if add_distance_gate)
        """
        from scipy.ndimage import distance_transform_edt

        B, C, H, W, D = dense.shape
        dist_maps = [] if self.add_distance_gate else None

        for b in range(B):
            occ = dense[b].abs().sum(dim=0) > 1e-6  # [H, W, D]
            if occ.all():
                if dist_maps is not None:
                    dist_maps.append(torch.zeros(1, H, W, D, device=dense.device))
                continue
            if not occ.any():
                if dist_maps is not None:
                    dist_maps.append(torch.ones(1, H, W, D, device=dense.device))
                continue

            if nn_indices is not None and nn_indices.shape[-3:] == (H, W, D):
                idx = nn_indices[b].long().to(dense.device)  # [3, H, W, D]
                dense[b] = dense[b, :, idx[0], idx[1], idx[2]]
                if dist_maps is not None:
                    # Compute distance from indices
                    coords = torch.stack(torch.meshgrid(
                        torch.arange(H), torch.arange(W), torch.arange(D), indexing='ij'
                    ), dim=0).float().to(dense.device)  # [3, H, W, D]
                    dist = ((coords - idx.float()) ** 2).sum(dim=0).sqrt()  # [H, W, D]
                    dist_maps.append(dist.unsqueeze(0))
            else:
                occ_np = occ.cpu().numpy()
                dist, idx = distance_transform_edt(~occ_np, return_indices=True)
                idx = torch.from_numpy(idx).long().to(dense.device)
                dense[b] = dense[b, :, idx[0], idx[1], idx[2]]
                if dist_maps is not None:
                    dist_maps.append(torch.from_numpy(dist).float().unsqueeze(0).to(dense.device))

        if dist_maps is not None:
            return dense, torch.stack(dist_maps, dim=0)  # [B, 1, H, W, D]
        return dense, None

    def forward(self, lidar: torch.Tensor, nn_indices=None) -> dict[str, torch.Tensor]:
        """
        Extract multi-scale features from sparse LiDAR.

        Args:
            lidar: Dense binary voxels [B, 1, H, W, D] (will be converted to sparse)
            nn_indices: Optional [B, 3, H, W, D] precomputed NN indices for level 0

        Returns:
            Dictionary with dense features at each level:
            - 'level0': [B, out_channels, 256, 256, 32]
            - 'level1': [B, out_channels, 128, 128, 16]
            - 'level2': [B, out_channels, 64, 64, 8]
            - 'level3': [B, out_channels, 32, 32, 4]
            - 'level4': [B, out_channels, 16, 16, 2]
        """
        B, C, H, W, D = lidar.shape

        # Convert dense to sparse
        sparse_input = self._dense_to_sparse(lidar)

        # Stem
        x = self.stem(sparse_input)
        x = x.replace_feature(self.stem_bn(x.features))
        x = x.replace_feature(F.relu(x.features))

        # Level 0
        x0 = self.level0(x)
        feat0 = self._sparse_to_dense(x0, (B, H, W, D))
        feat0 = self.proj0(feat0)
        if self.densify_nn:
            feat0, dist0 = self._nn_densify(feat0, nn_indices)
            if self.add_distance_gate and dist0 is not None:
                feat0 = feat0 * self.dist_gate0(dist0)

        x = self.down0(x0)
        x = x.replace_feature(self.down0_bn(x.features))
        x = x.replace_feature(F.relu(x.features))

        # Level 1
        x1 = self.level1(x)
        feat1 = self._sparse_to_dense(x1, (B, H//2, W//2, D//2))
        feat1 = self.proj1(feat1)
        if self.densify_nn:
            feat1, dist1 = self._nn_densify(feat1)
            if self.add_distance_gate and dist1 is not None:
                feat1 = feat1 * self.dist_gate1(dist1)

        x = self.down1(x1)
        x = x.replace_feature(self.down1_bn(x.features))
        x = x.replace_feature(F.relu(x.features))

        # Level 2
        x2 = self.level2(x)
        feat2 = self._sparse_to_dense(x2, (B, H//4, W//4, D//4))
        feat2 = self.proj2(feat2)
        if self.densify_nn:
            feat2, dist2 = self._nn_densify(feat2)
            if self.add_distance_gate and dist2 is not None:
                feat2 = feat2 * self.dist_gate2(dist2)

        x = self.down2(x2)
        x = x.replace_feature(self.down2_bn(x.features))
        x = x.replace_feature(F.relu(x.features))

        # Level 3
        x3 = self.level3(x)
        feat3 = self._sparse_to_dense(x3, (B, H//8, W//8, D//8))
        feat3 = self.proj3(feat3)
        if self.densify_nn:
            feat3, dist3 = self._nn_densify(feat3)
            if self.add_distance_gate and dist3 is not None:
                feat3 = feat3 * self.dist_gate3(dist3)

        x = self.down3(x3)
        x = x.replace_feature(self.down3_bn(x.features))
        x = x.replace_feature(F.relu(x.features))

        # Level 4 (bottleneck)
        x4 = self.level4(x)
        feat4 = self._sparse_to_dense(x4, (B, H//16, W//16, D//16))
        feat4 = self.proj4(feat4)
        if self.densify_nn:
            feat4, dist4 = self._nn_densify(feat4)
            if self.add_distance_gate and dist4 is not None:
                feat4 = feat4 * self.dist_gate4(dist4)

        return {
            'level0': feat0,  # 256×256×32
            'level1': feat1,  # 128×128×16
            'level2': feat2,  # 64×64×8
            'level3': feat3,  # 32×32×4
            'level4': feat4,  # 16×16×2
        }

    def _dense_to_sparse(self, dense: torch.Tensor) -> spconv.SparseConvTensor:
        """Convert dense tensor to sparse tensor.

        Handles both single-channel (binary LiDAR) and multi-channel
        (e.g. 20-ch LSK3DNet soft probs) inputs.

        Args:
            dense: [B, C, H, W, D] where C=1 for binary or C=20 for LSK3DNet
        """
        B, C, H, W, D = dense.shape
        device = dense.device

        # Find non-zero voxels: any channel > 0
        # For C=1: same as before (binary occupancy)
        # For C>1: occupied if any channel is non-zero
        occupancy = dense.abs().sum(dim=1)  # [B, H, W, D]

        batch_indices = []
        spatial_indices = []
        features = []

        for b in range(B):
            nz = torch.nonzero(occupancy[b] > 1e-6, as_tuple=False)  # [N, 3] (h, w, d)

            if nz.shape[0] > 0:
                batch_idx = torch.full((nz.shape[0], 1), b, dtype=torch.int32, device=device)
                batch_indices.append(batch_idx)
                spatial_indices.append(nz.int())
                # Extract C-dim features at occupied locations
                feat = dense[b, :, nz[:, 0], nz[:, 1], nz[:, 2]]  # [C, N]
                features.append(feat.t())  # [N, C]
            else:
                # Safety: spconv crashes on empty tensors. Add one dummy point
                # with zero features at (0,0,0) — won't affect output.
                batch_indices.append(torch.full((1, 1), b, dtype=torch.int32, device=device))
                spatial_indices.append(torch.zeros((1, 3), dtype=torch.int32, device=device))
                features.append(torch.zeros((1, C), device=device))

        batch_indices = torch.cat(batch_indices, dim=0)
        spatial_indices = torch.cat(spatial_indices, dim=0)
        features = torch.cat(features, dim=0)
        indices = torch.cat([batch_indices, spatial_indices], dim=1)

        sparse = spconv.SparseConvTensor(
            features=features,
            indices=indices,
            spatial_shape=[H, W, D],
            batch_size=B,
        )

        return sparse

    def _sparse_to_dense(
        self,
        sparse: spconv.SparseConvTensor,
        shape: tuple[int, int, int, int],  # (B, H, W, D)
    ) -> torch.Tensor:
        """Convert sparse tensor to dense tensor."""
        B, H, W, D = shape
        C = sparse.features.shape[1]
        device = sparse.features.device

        # Initialize dense tensor
        dense = torch.zeros(B, C, H, W, D, device=device)

        # Fill in values
        indices = sparse.indices  # [N, 4]: (batch, h, w, d)
        features = sparse.features  # [N, C]

        if indices.shape[0] > 0:
            batch_idx = indices[:, 0].long()
            h_idx = indices[:, 1].long()
            w_idx = indices[:, 2].long()
            d_idx = indices[:, 3].long()

            # Scatter features to dense tensor
            dense[batch_idx, :, h_idx, w_idx, d_idx] = features

        return dense


class SparseLiDAREncoderLite(nn.Module):
    """
    Lighter version of SparseLiDAREncoder for faster experiments.

    Uses fewer channels: 8 → 16 → 32 → 64 → 64
    """

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 8,
        out_channels: int = 16,
    ):
        super().__init__()

        self.base_channels = base_channels
        self.out_channels = out_channels

        ch = [base_channels, base_channels*2, base_channels*4, base_channels*8, base_channels*8]

        # Stem
        self.stem = spconv.SubMConv3d(in_channels, ch[0], 3, padding=1, bias=False, indice_key='stem', algo=_ALGO)
        self.stem_bn = nn.BatchNorm1d(ch[0])

        # Levels (same structure, fewer channels)
        self.level0 = SparseResBlock3D(ch[0], ch[0], indice_key='l0')
        self.down0 = spconv.SparseConv3d(ch[0], ch[1], 3, stride=2, padding=1, bias=False, indice_key='d0', algo=_ALGO)
        self.down0_bn = nn.BatchNorm1d(ch[1])

        self.level1 = SparseResBlock3D(ch[1], ch[1], indice_key='l1')
        self.down1 = spconv.SparseConv3d(ch[1], ch[2], 3, stride=2, padding=1, bias=False, indice_key='d1', algo=_ALGO)
        self.down1_bn = nn.BatchNorm1d(ch[2])

        self.level2 = SparseResBlock3D(ch[2], ch[2], indice_key='l2')
        self.down2 = spconv.SparseConv3d(ch[2], ch[3], 3, stride=2, padding=1, bias=False, indice_key='d2', algo=_ALGO)
        self.down2_bn = nn.BatchNorm1d(ch[3])

        self.level3 = SparseResBlock3D(ch[3], ch[3], indice_key='l3')
        self.down3 = spconv.SparseConv3d(ch[3], ch[4], 3, stride=2, padding=1, bias=False, indice_key='d3', algo=_ALGO)
        self.down3_bn = nn.BatchNorm1d(ch[4])

        self.level4 = SparseResBlock3D(ch[4], ch[4], indice_key='l4')

        # Projections
        self.proj0 = nn.Conv3d(ch[0], out_channels, 1)
        self.proj1 = nn.Conv3d(ch[1], out_channels, 1)
        self.proj2 = nn.Conv3d(ch[2], out_channels, 1)
        self.proj3 = nn.Conv3d(ch[3], out_channels, 1)
        self.proj4 = nn.Conv3d(ch[4], out_channels, 1)

    def forward(self, lidar: torch.Tensor) -> dict[str, torch.Tensor]:
        """Same interface as SparseLiDAREncoder."""
        B, C, H, W, D = lidar.shape

        sparse_input = self._dense_to_sparse(lidar)

        x = self.stem(sparse_input)
        x = x.replace_feature(self.stem_bn(x.features))
        x = x.replace_feature(F.relu(x.features))

        x0 = self.level0(x)
        feat0 = self._sparse_to_dense(x0, (B, H, W, D))
        feat0 = self.proj0(feat0)

        x = self.down0(x0)
        x = x.replace_feature(self.down0_bn(x.features))
        x = x.replace_feature(F.relu(x.features))

        x1 = self.level1(x)
        feat1 = self._sparse_to_dense(x1, (B, H//2, W//2, D//2))
        feat1 = self.proj1(feat1)

        x = self.down1(x1)
        x = x.replace_feature(self.down1_bn(x.features))
        x = x.replace_feature(F.relu(x.features))

        x2 = self.level2(x)
        feat2 = self._sparse_to_dense(x2, (B, H//4, W//4, D//4))
        feat2 = self.proj2(feat2)

        x = self.down2(x2)
        x = x.replace_feature(self.down2_bn(x.features))
        x = x.replace_feature(F.relu(x.features))

        x3 = self.level3(x)
        feat3 = self._sparse_to_dense(x3, (B, H//8, W//8, D//8))
        feat3 = self.proj3(feat3)

        x = self.down3(x3)
        x = x.replace_feature(self.down3_bn(x.features))
        x = x.replace_feature(F.relu(x.features))

        x4 = self.level4(x)
        feat4 = self._sparse_to_dense(x4, (B, H//16, W//16, D//16))
        feat4 = self.proj4(feat4)

        return {
            'level0': feat0,
            'level1': feat1,
            'level2': feat2,
            'level3': feat3,
            'level4': feat4,
        }

    # Reuse parent methods
    _dense_to_sparse = SparseLiDAREncoder._dense_to_sparse
    _sparse_to_dense = SparseLiDAREncoder._sparse_to_dense


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    print("=== SparseLiDAREncoder Test ===")

    # Test parameter count
    encoder = SparseLiDAREncoder(base_channels=16, out_channels=32)
    print(f"SparseLiDAREncoder parameters: {count_parameters(encoder):,}")

    encoder_lite = SparseLiDAREncoderLite(base_channels=8, out_channels=16)
    print(f"SparseLiDAREncoderLite parameters: {count_parameters(encoder_lite):,}")
