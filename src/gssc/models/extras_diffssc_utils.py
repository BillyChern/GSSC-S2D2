"""
DiffSSC utility functions: sparse tensor construction, KNN matching,
semantic encoding/decoding, and time embedding expansion.

Uses spconv v2 instead of MinkowskiEngine.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import spconv.pytorch as spconv


def points_to_sparse_tensor(point_feats_list, resolution=0.05, spatial_shape=None, coord_min=None):
    """Convert list of [N_i, F] point features to a SparseConvTensor.

    For each batch element, quantizes the first 3 channels (xyz) at `resolution`
    to get integer voxel coordinates. Duplicate voxels are averaged (UNWEIGHTED_AVERAGE
    equivalent). Returns the sparse tensor plus metadata for reconstruction.

    Args:
        point_feats_list: list of [N_i, F] tensors (F >= 3, first 3 = xyz)
        resolution: quantization resolution in meters
        spatial_shape: optional [X, Y, Z] spatial shape; if None, inferred from data
        coord_min: optional [3] int tensor to use as coordinate offset instead of
                   computing from data. Use this to ensure multiple sparse tensors
                   share the same coordinate frame for 1-NN matching.

    Returns:
        sparse_tensor: spconv.SparseConvTensor
        offsets: [3] int tensor, the coordinate offset applied (for reconstruction)
        inverse_maps: list of [N_i] long tensors mapping original points to voxel indices
    """
    device = point_feats_list[0].device
    num_features = point_feats_list[0].shape[1]

    all_coords = []  # [N_total, 4] int32: (batch, x, y, z)
    all_feats = []   # [N_total, F] float32
    batch_sizes = []

    for b, feats in enumerate(point_feats_list):
        # Quantize xyz at resolution
        xyz = feats[:, :3]
        coords_int = torch.round(xyz / resolution).int()
        batch_col = torch.full((len(coords_int), 1), b, dtype=torch.int32, device=device)
        all_coords.append(torch.cat([batch_col, coords_int], dim=1))
        all_feats.append(feats)
        batch_sizes.append(len(feats))

    all_coords = torch.cat(all_coords, dim=0)  # [N_total, 4]
    all_feats = torch.cat(all_feats, dim=0)     # [N_total, F]

    # Shift to non-negative (spconv requirement)
    if coord_min is None:
        coord_min = all_coords[:, 1:].min(dim=0).values  # [3]
    all_coords[:, 1:] -= coord_min

    # De-duplicate: average features for same voxel
    # Hash coords to find unique voxels
    coord_max = all_coords[:, 1:].max(dim=0).values + 1  # [3]
    hash_keys = (all_coords[:, 0].long() * coord_max[0].long() * coord_max[1].long() * coord_max[2].long() +
                 all_coords[:, 1].long() * coord_max[1].long() * coord_max[2].long() +
                 all_coords[:, 2].long() * coord_max[2].long() +
                 all_coords[:, 3].long())

    unique_hashes, inverse_idx, counts = torch.unique(hash_keys, return_inverse=True, return_counts=True)

    # Average features per unique voxel
    n_unique = len(unique_hashes)
    feat_sum = torch.zeros(n_unique, num_features, dtype=all_feats.dtype, device=device)
    feat_sum.scatter_add_(0, inverse_idx.unsqueeze(1).expand(-1, num_features), all_feats)
    avg_feats = feat_sum / counts.unsqueeze(1).float()

    # Get unique coords
    unique_coords = torch.zeros(n_unique, 4, dtype=torch.int32, device=device)
    # Use the first occurrence for coords (they're identical for same hash)
    first_occurrence = torch.zeros(n_unique, dtype=torch.long, device=device)
    first_occurrence.scatter_(0, inverse_idx, torch.arange(len(inverse_idx), device=device))
    unique_coords = all_coords[first_occurrence]

    # Compute spatial shape
    if spatial_shape is None:
        spatial_shape = (unique_coords[:, 1:].max(dim=0).values + 1).tolist()

    batch_size = len(point_feats_list)

    sparse_tensor = spconv.SparseConvTensor(
        features=avg_feats,
        indices=unique_coords,
        spatial_shape=spatial_shape,
        batch_size=batch_size,
    )

    # Build per-batch inverse maps
    inverse_maps = []
    offset = 0
    for b_size in batch_sizes:
        inverse_maps.append(inverse_idx[offset:offset + b_size])
        offset += b_size

    return sparse_tensor, coord_min, inverse_maps


def match_part_to_full(full_coords, full_batch_idx, part_sparse, chunk_size=4096):
    """1-NN matching: for each full-cloud voxel, find nearest partial-cloud voxel.

    Replaces PyKeOps LazyTensor with chunked torch.cdist.

    Args:
        full_coords: [N_full, 4] int32 sparse tensor indices (batch, x, y, z)
        full_batch_idx: unused (batch info is in full_coords[:, 0])
        part_sparse: spconv.SparseConvTensor for partial cloud
        chunk_size: number of full voxels to process at once

    Returns:
        matched_feats: [N_full, F_part] features from nearest partial voxels
    """
    full_c = full_coords.float()
    part_c = part_sparse.indices.float()
    part_f = part_sparse.features

    # Scale batch coordinate to separate batches in distance computation
    max_coord = max(full_c[:, 1:].abs().max().item(), part_c[:, 1:].abs().max().item(), 1.0)
    full_c_scaled = full_c.clone()
    part_c_scaled = part_c.clone()
    full_c_scaled[:, 0] *= max_coord * 2.0
    part_c_scaled[:, 0] *= max_coord * 2.0

    N_full = full_c_scaled.shape[0]
    matched_feats = torch.zeros(N_full, part_f.shape[1], device=part_f.device, dtype=part_f.dtype)

    for i in range(0, N_full, chunk_size):
        end = min(i + chunk_size, N_full)
        chunk = full_c_scaled[i:end]  # [chunk, 4]
        dists = torch.cdist(chunk.unsqueeze(0), part_c_scaled.unsqueeze(0)).squeeze(0)  # [chunk, N_part]
        nn_idx = dists.argmin(dim=1)  # [chunk]
        matched_feats[i:end] = part_f[nn_idx]

    return matched_feats


def expand_time_to_voxels(time_emb, sparse_tensor):
    """Expand [B, D] time embedding to [N_voxels, D] using batch indices.

    Args:
        time_emb: [B, D] time embedding
        sparse_tensor: spconv.SparseConvTensor

    Returns:
        expanded: [N_voxels, D]
    """
    batch_indices = sparse_tensor.indices[:, 0].long()
    return time_emb[batch_indices]


def get_timestep_embedding(timesteps, embed_dim):
    """Sinusoidal timestep embedding.

    Args:
        timesteps: [B] int tensor
        embed_dim: embedding dimension

    Returns:
        emb: [B, embed_dim]
    """
    assert len(timesteps.shape) == 1
    half_dim = embed_dim // 2
    emb_coeff = np.log(10000) / (half_dim - 1)
    emb_coeff = torch.from_numpy(np.exp(np.arange(0, half_dim) * -emb_coeff)).float().to(timesteps.device)
    emb = timesteps.float()[:, None] * emb_coeff[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embed_dim % 2 == 1:
        emb = F.pad(emb, (0, 1), "constant", 0)
    return emb


def semantic_encode(labels, num_classes=20, scale=5.0):
    """Convert integer labels to scaled logit representation for diffusion.

    The correct class gets +scale, others get -scale/(C-1), resulting in
    a zero-mean representation per point that's suitable for Gaussian noise.

    Args:
        labels: [N] int64 tensor of class indices (0 to num_classes-1)
        num_classes: number of classes

    Returns:
        logits: [N, num_classes] float tensor
    """
    one_hot = F.one_hot(labels.long(), num_classes).float()  # [N, C]
    logits = one_hot * scale - (1 - one_hot) * scale / (num_classes - 1)
    return logits


def probs_to_semantic_logits(probs, num_classes=20, scale=5.0):
    """Convert soft probabilities to semantic_encode logit scale.

    semantic_encode maps: correct class → +scale, others → -scale/(C-1).
    This function linearly maps probabilities to the same scale:
        logit[i] = scale/(C-1) * (prob[i] * C - 1)

    Args:
        probs: [..., C] float tensor of probabilities in [0, 1]
        num_classes: number of classes C
        scale: scale parameter (must match semantic_encode's scale)

    Returns:
        logits: [..., C] float tensor in [-scale/(C-1), +scale] range
    """
    return scale / (num_classes - 1) * (probs * num_classes - 1)


def semantic_decode(logits, temperature=1.0):
    """Convert predicted logits to semantic class probabilities via softmax.

    Args:
        logits: [N, C] or [B, N, C] float tensor
        temperature: softmax temperature

    Returns:
        probs: same shape as logits, softmax probabilities
    """
    return F.softmax(logits / temperature, dim=-1)


def point_cloud_to_voxel_grid(completed_points, num_classes=20,
                               point_cloud_range=(0, -25.6, -2, 51.2, 25.6, 4.4),
                               voxel_size=0.2,
                               grid_shape=(256, 256, 32)):
    """Convert completed point cloud to voxel grid for SSC evaluation.

    Args:
        completed_points: [N, 23] tensor (xyz + 20 semantic logits)
        num_classes: number of semantic classes
        point_cloud_range: (x_min, y_min, z_min, x_max, y_max, z_max)
        voxel_size: voxel size in meters
        grid_shape: (X, Y, Z) grid dimensions

    Returns:
        voxel_grid: [X, Y, Z] uint8 semantic labels (0=empty)
    """
    xyz = completed_points[:, :3]
    sem_logits = completed_points[:, 3:]

    # Get class predictions (1-indexed: class 0 is "unlabeled/empty")
    classes = sem_logits.argmax(dim=-1)  # [N], 0-19 range

    # Convert xyz to voxel coordinates
    x_min, y_min, z_min, x_max, y_max, z_max = point_cloud_range
    vx = ((xyz[:, 0] - x_min) / voxel_size).long()
    vy = ((xyz[:, 1] - y_min) / voxel_size).long()
    vz = ((xyz[:, 2] - z_min) / voxel_size).long()

    # Filter valid coordinates
    valid = ((vx >= 0) & (vx < grid_shape[0]) &
             (vy >= 0) & (vy < grid_shape[1]) &
             (vz >= 0) & (vz < grid_shape[2]))
    vx, vy, vz = vx[valid], vy[valid], vz[valid]
    classes = classes[valid]

    # Majority vote per voxel
    grid = torch.zeros(grid_shape, dtype=torch.uint8, device=completed_points.device)

    if len(vx) == 0:
        return grid

    # Hash for unique voxels
    hash_keys = vx * grid_shape[1] * grid_shape[2] + vy * grid_shape[2] + vz
    unique_hashes, inverse_idx = torch.unique(hash_keys, return_inverse=True)
    n_unique = len(unique_hashes)

    # Vote: accumulate class counts per voxel
    votes = torch.zeros(n_unique, num_classes, dtype=torch.int32, device=completed_points.device)
    votes.scatter_add_(0, inverse_idx.unsqueeze(1).expand(-1, num_classes),
                       F.one_hot(classes.long(), num_classes).int())

    # Majority vote (classes 0-19, but we want 1-19 for occupied voxels)
    # If max is class 0 (unlabeled), still assign it
    winner = votes.argmax(dim=1).to(torch.uint8)

    # Decode unique hashes back to coordinates
    uz = unique_hashes % grid_shape[2]
    uy = (unique_hashes // grid_shape[2]) % grid_shape[1]
    ux = unique_hashes // (grid_shape[1] * grid_shape[2])

    grid[ux.long(), uy.long(), uz.long()] = winner

    return grid
