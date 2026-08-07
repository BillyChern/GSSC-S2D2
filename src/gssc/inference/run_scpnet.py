#!/usr/bin/env python3
"""Run SCPNet inference on SemanticKITTI sequences and save 3D predictions + BEV.

Uses the ORIGINAL SCPNet network code with spconv v2 monkey-patches to reproduce
spconv v1 behavior:
  1. Allow direct .features assignment on SparseConvTensor
  2. Suppress kernel-size validation for shared indice_key
  3. Redirect `import spconv` to spconv.pytorch
  4. Force all layers sharing an indice_key to use the FIRST layer's kernel size
     (reproduces v1's shared pair data behavior)

Weight loading reshapes kernel dimensions to match, preserving flat element ordering
(weight[i] -> pair[i]) which is how v1 applies weights to shared pair data.

The upstream SCPNet network code is not vendored in this release. Clone it from
https://github.com/SCPNet/Codes-for-SCPNet and point this script at the checkout
either with the --scpnet-repo flag or the $SCPNET_REPO environment variable. The
default location is the repo-relative path 'external/Codes-for-SCPNet'.

Usage:
    python -m gssc.inference.run_scpnet --sequences 08 --eval --max_frames 10
    python -m gssc.inference.run_scpnet --sequences 08 --eval
    python -m gssc.inference.run_scpnet --sequences all
    SCPNET_REPO=/path/to/Codes-for-SCPNet python -m gssc.inference.run_scpnet --sequences 08
    python -m gssc.inference.run_scpnet --scpnet-repo /path/to/Codes-for-SCPNet --sequences 08
"""

import os
import sys

# =====================================================================
# Monkey-patch spconv v2.3 to reproduce spconv v1 shared indice_key behavior
# =====================================================================
import spconv.pytorch as _spconv_pytorch

# Patch 1: Allow direct feature assignment (x.features = val)
_SCT = _spconv_pytorch.SparseConvTensor
_orig_prop = _SCT.__dict__['features']
def _features_setter(self, val):
    self._features = val
_SCT.features = property(_orig_prop.fget, _features_setter)

# Patch 2: Suppress kernel-size check for shared indice_key
from spconv.pytorch.conv import SparseConvolutionBase


def _patched_check_subm(self, inp, spatial_shape, datas):
    assert datas.is_subm, "only support reuse subm indices"
    if inp.spatial_shape != datas.spatial_shape:
        raise ValueError("subm with same indice_key must have same spatial structure")
    if inp.indices.shape[0] != datas.indices.shape[0]:
        raise ValueError("subm with same indice_key must have same num of indices")
SparseConvolutionBase._check_subm_reuse_valid = _patched_check_subm

# Patch 3: Redirect `import spconv` to spconv.pytorch
sys.modules['spconv'] = _spconv_pytorch

# =====================================================================
# Imports (after spconv patches)
# =====================================================================
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Path to the upstream SCPNet checkout (https://github.com/SCPNet/Codes-for-SCPNet).
# Configurable via $SCPNET_REPO; defaults to the repo-relative 'external/Codes-for-SCPNet'.
# The --scpnet-repo CLI flag (see main()) overrides this at runtime.
SCPNET_ROOT = os.environ.get('SCPNET_REPO', 'external/Codes-for-SCPNet')
sys.path.insert(0, SCPNET_ROOT)

import argparse

import numpy as np
import torch
import yaml

# =====================================================================
# Patch 4: Fix ALL shared indice_key kernel mismatches
#
# In spconv v1, when multiple SubMConv3d layers share an indice_key,
# the FIRST layer creates spatial pair data and ALL subsequent layers
# REUSE it regardless of their declared kernel size. weight[i] maps
# to pair[i] by flat index.
#
# In spconv v2, different kernel shapes produce different dict entries
# even with the same indice_key string, so each layer gets its own
# pair data with its own spatial offsets — breaking trained behavior.
#
# Fix: Make ALL layers sharing a key use the SAME kernel as the FIRST
# layer that runs in the forward pass.
# =====================================================================
from network.segmentator_3d_asymm_spconv import (
    ReconBlock,
    ResBlock,
    ResContextBlock,
    UpBlock,
    conv1x3,
    conv3x1,
    conv3x1x1,
)

# ResContextBlock: conv1=conv1x3(1,3,3) runs first
_orig_resctx_init = ResContextBlock.__init__
def _patched_resctx_init(self, in_filters, out_filters, kernel_size=(3,3,3), stride=1, indice_key=None):
    _orig_resctx_init(self, in_filters, out_filters, kernel_size, stride, indice_key)
    self.conv1_2 = conv1x3(out_filters, out_filters, indice_key=indice_key + "bef")
    self.conv2 = conv1x3(in_filters, out_filters, indice_key=indice_key + "bef")
ResContextBlock.__init__ = _patched_resctx_init

# ResBlock: conv1=conv3x1(3,1,3) runs first
_orig_resblock_init = ResBlock.__init__
def _patched_resblock_init(self, in_filters, out_filters, dropout_rate, kernel_size=(3,3,3), stride=1,
                            pooling=True, drop_out=True, height_pooling=False, indice_key=None):
    _orig_resblock_init(self, in_filters, out_filters, dropout_rate, kernel_size, stride,
                         pooling, drop_out, height_pooling, indice_key)
    self.conv1_2 = conv3x1(out_filters, out_filters, indice_key=indice_key + "bef")
    self.conv2 = conv3x1(in_filters, out_filters, indice_key=indice_key + "bef")
ResBlock.__init__ = _patched_resblock_init

# UpBlock: conv1=conv1x3(1,3,3) runs first
_orig_upblock_init = UpBlock.__init__
def _patched_upblock_init(self, in_filters, out_filters, kernel_size=(3,3,3), indice_key=None, up_key=None):
    _orig_upblock_init(self, in_filters, out_filters, kernel_size, indice_key, up_key)
    self.conv2 = conv1x3(out_filters, out_filters, indice_key=indice_key)
    self.conv3 = conv1x3(out_filters, out_filters, indice_key=indice_key)
UpBlock.__init__ = _patched_upblock_init

# ReconBlock: conv1=conv3x1x1(3,1,1) runs first
_orig_reconblock_init = ReconBlock.__init__
def _patched_reconblock_init(self, in_filters, out_filters, kernel_size=(3,3,3), stride=1, indice_key=None):
    _orig_reconblock_init(self, in_filters, out_filters, kernel_size, stride, indice_key)
    self.conv1_2 = conv3x1x1(in_filters, out_filters, indice_key=indice_key + "bef")
    self.conv1_3 = conv3x1x1(in_filters, out_filters, indice_key=indice_key + "bef")
ReconBlock.__init__ = _patched_reconblock_init

# =====================================================================
# Rest of imports
# =====================================================================
from pathlib import Path

from network.cylinder_fea_generator import cylinder_fea
from network.cylinder_spconv_3d import get_model_class
from network.segmentator_3d_asymm_spconv import Asymm_3d_spconv
from tqdm import tqdm

# =====================================================================
# Weight loading with kernel reshape
# =====================================================================

# Layers where kernel shape was changed: suffix -> (v1_kernel, target_kernel)
KERNEL_RESHAPE = {}
# ResContextBlock: conv1_2, conv2 from (3,1,3) to (1,3,3)
KERNEL_RESHAPE['downCntx.conv1_2.weight'] = ((3,1,3), (1,3,3))
KERNEL_RESHAPE['downCntx.conv2.weight'] = ((3,1,3), (1,3,3))
# ResBlock: conv1_2, conv2 from (1,3,3) to (3,1,3)
for rb in ['resBlock2', 'resBlock3', 'resBlock4', 'resBlock5']:
    KERNEL_RESHAPE[f'{rb}.conv1_2.weight'] = ((1,3,3), (3,1,3))
    KERNEL_RESHAPE[f'{rb}.conv2.weight'] = ((1,3,3), (3,1,3))
# UpBlock: conv2 from (3,1,3) to (1,3,3), conv3 from (3,3,3) to (1,3,3)
for ub in ['upBlock0', 'upBlock1', 'upBlock2', 'upBlock3']:
    KERNEL_RESHAPE[f'{ub}.conv2.weight'] = ((3,1,3), (1,3,3))
    KERNEL_RESHAPE[f'{ub}.conv3.weight'] = ((3,3,3), (1,3,3))
# ReconBlock: conv1_2 from (1,3,1) to (3,1,1), conv1_3 from (1,1,3) to (3,1,1)
KERNEL_RESHAPE['ReconNet.conv1_2.weight'] = ((1,3,1), (3,1,1))
KERNEL_RESHAPE['ReconNet.conv1_3.weight'] = ((1,1,3), (3,1,1))


def reshape_kernel_weight(value, orig_kernel, target_kernel):
    """Reshape v1 weight to v2 format with kernel shape change.

    Preserves flat element ordering (weight[i] -> pair[i]).
    Truncates if target has fewer elements (e.g., 3x3x3 -> 1x3x3: 27 -> 9).

    v1: (kD, kH, kW, in_ch, out_ch) -> v2: (out_ch, kD_new, kH_new, kW_new, in_ch)
    """
    num_orig = orig_kernel[0] * orig_kernel[1] * orig_kernel[2]
    num_target = target_kernel[0] * target_kernel[1] * target_kernel[2]
    in_ch, out_ch = value.shape[-2], value.shape[-1]

    flat = value.reshape(num_orig, in_ch, out_ch)
    if num_target < num_orig:
        flat = flat[:num_target]

    reshaped = flat.reshape(target_kernel[0], target_kernel[1], target_kernel[2], in_ch, out_ch)
    return reshaped.permute(4, 0, 1, 2, 3).contiguous()


def build_scpnet_model(config_path):
    """Build SCPNet model from config."""
    with open(config_path) as f:
        configs = yaml.safe_load(f)

    model_config = configs['model_params']
    output_shape = model_config['output_shape']

    cylinder_3d_spconv_seg = Asymm_3d_spconv(
        output_shape=output_shape,
        use_norm=model_config['use_norm'],
        num_input_features=model_config['num_input_features'],
        init_size=model_config['init_size'],
        nclasses=model_config['num_class'],
    )

    cy_fea_net = cylinder_fea(
        grid_size=output_shape,
        fea_dim=model_config['fea_dim'],
        out_pt_fea_dim=model_config['out_fea_dim'],
        fea_compre=model_config['num_input_features'],
    )

    model = get_model_class(model_config["model_architecture"])(
        cylin_model=cy_fea_net,
        segmentator_spconv=cylinder_3d_spconv_seg,
        sparse_shape=output_shape,
    )

    return model, configs


def load_scpnet_checkpoint(model, checkpoint_path):
    """Load SCPNet checkpoint with kernel reshaping for v1->v2 compatibility."""
    pre_weight = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    my_model_dict = model.state_dict()

    part_load = {}
    stats = {'direct': 0, 'reshaped': 0, 'transposed': 0, 'skipped': 0}

    for k in pre_weight:
        if k not in my_model_dict:
            stats['skipped'] += 1
            continue

        value = pre_weight[k]
        target_shape = my_model_dict[k].shape

        # Check if this weight needs kernel reshaping
        reshaped = False
        for suffix, (orig_k, target_k) in KERNEL_RESHAPE.items():
            if k.endswith(suffix):
                value = reshape_kernel_weight(value, orig_k, target_k)
                reshaped = True
                stats['reshaped'] += 1
                break

        if reshaped:
            if value.shape == target_shape:
                part_load[k] = value
            else:
                stats['skipped'] += 1
                print(f"  Reshape mismatch: {k}: got {value.shape}, expected {target_shape}")
            continue

        # Fix any NaN values
        if value.is_floating_point() and torch.isnan(value).any():
            n = torch.isnan(value).sum().item()
            print(f"  Fixing {n} NaN values in {k}")
            value = torch.nan_to_num(value, nan=0.0)

        if value.shape == target_shape:
            stats['direct'] += 1
            part_load[k] = value
        elif len(value.shape) == 5 and len(target_shape) == 5:
            # Normal v1->v2 transpose: (kD,kH,kW,in,out) -> (out,kD,kH,kW,in)
            transposed = value.permute(4, 0, 1, 2, 3).contiguous()
            if transposed.shape == target_shape:
                stats['transposed'] += 1
                part_load[k] = transposed
            else:
                stats['skipped'] += 1
                print(f"  Shape mismatch: {k}: {value.shape} -> {transposed.shape} vs {target_shape}")
        else:
            stats['skipped'] += 1
            if value.shape != target_shape:
                print(f"  Shape mismatch: {k}: {value.shape} vs {target_shape}")

    print(f"  Loaded: {stats['direct']} direct, {stats['reshaped']} reshaped, "
          f"{stats['transposed']} transposed, {stats['skipped']} skipped")
    print(f"  Total: {len(part_load)}/{len(my_model_dict)} parameters loaded")

    my_model_dict.update(part_load)
    model.load_state_dict(my_model_dict)
    return model


def prepare_input(raw_data, grid_size, max_volume_space, min_volume_space):
    """Prepare point cloud for SCPNet inference (matches voxel_dataset logic)."""
    xyz = raw_data[:, :3]
    sig = raw_data[:, 3]

    max_bound = np.array(max_volume_space, dtype=np.float64)
    min_bound = np.array(min_volume_space, dtype=np.float64)

    # Cut points outside valid range (BEFORE x-axis translation)
    xyz0 = xyz.copy()
    for ci in range(3):
        xyz0[xyz[:, ci] < min_bound[ci], :] = 1000
        xyz0[xyz[:, ci] > max_bound[ci], :] = 1000
    valid = xyz0[:, 0] != 1000
    xyz = xyz[valid]
    sig = sig[valid]

    # Translate x-axis to center
    x_bias = (max_volume_space[0] - min_volume_space[0]) / 2
    min_bound[0] -= x_bias
    max_bound[0] -= x_bias
    xyz[:, 0] -= x_bias

    # Grid indices
    crop_range = max_bound - min_bound
    cur_grid_size = np.array(grid_size)
    intervals = crop_range / (cur_grid_size - 1)
    grid_ind = np.floor((np.clip(xyz, min_bound, max_bound) - min_bound) / intervals).astype(np.int64)

    # 7D features: (xyz_bias, xyz, intensity)
    voxel_centers = (grid_ind.astype(np.float32) + 0.5) * intervals + min_bound
    return_xyz = xyz - voxel_centers
    return_xyz = np.concatenate((return_xyz, xyz), axis=1)
    return_fea = np.concatenate((return_xyz, sig[..., np.newaxis]), axis=1)

    return return_fea, grid_ind


def compute_bev_majority_vote_fast(pred_3d, num_classes=20):
    """Fast vectorized majority vote BEV computation."""
    H, W, D = pred_3d.shape
    bev = np.zeros((H, W), dtype=np.uint8)
    counts = np.zeros((H, W, num_classes), dtype=np.int32)
    for c in range(1, num_classes):
        counts[:, :, c] = (pred_3d == c).sum(axis=2)
    has_occupied = counts[:, :, 1:].sum(axis=2) > 0
    bev[has_occupied] = counts[has_occupied, :].argmax(axis=1).astype(np.uint8)
    return bev


def compute_bev_top_view(pred_3d):
    """Compute BEV via top-view (topmost non-empty class)."""
    H, W, D = pred_3d.shape
    bev = np.zeros((H, W), dtype=np.uint8)
    for z in range(D - 1, -1, -1):
        mask = (pred_3d[:, :, z] > 0) & (bev == 0)
        bev[mask] = pred_3d[:, :, z][mask]
    return bev


class SSCMetrics:
    """Semantic Scene Completion metrics."""
    def __init__(self, num_classes=20):
        self.num_classes = num_classes
        self.confusion = np.zeros((num_classes, num_classes), dtype=np.int64)

    def update(self, pred, gt, invalid_mask=None):
        pred, gt = pred.ravel(), gt.ravel()
        mask = (gt >= 0) & (gt < self.num_classes) & (pred >= 0) & (pred < self.num_classes)
        if invalid_mask is not None:
            mask &= ~invalid_mask.ravel()
        np.add.at(self.confusion, (gt[mask], pred[mask]), 1)

    def get_miou(self):
        ious = []
        for c in range(1, self.num_classes):
            tp = self.confusion[c, c]
            denom = self.confusion[c, :].sum() + self.confusion[:, c].sum() - tp
            if denom > 0:
                ious.append(tp / denom)
        return np.mean(ious) * 100 if ious else 0.0

    def get_completion_iou(self):
        tp = self.confusion[1:, 1:].sum()
        fn = self.confusion[1:, 0].sum()
        fp = self.confusion[0, 1:].sum()
        denom = tp + fn + fp
        return (tp / denom * 100) if denom > 0 else 0.0

    def get_per_class_iou(self):
        ious = {}
        for c in range(1, self.num_classes):
            tp = self.confusion[c, c]
            denom = self.confusion[c, :].sum() + self.confusion[:, c].sum() - tp
            ious[c] = (tp / denom * 100) if denom > 0 else 0.0
        return ious


def unpack(compressed):
    """Unpack bit-encoded voxel grid."""
    uncompressed = np.zeros(compressed.shape[0] * 8, dtype=np.uint8)
    uncompressed[::8] = compressed[:] >> 7 & 1
    uncompressed[1::8] = compressed[:] >> 6 & 1
    uncompressed[2::8] = compressed[:] >> 5 & 1
    uncompressed[3::8] = compressed[:] >> 4 & 1
    uncompressed[4::8] = compressed[:] >> 3 & 1
    uncompressed[5::8] = compressed[:] >> 2 & 1
    uncompressed[6::8] = compressed[:] >> 1 & 1
    uncompressed[7::8] = compressed[:] & 1
    return uncompressed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scpnet-repo', type=str, default=None,
                        help='Path to the upstream SCPNet checkout '
                             '(https://github.com/SCPNet/Codes-for-SCPNet). '
                             'Overrides $SCPNET_REPO; default: external/Codes-for-SCPNet')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='SCPNet checkpoint (.pth); '
                             'default: <scpnet-repo>/model_load_dir/pretrained.pth')
    parser.add_argument('--config', type=str, default=None,
                        help='SCPNet config yaml; '
                             'default: <scpnet-repo>/config/semantickitti-multiscan.yaml')
    parser.add_argument('--data_root', type=str,
                        default='datasets/dataset_SemanticKITTI_SSC')
    parser.add_argument('--output_dir', type=str,
                        default='datasets/scpnet_predictions')
    parser.add_argument('--sequences', type=str, default='08',
                        help='Comma-separated sequence numbers or "all"')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--eval', action='store_true', help='Compute mIoU against GT')
    parser.add_argument('--max_frames', type=int, default=0,
                        help='Max frames per sequence (0=all)')
    parser.add_argument('--force', action='store_true',
                        help='Regenerate even if predictions exist')
    args = parser.parse_args()

    # Resolve the SCPNet checkout root: CLI flag > $SCPNET_REPO (already captured in
    # the module-level default) > repo-relative 'external/Codes-for-SCPNet'.
    global SCPNET_ROOT
    if args.scpnet_repo is not None:
        SCPNET_ROOT = args.scpnet_repo
        sys.path.insert(0, SCPNET_ROOT)
    if args.checkpoint is None:
        args.checkpoint = os.path.join(SCPNET_ROOT, 'model_load_dir', 'pretrained.pth')
    if args.config is None:
        args.config = os.path.join(SCPNET_ROOT, 'config', 'semantickitti-multiscan.yaml')

    device = torch.device(f'cuda:{args.gpu}')

    if args.sequences == 'all':
        sequences = [f'{i:02d}' for i in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]
    else:
        sequences = [s.strip().zfill(2) for s in args.sequences.split(',')]

    print("SCPNet Inference")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Sequences: {sequences}")
    print(f"  Output: {args.output_dir}")

    # Build and load model
    model, configs = build_scpnet_model(args.config)
    model = load_scpnet_checkpoint(model, args.checkpoint)
    model = model.to(device)
    model.eval()

    dataset_config = configs['dataset_params']
    max_volume_space = dataset_config['max_volume_space']
    min_volume_space = dataset_config['min_volume_space']
    grid_size = configs['model_params']['output_shape']

    # Load label remapping
    label_mapping_path = os.path.join(SCPNET_ROOT, dataset_config['label_mapping'])
    with open(label_mapping_path) as f:
        semkittiyaml = yaml.safe_load(f)

    remapdict = semkittiyaml['learning_map']
    maxkey = max(remapdict.keys())
    remap_lut = np.zeros((maxkey + 100), dtype=np.int32)
    remap_lut[list(remapdict.keys())] = list(remapdict.values())
    remap_lut[remap_lut == 0] = 255
    remap_lut[0] = 0

    metrics = SSCMetrics(20) if args.eval else None
    os.makedirs(args.output_dir, exist_ok=True)

    total_frames = 0
    for seq in sequences:
        seq_dir = os.path.join(args.data_root, 'sequences', seq)
        velodyne_dir = os.path.join(seq_dir, 'velodyne')
        voxels_dir = os.path.join(seq_dir, 'voxels')

        if not os.path.exists(velodyne_dir):
            print(f"  Skipping {seq}: no velodyne dir")
            continue

        voxel_bins = sorted(Path(voxels_dir).glob('*.bin'))
        frame_ids = [f.stem for f in voxel_bins]
        if args.max_frames > 0:
            frame_ids = frame_ids[:args.max_frames]
        print(f"\n  Sequence {seq}: {len(frame_ids)} frames")

        out_seq_dir = os.path.join(args.output_dir, seq)
        os.makedirs(out_seq_dir, exist_ok=True)

        with torch.no_grad():
            for frame_id in tqdm(frame_ids, desc=f"Seq {seq}"):
                pred_path = os.path.join(out_seq_dir, f'{frame_id}_pred.npy')

                if os.path.exists(pred_path) and not args.force:
                    if args.eval:
                        pred = np.load(pred_path)
                    total_frames += 1
                    if not args.eval:
                        continue
                else:
                    pc_path = os.path.join(velodyne_dir, f'{frame_id}.bin')
                    raw_data = np.fromfile(pc_path, dtype=np.float32).reshape(-1, 4)

                    pt_fea, grid_ind = prepare_input(raw_data, grid_size, max_volume_space, min_volume_space)

                    pt_fea_ten = torch.from_numpy(pt_fea).float().to(device)
                    grid_ind_ten = torch.from_numpy(grid_ind).to(device)

                    logits = model([pt_fea_ten], [grid_ind_ten], 1, use_tta=False)
                    pred = logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

                    np.save(pred_path, pred)

                    bev_vote = compute_bev_majority_vote_fast(pred)
                    bev_top = compute_bev_top_view(pred)
                    np.save(os.path.join(out_seq_dir, f'{frame_id}_bev_vote.npy'), bev_vote)
                    np.save(os.path.join(out_seq_dir, f'{frame_id}_bev_top.npy'), bev_top)

                    total_frames += 1

                if args.eval:
                    gt_label_path = os.path.join(voxels_dir, f'{frame_id}.label')
                    gt_invalid_path = os.path.join(voxels_dir, f'{frame_id}.invalid')

                    gt_labels = np.fromfile(gt_label_path, dtype=np.uint16).reshape(256, 256, 32)
                    gt_labels = remap_lut[gt_labels].astype(np.uint8)

                    invalid = unpack(np.fromfile(gt_invalid_path, dtype=np.uint8))
                    invalid = invalid[:256*256*32].reshape(256, 256, 32)
                    eval_mask = (gt_labels != 255) & (invalid == 0)

                    metrics.update(pred[eval_mask], gt_labels[eval_mask])

        if args.eval:
            print(f"  After seq {seq}: mIoU={metrics.get_miou():.2f}%, cIoU={metrics.get_completion_iou():.2f}%")

    print(f"\nDone! Processed {total_frames} frames")
    print(f"Predictions saved to: {args.output_dir}")

    if args.eval and metrics is not None:
        print(f"\n{'='*60}")
        print("SCPNet Evaluation Results")
        print(f"  mIoU: {metrics.get_miou():.2f}%")
        print(f"  Completion IoU: {metrics.get_completion_iou():.2f}%")
        print("\nPer-class IoU:")
        class_names = {
            1: 'car', 2: 'bicycle', 3: 'motorcycle', 4: 'truck',
            5: 'other-vehicle', 6: 'person', 7: 'bicyclist', 8: 'motorcyclist',
            9: 'road', 10: 'parking', 11: 'sidewalk', 12: 'other-ground',
            13: 'building', 14: 'fence', 15: 'vegetation', 16: 'trunk',
            17: 'terrain', 18: 'pole', 19: 'traffic-sign',
        }
        per_class = metrics.get_per_class_iou()
        for c, iou in per_class.items():
            print(f"  {class_names.get(c, f'class_{c}'):15s}: {iou:.2f}%")
        print(f"{'='*60}")


if __name__ == '__main__':
    main()
