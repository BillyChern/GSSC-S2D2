#!/usr/bin/env python3
"""Full D4 (dihedral-4) TTA for S²D² headline checkpoint.

Training used random {flip_x, flip_y, rot90 k=0..3} applied to lidar + BEV +
scpnet_pred (see scene_completion/s3_dskd_dataset.py::_augment). The full
8-element D4 group is therefore in-distribution for S²D². This script runs
all 8 transforms, averages the simplex predictions after inverse-transform,
and writes argmaxed .label files.

Transform inversion conventions (voxel grid axes):
  LiDAR [1,1,H=256,W=256,D=32]:  flip_x -> dim=2, flip_y -> dim=3, rot90(k) on dims (2,3)
  scp   [1,H=256,W=256,D=32]:    flip_x -> dim=1, flip_y -> dim=2, rot90(k) on dims (1,2)
  bev   [1,H=256,W=256]:         flip_x -> dim=1, flip_y -> dim=2, rot90(k) on dims (1,2)
  soft  [1,K=20,H=256,W=256,D=32]: invert — flip_x dim=2, flip_y dim=3, rot90(-k) on (2,3)

Usage:
  python tools/generate_tta_predictions_d4.py \
      --checkpoint outputs/scene_completion/Exp1_scpnet_synth_BEV/step_40000.pt \
      --cold_steps 4 \
      --output_dir outputs/official_predictions_exp1_31k_mf_40k_4step_tta_d4
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gssc.diffusion.multinomial import MultinomialDiffusion3DV2
from gssc.models.s2d2_unet import SceneCompletionUNetSparse

logger = logging.getLogger(__name__)

LEARNING_MAP_INV = {
    0: 0, 1: 10, 2: 11, 3: 15, 4: 18, 5: 20, 6: 30, 7: 31, 8: 32,
    9: 40, 10: 44, 11: 48, 12: 49, 13: 50, 14: 51, 15: 70, 16: 71,
    17: 72, 18: 80, 19: 81,
}
LEARNING_MAP_INV_ARRAY = np.zeros(20, dtype=np.int64)
for k, v in LEARNING_MAP_INV.items():
    LEARNING_MAP_INV_ARRAY[k] = v


def unpack_voxels(compressed: np.ndarray) -> np.ndarray:
    u = np.zeros(compressed.shape[0] * 8, dtype=np.uint8)
    u[::8]  = compressed >> 7 & 1
    u[1::8] = compressed >> 6 & 1
    u[2::8] = compressed >> 5 & 1
    u[3::8] = compressed >> 4 & 1
    u[4::8] = compressed >> 3 & 1
    u[5::8] = compressed >> 2 & 1
    u[6::8] = compressed >> 1 & 1
    u[7::8] = compressed & 1
    return u


def load_lidar_voxels(voxel_path: str | os.PathLike) -> torch.Tensor:
    """Decode a packed-binary `.bin` voxel grid into a [1, 1, 256, 256, 32] float tensor."""
    compressed = np.fromfile(voxel_path, dtype=np.uint8)
    binary = unpack_voxels(compressed).reshape(256, 256, 32).astype(np.float32)
    return torch.from_numpy(binary).unsqueeze(0).unsqueeze(0)


def derive_bev(scp_tensor: torch.Tensor) -> torch.Tensor:
    """Topmost-non-empty-class BEV projection of a [1, 256, 256, 32] SCPNet pred."""
    bev = torch.zeros(1, 256, 256, dtype=torch.long, device=scp_tensor.device)
    for z in range(scp_tensor.shape[3] - 1, -1, -1):
        layer = scp_tensor[0, :, :, z]
        mask = (layer > 0) & (bev[0] == 0)
        bev[0][mask] = layer[mask]
    return bev


def load_model(
    ckpt_path: str | os.PathLike,
    device: torch.device,
) -> tuple[SceneCompletionUNetSparse, MultinomialDiffusion3DV2]:
    """Load the headline checkpoint with EMA weights swapped in for inference."""
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    model = SceneCompletionUNetSparse(
        num_classes=20, base_channels=32, time_emb_dim=128,
        lidar_base_channels=16, lidar_out_channels=32, lidar_in_channels=1,
        no_bev=False, ssc_cond_channels=20, ssc_multiscale=False,
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'], strict=False)
    if 'ema_shadow' in ckpt:
        for name, p in model.named_parameters():
            if name in ckpt['ema_shadow']:
                p.data.copy_(ckpt['ema_shadow'][name])
    model.train(False)
    diffusion = MultinomialDiffusion3DV2(num_classes=20, num_timesteps=100, beta_max=0.1).to(device)
    return model, diffusion


def apply_d4(
    lidar: torch.Tensor,
    scp: torch.Tensor,
    bev: torch.Tensor,
    flip_x: bool,
    flip_y: bool,
    rot_k: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply D4 element to inputs. Returns transformed (lidar, scp, bev).

    Order matters: flip first, then rotate (mirrors training augment order).
    """
    if flip_x:
        lidar = torch.flip(lidar, dims=[2])
        scp   = torch.flip(scp,   dims=[1])
        bev   = torch.flip(bev,   dims=[1])
    if flip_y:
        lidar = torch.flip(lidar, dims=[3])
        scp   = torch.flip(scp,   dims=[2])
        bev   = torch.flip(bev,   dims=[2])
    if rot_k > 0:
        lidar = torch.rot90(lidar, k=rot_k, dims=[2, 3])
        scp   = torch.rot90(scp,   k=rot_k, dims=[1, 2])
        bev   = torch.rot90(bev,   k=rot_k, dims=[1, 2])
    return lidar, scp, bev


def invert_d4(soft: torch.Tensor, flip_x: bool, flip_y: bool, rot_k: int) -> torch.Tensor:
    """Invert D4 on [1, K, H, W, D] softmax. Inverse order: un-rotate first, then un-flip."""
    if rot_k > 0:
        soft = torch.rot90(soft, k=-rot_k, dims=[2, 3])
    if flip_y:
        soft = torch.flip(soft, dims=[3])
    if flip_x:
        soft = torch.flip(soft, dims=[2])
    return soft


def run_algo2_softmax(
    model: SceneCompletionUNetSparse,
    diffusion: MultinomialDiffusion3DV2,
    lidar: torch.Tensor,
    scp_tensor: torch.Tensor,
    bev: torch.Tensor,
    n_steps: int,
    device: torch.device,
) -> torch.Tensor:
    """Run a single Algo2 forward pass and return the soft-max distribution [1,K,H,W,D]."""
    scp_oh = F.one_hot(scp_tensor.long(), 20).float().permute(0, 4, 1, 2, 3)
    with torch.no_grad():
        soft = diffusion.sample_algo2(
            model, bev, lidar,
            scpnet_pred=scp_tensor,
            shape=(1, 256, 256, 32),
            device=device,
            n_steps=n_steps,
            show_progress=False,
            ssc_pred=scp_oh,
            return_softmax=True,
        )
    return soft


# D4 elements: (flip_x, flip_y, rot_k)
D4_ELEMENTS = [
    (False, False, 0),  # identity
    (True,  False, 0),
    (False, True,  0),
    (True,  True,  0),
    (False, False, 1),  # rot90
    (True,  False, 1),
    (False, True,  1),
    (True,  True,  1),
]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--cold_steps', type=int, default=4)
    p.add_argument('--output_dir', required=True)
    p.add_argument('--data_root', default='data/SemanticKITTI',
                   help='Root containing sequences/<SEQ>/voxels/*.bin')
    p.add_argument('--scpnet_dir', default='data/scpnet_predictions',
                   help='Root containing <SEQ>/<frame>_pred.npy')
    p.add_argument('--split', default='valid', choices=['train', 'valid', 'test'])
    p.add_argument('--skip_existing', action='store_true')
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, diffusion = load_model(args.checkpoint, device)

    if args.split == 'valid':
        seqs = ['08']
    elif args.split == 'train':
        seqs = ['00', '01', '02', '03', '04', '05', '06', '07', '09', '10']
    else:
        seqs = [f'{s:02d}' for s in range(11, 22)]

    total = 0
    for seq in seqs:
        voxels_dir = os.path.join(args.data_root, 'sequences', seq, 'voxels')
        scpnet_seq_dir = os.path.join(args.scpnet_dir, seq)
        out_pred_dir = os.path.join(args.output_dir, 'sequences', seq, 'predictions')
        os.makedirs(out_pred_dir, exist_ok=True)
        frame_ids = [f.stem for f in sorted(Path(voxels_dir).glob('*.bin'))]
        logger.info(
            "Seq %s: %d frames x 8 D4 passes -> %s",
            seq, len(frame_ids), out_pred_dir,
        )

        for frame_id in tqdm(frame_ids, desc=f'Seq {seq} D4'):
            out_path = os.path.join(out_pred_dir, f'{frame_id}.label')
            if args.skip_existing and os.path.exists(out_path):
                continue

            scpnet_path = os.path.join(scpnet_seq_dir, f'{frame_id}_pred.npy')
            if not os.path.exists(scpnet_path):
                logger.warning("Missing SCPNet pred for %s/%s", seq, frame_id)
                continue
            scpnet_pred = np.load(scpnet_path)
            voxel_path = os.path.join(voxels_dir, f'{frame_id}.bin')
            lidar_base = load_lidar_voxels(voxel_path).to(device)
            scp_base = torch.from_numpy(scpnet_pred.astype(np.int64)).unsqueeze(0).to(device)

            soft_sum = None
            for (fx, fy, rk) in D4_ELEMENTS:
                lidar_t, scp_t, _ = apply_d4(lidar_base, scp_base, torch.zeros(1, 256, 256, dtype=torch.long, device=device), fx, fy, rk)
                bev_t = derive_bev(scp_t)
                soft = run_algo2_softmax(model, diffusion, lidar_t, scp_t, bev_t, args.cold_steps, device)
                soft_back = invert_d4(soft, fx, fy, rk)
                soft_sum = soft_back if soft_sum is None else (soft_sum + soft_back)

            soft_avg = soft_sum / len(D4_ELEMENTS)
            pred = soft_avg.argmax(dim=1)

            pred_train = pred.cpu().numpy()[0].flatten()
            assert 0 <= pred_train.min() <= pred_train.max() <= 19, \
                f'bad range min={pred_train.min()} max={pred_train.max()}'
            pred_orig = LEARNING_MAP_INV_ARRAY[pred_train.astype(np.int64)]
            pred_orig.astype(np.uint16).tofile(out_path)
            total += 1

    logger.info("Done, wrote %d D4-TTA .label files", total)


if __name__ == '__main__':
    main()
