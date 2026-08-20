#!/usr/bin/env python3
"""Full D4 (dihedral-4) TTA for S²D² headline checkpoint.

Training used random {flip_x, flip_y, rot90 k=0..3} applied to lidar + BEV +
scpnet_pred (see scene_completion/s3_dskd_dataset.py::_augment). The full
8-element D4 group is therefore in-distribution for S²D². This script runs
all 8 transforms, averages the simplex predictions after inverse-transform,
and writes argmaxed .label files.

Transform inversion conventions (voxel grid axes):
  LiDAR [1,1,H=256,W=256,D=32]:  flip_x -> dim=2, flip_y -> dim=3, rot90(k) on dims (2,3)
  base  [1,H=256,W=256,D=32]:    flip_x -> dim=1, flip_y -> dim=2, rot90(k) on dims (1,2)
  bev   [1,H=256,W=256]:         flip_x -> dim=1, flip_y -> dim=2, rot90(k) on dims (1,2)
  soft  [1,K=20,H=256,W=256,D=32]: invert — flip_x dim=2, flip_y dim=3, rot90(-k) on (2,3)

The base prediction can come from any frozen SSC backbone (SCPNet for the
headline; JS3C-Net for the v1.1.0 cross-base reproduction).

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
from gssc.utils.checkpoint import assert_bound, config_value, load_checkpoint_config

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
    """Unpack a bit-packed SemanticKITTI ``.bin`` occupancy array into a flat uint8 voxel array (8 voxels per input byte, MSB first)."""
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


def derive_bev(base_tensor: torch.Tensor) -> torch.Tensor:
    """Topmost-non-empty-class BEV projection of a [1, 256, 256, 32] base-model prediction.

    Base-agnostic since v1.1.0: works on any per-voxel categorical prediction
    (SCPNet, JS3C-Net, ...). The diffusion-side kwarg
    ``sample_algo2(..., scpnet_pred=...)`` keeps its historical name (model-side
    conditioning), but this BEV projection has no SCPNet-specific assumptions.
    """
    bev = torch.zeros(1, 256, 256, dtype=torch.long, device=base_tensor.device)
    for z in range(base_tensor.shape[3] - 1, -1, -1):
        layer = base_tensor[0, :, :, z]
        mask = (layer > 0) & (bev[0] == 0)
        bev[0][mask] = layer[mask]
    return bev


def load_model(
    ckpt_path: str | os.PathLike,
    device: torch.device,
) -> tuple[SceneCompletionUNetSparse, MultinomialDiffusion3DV2]:
    """Load the headline checkpoint with EMA weights swapped in for inference.

    Supports both the v1.0.0 ``.pt`` layout (model_state_dict + ema_shadow)
    and the v1.1.0 ``.safetensors`` per-subdir layout (model_ema.safetensors
    holds the deployment weights directly).

    The architecture and the noise schedule are read from whatever the checkpoint
    declares about itself (its sibling ``config.json``, or ``ckpt["config"]``); the
    literals below are only the fallback for a checkpoint that declares nothing.
    Every shipped checkpoint declares exactly these values, so this is inert for
    them and load-bearing for anything else. The load is then checked: a partial
    ``strict=False`` load used to leave unmatched tensors at random initialisation
    and still produce a plausible score.

    Raises:
        RuntimeError: If the weights do not fully bind to the architecture.
    """
    ckpt_str = str(ckpt_path)
    ckpt = None
    if ckpt_str.endswith(".safetensors"):
        from safetensors.torch import load_file
        state_dict = load_file(ckpt_str)
    else:
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        state_dict = ckpt['model_state_dict']

    cfg = load_checkpoint_config(ckpt_path, ckpt)
    model = SceneCompletionUNetSparse(
        num_classes=config_value(cfg, 'num_classes', 20),
        base_channels=config_value(cfg, 'base_channels', 32),
        time_emb_dim=128,
        lidar_base_channels=16, lidar_out_channels=32, lidar_in_channels=1,
        no_bev=config_value(cfg, 'no_bev', False),
        ssc_cond_channels=20,
        ssc_multiscale=config_value(cfg, 'ssc_multiscale', False),
    ).to(device)

    load_result = model.load_state_dict(state_dict, strict=False)
    assert_bound("model", load_result, ckpt_path)
    if ckpt is not None and 'ema_shadow' in ckpt:
        for name, p in model.named_parameters():
            if name in ckpt['ema_shadow']:
                p.data.copy_(ckpt['ema_shadow'][name])
    model.train(False)
    diffusion = MultinomialDiffusion3DV2(
        num_classes=config_value(cfg, 'num_classes', 20),
        num_timesteps=config_value(cfg, 'num_timesteps', 100),
        beta_max=config_value(cfg, 'beta_max', 0.1),
    ).to(device)
    return model, diffusion


def apply_d4(
    lidar: torch.Tensor,
    base: torch.Tensor,
    bev: torch.Tensor,
    flip_x: bool,
    flip_y: bool,
    rot_k: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply D4 element to inputs. Returns transformed (lidar, base, bev).

    Order matters: flip first, then rotate (mirrors training augment order).
    Base-agnostic since v1.1.0; works on any per-voxel categorical prediction.
    """
    if flip_x:
        lidar = torch.flip(lidar, dims=[2])
        base  = torch.flip(base,  dims=[1])
        bev   = torch.flip(bev,   dims=[1])
    if flip_y:
        lidar = torch.flip(lidar, dims=[3])
        base  = torch.flip(base,  dims=[2])
        bev   = torch.flip(bev,   dims=[2])
    if rot_k > 0:
        lidar = torch.rot90(lidar, k=rot_k, dims=[2, 3])
        base  = torch.rot90(base,  k=rot_k, dims=[1, 2])
        bev   = torch.rot90(bev,   k=rot_k, dims=[1, 2])
    return lidar, base, bev


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
    base_tensor: torch.Tensor,
    bev: torch.Tensor,
    n_steps: int,
    device: torch.device,
    tau: float = 1.0,
) -> torch.Tensor:
    """Run a single S2D2 correction forward pass and return the soft-max distribution [1,K,H,W,D].

    Base-agnostic since v1.1.0; ``base_tensor`` may be any per-voxel categorical
    prediction (SCPNet, JS3C-Net). The diffusion kwarg ``scpnet_pred=`` keeps its
    historical model-side name; the kwarg is the conditioning channel for the
    denoiser, not a base-model assumption.
    """
    base_oh = F.one_hot(base_tensor.long(), 20).float().permute(0, 4, 1, 2, 3)
    with torch.no_grad():
        soft = diffusion.sample_algo2(
            model, bev, lidar,
            scpnet_pred=base_tensor,
            shape=(1, 256, 256, 32),
            device=device,
            n_steps=n_steps,
            tau=tau,
            show_progress=False,
            ssc_pred=base_oh,
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
    p.add_argument('--tau', type=float, default=1.0,
                   help='Sampling temperature on the denoiser logits. Inert at --cold_steps 1; '
                        'affects intermediate corrections at more steps, so it is not inert here '
                        '(this path defaults to 4).')
    p.add_argument('--output_dir', required=True)
    p.add_argument('--data_root', default='data/SemanticKITTI',
                   help='Root containing sequences/<SEQ>/voxels/*.bin')
    p.add_argument('--base_pred_dir', '--scpnet_dir', dest='base_pred_dir',
                   default='data/scpnet_predictions',
                   help='Root containing <SEQ>/<frame>_pred.npy. The --scpnet_dir alias '
                        'is kept for v1.0.0 callers and will be removed in v2.0.0.')
    p.add_argument('--split', default='valid', choices=['train', 'valid', 'test'])
    p.add_argument('--skip_existing', action='store_true')
    p.add_argument('--bev_source', default='derived', choices=['derived', 'gt'],
                   help=("'derived' uses topmost-non-empty BEV from the (D4-transformed) "
                         "base prediction (standard D4 TTA). 'gt' uses preprocessed GT "
                         "BEV — only valid for D4 elements that preserve BEV (identity)."))
    p.add_argument('--bev_root', default=None,
                   help='Root for GT BEV files when --bev_source gt.')
    args = p.parse_args()
    if args.bev_source == 'gt':
        # D4 TTA applies the same D4 transform to base_pred and BEV; with derived BEV
        # this works out of the box because `derive_bev` is applied AFTER the
        # transform. With GT BEV, the BEV is loaded from disk in canonical
        # orientation, so non-identity D4 elements would inject inconsistent
        # conditioning. Fall back to derived for D4 TTA.
        logger.warning(
            "D4 TTA + --bev_source gt is not supported (canonical-frame BEV "
            "cannot be D4-transformed); falling back to derived BEV."
        )
        args.bev_source = 'derived'

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
        base_seq_dir = os.path.join(args.base_pred_dir, seq)
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

            base_path = os.path.join(base_seq_dir, f'{frame_id}_pred.npy')
            if not os.path.exists(base_path):
                logger.warning("Missing base prediction for %s/%s", seq, frame_id)
                continue
            base_pred = np.load(base_path)
            voxel_path = os.path.join(voxels_dir, f'{frame_id}.bin')
            lidar_base = load_lidar_voxels(voxel_path).to(device)
            base_pred_t = torch.from_numpy(base_pred.astype(np.int64)).unsqueeze(0).to(device)

            assert D4_ELEMENTS, "D4 group must have >= 1 element"
            soft_sum: torch.Tensor | None = None
            for (fx, fy, rk) in D4_ELEMENTS:
                lidar_t, base_t, _ = apply_d4(lidar_base, base_pred_t, torch.zeros(1, 256, 256, dtype=torch.long, device=device), fx, fy, rk)
                bev_t = derive_bev(base_t)
                soft = run_algo2_softmax(model, diffusion, lidar_t, base_t, bev_t, args.cold_steps, device, tau=args.tau)
                soft_back = invert_d4(soft, fx, fy, rk)
                soft_sum = soft_back if soft_sum is None else (soft_sum + soft_back)

            assert soft_sum is not None  # guaranteed by D4_ELEMENTS non-empty assertion
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
