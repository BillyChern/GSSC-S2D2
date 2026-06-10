#!/usr/bin/env python3
"""Generate predictions in official SemanticKITTI SSC format for leaderboard submission.

Pipeline:
1. Load SCPNet 3D predictions (training space 0-19)
2. Load binary LiDAR voxels
3. Run cold diffusion inference (sample_cold)
4. Convert predictions from training space (0-19) to original space (0-255)
5. Save as uint16 .label files in official directory structure

Usage:
  # Val set (local evaluation):
  python -m gssc.inference.generate_predictions --split valid --output_dir outputs/official_predictions

  # Test set (leaderboard submission):
  python -m gssc.inference.generate_predictions --split test --output_dir outputs/official_predictions

  # Then evaluate locally (val only):
  python external/semantic_kitti_api/evaluate_completion.py \
      --dataset data/SemanticKITTI --predictions outputs/official_predictions --split valid

This module is normally driven through scripts/infer.py and scripts/eval.py
(see docs/MODEL_ZOO.md); the commands above are the equivalent direct entry points.
"""

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

# Official SemanticKITTI learning_map_inv: training space (0-19) → original space (0-255)
LEARNING_MAP_INV = np.array([
    0,   # 0: unlabeled/empty
    10,  # 1: car
    11,  # 2: bicycle
    15,  # 3: motorcycle
    18,  # 4: truck
    20,  # 5: other-vehicle
    30,  # 6: person
    31,  # 7: bicyclist
    32,  # 8: motorcyclist
    40,  # 9: road
    44,  # 10: parking
    48,  # 11: sidewalk
    49,  # 12: other-ground
    50,  # 13: building
    51,  # 14: fence
    70,  # 15: vegetation
    71,  # 16: trunk
    72,  # 17: terrain
    80,  # 18: pole
    81,  # 19: traffic-sign
], dtype=np.uint16)

SPLIT_SEQUENCES = {
    'valid': ['08'],
    'test': ['11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21'],
}


def unpack(compressed: np.ndarray) -> np.ndarray:
    """Unpack binary voxel grid from bit-encoded format."""
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


def load_lidar_voxels(voxel_path: str | os.PathLike) -> torch.Tensor:
    """Load binary LiDAR voxels from .bin file → [1, 1, 256, 256, 32] tensor."""
    compressed = np.fromfile(voxel_path, dtype=np.uint8)
    binary = unpack(compressed).reshape(256, 256, 32).astype(np.float32)
    return torch.from_numpy(binary).unsqueeze(0).unsqueeze(0)


def main() -> None:
    """Parse CLI arguments and run scene-completion inference, writing official SemanticKITTI ``.label`` predictions to the output directory."""
    parser = argparse.ArgumentParser(description='Generate official SemanticKITTI SSC predictions')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to B4b checkpoint (e.g., best_miou.pt)')
    parser.add_argument('--scpnet_dir', type=str, default='data/scpnet_predictions',
                        help='Directory with SCPNet 3D predictions (<SEQ>/<frame>_pred.npy)')
    parser.add_argument('--data_root', type=str, default='data/SemanticKITTI',
                        help='SemanticKITTI SSC root (contains sequences/<SEQ>/voxels/*.bin)')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for official predictions')
    parser.add_argument('--split', type=str, choices=['valid', 'test'], default='valid',
                        help='Which split to generate predictions for')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--use_train_weights', action='store_true',
                        help='Use training weights instead of EMA. WARNING: for correction sampling, EMA is strongly preferred')
    parser.add_argument('--cold_steps', type=int, default=100,
                        help='Number of cold diffusion steps (100=full, 1=one-step)')
    parser.add_argument('--algo2', action='store_true',
                        help='Use S2D2 correction sampling (Cold Diffusion non-noise specialisation) instead of D3PM posterior')
    parser.add_argument('--pred_dir', type=str, default=None,
                        help='Override prediction source dir (e.g., datasets/talos_predictions for TALoS inference)')
    parser.add_argument('--no_bev', action='store_true',
                        help='Disable BEV conditioning (for B4b-style no-BEV models)')
    parser.add_argument('--bev_from_base', action='store_true',
                        help='Use base-pred-derived BEV (height-pool of `--base_pred_dir` 3D)')
    parser.add_argument('--bev_source', type=str, default='derived',
                        choices=['derived', 'gt'],
                        help=('How to source the BEV conditioning. '
                              "'derived' (default): topmost-non-empty class from the base "
                              '3D prediction (matches `--bev_from_base` training-time '
                              'override). '
                              "'gt': load preprocessed GT BEV from `<voxel_root>/<seq>/<frame>_bev.npy`. "
                              'Used to reproduce the paper GT-BEV-conditioned protocol '
                              '(JS3C-Net + S²D² val mIoU 26.05%% under the official '
                              'semantic-kitti-api headline; 26.72%% under the paper internal '
                              'SSCMetrics, supp tab:supp_b6_val footnote).'))
    parser.add_argument('--bev_root', type=str, default=None,
                        help=('Root for GT BEV files when `--bev_source gt`. Defaults to '
                              '`<data_root>/../SemanticKITTI_3D/256` (mirrors the internal '
                              'preprocessing layout from `tools/prepare_256_data.py`).'))
    parser.add_argument('--scpnet_only', action='store_true',
                        help='Skip cold diffusion, just convert SCPNet predictions to .label format')
    parser.add_argument('--ssc_multiscale', action='store_true',
                        help='Enable 3D multiscale conditioning (for Exp5-style models)')
    parser.add_argument('--skip_existing', action='store_true',
                        help='Skip frames whose .label already exists (safe resume)')
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="[%(name)s] %(message)s",
    )

    # Use --pred_dir if specified, otherwise fall back to --scpnet_dir
    pred_source_dir = args.pred_dir if args.pred_dir else args.scpnet_dir
    if args.pred_dir:
        logger.info("Using prediction source: %s (TALoS/custom)", pred_source_dir)
    else:
        logger.info("Using prediction source: %s (SCPNet default)", pred_source_dir)

    device = torch.device(f'cuda:{args.gpu}')
    sequences = SPLIT_SEQUENCES[args.split]

    if not args.scpnet_only:
        logger.info("Loading checkpoint: %s", args.checkpoint)
        ckpt_path_str = args.checkpoint
        is_safetensors = ckpt_path_str.endswith(".safetensors")
        if is_safetensors:
            from safetensors.torch import load_file
            state_dict = load_file(ckpt_path_str)
            logger.info("  safetensors layout (v1.1.0): %d tensors", len(state_dict))
        else:
            ckpt = torch.load(ckpt_path_str, map_location='cpu', weights_only=False)
            logger.info(
                "  step=%s best_miou=%s",
                ckpt.get("global_step", "?"),
                f"{ckpt['best_miou']:.4f}" if "best_miou" in ckpt else "?",
            )

        model = SceneCompletionUNetSparse(
            num_classes=20, base_channels=32, time_emb_dim=128,
            lidar_base_channels=16, lidar_out_channels=32, lidar_in_channels=1,
            no_bev=args.no_bev, ssc_cond_channels=20, ssc_multiscale=args.ssc_multiscale,
        ).to(device)

        if is_safetensors:
            model.load_state_dict(state_dict, strict=False)
            logger.info("  Loaded deployment weights from safetensors")
        else:
            model.load_state_dict(ckpt['model_state_dict'], strict=False)
            if not args.use_train_weights:
                ema_count = 0
                for name, param in model.named_parameters():
                    if name in ckpt['ema_shadow']:
                        param.data.copy_(ckpt['ema_shadow'][name])
                        ema_count += 1
                logger.info("  Loaded EMA weights (%d parameters)", ema_count)
            else:
                logger.info("  Using training weights (not EMA)")

        model.train(False)

        diffusion = MultinomialDiffusion3DV2(
            num_classes=20, num_timesteps=100, beta_max=0.1
        ).to(device)

    # Resolve GT-BEV root once (only used when --bev_source gt).
    if args.bev_source == 'gt':
        if args.bev_root is not None:
            bev_root = args.bev_root
        else:
            # Default: <data_root>/../SemanticKITTI_3D/256 (mirrors prepare_256_data.py).
            bev_root = os.path.join(os.path.dirname(args.data_root.rstrip('/')),
                                    'SemanticKITTI_3D', '256')
        logger.info("Using GT BEV from %s (paper supp tab:supp_b6_val protocol)", bev_root)
    else:
        bev_root = None

    # Process each sequence
    total_frames = 0
    for seq in sequences:
        voxels_dir = os.path.join(args.data_root, 'sequences', seq, 'voxels')
        scpnet_seq_dir = os.path.join(pred_source_dir, seq)
        out_pred_dir = os.path.join(args.output_dir, 'sequences', seq, 'predictions')
        os.makedirs(out_pred_dir, exist_ok=True)

        # Get frame list from voxel .bin files
        voxel_files = sorted(Path(voxels_dir).glob('*.bin'))
        frame_ids = [f.stem for f in voxel_files]

        logger.info("Sequence %s: %d frames -> %s", seq, len(frame_ids), out_pred_dir)

        for frame_id in tqdm(frame_ids, desc=f'Seq {seq}'):
            out_path = os.path.join(out_pred_dir, f'{frame_id}.label')

            if args.skip_existing and os.path.exists(out_path):
                continue

            # Load SCPNet prediction (training space 0-19)
            scpnet_path = os.path.join(scpnet_seq_dir, f'{frame_id}_pred.npy')
            if not os.path.exists(scpnet_path):
                logger.warning("Missing SCPNet pred for %s/%s, skipping", seq, frame_id)
                continue

            scpnet_pred = np.load(scpnet_path)  # [256, 256, 32] uint8, values 0-19

            if args.scpnet_only:
                # Just convert SCPNet predictions to official format
                pred_train = scpnet_pred.flatten()
            else:
                # Load LiDAR voxels
                voxel_path = os.path.join(voxels_dir, f'{frame_id}.bin')
                lidar = load_lidar_voxels(voxel_path).to(device)

                # SCPNet pred to tensor
                scp_tensor = torch.from_numpy(scpnet_pred.astype(np.int64)).unsqueeze(0).to(device)
                scp_oh = F.one_hot(scp_tensor, 20).float().permute(0, 4, 1, 2, 3)

                # BEV conditioning
                if args.bev_source == 'gt':
                    bev_path = os.path.join(bev_root, seq, f'{frame_id}_bev.npy')
                    if not os.path.isfile(bev_path):
                        raise FileNotFoundError(
                            f"GT BEV missing for --bev_source gt: {bev_path}. "
                            "Run `python tools/prepare_256_data.py` to materialise "
                            "preprocessed BEV files, or switch to `--bev_source derived`."
                        )
                    bev_np = np.load(bev_path).astype(np.int64)
                    bev = torch.from_numpy(bev_np).unsqueeze(0).to(device)
                elif args.bev_from_base:
                    # Derive BEV from base-pred 3D voxels (topmost non-empty class).
                    bev = torch.zeros(1, 256, 256, dtype=torch.long, device=device)
                    for z in range(scp_tensor.shape[3] - 1, -1, -1):
                        layer = scp_tensor[0, :, :, z]
                        mask = (layer > 0) & (bev[0] == 0)
                        bev[0][mask] = layer[mask]
                else:
                    bev = torch.zeros(1, 256, 256, dtype=torch.long, device=device)

                # Run cold diffusion
                with torch.no_grad():
                    if args.algo2:
                        # S2D2 correction sampling (Cold Diffusion non-noise specialisation)
                        pred = diffusion.sample_algo2(
                            model, bev, lidar,
                            scpnet_pred=scp_tensor,
                            shape=(1, 256, 256, 32),
                            device=device,
                            n_steps=args.cold_steps,
                            show_progress=False,
                            ssc_pred=scp_oh,
                        )
                    elif args.cold_steps >= 100:
                        pred = diffusion.sample_cold(
                            model, bev, lidar,
                            scpnet_pred=scp_tensor,
                            shape=(1, 256, 256, 32),
                            device=device,
                            show_progress=False,
                            ssc_pred=scp_oh,
                        )
                    else:
                        # Few-step cold diffusion (posterior-based)
                        timesteps = list(np.linspace(99, 0, args.cold_steps, dtype=int))
                        bev_oh = diffusion._prepare_bev(bev).to(device)
                        x_t = scp_oh.clone()
                        for t in timesteps:
                            t_batch = torch.full((1,), t, device=device, dtype=torch.long)
                            x_t = diffusion.p_sample(
                                model, x_t, t_batch, bev_oh, lidar,
                                x_scpnet=scp_oh, ssc_pred=scp_oh,
                            )
                        pred = x_t.argmax(dim=1)

                pred_train = pred.cpu().numpy()[0].flatten()  # [2097152] values 0-19

            # Verify predictions are in valid range
            assert pred_train.min() >= 0 and pred_train.max() <= 19, \
                f'Prediction values out of range [0-19]: min={pred_train.min()}, max={pred_train.max()}'

            # Convert from training space (0-19) to original space (0-255)
            pred_original = LEARNING_MAP_INV[pred_train.astype(np.int64)]

            # Save as uint16 .label file
            pred_original.astype(np.uint16).tofile(out_path)

            total_frames += 1

    logger.info("Done. Generated %d prediction files in %s", total_frames, args.output_dir)
    logger.info(
        "To score: python external/semantic_kitti_api/evaluate_completion.py "
        "--dataset %s --predictions %s --split %s",
        os.path.abspath(args.data_root),
        os.path.abspath(args.output_dir),
        args.split,
    )


if __name__ == '__main__':
    main()
