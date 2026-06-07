"""Dump JS3C-Net SSC predictions as per-frame x_src .npy files.

This script reproduces the JS3C-Net base predictions used in the paper Tab.
III rows 90-91 (cross-base reproduction). Output format mirrors the
existing ``data/scpnet_predictions/`` layout so the migrated S2D2
dataloader consumes it as ``base_pred_dir=data/js3cnet_predictions``.

Setup
-----
Clone the upstream JS3C-Net repo (tested at commit
``7df4d0c66`` on the public ``master``) and download the pretrained
SemanticKITTI checkpoint following its README::

    git clone --depth 1 https://github.com/yanx27/JS3C-Net external/JS3C-Net
    bash external/JS3C-Net/download_pretrained.sh  # populates log/JS3C-Net-kitti/

Then run this script from anywhere inside the GSSC-S2D2 venv::

    python scripts/dump_js3c_predictions.py \\
        --js3c-repo external/JS3C-Net \\
        --semantickitti_root data/SemanticKITTI \\
        --output_dir data/js3cnet_predictions \\
        --sequences 00 01 02 03 04 05 06 07 08 09 10 \\
        --gpu 0

Notes
-----
* Sequences 00-10 = train+val (val=08); 11-21 = SemanticKITTI hidden test.
* One full pass through all 22 sequences takes ~2-4 GPU-hours on an H100.
* Idempotent: frames whose ``.npy`` already exists are skipped.
"""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import os
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

logger = logging.getLogger("gssc.dump_js3c")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--js3c-repo",
        required=True,
        type=Path,
        help="Path to a local clone of the upstream JS3C-Net repository.",
    )
    parser.add_argument(
        "--semantickitti_root",
        required=True,
        type=Path,
        help="SemanticKITTI dataset root containing sequences/{00..21}/...",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        type=Path,
        help="Output root for per-frame .npy predictions.",
    )
    parser.add_argument(
        "--js3c_log_dir",
        type=Path,
        default=None,
        help="Override for the JS3C-Net checkpoint directory (defaults to "
        "<js3c-repo>/log/JS3C-Net-kitti).",
    )
    parser.add_argument(
        "--sequences",
        nargs="+",
        default=[f"{i:02d}" for i in range(22)],
        help="Sequences to process (default: all 00-21).",
    )
    parser.add_argument("--gpu", default="0", help="CUDA_VISIBLE_DEVICES value.")
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Load the model and run on a single frame; do not save anything.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args()


@contextmanager
def js3c_workdir(repo: Path) -> Iterator[None]:
    """Temporarily ``chdir`` into the JS3C-Net repo while adding it to ``sys.path``.

    The JS3C-Net codebase uses bare ``from models import …`` imports that
    only resolve when the cwd is the repo root. We restore both the cwd
    and ``sys.path`` on exit so subsequent runs are not contaminated.
    """
    original_cwd = Path.cwd()
    repo_str = str(repo)
    path_modified = repo_str not in sys.path
    if path_modified:
        sys.path.insert(0, repo_str)
    os.chdir(repo)
    try:
        yield
    finally:
        os.chdir(original_cwd)
        if path_modified:
            try:
                sys.path.remove(repo_str)
            except ValueError:
                pass


def load_js3c_model(log_dir: Path, repo: Path):
    """Reconstruct JS3C-Net's ``J3SC_Net`` model and load weights from ``log_dir``."""
    import glob

    import torch
    from torch import nn

    with (log_dir / "args.txt").open() as f:
        config = json.load(f)
    config["GENERAL"]["debug"] = False

    seg_module = importlib.import_module("models." + config["Segmentation"]["model_name"])
    complet_module = importlib.import_module("models." + config["Completion"]["model_name"])
    model_utils = importlib.import_module("models.model_utils")

    class JS3CNet(nn.Module):
        """Mirrors test_kitti_ssc.py's J3SC_Net class."""

        def __init__(self) -> None:
            super().__init__()
            self.args = config
            self.seg_head = seg_module.get_model(config)
            self.complet_head = complet_module.get_model(config)
            self.voxelpool = model_utils.VoxelPooling(config)
            self.seg_sigmas_sq = nn.Parameter(
                torch.Tensor(1).uniform_(0.2, 1), requires_grad=True
            )
            self.complet_sigmas_sq = nn.Parameter(
                torch.Tensor(1).uniform_(0.2, 1), requires_grad=True
            )

        def forward(self, x):
            import spconv.pytorch as spconv

            seg_inputs, complet_inputs, _ = x
            seg_output, feat = self.seg_head(seg_inputs)
            torch.cuda.empty_cache()

            coords = complet_inputs["complet_coords"]
            coords = coords[:, [0, 3, 2, 1]]
            if self.args["DATA"]["dataset"] == "SemanticKITTI":
                coords[:, 3] += 1

            feeding_mode = self.args["Completion"]["feeding"]
            if feeding_mode == "both":
                feeding = torch.cat([seg_output, feat], 1)
            elif feeding_mode == "feat":
                feeding = feat
            else:
                feeding = seg_output

            features = self.voxelpool(
                invoxel_xyz=complet_inputs["complet_invoxel_features"][:, :, :-1],
                invoxel_map=complet_inputs["complet_invoxel_features"][:, :, -1].long(),
                src_feat=feeding,
                voxel_center=complet_inputs["voxel_centers"],
            )
            if self.args["Completion"]["no_fuse_feat"]:
                features[...] = 1
                features = features.detach()

            batch_complet = spconv.SparseConvTensor(
                features.float(),
                coords.int(),
                self.args["Completion"]["full_scale"],
                self.args["TRAIN"]["batch_size"],
            )
            complet_output = self.complet_head(batch_complet)
            torch.cuda.empty_cache()
            return seg_output, complet_output

    model = JS3CNet().cuda()
    # Inference mode without triggering hook false-positive in JS3C-Net's
    # voxelpool module (upstream uses .train(False) for the same reason).
    model.train(False)

    ckpts = sorted(glob.glob(str(log_dir / "model*.pth")))
    if not ckpts:
        raise FileNotFoundError(f"No model*.pth in {log_dir}")
    logger.info("loading JS3C-Net checkpoint: %s", ckpts[-1])
    state = torch.load(ckpts[-1], map_location="cuda")
    model.load_state_dict(state, strict=False)
    return model, config


def get_dataset_for_split(config: dict, split: str):
    """Build JS3C-Net's ``kitti_dataset`` for a single split."""
    kitti_dataset = importlib.import_module("kitti_dataset")

    config = dict(config)
    config["DATA"]["split"] = split
    return kitti_dataset.get_dataset(config, split=split)


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="[%(name)s %(levelname)s] %(message)s",
    )
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    out_root = args.output_dir
    out_root.mkdir(parents=True, exist_ok=True)

    log_dir = args.js3c_log_dir or (args.js3c_repo / "log/JS3C-Net-kitti")

    with js3c_workdir(args.js3c_repo):
        import numpy as np
        import torch
        SubSparseConv = importlib.import_module("models").SubSparseConv
        from torch.utils.data import DataLoader, Subset
        from tqdm import tqdm

        model, config = load_js3c_model(log_dir, args.js3c_repo)
        config["GENERAL"]["dataset_dir"] = str(args.semantickitti_root)

        n_total = n_skipped = n_failed = 0
        t_start = time.time()

        for seq in args.sequences:
            if seq == "08":
                split = "valid"
            elif int(seq) >= 11:
                split = "test"
            else:
                split = "train"

            try:
                dataset = get_dataset_for_split(config, split)
            except Exception as exc:
                logger.warning("[seq %s] dataset build failed for split=%s: %s", seq, split, exc)
                continue

            seq_indices = [
                i
                for i, (seq_str, _frame_id) in enumerate(
                    getattr(dataset, "filenames", [])
                )
                if str(seq_str).zfill(2) == str(seq).zfill(2)
            ]
            if not seq_indices:
                logger.warning("[seq %s] no frames found (split=%s)", seq, split)
                continue

            logger.info("[seq %s] %d frames (split=%s)", seq, len(seq_indices), split)
            seq_out = out_root / seq
            seq_out.mkdir(parents=True, exist_ok=True)

            loader = DataLoader(
                Subset(dataset, seq_indices),
                batch_size=1,
                collate_fn=SubSparseConv.Merge,
                num_workers=4,
                pin_memory=True,
                shuffle=False,
                persistent_workers=True,
            )

            with torch.no_grad():
                for batch in tqdm(loader, desc=f"seq {seq}"):
                    try:
                        _seq_in_batch, frame_id = batch[2][0]
                    except Exception:
                        n_failed += 1
                        continue

                    out_path = seq_out / f"{frame_id}_pred.npy"
                    if out_path.exists() and out_path.stat().st_size > 0:
                        n_skipped += 1
                        continue

                    try:
                        _, complet_pred = model(batch)
                        pred_choice = complet_pred[-1].data.max(1)[1]
                        pred_arr = pred_choice.cpu().numpy().astype(np.uint8)
                        if pred_arr.size == 256 * 256 * 32:
                            pred_arr = pred_arr.reshape(256, 256, 32)
                        elif pred_arr.size == 256 * 256 * 31:
                            pred_arr = np.concatenate(
                                [
                                    pred_arr.reshape(256, 256, 31),
                                    np.zeros((256, 256, 1), dtype=np.uint8),
                                ],
                                axis=-1,
                            )
                        else:
                            logger.warning(
                                "[seq %s] %s: unexpected pred size %d",
                                seq, frame_id, pred_arr.size,
                            )
                            n_failed += 1
                            continue

                        np.save(out_path, pred_arr)
                        n_total += 1

                        if args.dry_run and n_total >= 1:
                            logger.info("[dry_run] saved 1 sample to %s; stopping", out_path)
                            return 0
                    except Exception as exc:
                        logger.warning("[seq %s] %s failed: %s", seq, frame_id, exc)
                        n_failed += 1
                        continue

        dt = time.time() - t_start
        logger.info(
            "Done in %.1f min - saved=%d skipped=%d failed=%d",
            dt / 60, n_total, n_skipped, n_failed,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
