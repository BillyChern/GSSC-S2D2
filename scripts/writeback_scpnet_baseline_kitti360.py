#!/usr/bin/env python3
"""Write SCPNet-only zero-shot predictions to KITTI-360 .label format (baseline).

Reads the SemanticKITTI-train-space SCPNet predictions dumped by
tools/run_scpnet_inference.py (<seq>/<frame>_pred.npy, values 0-19), projects
them SemanticKITTI->KITTI-360 train space (remap_scpnet_to_kitti360), then to
KITTI-360 ORIGINAL 0-255 labels (KITTI360_LEARNING_MAP_INV), and writes
<frame>.label uint16 flat — the SCPNet-only baseline (no S2D2 refinement).

This is identical to the class-projection writeback in
scripts/eval_kitti360.run_s2d2_and_writeback, minus the diffusion correction
step, so the baseline and S2D2 rows are scored under the exact same protocol.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from gssc.data.kitti360_class_map import (
    KITTI360_LEARNING_MAP_INV,
    remap_scpnet_to_kitti360,
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scpnet_dir", required=True, help="Dir with <frame>_pred.npy (0-19)")
    ap.add_argument("--out_dir", required=True, help="Dir to write <frame>.label (orig 0-255)")
    args = ap.parse_args()

    scpnet_dir = Path(args.scpnet_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    preds = sorted(scpnet_dir.glob("*_pred.npy"))
    if not preds:
        raise SystemExit(f"No *_pred.npy under {scpnet_dir}")

    for pf in preds:
        fid = pf.stem.replace("_pred", "")
        pred_sk = np.load(pf)  # [256,256,32] SemanticKITTI train 0-19
        pred_k360_train = remap_scpnet_to_kitti360(pred_sk)
        pred_original = KITTI360_LEARNING_MAP_INV[pred_k360_train.astype(np.int64)]
        pred_original.reshape(-1).astype(np.uint16).tofile(out_dir / f"{fid}.label")

    print(f"Wrote {len(preds)} SCPNet-only .label predictions to {out_dir}")


if __name__ == "__main__":
    main()
