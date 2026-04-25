"""GSSC-S2D2 evaluation entry point.

Loads a checkpoint, runs Algo2 correction sampling on SemanticKITTI val, and
reports per-class IoU + mIoU + completion IoU + safety-aware metrics
(SC-mIoU, VRU-IoU, DW-IoU).

Examples
--------
Reproduce the headline 38.54% val mIoU (1-step Algo2)::

    python scripts/eval.py +config=eval/val_1step \
        --checkpoint checkpoints/gssc_31k_mf_step40000.safetensors

Reproduce the 38.73% val mIoU under D4 TTA::

    python scripts/eval.py +config=eval/val_d4tta \
        --checkpoint checkpoints/gssc_31k_mf_step40000.safetensors

Step sweep (Tab. V)::

    python scripts/eval.py +config=eval/step_sweep \
        --checkpoint checkpoints/gssc_31k_mf_step40000.safetensors
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))


def main() -> None:
    parser = argparse.ArgumentParser(description="GSSC-S2D2 evaluation entry point.")
    parser.add_argument("config", help="Hydra-style config name, e.g. eval/val_1step")
    parser.add_argument("--checkpoint", required=True, help="Path to trained checkpoint")
    parser.add_argument("--data-root", default=str(REPO_ROOT / "data"))
    parser.add_argument("--output", default=None, help="Where to dump per-class CSV")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--steps", type=int, default=None, help="Algo2 step count override (1, 4, 100)")
    parser.add_argument("--tta", choices=["none", "flip_y", "d4"], default=None)
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["miou", "completion_iou", "per_class"],
        help="Metric set: miou, completion_iou, per_class, safety, dwiou",
    )
    args = parser.parse_args()

    from gssc.inference.evaluate import run_evaluation

    run_evaluation(
        checkpoint=args.checkpoint,
        config=args.config,
        data_root=args.data_root,
        output=args.output,
        gpu=args.gpu,
        steps=args.steps,
        tta=args.tta,
        metrics=args.metrics,
    )


if __name__ == "__main__":
    main()
