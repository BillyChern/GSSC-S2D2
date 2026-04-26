"""GSSC-S2D2 evaluation entry point.

Loads a checkpoint, runs Algo2 correction sampling on SemanticKITTI val, and
reports per-class IoU + mIoU + completion IoU.

Examples
--------
Reproduce the headline 38.54% val mIoU (1-step Algo2)::

    python scripts/eval.py eval/val_1step \
        --checkpoint data/checkpoints/gssc_31k_mf_step40000.pt

Reproduce the 38.73% val mIoU under D4 TTA::

    python scripts/eval.py eval/val_d4tta \
        --checkpoint data/checkpoints/gssc_31k_mf_step40000.pt
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))


def main() -> None:
    parser = argparse.ArgumentParser(description="GSSC-S2D2 evaluation entry point.")
    parser.add_argument("config", help="Hydra-style config name, e.g. eval/val_1step")
    parser.add_argument("--checkpoint", required=True, help="Path to trained checkpoint")
    parser.add_argument("--data-root", default=str(REPO_ROOT / "data"))
    parser.add_argument("--output", default=None, help="Where to dump per-class JSON")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--steps", type=int, default=None, help="Algo2 step count override (1, 4, 100)")
    parser.add_argument("--tta", choices=["none", "flip_y", "d4"], default=None)
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["miou", "completion_iou", "per_class"],
        help="Metric set: miou, completion_iou, per_class, safety, dwiou",
    )
    parser.add_argument(
        "--keep-predictions",
        action="store_true",
        help="Persist generated .label files under <data-root>/predictions/<config>/ "
        "instead of using a temporary directory. Useful for inspection or test-server "
        "submission.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="[%(name)s] %(message)s",
    )

    from gssc.inference.evaluate import run_evaluation

    metrics = run_evaluation(
        checkpoint=args.checkpoint,
        config=args.config,
        data_root=args.data_root,
        output=args.output,
        gpu=args.gpu,
        steps=args.steps,
        tta=args.tta,
        metrics=args.metrics,
        keep_predictions=args.keep_predictions,
    )

    print()
    print("=" * 60)
    print(f" GSSC-S2D2 evaluation: {args.config}")
    print("=" * 60)
    if "mIoU" in metrics:
        print(f"  mIoU       : {metrics['mIoU']:.2f} %")
    if "IoU_cmpl" in metrics:
        print(f"  Completion : {metrics['IoU_cmpl']:.2f} %")
    print("-" * 60)
    print(" Per-class IoU:")
    for k, v in sorted(metrics.items()):
        if k.startswith("IoU_") and k != "IoU_cmpl":
            print(f"  {k[4:]:<20s} {v:6.2f} %")
    print("=" * 60)
    if args.output:
        print(f" JSON: {args.output}")
    sys.stdout.flush()
    # Compact JSON dump always (last line, easy for CI to parse)
    print(json.dumps(metrics))


if __name__ == "__main__":
    main()
