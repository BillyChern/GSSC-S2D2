"""GSSC-S2D2 evaluation entry point.

Loads a checkpoint, runs S2D2 correction sampling on SemanticKITTI val, and
reports per-class IoU + mIoU + completion IoU.

Examples
--------
Checkpoints since v1.1.0 use a per-checkpoint subdir layout
``data/checkpoints/<group>/<name>/{model.safetensors, model_ema.safetensors,
config.json}`` (the layout ``scripts/download_assets.py`` writes). Inference
defaults to the EMA weights ``model_ema.safetensors``.

Reproduce the headline 38.54% val mIoU (single correction step, N=1)::

    python scripts/eval.py eval/val_1step \
        --checkpoint data/checkpoints/gssc_mf/gssc_31k_mf_step40000/model_ema.safetensors

Reproduce the 38.73% val mIoU under D4 TTA::

    python scripts/eval.py eval/val_d4tta \
        --checkpoint data/checkpoints/gssc_mf/gssc_31k_mf_step40000/model_ema.safetensors

Reproduce the 36.09% BEV mIoU (secondary task, paper Sec. 4)::

    python scripts/eval.py eval/bev_secondary \
        --checkpoint data/checkpoints/bev/bev_perception_net/model.safetensors
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
    parser.add_argument("--steps", type=int, default=None, help="Correction-step count override (1, 4, 100)")
    parser.add_argument("--tta", choices=["none", "flip_y", "d4"], default=None)
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["miou", "completion_iou", "per_class"],
        help="Metric set: miou, completion_iou, per_class",
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

    # Dispatch by config family. The 3D SSC headline path uses run_evaluation
    # (subprocess into generate_predictions + semantic_kitti_api scoring).
    # The BEV second-task path needs its own loader because it never produces
    # .label files in SemanticKITTI's 3D layout.
    is_bev = args.config.startswith("eval/bev_")

    if is_bev:
        from gssc.inference.evaluate_bev import evaluate_bev
        metrics = evaluate_bev(
            checkpoint=args.checkpoint,
            data_root=args.data_root,
            n_steps=args.steps if args.steps is not None else 1,
            sequence="08",
            output=args.output,
            gpu=args.gpu,
        )
    else:
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
