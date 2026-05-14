"""Reproduce any per-table number from the paper.

Each table maps to a specific eval recipe. Running this script will:
1. Verify the right checkpoint subdir is downloaded.
2. Run the appropriate eval pipeline.
3. Print a CSV that matches the LaTeX numbers within 0.1 pp.

Checkpoints since v1.1.0 live in a per-checkpoint subdir layout
``data/checkpoints/<group>/<name>/{model.safetensors, model_ema.safetensors,
config.json}`` (HF convention). Inference defaults to ``model_ema.safetensors``.

Examples::

    python scripts/reproduce_table.py tab:perclass
    python scripts/reproduce_table.py tab:main_results
    python scripts/reproduce_table.py tab:cross_base_js3c   # NEW v1.1.0
    python scripts/reproduce_table.py tab:bev_results
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]

TABLE_MAP: dict[str, dict[str, Any]] = {
    "tab:perclass":         {"config": "eval/val_1step",     "checkpoint": "gssc_mf/gssc_31k_mf_step40000",  "metrics": "miou per_class completion_iou"},
    "tab:main_results":     {"config": "infer/test_d4tta",   "checkpoint": "gssc_mf/gssc_31k_mf_step40000",  "metrics": "miou per_class completion_iou", "submit": True},
    "tab:safety_metrics":   {"config": "eval/val_1step",     "checkpoint": "gssc_mf/gssc_31k_mf_step40000",  "metrics": "safety"},
    "tab:dwiou":            {"config": "eval/val_1step",     "checkpoint": "gssc_mf/gssc_31k_mf_step40000",  "metrics": "dwiou"},
    "tab:step_reduction":   {"config": "eval/step_sweep",    "checkpoint": "gssc_mf/gssc_31k_mf_step40000",  "metrics": "miou completion_iou"},
    "tab:train_timesteps_curriculum": {"config": "eval/timestep_ablation", "checkpoint": "[gssc_timesteps/gssc_T10|gssc_timesteps/gssc_T50|gssc_timesteps/gssc_T100skewed|gssc_mf/gssc_31k_mf_step40000]", "metrics": "miou"},
    "tab:loss_ablation":    {"config": "eval/loss_ablation", "checkpoint": "[per-row variant]", "metrics": "miou"},
    "tab:s2d2_ablation":    {"config": "eval/s2d2_ablation", "checkpoint": "[per-row variant]", "metrics": "miou"},
    "tab:bev_results":      {"config": "eval/bev_secondary", "checkpoint": "bev/bev_perception_net",         "metrics": "miou"},
    "tab:data_scaling":     {"config": "eval/data_scaling_sf", "checkpoint": "[gssc_sf/gssc_{0K,10K,20K,31K,57K}_sf_step100000]", "metrics": "miou"},
    "tab:cross_base_js3c":  {
        "config": "eval/js3c_val_1step",
        "checkpoint": "gssc_js3c/gssc_js3c_s2d2_real",
        "metrics": "miou per_class completion_iou",
        "base_pred_dir_required": "data/js3cnet_predictions",
        "expected_mIoU": 26.72,
    },
}


def _check_js3c_predictions(required_dir: Path) -> None:
    """Pre-flight: ensure JS3C-Net predictions are dumped before cross-base eval.

    The cross-base eval needs ``data/js3cnet_predictions/08/<frame>_pred.npy``.
    If the directory is missing or empty, print the exact dumper command and
    exit non-zero rather than crashing inside the dataloader.
    """
    val_seq = required_dir / "08"
    if not val_seq.is_dir() or not any(val_seq.glob("*_pred.npy")):
        print("=" * 78)
        print("Missing JS3C-Net predictions for the cross-base reproduction.")
        print("Expected at least one *_pred.npy file under:")
        print(f"   {val_seq}")
        print()
        print("Reproduce by running the dumper against your JS3C-Net clone:")
        print()
        print("   git clone --depth 1 https://github.com/yangyangyang127/JS3C-Net external/JS3C-Net")
        print("   bash external/JS3C-Net/download_pretrained.sh")
        print("   python scripts/dump_js3c_predictions.py \\")
        print("       --js3c-repo external/JS3C-Net \\")
        print("       --semantickitti_root data/SemanticKITTI/dataset \\")
        print(f"       --output_dir {required_dir} \\")
        print("       --sequences 08")
        print()
        print("Then re-run this script. See docs/REPRODUCIBILITY.md for the full")
        print("cross-base protocol.")
        print("=" * 78)
        sys.exit(2)


def _resolve_checkpoint(ckpt_dir: Path, name: str, prefer_ema: bool = True) -> Path:
    """Resolve a checkpoint subdir to its preferred .safetensors file.

    Layout (v1.1.0+)::

        <ckpt_dir>/<group>/<name>/
            model.safetensors        # training weights
            model_ema.safetensors    # deployment weights (paper convention)
            config.json
    """
    subdir = ckpt_dir / name
    ema = subdir / "model_ema.safetensors"
    plain = subdir / "model.safetensors"
    if prefer_ema and ema.is_file():
        return ema
    if plain.is_file():
        return plain
    raise FileNotFoundError(
        f"checkpoint missing: expected {ema} or {plain}; "
        "run `python scripts/download_assets.py --checkpoints`"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("table", choices=list(TABLE_MAP), help="Table label from the paper")
    parser.add_argument("--checkpoints-dir", default=str(REPO_ROOT / "data" / "checkpoints"))
    parser.add_argument("--data-root", default=str(REPO_ROOT / "data"))
    args = parser.parse_args()

    spec = TABLE_MAP[args.table]
    ckpt_dir = Path(args.checkpoints_dir)
    data_root = Path(args.data_root)

    print(f"=== Reproducing {args.table} ===")
    print(f"  config:     {spec['config']}")
    print(f"  checkpoint: {spec['checkpoint']}")
    print(f"  metrics:    {spec['metrics']}")
    if "expected_mIoU" in spec:
        print(f"  expected:   {spec['expected_mIoU']}% mIoU (paper)")
    print()

    if "base_pred_dir_required" in spec:
        _check_js3c_predictions(data_root.parent / spec["base_pred_dir_required"]
                                if not Path(spec["base_pred_dir_required"]).is_absolute()
                                else Path(spec["base_pred_dir_required"]))

    if "[" in spec["checkpoint"]:
        print("Multi-row table. Each row variant must be run with its own checkpoint.")
        print("See docs/MODEL_ZOO.md for the row-by-row checkpoint list.")
        return

    ckpt_path = _resolve_checkpoint(ckpt_dir, spec["checkpoint"])
    cmd = [
        sys.executable, "scripts/eval.py", spec["config"],
        "--checkpoint", str(ckpt_path),
        "--metrics", *spec["metrics"].split(),
    ]
    print(f"$ {' '.join(cmd)}")
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


if __name__ == "__main__":
    main()
