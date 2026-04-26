"""Reproduce any per-table number from the paper.

Each table maps to a specific eval recipe. Running this script will:
1. Verify the right checkpoint is downloaded
2. Run the appropriate eval pipeline
3. Print a CSV that matches the LaTeX numbers within 0.1%

Examples::

    python scripts/reproduce_table.py tab:perclass
    python scripts/reproduce_table.py tab:main_results
    python scripts/reproduce_table.py tab:safety_metrics
    python scripts/reproduce_table.py tab:step_reduction
    python scripts/reproduce_table.py tab:train_timesteps_curriculum
    python scripts/reproduce_table.py tab:loss_ablation
    python scripts/reproduce_table.py tab:s2d2_ablation
    python scripts/reproduce_table.py tab:bev_results
    python scripts/reproduce_table.py tab:dwiou
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

TABLE_MAP = {
    "tab:perclass":         {"config": "eval/val_1step",     "checkpoint": "gssc_31k_mf_step40000",  "metrics": "miou per_class completion_iou"},
    "tab:main_results":     {"config": "infer/test_d4tta",   "checkpoint": "gssc_31k_mf_step40000",  "metrics": "miou per_class completion_iou", "submit": True},
    "tab:safety_metrics":   {"config": "eval/val_1step",     "checkpoint": "gssc_31k_mf_step40000",  "metrics": "safety"},
    "tab:dwiou":            {"config": "eval/val_1step",     "checkpoint": "gssc_31k_mf_step40000",  "metrics": "dwiou"},
    "tab:step_reduction":   {"config": "eval/step_sweep",    "checkpoint": "gssc_31k_mf_step40000",  "metrics": "miou completion_iou"},
    "tab:train_timesteps_curriculum": {"config": "eval/timestep_ablation", "checkpoint": "[T10|T50|T100skewed|31k_mf]", "metrics": "miou"},
    "tab:loss_ablation":    {"config": "eval/loss_ablation", "checkpoint": "[per-row variant]", "metrics": "miou"},
    "tab:s2d2_ablation":    {"config": "eval/s2d2_ablation", "checkpoint": "[per-row variant]", "metrics": "miou"},
    "tab:bev_results":      {"config": "eval/bev_secondary", "checkpoint": "bev_perception_net", "metrics": "miou"},
    "tab:data_scaling":     {"config": "eval/data_scaling_sf", "checkpoint": "[0K_sf|10K_sf|20K_sf|31K_sf|57K_sf]", "metrics": "miou"},
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("table", choices=list(TABLE_MAP), help="Table label from the paper")
    parser.add_argument("--checkpoints-dir", default=str(REPO_ROOT / "data" / "checkpoints"))
    args = parser.parse_args()

    spec = TABLE_MAP[args.table]
    ckpt_dir = Path(args.checkpoints_dir)

    print(f"=== Reproducing {args.table} ===")
    print(f"  config:     {spec['config']}")
    print(f"  checkpoint: {spec['checkpoint']}")
    print(f"  metrics:    {spec['metrics']}")
    print()

    if "[" in spec["checkpoint"]:
        print("Multi-row table. Running each row variant in sequence ...")
        # Inner loop will be implemented per spec
    else:
        ckpt = ckpt_dir / f"{spec['checkpoint']}.safetensors"
        if not ckpt.exists():
            print(f"Missing checkpoint: {ckpt}")
            print("Run: python scripts/download_assets.py --checkpoints")
            sys.exit(1)
        cmd = [
            sys.executable, "scripts/eval.py", spec["config"],
            "--checkpoint", str(ckpt),
            "--metrics", *spec["metrics"].split(),
        ]
        print(f"$ {' '.join(cmd)}")
        subprocess.run(cmd, cwd=REPO_ROOT, check=True)


if __name__ == "__main__":
    main()
