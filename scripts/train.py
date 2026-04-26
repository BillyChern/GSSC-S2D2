"""GSSC-S2D2 training entry point.

Hydra-driven dispatcher to the underlying trainers. Every reproducible recipe
in the paper has a corresponding YAML under ``configs/train/``.

Examples
--------
Reproduce the headline 31K-MF run that yields 38.54% val mIoU and 39.2% test
mIoU (with D4 TTA at inference)::

    python scripts/train.py +config=train/31k_mf

Single-frame retrains (Tab. VII data-scaling row)::

    python scripts/train.py +config=train/0K_sf
    python scripts/train.py +config=train/10K_sf
    python scripts/train.py +config=train/20K_sf
    python scripts/train.py +config=train/31K_sf
    python scripts/train.py +config=train/57K_sf

Training-timestep ablation rows (Tab. XII)::

    python scripts/train.py +config=train/T10
    python scripts/train.py +config=train/T50
    python scripts/train.py +config=train/T100skewed
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="GSSC-S2D2 training entry point.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "config",
        type=str,
        help="Hydra-style config name, e.g. train/31k_mf",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=str(REPO_ROOT / "data"),
        help="SemanticKITTI dataset root (default: ./data).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Where to write checkpoints + logs (default: ./outputs/<config-name>).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42, matching paper headline).",
    )
    parser.add_argument(
        "--gpu",
        type=str,
        default="0",
        help="CUDA device id (default: 0). Multi-GPU: comma-separated, e.g. 0,1.",
    )
    # Use parse_known_args so the named flags (--gpu, --seed, ...) are picked
    # up wherever they appear and the leftovers are forwarded to the trainer.
    args, remainder = parser.parse_known_args()

    # Translate config alias to the underlying trainer command.
    cfg_path = REPO_ROOT / "configs" / f"{args.config}.yaml"
    if not cfg_path.exists():
        raise SystemExit(
            f"Config not found: {cfg_path}\n"
            f"Available: {sorted(p.relative_to(REPO_ROOT / 'configs') for p in (REPO_ROOT / 'configs').rglob('*.yaml'))}"
        )

    out_dir = Path(args.output_dir or REPO_ROOT / "outputs" / args.config.replace("/", "_"))
    out_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.gpu
    env["PYTHONPATH"] = f"{REPO_ROOT / 'src'}:{env.get('PYTHONPATH', '')}"

    # Import here so ``--help`` doesn't pay the import cost.
    from gssc.utils.config_loader import load_yaml_to_args

    forwarded = load_yaml_to_args(cfg_path)
    # Override the YAML's output_dir / data_root with the CLI flags. Done after
    # load_yaml_to_args so user CLI flags take precedence.
    forwarded.extend(["--output_dir", str(out_dir)])
    forwarded.extend(["--data_root", args.data_root])
    forwarded.extend(["--seed", str(args.seed)])
    forwarded.extend(remainder)

    cmd = [sys.executable, "-m", "gssc.training.train_scene_completion", *forwarded]
    print(f"$ {' '.join(cmd)}")
    subprocess.run(cmd, env=env, check=True)


if __name__ == "__main__":
    main()
