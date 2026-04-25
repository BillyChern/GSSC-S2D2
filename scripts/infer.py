"""GSSC-S2D2 inference entry point - generate .label predictions for SemanticKITTI evaluation server."""
from __future__ import annotations
import argparse, sys, subprocess
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))


def main():
    p = argparse.ArgumentParser(description="GSSC-S2D2 inference (.label generator).")
    p.add_argument("config", help="Hydra config: infer/val_d4tta, infer/test_d4tta, infer/val_1step")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True, help="Where .label predictions land")
    p.add_argument("--data-root", default=str(REPO_ROOT / "data"))
    p.add_argument("--gpu", default="0")
    p.add_argument("--steps", type=int, default=4, help="Algo2 correction steps (1, 4, 100)")
    p.add_argument("--tta", choices=["none", "flip_y", "d4"], default="d4")
    a = p.parse_args()

    if a.tta == "d4":
        cmd = [sys.executable, "-m", "gssc.inference.d4_tta",
               "--checkpoint", a.checkpoint,
               "--cold_steps", str(a.steps),
               "--output_dir", a.output]
    else:
        cmd = [sys.executable, "-m", "gssc.inference.generate_predictions",
               "--checkpoint", a.checkpoint,
               "--cold_steps", str(a.steps),
               "--output_dir", a.output,
               "--algo2"]
    print(f"$ {' '.join(cmd)}")
    import os
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = a.gpu
    env["PYTHONPATH"] = f"{REPO_ROOT / 'src'}:{env.get('PYTHONPATH', '')}"
    subprocess.run(cmd, env=env, check=True)


if __name__ == "__main__":
    main()
