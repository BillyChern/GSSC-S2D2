"""Download pretrained checkpoints + reference datasets from Hugging Face Hub.

Usage::

    python scripts/download_assets.py --checkpoints     # ~3 GB models
    python scripts/download_assets.py --predictions     # ~50 GB SCPNet val/test predictions
    python scripts/download_assets.py --object-bank     # ~448 MB rare-class object bank
    python scripts/download_assets.py --synthetic-pool 31K   # ~120 GB headline synth pool
    python scripts/download_assets.py --all              # everything

Hugging Face host: [HF_ORG_URL]

The full 230 GB synthetic pool is mirrored to IEEE DataPort; see
docs/DATASET.md for direct download links.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

HF_REPO_MODELS = "[HF_REPO_CHECKPOINTS]"
HF_REPO_DATA = "[HF_REPO_DATASETS]"
DATAPORT_URL = "[SYNTHETIC_POOL_URL]"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", action="store_true", help="Download model checkpoints (~3 GB)")
    parser.add_argument("--predictions", action="store_true", help="Download SCPNet predictions (~50 GB)")
    parser.add_argument("--object-bank", action="store_true", help="Download rare-class object bank (~448 MB)")
    parser.add_argument("--synthetic-pool", choices=["0K", "10K", "20K", "31K", "57K"],
                        default=None, help="Download a synthetic pool variant")
    parser.add_argument("--all", action="store_true", help="Download everything (~3 GB models + ~50 GB predictions)")
    parser.add_argument("--root", default=str(REPO_ROOT / "data"), help="Where to store downloads")
    args = parser.parse_args()

    if not any([args.checkpoints, args.predictions, args.object_bank, args.synthetic_pool, args.all]):
        parser.print_help()
        sys.exit(0)

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        sys.exit("Install huggingface_hub: pip install huggingface_hub")

    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)

    if args.checkpoints or args.all:
        print(f"Downloading checkpoints from {HF_REPO_MODELS} ...")
        snapshot_download(repo_id=HF_REPO_MODELS, local_dir=root / "checkpoints", local_dir_use_symlinks=False)
    if args.predictions or args.all:
        print(f"Downloading SCPNet predictions from {HF_REPO_DATA} ...")
        snapshot_download(repo_id=HF_REPO_DATA, repo_type="dataset",
                          allow_patterns=["scpnet_predictions/*"],
                          local_dir=root / "scpnet_predictions", local_dir_use_symlinks=False)
    if args.object_bank or args.all:
        print(f"Downloading object bank from {HF_REPO_DATA} ...")
        snapshot_download(repo_id=HF_REPO_DATA, repo_type="dataset",
                          allow_patterns=["object_bank/*"],
                          local_dir=root / "object_bank", local_dir_use_symlinks=False)
    if args.synthetic_pool:
        print(f"Synthetic pool '{args.synthetic_pool}' is hosted on IEEE DataPort.")
        print(f"  → {DATAPORT_URL}")
        print("Direct command:")
        print(f"  wget -O {root}/synthetic_pool_{args.synthetic_pool}.tar.gz {DATAPORT_URL}/synthetic_pool_{args.synthetic_pool}.tar.gz")
        print(f"  tar -xzf {root}/synthetic_pool_{args.synthetic_pool}.tar.gz -C {root}/")

    print("Done.")


if __name__ == "__main__":
    main()
