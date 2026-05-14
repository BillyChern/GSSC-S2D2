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
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

HF_REPO_MODELS = "[HF_REPO_CHECKPOINTS]"
HF_REPO_DATA = "[HF_REPO_DATASETS]"
DATAPORT_URL = "[SYNTHETIC_POOL_URL]"

logger = logging.getLogger("gssc.download")


def _ensure_url_configured(url: str, label: str) -> None:
    """Bail out early when the asset URL is still a placeholder.

    The hosted mirrors come online on paper acceptance; until then,
    direct visitors at the manual-download docs instead of failing
    inside huggingface_hub with a confusing 'Repository not found'.
    """
    if url.startswith("[") and url.endswith("]"):
        sys.exit(
            f"\n{label} URL is not yet configured (placeholder: {url}).\n"
            "The hosted mirrors come online on paper acceptance. Until then:\n"
            "  - Manual instructions:    docs/DATASET.md\n"
            "  - Reproducibility guide:  docs/REPRODUCIBILITY.md\n"
            "  - Issues:                 https://github.com/BillyChern/GSSC-S2D2/issues\n"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", action="store_true", help="Download model checkpoints (~3 GB)")
    parser.add_argument("--predictions", action="store_true", help="Download SCPNet predictions (~178 GB, real + synth)")
    parser.add_argument("--js3c-predictions", action="store_true", help="Download JS3C-Net predictions (~54 GB, cross-base eval)")
    parser.add_argument("--object-bank", action="store_true", help="Download rare-class object bank (~448 MB)")
    parser.add_argument("--synthetic-pool", choices=["0K", "10K", "20K", "31K", "57K"],
                        default=None, help="Download a synthetic pool variant")
    parser.add_argument("--all", action="store_true", help="Download everything (~3 GB models + ~50 GB predictions)")
    parser.add_argument("--root", default=str(REPO_ROOT / "data"), help="Where to store downloads")
    args = parser.parse_args()

    if not any([args.checkpoints, args.predictions, args.js3c_predictions, args.object_bank, args.synthetic_pool, args.all]):
        parser.print_help()
        sys.exit(0)

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        sys.exit("Install huggingface_hub: pip install huggingface_hub")

    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s")

    if args.checkpoints or args.all:
        _ensure_url_configured(HF_REPO_MODELS, "Checkpoints")
        logger.info("Downloading checkpoints from %s ...", HF_REPO_MODELS)
        snapshot_download(repo_id=HF_REPO_MODELS, local_dir=root / "checkpoints", local_dir_use_symlinks=False)
    if args.predictions or args.all:
        _ensure_url_configured(HF_REPO_DATA, "Datasets (predictions)")
        logger.info("Downloading SCPNet predictions from %s ...", HF_REPO_DATA)
        snapshot_download(repo_id=HF_REPO_DATA, repo_type="dataset",
                          allow_patterns=["scpnet_predictions/*"],
                          local_dir=root / "scpnet_predictions", local_dir_use_symlinks=False)
    if args.js3c_predictions or args.all:
        _ensure_url_configured(HF_REPO_DATA, "Datasets (JS3C-Net predictions)")
        logger.info("Downloading JS3C-Net predictions from %s ...", HF_REPO_DATA)
        snapshot_download(repo_id=HF_REPO_DATA, repo_type="dataset",
                          allow_patterns=["js3cnet_predictions/*"],
                          local_dir=root / "js3cnet_predictions", local_dir_use_symlinks=False)
    if args.object_bank or args.all:
        _ensure_url_configured(HF_REPO_DATA, "Datasets (object bank)")
        logger.info("Downloading object bank from %s ...", HF_REPO_DATA)
        snapshot_download(repo_id=HF_REPO_DATA, repo_type="dataset",
                          allow_patterns=["object_bank/*"],
                          local_dir=root / "object_bank", local_dir_use_symlinks=False)
    if args.synthetic_pool:
        _ensure_url_configured(DATAPORT_URL, "Synthetic pool (IEEE DataPort)")
        logger.info("Synthetic pool '%s' is hosted on IEEE DataPort.", args.synthetic_pool)
        logger.info("  -> %s", DATAPORT_URL)
        logger.info(
            "Direct: wget -O %s/synthetic_pool_%s.tar.gz %s/synthetic_pool_%s.tar.gz "
            "&& tar -xzf %s/synthetic_pool_%s.tar.gz -C %s/",
            root, args.synthetic_pool, DATAPORT_URL, args.synthetic_pool,
            root, args.synthetic_pool, root,
        )

    logger.info("Done.")


if __name__ == "__main__":
    main()
