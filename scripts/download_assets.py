"""Download pretrained checkpoints + reference datasets from Hugging Face Hub.

Usage::

    python scripts/download_assets.py --checkpoints          # ~4 GB models
    python scripts/download_assets.py --predictions          # ~178 GB SCPNet val/test/synth predictions
    python scripts/download_assets.py --js3c-predictions     # ~190 GB JS3C-Net cross-base predictions
    python scripts/download_assets.py --lmscnet-predictions  # ~46 GB LMSCNet cross-base predictions
    python scripts/download_assets.py --object-bank          # ~448 MB rare-class object bank
    python scripts/download_assets.py --synthetic-pool 31K   # ~120 GB headline synth pool
    python scripts/download_assets.py --all                  # everything EXCEPT the synthetic pool (~4 GB models + ~414 GB predictions; see docs/DATASET.md)

Checkpoints, base-model predictions and the object bank resolve against the two
Hugging Face mirrors below. The synthetic pool is archived separately on IEEE
DataPort; until its DOI is filled in, ``--synthetic-pool`` exits with the
manual-build instructions in docs/DATASET.md / docs/REPRODUCIBILITY.md.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

HF_REPO_MODELS = "BillyChern/GSSC-S2D2-checkpoints"
HF_REPO_DATA = "BillyChern/GSSC-S2D2-datasets"
# The synthetic pool (~128-230 GB) is archived on IEEE DataPort rather than the two
# Hugging Face mirrors above. Fill in the DOI the archive is minted under.
DATAPORT_URL = "[SYNTHETIC_POOL_URL]"

logger = logging.getLogger("gssc.download")


def _ensure_url_configured(url: str, label: str) -> None:
    """Bail out early when the asset URL is still a placeholder.

    Direct visitors at the manual-download docs rather than failing inside
    huggingface_hub with a confusing 'Repository not found'.
    """
    if url.startswith("[") and url.endswith("]"):
        sys.exit(
            f"\n{label} URL is not yet configured (placeholder: {url}).\n"
            "Build it locally instead:\n"
            "  - Manual instructions:    docs/DATASET.md\n"
            "  - Reproducibility guide:  docs/REPRODUCIBILITY.md\n"
            "  - Issues:                 https://github.com/BillyChern/GSSC-S2D2/issues\n"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", action="store_true", help="Download model checkpoints (~4 GB)")
    parser.add_argument("--predictions", action="store_true", help="Download SCPNet predictions (~178 GB, real + synth)")
    parser.add_argument("--js3c-predictions", action="store_true", help="Download JS3C-Net predictions (~190 GB, cross-base eval)")
    parser.add_argument("--lmscnet-predictions", action="store_true", help="Download LMSCNet predictions (~46 GB, cross-base eval)")
    parser.add_argument("--object-bank", action="store_true", help="Download rare-class object bank (~448 MB)")
    parser.add_argument("--synthetic-pool", choices=["0K", "10K", "20K", "31K", "57K"],
                        default=None, help="Download a synthetic pool variant")
    parser.add_argument("--all", action="store_true", help="Download everything EXCEPT the synthetic pool (~4 GB models + ~414 GB predictions [SCPNet 178 + JS3C 190 + LMSCNet 46]; see docs/DATASET.md disk-space table). The synthetic pool is opt-in via --synthetic-pool because it is only needed to retrain from scratch.")
    parser.add_argument("--root", default=str(REPO_ROOT / "data"), help="Where to store downloads")
    args = parser.parse_args()

    if not any([args.checkpoints, args.predictions, args.js3c_predictions, args.lmscnet_predictions, args.object_bank, args.synthetic_pool, args.all]):
        parser.print_help()
        sys.exit(0)

    # Validate that the hosting URLs for every requested asset group are
    # configured BEFORE attempting the huggingface_hub import, so a user
    # probing the script in a bare (pre-`uv sync`) environment gets the
    # documented "see docs/DATASET.md" pointer rather than a misleading
    # "Install huggingface_hub" message.
    if args.checkpoints or args.all:
        _ensure_url_configured(HF_REPO_MODELS, "Checkpoints")
    if args.predictions or args.all:
        _ensure_url_configured(HF_REPO_DATA, "Datasets (predictions)")
    if args.js3c_predictions or args.all:
        _ensure_url_configured(HF_REPO_DATA, "Datasets (JS3C-Net predictions)")
    if args.lmscnet_predictions or args.all:
        _ensure_url_configured(HF_REPO_DATA, "Datasets (LMSCNet predictions)")
    if args.object_bank or args.all:
        _ensure_url_configured(HF_REPO_DATA, "Datasets (object bank)")
    if args.synthetic_pool:
        _ensure_url_configured(DATAPORT_URL, "Synthetic pool (IEEE DataPort)")

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        sys.exit(
            "huggingface_hub is required for downloads. Install it with:\n"
            "  uv pip install huggingface-hub\n"
            "or install the project with its dependencies:\n"
            "  pip install -e ."
        )

    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s")

    if args.checkpoints or args.all:
        logger.info("Downloading checkpoints from %s ...", HF_REPO_MODELS)
        snapshot_download(repo_id=HF_REPO_MODELS, local_dir=root / "checkpoints")
    # snapshot_download preserves the matched pattern prefix in the output tree,
    # so the local_dir must be `root` (not `root / "<name>"`) or the files land
    # at root/<name>/<name>/... (double-nested). The allow_patterns prefix is the
    # single directory level we want.
    if args.predictions or args.all:
        logger.info("Downloading SCPNet predictions from %s ...", HF_REPO_DATA)
        snapshot_download(repo_id=HF_REPO_DATA, repo_type="dataset",
                          allow_patterns=["scpnet_predictions/*"],
                          local_dir=root)
    if args.js3c_predictions or args.all:
        logger.info("Downloading JS3C-Net predictions from %s ...", HF_REPO_DATA)
        snapshot_download(repo_id=HF_REPO_DATA, repo_type="dataset",
                          allow_patterns=["js3cnet_predictions/*"],
                          local_dir=root)
    if args.lmscnet_predictions or args.all:
        logger.info("Downloading LMSCNet predictions from %s ...", HF_REPO_DATA)
        snapshot_download(repo_id=HF_REPO_DATA, repo_type="dataset",
                          allow_patterns=["lmscnet_predictions/*"],
                          local_dir=root)
    if args.object_bank or args.all:
        logger.info("Downloading object bank from %s ...", HF_REPO_DATA)
        snapshot_download(repo_id=HF_REPO_DATA, repo_type="dataset",
                          allow_patterns=["object_bank/*"],
                          local_dir=root)
    if args.synthetic_pool:
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
