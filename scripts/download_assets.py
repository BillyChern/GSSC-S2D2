"""Download pretrained checkpoints + reference datasets from Hugging Face Hub.

Usage::

    python scripts/download_assets.py --checkpoints          # 4.58 GiB / 4.9 GB models
    python scripts/download_assets.py --predictions          # 177 GiB / 190 GB SCPNet real+synth predictions
    python scripts/download_assets.py --js3c-predictions     # 189 GiB / 203 GB JS3C-Net cross-base predictions
    python scripts/download_assets.py --lmscnet-predictions  # 45 GiB / 49 GB LMSCNet cross-base predictions
    python scripts/download_assets.py --object-bank          # 313 MiB / 328 MB rare-class object bank
    python scripts/download_assets.py --synthetic-pool 31K   # 127 GiB / 136 GB headline synth pool
    python scripts/download_assets.py --all                  # everything EXCEPT the synthetic pool (4.9 GB models + ~442 GB predictions [SCPNet 190 + JS3C 203 + LMSCNet 49] + 0.33 GB object bank; see docs/DATASET.md)

Every size here is the `GiB / GB` pair measured on the staged release payload and
tabulated in docs/DATASET.md's disk-space table -- keep the two in step. The
SCPNet figure is UNIQUE content: three of its `synthetic*` farms are symlinks
into a fourth, so an upload that materialises links expands to 324 GiB / 348 GB
(docs/DATASET.md explains which figure to size a transfer against).

Prediction groups are whole-prefix downloads by default. To take a subset -- one
sequence, or the handful of frames the quickstart notebook needs -- narrow the fetch
with ``--include``, whose patterns are passed straight through to
``huggingface_hub.snapshot_download(allow_patterns=...)`` over the same repo tree
docs/DATASET.md documents::

    # val seq 08 only (8.5 GiB / 9.1 GB), instead of the whole 177 GiB / 190 GB prefix
    python scripts/download_assets.py --predictions --include 'scpnet_predictions/08/*'

    # val 08 + the hidden test sequences, for a leaderboard submission (16.6 GiB / 17.8 GB)
    python scripts/download_assets.py --predictions \
        --include 'scpnet_predictions/08/*' 'scpnet_predictions/1*/*' 'scpnet_predictions/2*/*'

    # just the frame examples/quickstart.ipynb uses
    python scripts/download_assets.py --predictions --include 'scpnet_predictions/08/000000_*'

``--include`` applies only to the prediction / object-bank groups (the checkpoint
group is a whole-repo snapshot); it is ignored with ``--all``, which is by definition
the complete fetch. The raw SemanticKITTI voxels the notebook also needs are NOT
hosted here -- they require registration at semantic-kitti.org (docs/DATASET.md).

Checkpoints, base-model predictions and the object bank resolve against the two
Hugging Face mirrors below. The synthetic pool is archived separately on IEEE
DataPort, and its DOI is the one asset URL in this release that does not exist
yet; until it is filled in, ``--synthetic-pool`` exits with the manual-build
instructions in docs/DATASET.md / docs/REPRODUCIBILITY.md. Nothing here waits on
paper acceptance: the paper's availability footnote already sends readers to
github.com/BillyChern/GSSC-S2D2 for the code, the pre-trained models and the
PS3 dataset.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

HF_REPO_MODELS = "Stone-Chern/GSSC-S2D2-checkpoints"
HF_REPO_DATA = "Stone-Chern/GSSC-S2D2-datasets"
# The synthetic pool (127 GiB / 136 GB for 31K, 229 GiB / 246 GB for 57K) is archived
# on IEEE DataPort rather than the two
# Hugging Face mirrors above. This stays a [PLACEHOLDER] because the archive's DOI has not
# been minted yet -- the pool is not embargoed until publication, so do NOT describe it that
# way in the docs. While it is a placeholder, _ensure_url_configured routes `--synthetic-pool`
# to the manual build in docs/DATASET.md; replace it with the real DOI once minted.
DATAPORT_URL = "[SYNTHETIC_POOL_URL]"

logger = logging.getLogger("gssc.download")


#: The three places a reader can get an asset when this script cannot fetch it. Every exit
#: path in this file ends here, so no failure mode can leave the user without a next step.
_MANUAL_ROUTES = (
    "  - Manual instructions:    docs/DATASET.md\n"
    "  - Reproducibility guide:  docs/REPRODUCIBILITY.md\n"
    "  - Issues:                 https://github.com/BillyChern/GSSC-S2D2/issues\n"
)


def _ensure_url_configured(url: str, label: str) -> None:
    """Bail out early when the asset URL is still a placeholder.

    Direct visitors at the manual-download docs rather than failing inside
    huggingface_hub with a confusing 'Repository not found'.
    """
    if url.startswith("[") and url.endswith("]"):
        sys.exit(
            f"\n{label} URL is not yet configured (placeholder: {url}).\n"
            "Build it locally instead:\n" + _MANUAL_ROUTES
        )


def _fetch(snapshot_download, label: str, repo_id: str, **kwargs) -> None:
    """One snapshot_download, with every failure turned into the documented pointer.

    ``_ensure_url_configured`` guards exactly one shape of unavailability -- a
    ``[PLACEHOLDER]`` URL -- and only ``DATAPORT_URL`` has that shape. The two Hugging
    Face repo ids are real-LOOKING strings, so they sail past that guard and any problem
    reaching them (repo missing, gated, network down, no auth token, hub API change)
    surfaced as a raw ``huggingface_hub`` traceback: precisely the outcome the guard's
    docstring, README.md, docs/MODEL_ZOO.md and examples/quickstart.ipynb all promise the
    user will not get. Catching ``BaseException`` rather than ``Exception`` is deliberate:
    huggingface_hub has shipped error classes built through ``__new__`` that do not
    inherit the usual way, and a fix that only covers the errors we predicted is the same
    defect one release later.
    """
    try:
        snapshot_download(repo_id=repo_id, **kwargs)
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException as exc:  # broad on purpose: see the docstring above
        reason = f"{type(exc).__name__}: {exc}".strip()
        sys.exit(
            f"\n{label}: could not download from the Hugging Face repo '{repo_id}'.\n"
            f"  reason: {reason.splitlines()[0][:300] if reason else 'unknown error'}\n"
            "Common causes: no network or a blocking proxy; the repo requires "
            "`huggingface-cli login`; or a stale huggingface_hub.\n"
            "Get the assets another way:\n" + _MANUAL_ROUTES
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", action="store_true", help="Download model checkpoints (4.58 GiB / 4.9 GB)")
    parser.add_argument("--predictions", action="store_true", help="Download SCPNet predictions (177 GiB / 190 GB unique, real + synth)")
    parser.add_argument("--js3c-predictions", action="store_true", help="Download JS3C-Net predictions (189 GiB / 203 GB, cross-base eval)")
    parser.add_argument("--lmscnet-predictions", action="store_true", help="Download LMSCNet predictions (45 GiB / 49 GB, cross-base eval)")
    parser.add_argument("--object-bank", action="store_true", help="Download rare-class object bank (313 MiB / 328 MB of data; 448 MB of disk blocks)")
    # Only 31K and 57K are staged for release. 0K means real-only, so no tarball can
    # ever exist for it; 10K and 20K exist locally but are not part of the release
    # surface. Offering a choice with nothing behind it is a promise the script
    # cannot keep, so the choices are the two that ship.
    parser.add_argument("--synthetic-pool", choices=["31K", "57K"],
                        default=None,
                        help="Download a released synthetic pool variant "
                             "(31K = 127 GiB / 136 GB, 57K = 229 GiB / 246 GB)")
    parser.add_argument("--all", action="store_true", help="Download everything EXCEPT the synthetic pool (4.9 GB models + ~442 GB predictions [SCPNet 190 + JS3C 203 + LMSCNet 49] + 0.33 GB object bank; see docs/DATASET.md disk-space table). The synthetic pool is opt-in via --synthetic-pool because it is only needed to retrain from scratch.")
    parser.add_argument("--root", default=str(REPO_ROOT / "data"), help="Where to store downloads")
    parser.add_argument(
        "--include", nargs="+", metavar="PATTERN", default=None,
        help="Restrict a prediction / object-bank download to these glob patterns "
             "(huggingface_hub allow_patterns over the dataset repo, whose tree matches "
             "the layout in docs/DATASET.md). Example: "
             "--predictions --include 'scpnet_predictions/08/*' fetches val seq 08 only "
             "(8.5 GiB / 9.1 GB) instead of the whole 177 GiB / 190 GB prefix. Ignored "
             "with --all and with --checkpoints.",
    )
    args = parser.parse_args()

    if not any([args.checkpoints, args.predictions, args.js3c_predictions, args.lmscnet_predictions, args.object_bank, args.synthetic_pool, args.all]):
        parser.print_help()
        sys.exit(0)

    if args.include and args.all:
        logger.warning("--include is ignored with --all (which is the complete fetch).")
    if args.include and not any([args.predictions, args.js3c_predictions,
                                 args.lmscnet_predictions, args.object_bank]):
        sys.exit("--include only applies to --predictions / --js3c-predictions / "
                 "--lmscnet-predictions / --object-bank.")

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

    def _patterns(prefix: str) -> list[str]:
        """allow_patterns for one asset prefix, narrowed by --include when given.

        Without --include this is the whole prefix (the historical behaviour). With
        it, each user pattern that already names the prefix is passed through as-is
        and anything else is anchored under it, so both
        ``--include 'scpnet_predictions/08/*'`` and ``--include '08/*'`` work.
        """
        if not args.include or args.all:
            return [f"{prefix}/*"]
        out = []
        for pat in args.include:
            pat = pat.lstrip("/")
            out.append(pat if pat.startswith(f"{prefix}/") else f"{prefix}/{pat}")
        logger.info("Restricting %s to %s", prefix, out)
        return out

    if args.checkpoints or args.all:
        logger.info("Downloading checkpoints from %s ...", HF_REPO_MODELS)
        _fetch(snapshot_download, "Checkpoints", HF_REPO_MODELS,
               local_dir=root / "checkpoints")
    # snapshot_download preserves the matched pattern prefix in the output tree,
    # so the local_dir must be `root` (not `root / "<name>"`) or the files land
    # at root/<name>/<name>/... (double-nested). The allow_patterns prefix is the
    # single directory level we want.
    if args.predictions or args.all:
        logger.info("Downloading SCPNet predictions from %s ...", HF_REPO_DATA)
        _fetch(snapshot_download, "Datasets (predictions)", HF_REPO_DATA,
               repo_type="dataset", allow_patterns=_patterns("scpnet_predictions"),
               local_dir=root)
    if args.js3c_predictions or args.all:
        logger.info("Downloading JS3C-Net predictions from %s ...", HF_REPO_DATA)
        _fetch(snapshot_download, "Datasets (JS3C-Net predictions)", HF_REPO_DATA,
               repo_type="dataset", allow_patterns=_patterns("js3cnet_predictions"),
               local_dir=root)
    if args.lmscnet_predictions or args.all:
        logger.info("Downloading LMSCNet predictions from %s ...", HF_REPO_DATA)
        _fetch(snapshot_download, "Datasets (LMSCNet predictions)", HF_REPO_DATA,
               repo_type="dataset", allow_patterns=_patterns("lmscnet_predictions"),
               local_dir=root)
    if args.object_bank or args.all:
        logger.info("Downloading object bank from %s ...", HF_REPO_DATA)
        _fetch(snapshot_download, "Datasets (object bank)", HF_REPO_DATA,
               repo_type="dataset", allow_patterns=_patterns("object_bank"),
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
