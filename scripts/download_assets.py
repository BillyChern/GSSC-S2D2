"""Download pretrained checkpoints + reference datasets from Hugging Face Hub.

Usage::

    python scripts/download_assets.py --checkpoints          # 4.58 GiB / 4.9 GB models
    python scripts/download_assets.py --predictions          # 177 GiB / 190 GB SCPNet real+synth predictions
    python scripts/download_assets.py --js3c-predictions     # 189 GiB / 203 GB JS3C-Net cross-base predictions
    python scripts/download_assets.py --lmscnet-predictions  # 45 GiB / 49 GB LMSCNet cross-base predictions
    python scripts/download_assets.py --object-bank          # 313 MiB / 328 MB rare-class object bank
    python scripts/download_assets.py --synthetic-pool 31K   # PRINTS how to get the 127 GiB / 136 GB headline synth pool; cannot fetch it
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
DataPort under doi:10.21227/nqgf-9k39. That DOI is minted and resolves, but a live
DOI does not make the archives fetchable: DataPort serves dataset files only to an
authenticated session and publishes no direct file URL, so this script cannot
retrieve the pool. ``--synthetic-pool`` therefore prints the DOI, the archive names
and the manual retrieval steps, then exits non-zero with the manual-build
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
# on IEEE DataPort rather than the two Hugging Face mirrors above. The DOI was minted on
# 2026-08-23 and resolves, so this is no longer a [PLACEHOLDER] -- but filling it in did NOT
# make the pool downloadable, and the reason `--synthetic-pool` still refuses has simply
# changed. IEEE DataPort serves dataset files only to an authenticated session: the landing
# page renders both archives as login modals rather than as hrefs, so there is no file URL to
# fetch and no anonymous request of any shape succeeds. The graceful exit that used to come
# from _ensure_url_configured spotting a placeholder now lives in the `--synthetic-pool`
# branch of main(), which owns that mode unconditionally.
# The pool is NOT embargoed until publication, so do NOT describe it that way in the docs:
# what stands between a reader and the archive is IEEE's access gate, not our release date.
DATAPORT_URL = "https://dx.doi.org/10.21227/nqgf-9k39"

#: The two released archives on the deposit. Sizes are the `GiB / GB` pair this repo quotes
#: everywhere; note that DataPort's own file list labels them "2.15 GB" and "3.88 GB", which
#: are the GiB figures under a GB label. 31K is a STRICT SUBSET of 57K -- nobody needs both.
DATAPORT_ARCHIVES = {
    "31K": ("synthetic_pool_31K.tar.gz", "2.15 GiB / 2.31 GB", "32,039"),
    "57K": ("synthetic_pool_57K.tar.gz", "3.88 GiB / 4.16 GB", "57,650"),
}

logger = logging.getLogger("gssc.download")


#: The three places a reader can get an asset when this script cannot fetch it. Every exit
#: path in this file ends here, so no failure mode can leave the user without a next step.
_MANUAL_ROUTES = (
    "  - Manual instructions:    docs/DATASET.md\n"
    "  - Reproducibility guide:  docs/REPRODUCIBILITY.md\n"
    "  - Issues:                 https://github.com/BillyChern/GSSC-S2D2/issues\n"
)


def _ensure_url_configured(url: str, label: str) -> None:
    """Bail out early when an asset URL is still a `[PLACEHOLDER]` token.

    Direct visitors at the manual-download docs rather than failing inside
    huggingface_hub with a confusing 'Repository not found'.

    THIS IS NOW A REGRESSION GUARD, NOT THE LIVE MECHANISM FOR ANY MODE. Until the
    DataPort DOI was minted, `DATAPORT_URL` was the one placeholder in the file and this
    function was what made `--synthetic-pool` exit gracefully. It no longer is: every URL
    here is real, so no call below can currently trip, and the graceful exit for
    `--synthetic-pool` is the explicit `sys.exit` in that mode's own branch. The function
    is kept because it is cheap and because the failure it catches -- a URL edited back
    into a placeholder, or a new asset group added with its host still to be decided --
    is one a reader would otherwise meet as a huggingface_hub traceback.
    """
    if url.startswith("[") and url.endswith("]"):
        sys.exit(
            f"\n{label} URL is not yet configured (placeholder: {url}).\n"
            "Build it locally instead:\n" + _MANUAL_ROUTES
        )


def _synthetic_pool_notice(root: Path, variant: str) -> str:
    """What `--synthetic-pool` exits with. It cannot fetch, so it must instruct.

    DELIBERATELY NOT A COPY-PASTEABLE FETCH. This branch used to print

        wget -O <root>/synthetic_pool_31K.tar.gz <DATAPORT_URL>/synthetic_pool_31K.tar.gz

    which was wrong twice over even once a real DOI was substituted for the placeholder.
    IEEE DataPort publishes no ``<landing-page>/<filename>`` URL -- the deposit page renders
    each archive as a login modal, never as an href -- and it serves files only to an
    authenticated session, so no anonymous request of any shape can succeed. A command that
    404s is worse than no command: it reads as a supported path and fails only after the
    reader has trusted it. Hence prose steps, with ``tar -xzf`` given separately because
    extraction is the one half the user actually runs locally.
    """
    name, size, scenes = DATAPORT_ARCHIVES[variant]
    other = "57K" if variant == "31K" else "31K"
    return (
        f"\nSynthetic pool '{variant}' is archived on IEEE DataPort, which this script "
        f"cannot download from.\n"
        f"  DOI:      doi:10.21227/nqgf-9k39\n"
        f"  Landing:  {DATAPORT_URL}\n"
        f"  Archive:  {name}  ({size} compressed, {scenes} scenes)\n\n"
        "IEEE DataPort serves dataset files only to a signed-in session, and it exposes no "
        "direct file URL to fetch. As deposited, downloading the archives requires an IEEE "
        "DataPort subscription (IEEE Society membership included); check the access notice "
        "on the landing page before planning around it. Retrieve it by hand:\n"
        f"  1. Open {DATAPORT_URL} in a browser and sign in.\n"
        f"  2. Download {name} from the dataset's file list.\n"
        "  3. Extract it where this script would have put it:\n"
        f"       tar -xzf {name} -C {root}/\n\n"
        f"The {other} pool is the other released variant, and 31K is a strict subset of 57K "
        "-- take one, never both.\n"
        "No IEEE credentials? Rebuilding the pool locally with the PS3 ray-tracer needs none, "
        "and is documented with everything else here:\n" + _MANUAL_ROUTES
    )


def _fetch(snapshot_download, label: str, repo_id: str, **kwargs) -> None:
    """One snapshot_download, with every failure turned into the documented pointer.

    ``_ensure_url_configured`` guards exactly one shape of unavailability -- a
    ``[PLACEHOLDER]`` URL -- and nothing in this file has that shape any more. The two
    Hugging Face repo ids are real-LOOKING strings, so they sail past that guard and any
    problem reaching them (repo missing, gated, network down, no auth token, hub API change)
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
    parser.add_argument("--object-bank", action="store_true", help="Download rare-class object bank (313 MiB / 328 MB of data; 448 MiB / 470 MB of disk blocks)")
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

    # The pool is the one group nothing on Hugging Face can serve, so its mode must not be
    # made to depend on huggingface_hub. Answering it BEFORE the import below keeps
    # `--synthetic-pool` working in a bare (pre-`uv sync`) environment -- exactly the
    # property the comment above says the early URL validation exists to preserve. Without
    # this, a reader with no huggingface_hub installed was told to `uv pip install
    # huggingface-hub` for an asset that Hugging Face does not host. Requested alongside a
    # Hugging Face group, it falls through and is reported after those downloads instead.
    if args.synthetic_pool and not any([args.checkpoints, args.predictions,
                                        args.js3c_predictions, args.lmscnet_predictions,
                                        args.object_bank, args.all]):
        sys.exit(_synthetic_pool_notice(Path(args.root), args.synthetic_pool))

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
    # Reached only when the pool was requested ALONGSIDE a Hugging Face group (the
    # pool-only case exited above, before the import). Those groups are done; the pool
    # still cannot be provisioned, so the run is not a success and must not exit 0.
    if args.synthetic_pool:
        sys.exit(_synthetic_pool_notice(root, args.synthetic_pool))

    logger.info("Done.")


if __name__ == "__main__":
    main()
