"""Download pretrained checkpoints + reference datasets from Hugging Face Hub.

Usage::

    python scripts/download_assets.py --checkpoints          # 4.58 GiB / 4.9 GB models
    python scripts/download_assets.py --predictions          # 177 GiB / 190 GB SCPNet real+synth predictions
    python scripts/download_assets.py --js3c-predictions     # 189 GiB / 203 GB JS3C-Net cross-base predictions
    python scripts/download_assets.py --lmscnet-predictions  # 45 GiB / 49 GB LMSCNet cross-base predictions
    python scripts/download_assets.py --object-bank          # 313 MiB / 328 MB rare-class object bank
    python scripts/download_assets.py --synthetic-pool 31K   # 2.15 GiB / 2.31 GB archive -> the 127 GiB / 136 GB headline synth pool
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
group is a whole-repo snapshot, and ``--synthetic-pool`` is a single named archive);
it is ignored with ``--all``, which is by definition the complete fetch. The raw SemanticKITTI voxels the notebook also needs are NOT
hosted here -- they require registration at semantic-kitti.org (docs/DATASET.md).

Checkpoints, base-model predictions and the object bank resolve against the two
Hugging Face mirrors below. The synthetic pool is a separately citable artefact with
its own DOI, and the two are different questions: CITE doi:10.21227/nqgf-9k39
(IEEE DataPort, *PS3-SemanticKITTI*), DOWNLOAD from either host. ``--synthetic-pool``
fetches the requested archive from the free Hugging Face mirror
``Stone-Chern/PS3-SemanticKITTI``, whose tarballs are byte-identical to the DataPort
deposit and carry the same CC-BY-NC-SA 4.0 LICENSE, so no IEEE subscription is needed.
The DataPort record remains the alternative for anyone who prefers the archival
deposit, but it is not the free route and this script cannot fetch from it: it is
marked "Subscription Required", serves dataset files only to a signed-in session and
publishes no direct file URL. The mirror is private until the release repos are
flipped public together, so until then an anonymous fetch fails -- into the same
docs/DATASET.md pointer every other unreachable repo produces, never a traceback.
Nothing here waits on paper acceptance: the paper's availability footnote already
sends readers to github.com/BillyChern/GSSC-S2D2 for the code, the pre-trained models
and the PS3 dataset.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

HF_REPO_MODELS = "Stone-Chern/GSSC-S2D2-checkpoints"
HF_REPO_DATA = "Stone-Chern/GSSC-S2D2-datasets"
# THE SYNTHETIC POOL'S FREE ROUTE. The pool (127 GiB / 136 GB for 31K, 229 GiB / 246 GB for
# 57K, uncompressed) is deposited on IEEE DataPort under the DOI below, and that DOI is what
# you CITE. It is not a download: the record is marked "Subscription Required -- This dataset
# requires an IEEE DataPort Subscription to access", it serves files only to a signed-in
# session, and its landing page renders each archive as a login modal rather than an href, so
# no anonymous request of any shape succeeds and no script can retrieve it from there.
# The same two archives -- BYTE-IDENTICAL to the deposit, shipping the same CC-BY-NC-SA 4.0
# LICENSE, which is what permits the redistribution -- are mirrored on the Hugging Face
# dataset repo below, which needs no IEEE subscription. That is what `--synthetic-pool`
# downloads. Cite the DOI; take the bytes from whichever host suits you.
# The mirror is PRIVATE today and flips public with the other release repos, so an anonymous
# fetch fails until then; `_fetch` turns that into the documented docs/DATASET.md pointer.
# The pool is NOT embargoed until publication, so do NOT describe it that way in the docs:
# what stood between a reader and the archive was IEEE's access gate, not our release date.
HF_REPO_SYNTH = "Stone-Chern/PS3-SemanticKITTI"
#: The pool's CITABLE IDENTIFIER OF RECORD, and the alternative download route for anyone who
#: prefers the archival deposit. Minted 2026-08-23; resolves 200. A live DOI is not a fetchable
#: file URL -- see HF_REPO_SYNTH above for why the download goes to the mirror instead.
DATAPORT_URL = "https://dx.doi.org/10.21227/nqgf-9k39"

#: The two released archives, identical on BOTH hosts. Sizes are the `GiB / GB` pair this repo
#: quotes everywhere; note that DataPort's own file list labels them "2.15 GB" and "3.88 GB",
#: which are the GiB figures under a GB label. 31K is a STRICT SUBSET of 57K -- nobody needs
#: both, which is also why `--synthetic-pool` filters the mirror down to one archive instead of
#: snapshotting the whole repo (6.03 GiB / 6.47 GB) to deliver half of it.
POOL_ARCHIVES = {
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
    here is real, so no call below can currently trip, and `--synthetic-pool` now resolves
    against the Hugging Face mirror like every other group, failing through `_fetch` when the
    repo cannot be reached. The function is kept, and every asset group including the pool
    still calls it, because it is cheap and because the failure it catches -- a URL edited
    back into a placeholder, or a new asset group added with its host still to be decided --
    is one a reader would otherwise meet as a huggingface_hub traceback.
    """
    if url.startswith("[") and url.endswith("]"):
        sys.exit(
            f"\n{label} URL is not yet configured (placeholder: {url}).\n"
            "Build it locally instead:\n" + _MANUAL_ROUTES
        )


def _synthetic_pool_note(root: Path, variant: str) -> str:
    """What `--synthetic-pool` prints AFTER the fetch: how to cite it, how to unpack it.

    THE DOI IS THE CITATION, THE MIRROR IS THE DOWNLOAD, AND THIS MESSAGE MUST KEEP THEM
    APART. The pool is deposited on IEEE DataPort under doi:10.21227/nqgf-9k39, which is its
    citable identifier of record and belongs here whichever host the bytes came from; the
    Hugging Face mirror is the route that needs no IEEE subscription and is never presented
    as the thing to cite. DataPort is named as the alternative, never as freely downloadable.

    NO COPY-PASTEABLE DATAPORT FETCH, DELIBERATELY. This branch used to print

        wget -O <root>/synthetic_pool_31K.tar.gz <DATAPORT_URL>/synthetic_pool_31K.tar.gz

    which was wrong even with a real DOI substituted for the placeholder: IEEE DataPort
    publishes no ``<landing-page>/<filename>`` URL -- the deposit page renders each archive as
    a login modal, never as an href -- and it serves files only to a signed-in, subscribed
    session, so no anonymous request of any shape can succeed. A command that 404s is worse
    than no command: it reads as a supported path and fails only after the reader has trusted
    it. ``tar -xzf`` is spelled out because extraction is the half the user still runs locally.
    """
    name, size, scenes = POOL_ARCHIVES[variant]
    other = "57K" if variant == "31K" else "31K"
    return (
        f"\nSynthetic pool '{variant}' fetched from the free Hugging Face mirror.\n"
        f"  Archive:  {root}/{name}  ({size} compressed, {scenes} scenes)\n"
        f"  Mirror:   https://huggingface.co/datasets/{HF_REPO_SYNTH}  "
        f"(no IEEE subscription needed)\n"
        f"  Cite as:  Shi Chen, Weifeng Ge, \"PS3-SemanticKITTI: Paired Sparse-Dense "
        f"Synthetic Scenes\n"
        f"            for LiDAR Semantic Scene Completion\", IEEE Dataport, August 23, 2026,\n"
        f"            doi:10.21227/nqgf-9k39  ({DATAPORT_URL})\n\n"
        "Unpack it where the rest of the data lives:\n"
        f"    tar -xzf {root}/{name} -C {root}/\n\n"
        f"The {other} pool is the other released variant, and 31K is a strict subset of 57K "
        "-- take one, never both.\n"
        f"Prefer the archival deposit? The same two archives, byte-identical, are on IEEE "
        f"DataPort at {DATAPORT_URL} -- but that record requires an IEEE DataPort "
        "subscription (IEEE Society membership included) and releases files only to a "
        "signed-in session, so it has to be fetched by hand from a browser.\n"
        "Neither host? Rebuilding the pool locally with the PS3 ray-tracer needs no account "
        "at all, and is documented with everything else here:\n" + _MANUAL_ROUTES
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
                        help="Download a released synthetic pool variant from the free "
                             "Hugging Face mirror; cite it by its IEEE DataPort DOI "
                             "(10.21227/nqgf-9k39). 31K = 2.15 GiB / 2.31 GB compressed "
                             "-> 127 GiB / 136 GB unpacked, 57K = 3.88 GiB / 4.16 GB "
                             "-> 229 GiB / 246 GB unpacked")
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
        _ensure_url_configured(HF_REPO_SYNTH, "Synthetic pool (Hugging Face mirror)")
        _ensure_url_configured(DATAPORT_URL, "Synthetic pool citation (IEEE DataPort DOI)")

    # `--synthetic-pool` USED TO BE ANSWERED HERE, before the import, because nothing on
    # Hugging Face served the pool and it would otherwise have told a reader in a bare
    # (pre-`uv sync`) environment to install huggingface-hub for an asset the Hub did not
    # host. The free mirror removed that asymmetry: the pool is now fetched by the same
    # machinery as every other group, so it needs huggingface_hub exactly as they do and is
    # dispatched with them below. The early exit is gone rather than kept as a special case;
    # what it protected -- a bare environment getting a useful message -- is now served by
    # the ImportError branch immediately below, which is the same message every other mode
    # gets.

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
    # The synthetic pool: one archive out of its own mirror, then the DOI to cite it by.
    if args.synthetic_pool:
        _name, size, scenes = POOL_ARCHIVES[args.synthetic_pool]
        logger.info("Downloading synthetic pool %s (%s compressed, %s scenes) from %s ...",
                    args.synthetic_pool, size, scenes, HF_REPO_SYNTH)
        # ONE ARCHIVE PER VARIANT, AND THE FILENAME IS SPELLED OUT AT EACH CALL SITE.
        # `allow_patterns` has to be a literal here, not `_name`:
        # .release_checks/check_asset_namespace.py reads these call sites with `ast` to prove
        # that the file this script asks the mirror for is the file the upload procedure puts
        # into that repo, and a variable makes it report PARSE DRIFT instead of a filter set
        # (its `_pattern_strings` docstring records two occasions when exactly that misread
        # shipped as a confident finding). Two branches also keep the mode's dispatch
        # explicit, which is what check_cli_surface.py measures.
        if args.synthetic_pool == "31K":
            _fetch(snapshot_download, "Synthetic pool 31K", HF_REPO_SYNTH,
                   repo_type="dataset", allow_patterns=["synthetic_pool_31K.tar.gz"],
                   local_dir=root)
        else:
            _fetch(snapshot_download, "Synthetic pool 57K", HF_REPO_SYNTH,
                   repo_type="dataset", allow_patterns=["synthetic_pool_57K.tar.gz"],
                   local_dir=root)
        logger.info("%s", _synthetic_pool_note(root, args.synthetic_pool))

    logger.info("Done.")


if __name__ == "__main__":
    main()
