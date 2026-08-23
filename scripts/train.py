"""GSSC-S2D2 training entry point.

Dispatcher from a Hydra-*style* YAML (plain YAML, no Hydra runtime) to the
underlying trainers. Every reproducible recipe in the paper has a corresponding
YAML under ``configs/train/``.

The three trainers this dispatches to use DIFFERENT flag spellings:
``train_scene_completion`` and ``train_bev_secondary`` declare underscored flags
(``--output_dir``, ``--data_root``), while ``train_pyramid_s2`` /
``train_pyramid_s3`` declare hyphenated ones (``--output-dir``, ``--data-root``,
``--ssc-root``, ``--quantized-root``) and no ``--seed`` at all. argparse rejects
the wrong spelling outright, so the roots are appended *per entry point* below,
not once for all of them.

Examples
--------
The config name is a positional argument (the on-disk YAML stem under
``configs/``, e.g. ``train/31k_mf`` -> ``configs/train/31k_mf.yaml``).

Reproduce the headline 31K-MF run that yields 38.54% val mIoU and, on the hidden
test set, 38.8% mIoU at N=1 with no TTA (the paper's headline row) / 39.2% with
four correction steps and 8-view D4 TTA::

    python scripts/train.py train/31k_mf

Single-frame data-scaling retrains (a companion to supplementary Tab. VII, which
is the MULTI-frame sweep; the paper carries this single-frame sweep as prose in
supplementary App. C-B, not as a table)::

    python scripts/train.py train/0K_sf
    python scripts/train.py train/10K_sf
    python scripts/train.py train/20K_sf
    python scripts/train.py train/31k_sf
    python scripts/train.py train/57K_sf

Training-timestep ablation rows (internal; the paper prints no table for them)::

    python scripts/train.py train/T10
    python scripts/train.py train/T50
    python scripts/train.py train/T100skewed

Pyramid data-augmentation stages (offline, run once before any of the above; the
synthetic pool they produce ships as data, so no paper number depends on
rerunning them). Both need the pre-quantized corpus described in
``docs/TRAIN.md`` -- without it the loaders enumerate zero frames::

    python scripts/train.py train/pyramid_s2
    python scripts/train.py train/pyramid_s3

The dispatcher derives the pyramid roots from ``--data-root`` (default
``<repo>/data``): S2 gets ``<data-root>/SemanticKITTI_quantized``, S3 gets that
plus ``<data-root>/SemanticKITTI``. Add ``--dry-run`` to print the resolved
trainer command without launching it.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))


def _seed_was_explicit() -> bool:
    """True when --seed appeared on the command line, as opposed to defaulting to 42.

    argparse cannot tell the two apart after parsing, and the difference matters: warning on the
    default would fire on every pyramid run and teach people to ignore the warning.
    """
    return any(a == "--seed" or a.startswith("--seed=") for a in sys.argv[1:])


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
        help=(
            "Random seed. Default: 42, the verified reproducible recipe — "
            "produces 38.05%% val 1-step mIoU on the migrated codebase, "
            "within ~0.5%% of the paper's 38.54%% headline. The paper's "
            "published 38.54%% checkpoint was trained without seeding on the "
            "original repo and is not bit-reproducible by design."
        ),
    )
    parser.add_argument(
        "--gpu",
        type=str,
        default="0",
        help=(
            "CUDA device id, set as CUDA_VISIBLE_DEVICES for the trainer "
            "(default: 0). Use ONE id: train_scene_completion has no "
            "DataParallel / DDP / torch.distributed path, so a comma-separated "
            "list only makes extra devices visible, it does not use them. The "
            "pyramid trainers do support DDP, but only under a torchrun "
            "launcher, not through this dispatcher."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve the config and print the trainer command without launching it.",
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
    if not args.dry_run:
        # --dry-run resolves and prints; it must not leave an empty run directory behind.
        out_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.gpu
    env["PYTHONPATH"] = f"{REPO_ROOT / 'src'}:{env.get('PYTHONPATH', '')}"

    # Import here so ``--help`` doesn't pay the import cost.
    from gssc.utils.config_loader import load_yaml_to_args

    forwarded = load_yaml_to_args(cfg_path)

    # Pick the trainer FIRST: which root flags are legal depends on it.
    # bev_* -> the BEV second-task trainer, pyramid_* -> the pyramid stages,
    # everything else -> the canonical 3D SSC trainer.
    if args.config.startswith("train/bev_"):
        entry = "gssc.training.train_bev_secondary"
    elif args.config.startswith("train/pyramid_"):
        # Explicit names, not an else: the release ships no train_pyramid_s1 module, and
        # an open else would silently launch the S3 trainer for any future pyramid_s1 or
        # pyramid_s4 config (the paper's tab:supp_pyramid_hyperparams does print an S1 row).
        if "_s2" in args.config or args.config.endswith("/s2"):
            entry = "gssc.training.train_pyramid_s2"
        elif "_s3" in args.config or args.config.endswith("/s3"):
            entry = "gssc.training.train_pyramid_s3"
        else:
            raise SystemExit(
                f"No pyramid trainer for {args.config}: this release ships only "
                "train_pyramid_s2 and train_pyramid_s3."
            )
    else:
        entry = "gssc.training.train_scene_completion"

    # Override the YAML's roots with the CLI flags. Appended AFTER
    # load_yaml_to_args so the CLI wins over the config.
    if entry.startswith("gssc.training.train_pyramid_"):
        # Hyphenated flags, and no --seed on either pyramid parser. The
        # underscored spellings this dispatcher used until 2026-08-22 made
        # `train/pyramid_*` die on "unrecognized arguments: --output_dir" before
        # the trainer ran a single step.
        data_root = Path(args.data_root)
        quantized_root = data_root / "SemanticKITTI_quantized"
        forwarded.extend(["--output-dir", str(out_dir)])
        if entry.endswith("_s2"):
            # S2 reads only the pre-quantized pyramid corpus (s1/ as conditioning,
            # s2/ as target).
            forwarded.extend(["--data-root", str(quantized_root)])
        else:
            # S3 reads two trees: the raw full-resolution SSC labels and the
            # quantized s2/ (plus optional s3_cond/) levels.
            forwarded.extend(["--ssc-root", str(data_root / "SemanticKITTI")])
            forwarded.extend(["--quantized-root", str(quantized_root)])
    else:
        forwarded.extend(["--output_dir", str(out_dir)])
        forwarded.extend(["--data_root", args.data_root])
        if args.seed is not None:
            forwarded.extend(["--seed", str(args.seed)])
    forwarded.extend(remainder)

    # The pyramid trainers expose no --seed, so the branch above cannot forward one and the flag
    # would otherwise vanish without a word. Say so: someone passing --seed is asking for
    # reproducibility, and silently not seeding is the one outcome they must not get unknowingly.
    if "--seed" not in forwarded and _seed_was_explicit():
        print(
            f"WARNING: --seed {args.seed} was given, but {entry} accepts no --seed, so this run "
            f"is NOT seeded by it. The pyramid trainers set no seed at all, so those runs "
            f"are unseeded; the "
            f"released checkpoints are reproducible from their recorded epoch and step "
            f"(data/checkpoints/pyramid/*/config.json), not from a seed passed here.",
            file=sys.stderr,
        )

    cmd = [sys.executable, "-m", entry, *forwarded]
    print(f"$ {' '.join(cmd)}")
    if args.dry_run:
        print("(--dry-run: resolved above, not launched)")
        return
    subprocess.run(cmd, env=env, check=True)


if __name__ == "__main__":
    main()
