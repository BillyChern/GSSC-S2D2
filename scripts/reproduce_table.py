"""Reproduce any per-table number from the paper.

Each table maps to a specific eval recipe. Running this script will:
1. Verify the right checkpoint subdir is downloaded.
2. Run the appropriate eval pipeline.
3. Print a CSV that matches the LaTeX numbers within 0.1 pp.

Checkpoints since v1.1.0 live in a per-checkpoint subdir layout
``data/checkpoints/<group>/<name>/{model.safetensors, model_ema.safetensors,
config.json}`` (HF convention). Inference defaults to ``model_ema.safetensors``.

Examples::

    python scripts/reproduce_table.py tab:perclass
    python scripts/reproduce_table.py tab:main_results
    python scripts/reproduce_table.py tab:cross_base_js3c   # cross-base v1.1.0
    python scripts/reproduce_table.py tab:cross_base_lmsc   # cross-base v2.1.0
    python scripts/reproduce_table.py tab:bev_results
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]

TABLE_MAP: dict[str, dict[str, Any]] = {
    "tab:perclass":         {"config": "eval/val_1step",     "checkpoint": "gssc_mf/gssc_31k_mf_step40000",  "metrics": "miou per_class completion_iou"},
    "tab:main_results":     {"config": "infer/test_d4tta",   "checkpoint": "gssc_mf/gssc_31k_mf_step40000",  "metrics": "miou per_class completion_iou", "submit": True},
    "tab:step_reduction":   {"config": "eval/step_sweep",    "checkpoint": "gssc_mf/gssc_31k_mf_step40000",  "metrics": "miou completion_iou"},
    "tab:train_timesteps_curriculum": {"config": "eval/timestep_ablation", "checkpoint": "[gssc_timesteps/gssc_T10|gssc_timesteps/gssc_T50|gssc_timesteps/gssc_T100skewed|gssc_mf/gssc_31k_mf_step40000]", "metrics": "miou"},
    # The 36.1% BEV entry was produced by bev/bev_s2d2_scpnet, NOT by
    # bev/bev_perception_net (a different, 938K-param 2D refinement model that the BEV
    # evaluator cannot load). "protocol" is printed with the recipe because this row is
    # NOT scored the way every other row here is -- see the checkpoint's config.json.
    "tab:bev_results":      {"config": "eval/bev_secondary", "checkpoint": "bev/bev_s2d2_scpnet",           "metrics": "miou",
                             "protocol": "training-time 2D BEV evaluator, 100 fixed val samples (seed 42) "
                                         "-- NOT the 4071-frame semantic-kitti-api protocol used by every "
                                         "other row; mIoU over the 19 evaluation classes, class 0 excluded"},
    # NOT the paper's tab:data_scaling. That table is the MULTI-frame sweep and only the
    # headline 31K multi-frame checkpoint ships, so it cannot be regenerated. What the released
    # per-row checkpoints reproduce is the SINGLE-frame sweep reported in prose in supp. App. C.
    "data_scaling_sf":      {"config": "eval/data_scaling_sf", "checkpoint": "[gssc_sf/gssc_{0K,10K,20K,31K,57K}_sf_step{93000,87000,85000,72000,69000}]", "metrics": "miou"},
    "tab:cross_base_js3c":  {
        # JS3C-Net cross-base. The PAPER'S headline for this base is 24.32
        # (derived BEV, official semantic-kitti-api, +1.59 pp), which
        # eval/js3c_val_realistic reproduces. 26.05 is a GT-BEV DIAGNOSTIC
        # (official api, +3.32 pp) and 26.72 the same protocol under the paper's
        # internal SSCMetrics (+3.99 pp); neither is the paper's headline. This key is named
        # after a PAPER TABLE, so it must dispatch the config that reproduces that table's row:
        # main Tab. III prints "JS3C-Net + S2D2  24.3  +1.6". Until 2026-08-15 it shipped
        # eval/js3c_val_1step with expected 26.05 -- the GT-BEV diagnostic -- so a reader
        # reproducing the cross-base row got a number the paper does not print there. For the
        # GT-BEV diagnostic run eval/js3c_val_1step explicitly (see docs/REPRODUCIBILITY.md).
        "config": "eval/js3c_val_realistic",
        "checkpoint": "gssc_js3c/gssc_js3c_s2d2_real",
        "metrics": "miou per_class completion_iou",
        "base_pred_dir_required": "data/js3cnet_predictions",
        "base_kind": "js3c",
        "expected_mIoU": 24.32,
    },
    "tab:cross_base_lmsc":  {
        "config": "eval/lmscnet_val_1step",
        "checkpoint": "gssc_lmsc/gssc_lmsc_s2d2_real",
        "metrics": "miou per_class completion_iou",
        "base_pred_dir_required": "data/lmscnet_predictions",
        "base_kind": "lmscnet",
        "expected_mIoU": 16.59,
    },
}

BASE_DUMPER_INFO: dict[str, dict[str, str]] = {
    "js3c": {
        "name": "JS3C-Net",
        "dumper": "scripts/dump_js3c_predictions.py",
        "extra_setup": (
            "   git clone --depth 1 https://github.com/yanx27/JS3C-Net external/JS3C-Net\n"
            "   bash external/JS3C-Net/download_pretrained.sh"
        ),
        "args_hint": "--js3c-repo external/JS3C-Net",
    },
    "lmscnet": {
        "name": "LMSCNet",
        "dumper": "scripts/dump_lmscnet_predictions.py",
        "extra_setup": (
            "   git clone --depth 1 https://github.com/astra-vision/LMSCNet external/LMSCNet\n"
            "   # download LMSCNet.pth from the upstream Google Drive folder into external/LMSCNet/pretrained_models/"
        ),
        "args_hint": "--lmscnet-repo external/LMSCNet --checkpoint external/LMSCNet/pretrained_models/LMSCNet.pth",
    },
}


def _check_base_predictions(required_dir: Path, base_kind: str) -> None:
    """Pre-flight: ensure base-model predictions are dumped before cross-base eval.

    Each cross-base eval needs ``<required_dir>/08/<frame>_pred.npy``. If the
    directory is missing or empty, print the exact dumper command for the
    relevant base (``base_kind in {'js3c', 'lmscnet'}``) and exit non-zero
    rather than crashing inside the dataloader.
    """
    val_seq = required_dir / "08"
    if val_seq.is_dir() and any(val_seq.glob("*_pred.npy")):
        return
    info = BASE_DUMPER_INFO.get(base_kind)
    print("=" * 78)
    if info is None:
        print(f"Missing base predictions for cross-base reproduction (base_kind={base_kind}).")
        print(f"Expected at least one *_pred.npy file under: {val_seq}")
        print("=" * 78)
        sys.exit(2)
    print(f"Missing {info['name']} predictions for the cross-base reproduction.")
    print("Expected at least one *_pred.npy file under:")
    print(f"   {val_seq}")
    print()
    print(f"Reproduce by running the dumper against your {info['name']} clone:")
    print()
    print(info["extra_setup"])
    print(f"   python {info['dumper']} \\")
    print(f"       {info['args_hint']} \\")
    print("       --semantickitti_root data/SemanticKITTI \\")
    print(f"       --output_dir {required_dir} \\")
    print("       --sequences 08")
    print()
    print("Then re-run this script. See docs/REPRODUCIBILITY.md for the full")
    print("cross-base protocol.")
    print("=" * 78)
    sys.exit(2)


def _resolve_checkpoint(ckpt_dir: Path, name: str, prefer_ema: bool = True) -> Path:
    """Resolve a checkpoint subdir to its preferred .safetensors file.

    Layout (v1.1.0+)::

        <ckpt_dir>/<group>/<name>/
            model.safetensors        # training weights
            model_ema.safetensors    # deployment weights (paper convention)
            config.json
    """
    subdir = ckpt_dir / name
    ema = subdir / "model_ema.safetensors"
    plain = subdir / "model.safetensors"
    if prefer_ema and ema.is_file():
        return ema
    if plain.is_file():
        return plain
    raise FileNotFoundError(
        f"checkpoint missing: expected {ema} or {plain}; "
        "run `python scripts/download_assets.py --checkpoints`"
    )


# Aliases: the labels as they appear in the paper, mapped onto the keys above. Additive by design --
# the original keys keep working. Two paper labels have no 1:1 key and are handled explicitly:
# tab:portable_s2d2 spans three bases, so it resolves to the per-base keys and reports both; and the
# training-timestep comparison is no longer a table in the paper (it was folded into prose in
# Appendix C), so its key is kept for the experiment but carries no paper label.
PAPER_LABEL_ALIASES: dict[str, str] = {
    "tab:perclass_delta": "tab:perclass",          # main Table II
}
UNREPRODUCIBLE: dict[str, str] = {
    "tab:data_scaling": (
        "the multi-frame data-scaling sweep cannot be regenerated: only the headline 31K "
        "multi-frame checkpoint ships. Run 'data_scaling_sf' for the single-frame sweep "
        "reported in prose in supplementary Appendix C."
    ),
}
MULTI_KEY_LABELS: dict[str, list[str]] = {
    "tab:portable_s2d2": ["tab:cross_base_lmsc", "tab:cross_base_js3c"],
}


def resolve_table(name: str) -> list[str]:
    """Map a user-supplied name onto one or more TABLE_MAP keys."""
    if name in TABLE_MAP:
        return [name]
    if name in PAPER_LABEL_ALIASES:
        return [PAPER_LABEL_ALIASES[name]]
    if name in MULTI_KEY_LABELS:
        return MULTI_KEY_LABELS[name]
    if name in UNREPRODUCIBLE:
        raise KeyError(f"{name}: {UNREPRODUCIBLE[name]}")
    raise KeyError(name)


def selectable_names() -> list[str]:
    return sorted({*TABLE_MAP, *PAPER_LABEL_ALIASES, *MULTI_KEY_LABELS})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("table", choices=selectable_names(),
                        help="Table label from the paper, or a driver key (see PAPER_LABEL_ALIASES)")
    parser.add_argument("--checkpoints-dir", default=str(REPO_ROOT / "data" / "checkpoints"))
    parser.add_argument("--data-root", default=str(REPO_ROOT / "data"))
    args = parser.parse_args()

    keys = resolve_table(args.table)
    ckpt_dir = Path(args.checkpoints_dir)
    data_root = Path(args.data_root)
    if len(keys) > 1:
        print(f"{args.table} spans {len(keys)} driver entries: {', '.join(keys)}; "
              f"reproducing each in turn.")
    for key in keys:
        _reproduce_one(key, args.table, ckpt_dir, data_root)


def _reproduce_one(key: str, label: str, ckpt_dir: Path, data_root: Path) -> None:
    """Reproduce ONE driver entry.

    Split out of :func:`main` because the announcement and the behaviour disagreed:
    main() printed "reproducing each in turn" for a multi-entry label and then ran
    ``TABLE_MAP[keys[0]]`` only, so every entry after the first was silently skipped.
    A reader trusting that message would have reported a partial table as a full one.

    Args:
        key: Driver key into :data:`TABLE_MAP`.
        label: The table label the user asked for, echoed so multi-entry output stays
            attributable to the request that produced it.
        ckpt_dir: Root of the checkpoint tree.
        data_root: Root of the data tree.
    """
    spec = TABLE_MAP[key]

    print(f"=== Reproducing {label} [{key}] ===")
    print(f"  config:     {spec['config']}")
    print(f"  checkpoint: {spec['checkpoint']}")
    print(f"  metrics:    {spec['metrics']}")
    if "expected_mIoU" in spec:
        print(f"  expected:   {spec['expected_mIoU']}% mIoU (paper)")
    if "protocol" in spec:
        # A number quoted without its protocol is a different claim from the number.
        print(f"  protocol:   {spec['protocol']}")
    print()

    if "base_pred_dir_required" in spec:
        required = (
            data_root.parent / spec["base_pred_dir_required"]
            if not Path(spec["base_pred_dir_required"]).is_absolute()
            else Path(spec["base_pred_dir_required"])
        )
        _check_base_predictions(required, spec.get("base_kind", "js3c"))

    if "[" in spec["checkpoint"]:
        print("Multi-row table. Each row variant must be run with its own checkpoint.")
        print("See docs/MODEL_ZOO.md for the row-by-row checkpoint list.")
        return

    ckpt_path = _resolve_checkpoint(ckpt_dir, spec["checkpoint"])
    if spec.get("submit"):
        # Hidden-test row: the SemanticKITTI test GT is server-side, so we
        # generate prediction files for leaderboard submission rather than
        # scoring locally (local scoring would fail — sequences 11-21 have no
        # public .label GT).
        out_dir = REPO_ROOT / "outputs" / "test_submission"
        cmd = [
            sys.executable, "scripts/infer.py", spec["config"],
            "--checkpoint", str(ckpt_path),
            "--output", str(out_dir),
        ]
        print(f"$ {' '.join(cmd)}")
        subprocess.run(cmd, cwd=REPO_ROOT, check=True)
        print(
            f"\nTest predictions written under {out_dir}/sequences/<seq>/predictions/.\n"
            "Zip and submit to the SemanticKITTI SSC leaderboard (Codabench) for the "
            "hidden-test mIoU (paper: 38.8% at N=1, 39.2% with D4 TTA); see README.md."
        )
        return
    cmd = [
        sys.executable, "scripts/eval.py", spec["config"],
        "--checkpoint", str(ckpt_path),
        "--metrics", *spec["metrics"].split(),
    ]
    print(f"$ {' '.join(cmd)}")
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


if __name__ == "__main__":
    main()
