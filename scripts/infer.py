"""GSSC-S2D2 inference entry point - generate .label predictions for SemanticKITTI evaluation server.

Unlike :mod:`scripts.eval`, this script does *not* score the predictions; it only
writes ``.label`` files (e.g. for the hidden-test leaderboard submission, sequences
11-21). The positional ``config`` selects the split / correction-step / TTA recipe
from ``configs/infer/<name>.yaml``:

================  =====  ================  =====  ====  ====================================
config            split  gen sequences     steps  tta   paper row
================  =====  ================  =====  ====  ====================================
infer/val_1step   val    08                1      none  tab:perclass (val per-class)
infer/val_d4tta   val    08                1      d4    tab:main_results (val + D4 TTA)
infer/test_d4tta  test   11..21 (hidden)   4      d4    tab:main_results (39.2 test row)
================  =====  ================  =====  ====  ====================================

The YAML is the source of truth for ``split`` / ``correction_steps`` / ``tta``;
the historical bug where ``infer/test_d4tta`` silently ran on val seq-08 (because
``--split`` was never forwarded) is fixed by loading + forwarding the config here.
CLI flags override the config (``--split`` / ``--steps`` / ``--tta``).

Examples
--------
Reproduce the 39.2% test leaderboard row (hidden sequences 11-21)::

    python scripts/infer.py infer/test_d4tta \
        --checkpoint data/checkpoints/gssc_mf/gssc_31k_mf_step40000/model_ema.safetensors \
        --output outputs/test_submission

Generate val per-class predictions (single correction step, seq 08)::

    python scripts/infer.py infer/val_1step \
        --checkpoint data/checkpoints/gssc_mf/gssc_31k_mf_step40000/model_ema.safetensors \
        --output outputs/val_preds
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

# Config split vocabulary ("val"/"test"/"train") -> generator --split vocabulary
# ("valid"/"test"/"train"). Mirrors run_evaluation's gen_split mapping so the
# infer and eval paths resolve the same sequences from the same YAML key.
_SPLIT_TO_GEN = {"val": "valid", "valid": "valid", "test": "test", "train": "train"}

# Hidden-test / val / train sequence sets, for the static dry-run banner and so a
# stale/missing `sequences` key can never disagree with the resolved `split`.
_GEN_SEQUENCES = {
    "valid": ["08"],
    "test": ["11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21"],
    "train": ["00", "01", "02", "03", "04", "05", "06", "07", "09", "10"],
}


def _resolve_config(config: str) -> dict:
    """Load a Hydra-style ``configs/<config>.yaml`` into a dict.

    Reuses :func:`gssc.inference.evaluate._resolve_config` so infer and eval share
    one resolution + not-found-listing path.

    Args:
        config: Config name relative to ``configs/`` without the ``.yaml`` suffix
            (e.g. ``"infer/test_d4tta"``).

    Returns:
        The parsed YAML mapping.

    Raises:
        FileNotFoundError: If the config does not exist (lists available configs).
    """
    from gssc.inference.evaluate import _resolve_config as _resolve

    return _resolve(config)


def main() -> None:
    """Resolve the named infer config, then write ``.label`` predictions for its split.

    The YAML supplies ``split`` / ``correction_steps`` / ``tta`` (and optional
    ``bev_source`` / ``base_pred_dir``); matching CLI flags override the config.
    """
    p = argparse.ArgumentParser(
        description="GSSC-S2D2 inference (.label generator).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("config", help="Infer config: infer/val_1step, infer/val_d4tta, infer/test_d4tta")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True, help="Where .label predictions land")
    p.add_argument("--data-root", default=str(REPO_ROOT / "data"))
    p.add_argument("--gpu", default="0")
    # Overrides default to None so "unset" means "use the config value".
    p.add_argument("--split", default=None, choices=["val", "valid", "test", "train"],
                   help="Override config split (val/valid -> seq 08, test -> 11-21)")
    p.add_argument("--steps", type=int, default=None,
                   help="Override config correction_steps (1, 4, 100)")
    p.add_argument("--tau", type=float, default=1.0,
                   help="Sampling temperature on the denoiser logits; provably inert at steps=1")
    p.add_argument("--tta", choices=["none", "flip_y", "d4"], default=None,
                   help="Override config tta mode")
    p.add_argument("--dry-run", action="store_true",
                   help="Resolve + print the command without launching it (no GPU).")
    a = p.parse_args()

    cfg = _resolve_config(a.config)

    # YAML is the source of truth; CLI flags win when explicitly set.
    cfg_split = str(cfg.get("split", "val"))
    split = a.split if a.split is not None else cfg_split
    if split not in _SPLIT_TO_GEN:
        raise ValueError(
            f"Unknown split {split!r} (config {a.config!r}); "
            f"expected one of {sorted(_SPLIT_TO_GEN)}"
        )
    gen_split = _SPLIT_TO_GEN[split]
    gen_sequences = _GEN_SEQUENCES.get(gen_split, [])

    # Cross-check the informational `sequences` key against the resolved split so a
    # stale/edited `sequences` (e.g. seq 08 left on a `split: test` config) cannot
    # silently disagree with the split that actually drives the generator. The
    # generator derives its sequences from --split; `sequences` is documentation.
    cfg_sequences = cfg.get("sequences")
    if cfg_sequences is not None:
        declared = {s.strip() for s in str(cfg_sequences).replace(",", " ").split() if s.strip()}
        resolved = set(gen_sequences)
        if a.split is None and declared and declared != resolved:
            raise ValueError(
                f"Config {a.config!r} declares sequences={sorted(declared)} but "
                f"split={split!r} resolves to {sorted(resolved)}. Fix the config's "
                f"`split`/`sequences` so they agree, or pass --split to override."
            )

    cfg_steps = cfg.get("correction_steps", cfg.get("algo2_steps", 4))
    steps = a.steps if a.steps is not None else int(cfg_steps)

    cfg_tta = str(cfg.get("tta", "none"))
    tta = a.tta if a.tta is not None else cfg_tta

    # Optional conditioning keys (consumed when present so no infer-config key is inert).
    bev_source = str(cfg.get("bev_source", "derived"))
    base_pred_dir_cfg = cfg.get("base_pred_dir") or cfg.get("scpnet_pred_dir")

    data_root = Path(a.data_root).resolve()
    semantic_kitti_root = data_root / "SemanticKITTI"
    if not semantic_kitti_root.exists():
        semantic_kitti_root = data_root

    # Resolve base-prediction dir the same way run_evaluation does so cross-base
    # infer configs (JS3C-Net, ...) reach the right <seq>/<frame>_pred.npy tree.
    if base_pred_dir_cfg is None:
        scpnet_dir = data_root / "scpnet_predictions"
    else:
        base_pred_dir_path = Path(str(base_pred_dir_cfg))
        if base_pred_dir_path.is_absolute():
            scpnet_dir = base_pred_dir_path
        else:
            scpnet_dir = data_root.parent / base_pred_dir_path
            if not scpnet_dir.exists():
                scpnet_dir = data_root / base_pred_dir_path.name

    # generate_predictions only knows {valid, test}; only d4_tta handles `train`.
    if tta != "d4" and gen_split == "train":
        raise ValueError(
            "split=train is only supported with tta=d4 (gssc.inference.d4_tta); "
            "the non-TTA generate_predictions path accepts valid/test only."
        )

    if tta == "d4":
        cmd = [sys.executable, "-m", "gssc.inference.d4_tta",
               "--checkpoint", a.checkpoint,
               "--cold_steps", str(steps),
               "--tau", str(a.tau),
               "--output_dir", a.output,
               "--data_root", str(semantic_kitti_root),
               "--scpnet_dir", str(scpnet_dir),
               "--split", gen_split,
               "--bev_source", bev_source]
    else:
        cmd = [sys.executable, "-m", "gssc.inference.generate_predictions",
               "--checkpoint", a.checkpoint,
               "--cold_steps", str(steps),
               "--tau", str(a.tau),
               "--output_dir", a.output,
               "--data_root", str(semantic_kitti_root),
               "--scpnet_dir", str(scpnet_dir),
               "--split", gen_split,
               "--bev_from_base",
               "--bev_source", bev_source,
               "--algo2"]

    print(f"# config={a.config} split={split} -> gen_split={gen_split} "
          f"sequences={','.join(gen_sequences)} steps={steps} tta={tta} bev_source={bev_source}")
    print(f"$ {' '.join(cmd)}")
    if a.dry_run:
        return

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = a.gpu
    env["PYTHONPATH"] = f"{REPO_ROOT / 'src'}:{env.get('PYTHONPATH', '')}"
    subprocess.run(cmd, env=env, check=True)


if __name__ == "__main__":
    main()
