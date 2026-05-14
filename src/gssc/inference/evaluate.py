"""GSSC-S2D2 evaluation pipeline.

Loads a checkpoint, runs S2D2 correction sampling on SemanticKITTI val, and
returns per-class IoU + mIoU + completion IoU.

The evaluation is a two-stage pipeline:

1. ``gssc.inference.generate_predictions`` (S2D2 correction sampler) writes ``.label``
   files for each frame in the requested split.
2. ``external/semantic_kitti_api/evaluate_completion.py`` reads those
   ``.label`` files and produces the official scoring numbers.

This module wires the two together into a single :func:`run_evaluation`
call, matching the eval/ Hydra configs in the repo.
"""
from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]


def _parse_eval_completion_output(text: str) -> dict[str, float]:
    """Extract mIoU + IoU_cmpl + per-class IoU from evaluate_completion.py output.

    The official semantic_kitti_api script prints lines like::

        IoU class 1 [car] = 0.514
        IoU class 8 [motorcyclist] = 0.124
        Precision =\t86.04
        Recall   =\t57.59
        IoU Cmpltn =\t52.66
        mIoU SSC =\t38.54

    Per-class IoU is printed as a fraction in [0, 1]; aggregate metrics
    are already in percent. We normalise everything to percent.

    Args:
        text: Captured stdout/stderr of the scoring process.

    Returns:
        Mapping from metric name to percentage value (e.g. ``{"mIoU": 38.54,
        "IoU_cmpl": 52.66, "IoU_car": 51.4, "IoU_motorcyclist": 12.4, ...}``).
    """
    metrics: dict[str, float] = {}
    for line in text.splitlines():
        # Aggregate metrics: tab-separated, already in percent.
        m = re.match(r"\s*mIoU\s+SSC\s*=\s*([\d.]+)", line)
        if m:
            metrics["mIoU"] = float(m.group(1))
            continue
        m = re.match(r"\s*IoU\s+Cmpltn\s*=\s*([\d.]+)", line)
        if m:
            metrics["IoU_cmpl"] = float(m.group(1))
            continue
        m = re.match(r"\s*Precision\s*=\s*([\d.]+)", line)
        if m:
            metrics["Precision"] = float(m.group(1))
            continue
        m = re.match(r"\s*Recall\s*=\s*([\d.]+)", line)
        if m:
            metrics["Recall"] = float(m.group(1))
            continue
        # Per-class IoU: "IoU class 1 [car] = 0.514" — fraction → percent.
        m = re.match(r"\s*IoU class \d+\s*\[([^\]]+)\]\s*=\s*([\d.]+)", line)
        if m:
            cls_name = m.group(1).strip().replace("-", "_").replace(" ", "_")
            metrics[f"IoU_{cls_name}"] = float(m.group(2)) * 100.0
    return metrics


def _resolve_config(config: str) -> dict:
    """Load a Hydra-style YAML config from the configs/ tree."""
    cfg_path = REPO_ROOT / "configs" / f"{config}.yaml"
    if not cfg_path.exists():
        candidates = sorted(p.relative_to(REPO_ROOT / "configs")
                            for p in (REPO_ROOT / "configs").rglob("*.yaml"))
        raise FileNotFoundError(
            f"Config not found: {cfg_path}\n"
            f"Available: {candidates}"
        )
    return yaml.safe_load(cfg_path.read_text())


def run_evaluation(
    checkpoint: str,
    config: str,
    data_root: str,
    output: str | None = None,
    gpu: str = "0",
    steps: int | None = None,
    tta: str | None = None,
    metrics: list[str] | None = None,
    keep_predictions: bool = False,
) -> dict:
    """Run a SemanticKITTI evaluation end-to-end.

    Args:
        checkpoint: Path to the trained checkpoint (``.pt`` or ``.safetensors``).
        config: Hydra-style config name, e.g. ``"eval/val_1step"``.
        data_root: Repository data root (must contain ``SemanticKITTI/`` and
            ``scpnet_predictions/``).
        output: Optional path to dump per-class metrics as JSON.
        gpu: CUDA device id (default ``"0"``).
        steps: Correction-step count override; ``None`` uses the config default.
        tta: TTA mode override; one of ``{"none", "flip_y", "d4"}``.
        metrics: Reserved for future safety-metric subset selection.
        keep_predictions: If True, keep the intermediate ``.label`` files
            in ``<data_root>/predictions/<config_name>/``. If False
            (default), they go to a temporary directory.

    Returns:
        Mapping from metric name (``"mIoU"``, ``"IoU_cmpl"``, per-class
        ``"IoU_<class>"``) to percentage value.

    Raises:
        FileNotFoundError: If the checkpoint or data root is missing.
        RuntimeError: If generate_predictions or evaluate_completion fails.
    """
    metrics = metrics or ["miou", "completion_iou", "per_class"]
    ckpt_path = Path(checkpoint).resolve()
    data_root_path = Path(data_root).resolve()
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    if not data_root_path.exists():
        raise FileNotFoundError(f"Data root not found: {data_root_path}")

    cfg = _resolve_config(config)
    sequences = str(cfg.get("sequences", "08"))
    n_steps = steps if steps is not None else int(
        cfg.get("correction_steps", cfg.get("algo2_steps", 1))
    )
    tta_mode = tta if tta is not None else str(cfg.get("tta", "none"))

    logger.info("Eval config: %s", config)
    logger.info("Checkpoint: %s", ckpt_path)
    logger.info("Sequences=%s steps=%d tta=%s", sequences, n_steps, tta_mode)

    pred_dir_ctx: tempfile.TemporaryDirectory[str] | None = None
    if keep_predictions:
        pred_dir = data_root_path / "predictions" / config.replace("/", "_")
        pred_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Predictions kept at: %s", pred_dir)
    else:
        pred_dir_ctx = tempfile.TemporaryDirectory(prefix="gssc_eval_preds_")
        pred_dir = Path(pred_dir_ctx.name)
        logger.info("Predictions in temp dir: %s (auto-cleaned on success)", pred_dir)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu
    env["PYTHONPATH"] = f"{REPO_ROOT / 'src'}:{env.get('PYTHONPATH', '')}"

    # Resolve the base-prediction directory. Modern configs (since v1.1.0)
    # specify ``base_pred_dir`` to support cross-base reproduction (JS3C-Net,
    # PaSCo, ...). Legacy configs fall back to ``scpnet_pred_dir``; both feed
    # the same downstream subprocess.
    base_pred_dir_cfg = cfg.get("base_pred_dir") or cfg.get("scpnet_pred_dir")
    if base_pred_dir_cfg is None:
        scpnet_dir = data_root_path / "scpnet_predictions"
    else:
        base_pred_dir_path = Path(base_pred_dir_cfg)
        if base_pred_dir_path.is_absolute():
            scpnet_dir = base_pred_dir_path
        else:
            scpnet_dir = data_root_path.parent / base_pred_dir_path
            if not scpnet_dir.exists():
                scpnet_dir = data_root_path / base_pred_dir_path.name
    semantic_kitti_root = data_root_path / "SemanticKITTI"
    if not semantic_kitti_root.exists():
        semantic_kitti_root = data_root_path

    cfg_split = str(cfg.get("split", "valid"))
    gen_split = "valid" if cfg_split == "val" else cfg_split
    bev_source = str(cfg.get("bev_source", "derived"))
    bev_root_cfg = cfg.get("bev_root")

    if tta_mode == "d4":
        cmd = [
            sys.executable, "-m", "gssc.inference.d4_tta",
            "--checkpoint", str(ckpt_path),
            "--cold_steps", str(n_steps),
            "--output_dir", str(pred_dir),
            "--data_root", str(semantic_kitti_root),
            "--scpnet_dir", str(scpnet_dir),
            "--split", gen_split,
            "--bev_source", bev_source,
        ]
    else:
        cmd = [
            sys.executable, "-m", "gssc.inference.generate_predictions",
            "--checkpoint", str(ckpt_path),
            "--cold_steps", str(n_steps),
            "--output_dir", str(pred_dir),
            "--data_root", str(semantic_kitti_root),
            "--scpnet_dir", str(scpnet_dir),
            "--split", gen_split,
            "--algo2",
            "--bev_from_scpnet",
            "--bev_source", bev_source,
        ]
    if bev_root_cfg:
        bev_root_path = Path(str(bev_root_cfg))
        if not bev_root_path.is_absolute():
            bev_root_path = data_root_path / bev_root_path
        cmd.extend(["--bev_root", str(bev_root_path)])
    # When predictions are persisted, an interrupted/repeat run can resume by
    # skipping frames that already have a .label written.
    if keep_predictions:
        cmd.append("--skip_existing")

    logger.info("Stage 1 (generate): %s", " ".join(cmd))
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if proc.returncode != 0:
        if pred_dir_ctx is not None:
            pred_dir_ctx.cleanup()
        raise RuntimeError(
            f"generate_predictions failed (exit {proc.returncode})\n"
            f"STDOUT: {proc.stdout[-2000:]}\n"
            f"STDERR: {proc.stderr[-2000:]}"
        )
    logger.info("Stage 1 done. Predictions written under %s", pred_dir)

    eval_script = REPO_ROOT / "external" / "semantic_kitti_api" / "evaluate_completion.py"
    eval_dir = eval_script.parent
    eval_datacfg = eval_dir / "config" / "semantic-kitti.yaml"

    score_cmd = [
        sys.executable, str(eval_script),
        "--dataset", str(semantic_kitti_root),
        "--predictions", str(pred_dir),
        "--split", gen_split,
        "--datacfg", str(eval_datacfg),
    ]
    logger.info("Stage 2 (score): %s", " ".join(score_cmd))
    score_proc = subprocess.run(score_cmd, env=env, capture_output=True, text=True, cwd=str(eval_dir))
    combined = score_proc.stdout + "\n" + score_proc.stderr
    if score_proc.returncode != 0:
        # Keep predictions on scoring failure so the user can rerun stage 2
        # without spending another ~10 min on stage 1.
        if pred_dir_ctx is not None:
            persist = data_root_path / "predictions" / f"{config.replace('/', '_')}_failed"
            persist.parent.mkdir(parents=True, exist_ok=True)
            try:
                Path(pred_dir).rename(persist)
                logger.error("Stage 2 failed; predictions retained at: %s", persist)
            except OSError:
                logger.error("Stage 2 failed; predictions still under: %s", pred_dir)
        raise RuntimeError(
            f"evaluate_completion failed (exit {score_proc.returncode})\n"
            f"STDOUT: {score_proc.stdout[-2000:]}\n"
            f"STDERR: {score_proc.stderr[-2000:]}"
        )

    parsed = _parse_eval_completion_output(combined)
    if not parsed:
        if pred_dir_ctx is not None:
            pred_dir_ctx.cleanup()
        raise RuntimeError(
            "Could not parse mIoU from evaluate_completion output. "
            f"Last 2000 chars:\n{combined[-2000:]}"
        )

    if output:
        out_path = Path(output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(parsed, indent=2))
        logger.info("Wrote per-class metrics to %s", out_path)

    for k in ("mIoU", "IoU_cmpl"):
        if k in parsed:
            logger.info("%-12s %.2f %%", k, parsed[k])

    if pred_dir_ctx is not None:
        pred_dir_ctx.cleanup()

    return parsed
