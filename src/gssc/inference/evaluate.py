"""GSSC-S2D2 evaluation pipeline.

Loads a checkpoint, runs Algo2 correction sampling on SemanticKITTI val, and
returns per-class IoU + mIoU + completion IoU.

The evaluation is a two-stage pipeline:

1. ``gssc.inference.generate_predictions`` (Algo2 sampler) writes ``.label``
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
from typing import List, Optional

import yaml

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]


def _parse_eval_completion_output(text: str) -> dict[str, float]:
    """Extract mIoU + IoU_cmpl + per-class IoU from evaluate_completion.py output.

    The official semantic_kitti_api script prints lines like::

        IoU class motorcyclist                = 12.43%
        Completion IoU = 55.45%
        Mean IoU = 38.54%

    Args:
        text: Captured stdout/stderr of the scoring process.

    Returns:
        Mapping from metric name to percentage value.
    """
    metrics: dict[str, float] = {}
    for line in text.splitlines():
        m = re.match(r"\s*Mean IoU\s*=\s*([\d.]+)%", line)
        if m:
            metrics["mIoU"] = float(m.group(1))
        m = re.match(r"\s*Completion IoU\s*=\s*([\d.]+)%", line)
        if m:
            metrics["IoU_cmpl"] = float(m.group(1))
        m = re.match(r"\s*IoU class (\S+(?:\s+\S+)?)\s*=\s*([\d.]+)%", line)
        if m:
            cls_name = m.group(1).strip().replace(" ", "_")
            metrics[f"IoU_{cls_name}"] = float(m.group(2))
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
    output: Optional[str] = None,
    gpu: str = "0",
    steps: Optional[int] = None,
    tta: Optional[str] = None,
    metrics: Optional[List[str]] = None,
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
        steps: Algo2 step count override; ``None`` uses the config default.
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
    n_steps = steps if steps is not None else int(cfg.get("algo2_steps", 1))
    tta_mode = tta if tta is not None else str(cfg.get("tta", "none"))

    logger.info("Eval config: %s", config)
    logger.info("Checkpoint: %s", ckpt_path)
    logger.info("Sequences=%s steps=%d tta=%s", sequences, n_steps, tta_mode)

    pred_dir_ctx: Optional[tempfile.TemporaryDirectory[str]] = None
    if keep_predictions:
        pred_dir = data_root_path / "predictions" / config.replace("/", "_")
        pred_dir.mkdir(parents=True, exist_ok=True)
    else:
        pred_dir_ctx = tempfile.TemporaryDirectory(prefix="gssc_eval_preds_")
        pred_dir = Path(pred_dir_ctx.name)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu
    env["PYTHONPATH"] = f"{REPO_ROOT / 'src'}:{env.get('PYTHONPATH', '')}"

    scpnet_dir = data_root_path / "scpnet_predictions"
    semantic_kitti_root = data_root_path / "SemanticKITTI"
    if not semantic_kitti_root.exists():
        semantic_kitti_root = data_root_path

    cfg_split = str(cfg.get("split", "valid"))
    gen_split = "valid" if cfg_split == "val" else cfg_split

    if tta_mode == "d4":
        cmd = [
            sys.executable, "-m", "gssc.inference.d4_tta",
            "--checkpoint", str(ckpt_path),
            "--cold_steps", str(n_steps),
            "--output_dir", str(pred_dir),
            "--data_root", str(semantic_kitti_root),
            "--scpnet_dir", str(scpnet_dir),
            "--split", gen_split,
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
        ]

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

    eval_script = REPO_ROOT / "external" / "semantic_kitti_api" / "evaluate_completion.py"

    score_cmd = [
        sys.executable, str(eval_script),
        "--dataset", str(semantic_kitti_root),
        "--predictions", str(pred_dir),
        "--split", gen_split,
    ]
    logger.info("Stage 2 (score): %s", " ".join(score_cmd))
    score_proc = subprocess.run(score_cmd, env=env, capture_output=True, text=True)
    combined = score_proc.stdout + "\n" + score_proc.stderr
    if score_proc.returncode != 0:
        if pred_dir_ctx is not None:
            pred_dir_ctx.cleanup()
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
