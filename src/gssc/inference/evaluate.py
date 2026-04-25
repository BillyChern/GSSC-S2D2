"""GSSC-S2D2 evaluation pipeline.

Loads a checkpoint, runs Algo2 correction sampling on SemanticKITTI val, and
returns per-class IoU + mIoU + completion IoU, plus optional safety-aware
metrics (SC-mIoU, VRU-IoU, DW-IoU).

Public API:
    run_evaluation(checkpoint, config, ...) -> dict[str, float]
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


def run_evaluation(
    checkpoint: str,
    config: str,
    data_root: str,
    output: Optional[str] = None,
    gpu: str = "0",
    steps: Optional[int] = None,
    tta: Optional[str] = None,
    metrics: Optional[List[str]] = None,
) -> dict:
    """Run a full SemanticKITTI evaluation.

    Args:
        checkpoint: Path to the trained checkpoint (.safetensors or .pt).
        config: Hydra-style config name, e.g. "eval/val_1step".
        data_root: SemanticKITTI dataset root.
        output: Where to write per-class CSV (None = stdout only).
        gpu: CUDA device id (default "0").
        steps: Algo2 step count override.
        tta: TTA mode override (none / flip_y / d4).
        metrics: Which metric set to compute.

    Returns:
        Dictionary of metric name to value.

    Raises:
        FileNotFoundError: If the checkpoint or data root is missing.
    """
    metrics = metrics or ["miou", "completion_iou", "per_class"]
    ckpt_path = Path(checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    if not Path(data_root).exists():
        raise FileNotFoundError(f"Data root not found: {data_root}")

    logger.info("Loading checkpoint: %s", ckpt_path)
    logger.info("Config: %s", config)
    logger.info("Metrics: %s", metrics)

    # NOTE: this thin module dispatches to the legacy generate_predictions +
    # external/semantic_kitti_api/evaluate_completion pipeline. The full
    # in-process implementation will be wired in via P2.6 (see CONTRIBUTING.md
    # quality bar before re-implementing).
    raise NotImplementedError(
        "evaluate.run_evaluation: route to legacy CLI for now. "
        "Use scripts/infer.py to generate predictions, then "
        "external/semantic_kitti_api/evaluate_completion.py to score."
    )
