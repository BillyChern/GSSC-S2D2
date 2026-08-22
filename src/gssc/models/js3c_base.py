"""JS3C-Net base-prediction reader for the cross-base S²D² pipeline.

JS3C-Net (Yan et al., 2021) is shipped as a *prediction-only* alt-base: the
public release contains 22-sequence per-frame voxel-grid predictions
pre-dumped into ``data/js3cnet_predictions/`` rather than model weights.
Users who want to regenerate predictions from scratch can run
``scripts/dump_js3c_predictions.py`` (which depends on a local clone of the
upstream JS3C-Net repo).

The reader API mirrors the sibling :mod:`gssc.models.lmscnet_base` reader: a
small free function that reads a per-frame ``.npy`` file and returns a
``(256, 256, 32)`` ``int64`` array of 20-class semantic labels.

Reproducing the JS3C-Net cross-base row of paper tab:portable_s2d2, whose
headline is 24.32 % val mIoU (derived BEV, official ``semantic-kitti-api``,
+1.59 pp over the 22.7 % base). The supplement's 26.72 % (printed 26.7) is the
**same derived-BEV run** scored with the paper's internal training-time
evaluator -- the evaluator is what separates 26.7 from 24.3, not the BEV
source. GT BEV is a *separate* diagnostic (``eval/js3c_val_paper``) and is not
what either figure refers to::

    python scripts/dump_js3c_predictions.py --js3c-repo external/JS3C-Net \
        --semantickitti_root data/SemanticKITTI/dataset \
        --output_dir data/js3cnet_predictions \
        --sequences 08
    python scripts/eval.py eval/js3c_val_realistic \
        --checkpoint data/checkpoints/gssc_js3c/gssc_js3c_s2d2_real/model_ema.safetensors
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["load_js3c_predictions"]

# Voxel-grid contract shared with SCPNet predictions (paper §III).
EXPECTED_SHAPE: tuple[int, int, int] = (256, 256, 32)
NUM_CLASSES: int = 20  # 0=unlabeled, 1-19=semantic classes per SemanticKITTI


def load_js3c_predictions(
    base_pred_dir: str | Path,
    seq: str,
    frame_id: str,
) -> np.ndarray:
    """Load a single JS3C-Net per-frame voxel-grid prediction.

    Args:
        base_pred_dir: Root of the predictions tree (typically
            ``data/js3cnet_predictions``). Per-sequence subdirs ``{00..21}/``
            contain real-frame predictions; ``synthetic_31k/`` and
            ``synthetic_filtered/`` contain the synth-pool predictions.
        seq: Sequence id, e.g. ``"08"`` (zero-padded).
        frame_id: Six-digit frame id, e.g. ``"003096"``.

    Returns:
        A ``(256, 256, 32)`` array of ``int64`` semantic labels in
        ``[0, 19]``.

    Raises:
        FileNotFoundError: If the prediction file does not exist (run the
            dumper, see module docstring).
        ValueError: If the loaded array has the wrong shape, dtype, or
            value range.
    """
    root = Path(base_pred_dir)
    path = root / seq / f"{frame_id}_pred.npy"
    if not path.is_file():
        raise FileNotFoundError(
            f"js3c prediction missing: expected {path}; run "
            "scripts/dump_js3c_predictions.py (see docs/REPRODUCIBILITY.md)"
        )

    arr = np.load(path)
    if arr.shape != EXPECTED_SHAPE:
        raise ValueError(
            f"js3c prediction shape mismatch: expected {EXPECTED_SHAPE}, "
            f"got {arr.shape} (at {path})"
        )
    if arr.dtype != np.int64:
        try:
            arr = arr.astype(np.int64, copy=False)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"js3c prediction dtype mismatch: expected int64-castable, "
                f"got {arr.dtype} (at {path})"
            ) from exc

    arr_min = int(arr.min())
    arr_max = int(arr.max())
    if arr_min < 0 or arr_max >= NUM_CLASSES:
        raise ValueError(
            f"js3c prediction value range mismatch: expected "
            f"[0, {NUM_CLASSES - 1}], got [{arr_min}, {arr_max}] (at {path})"
        )
    return arr
