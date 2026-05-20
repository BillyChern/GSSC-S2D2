"""LMSCNet base-prediction reader for the cross-base S2D2 pipeline.

Mirrors :mod:`gssc.models.js3c_base`: a small free function that reads a
per-frame ``.npy`` file dumped by ``scripts/dump_lmscnet_predictions.py``
and returns a ``(256, 256, 32)`` ``int64`` array of 20-class semantic
labels in SemanticKITTI (X, Y, Z) axis order.

LMSCNet (Roldao et al., 2020) is a lightweight (394K params) 2D U-Net with
the height-as-channels trick. We use the publicly released multiscale
checkpoint (Google Drive folder linked in the official LMSCNet README).
The paper's Tab. III row cites the single-scale variant LMSCNet-SS which
is not publicly released; our val reproduction uses the released
multiscale variant and a footnote in the table documents the difference.

GT data invariant
-----------------
Predictions are pure forward-pass outputs of LMSCNet on the SemanticKITTI
.bin (sparse LiDAR input). No .label / GT files are read in the dump
pipeline. The S2D2 trainer uses this prediction as x_src; it never sees GT.

Reproducing paper Tab. III LMSCNet+S2D2 row::

    python scripts/dump_lmscnet_predictions.py --sequences trainval
    python scripts/train.py train/lmscnet_real
    python scripts/eval.py eval/lmscnet_val_1step \
        --checkpoint outputs/Exp_LMSCNet_S2D2_REAL/model_ema.safetensors
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["load_lmscnet_predictions"]

EXPECTED_SHAPE: tuple[int, int, int] = (256, 256, 32)
NUM_CLASSES: int = 20


def load_lmscnet_predictions(base_pred_dir, seq, frame_id):
    """Load a single LMSCNet per-frame voxel-grid prediction.

    Args:
        base_pred_dir: Root of the predictions tree (typically
            ``data/lmscnet_predictions``). Per-sequence subdirs ``{00..10}/``
            contain real-frame predictions.
        seq: Sequence id, e.g. ``"08"`` (zero-padded).
        frame_id: Six-digit frame id, e.g. ``"003096"``.

    Returns:
        A ``(256, 256, 32)`` array of ``int64`` semantic labels in [0, 19].

    Raises:
        FileNotFoundError: If the prediction file does not exist.
        ValueError: If shape, dtype, or value range is wrong.
    """
    root = Path(base_pred_dir)
    path = root / seq / f"{frame_id}_pred.npy"
    if not path.is_file():
        raise FileNotFoundError(
            f"lmscnet prediction missing: expected {path}; run "
            "scripts/dump_lmscnet_predictions.py first"
        )

    arr = np.load(path)
    if arr.shape != EXPECTED_SHAPE:
        raise ValueError(
            f"lmscnet prediction shape mismatch: expected {EXPECTED_SHAPE}, "
            f"got {arr.shape} (at {path})"
        )
    if arr.dtype != np.int64:
        try:
            arr = arr.astype(np.int64, copy=False)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"lmscnet prediction dtype mismatch: expected int64-castable, "
                f"got {arr.dtype} (at {path})"
            ) from exc

    arr_min = int(arr.min())
    arr_max = int(arr.max())
    if arr_min < 0 or arr_max >= NUM_CLASSES:
        raise ValueError(
            f"lmscnet prediction value range mismatch: expected "
            f"[0, {NUM_CLASSES - 1}], got [{arr_min}, {arr_max}] (at {path})"
        )
    return arr
