"""Convert ``infer.py`` ``.label`` output into the ``_pred.npy`` tree ``eval.py`` reads.

``scripts/infer.py`` writes SemanticKITTI *submission* format::

    <output>/sequences/<SEQ>/predictions/<frame>.label   # flat uint16, ORIGINAL label space

``scripts/eval.py`` sources ``x_src`` from a *base prediction* tree
(:mod:`gssc.data.semantickitti`, ``base_pred_dir``)::

    <base_pred_dir>/<SEQ>/<frame>_pred.npy               # (256,256,32) uint8, LEARNING-MAP space

The two disagree in three ways -- an extra ``predictions/`` path level, the file
extension/container, and the label space -- so pointing ``base_pred_dir`` straight at an
``infer.py`` output makes every frame miss the ``scpnet_file.exists()`` test and get
dropped by the ``continue`` at ``semantickitti.py:256``. The eval then reports a metric
over zero frames rather than failing, which is why the mismatch reads as a silent no-op.

This script closes that gap. It is what makes ``configs/eval/round2_a.yaml`` runnable:
Round-1 is produced by ``infer.py``, converted here, and fed back as Round-2's source.

Usage::

    python scripts/infer.py infer/val_d4tta --checkpoint <ckpt> --output round1/
    python scripts/label_to_base_pred.py round1/ --output round1_src/
    python scripts/eval.py eval/round2_a --checkpoint <ckpt>   # base_pred_dir: round1_src

Self-test (no data required)::

    python scripts/label_to_base_pred.py --selftest
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from gssc.data.learning_map import LEARNING_MAP_ARRAY

#: SemanticKITTI SSC voxel grid. ``infer.py`` flattens this in C order before writing.
GRID_SHAPE = (256, 256, 32)
GRID_SIZE = GRID_SHAPE[0] * GRID_SHAPE[1] * GRID_SHAPE[2]

logger = logging.getLogger("gssc.label2pred")


def convert_label_array(raw: np.ndarray) -> np.ndarray:
    """Map one flat ``.label`` buffer to a ``(256,256,32)`` learning-map volume.

    Args:
        raw: Flat ``uint16`` labels in the ORIGINAL SemanticKITTI space, as written by
            ``gssc.inference.d4_tta`` / ``generate_predictions`` via ``LEARNING_MAP_INV``.

    Returns:
        ``(256,256,32)`` ``uint8`` array with values in ``0..19``.

    Raises:
        ValueError: If the buffer is not exactly one voxel grid, or carries a label
            outside the table the official ``learning_map`` defines.
    """
    if raw.size != GRID_SIZE:
        raise ValueError(
            f"expected {GRID_SIZE} voxels ({GRID_SHAPE}), got {raw.size}"
        )
    idx = raw.astype(np.int64)
    if idx.min() < 0 or idx.max() >= LEARNING_MAP_ARRAY.shape[0]:
        raise ValueError(
            f"label out of range for the official learning_map: "
            f"min={idx.min()} max={idx.max()} (table covers 0..{LEARNING_MAP_ARRAY.shape[0] - 1})"
        )
    return LEARNING_MAP_ARRAY[idx].reshape(GRID_SHAPE).astype(np.uint8)


def convert_tree(src: Path, dst: Path, sequences: list[str] | None = None,
                 overwrite: bool = False) -> int:
    """Convert every ``.label`` under an ``infer.py`` output into ``<SEQ>/<frame>_pred.npy``.

    Args:
        src: An ``infer.py`` ``--output`` directory (contains ``sequences/<SEQ>/predictions/``).
        dst: Destination root; becomes a valid ``base_pred_dir``.
        sequences: Restrict to these sequence ids. ``None`` converts all found.
        overwrite: Re-convert frames whose ``_pred.npy`` already exists.

    Returns:
        Number of frames written.

    Raises:
        FileNotFoundError: If ``src`` has no ``sequences/`` level to walk.
    """
    seq_root = src / "sequences" if (src / "sequences").is_dir() else src
    if not seq_root.is_dir():
        raise FileNotFoundError(f"no sequences/ under {src}")

    total = 0
    for seq_dir in sorted(p for p in seq_root.iterdir() if p.is_dir()):
        seq = seq_dir.name
        if sequences and seq not in sequences:
            continue
        pred_dir = seq_dir / "predictions" if (seq_dir / "predictions").is_dir() else seq_dir
        labels = sorted(pred_dir.glob("*.label"))
        if not labels:
            logger.warning("Seq %s: no .label files under %s", seq, pred_dir)
            continue
        out_dir = dst / seq
        out_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Seq %s: %d frame(s) -> %s", seq, len(labels), out_dir)

        for label_path in labels:
            out_path = out_dir / f"{label_path.stem}_pred.npy"
            if out_path.exists() and not overwrite:
                continue
            raw = np.fromfile(label_path, dtype=np.uint16)
            try:
                vol = convert_label_array(raw)
            except ValueError as exc:
                logger.error("%s: %s", label_path, exc)
                raise
            np.save(out_path, vol)
            total += 1

    logger.info("Done, wrote %d _pred.npy file(s)", total)
    return total


def _selftest() -> int:
    """Replay the historical defect: a round-trip that must be exact, and a silent no-op."""
    import tempfile

    from gssc.inference.d4_tta import LEARNING_MAP_INV_ARRAY as INV

    failures = 0

    def check(name: str, got: object, want: object) -> None:
        nonlocal failures
        ok = got == want
        print(f"  [{'ok' if ok else 'FAIL'}] {name}" + ("" if ok else f"  got={got!r} want={want!r}"))
        if not ok:
            failures += 1

    # 1. THE defect this script exists for: every class must survive
    #    learning-map -> INV (what infer.py writes) -> learning-map (what we read back).
    broken = [c for c in range(20) if int(LEARNING_MAP_ARRAY[INV[c]]) != c]
    check("round-trip class -> .label -> _pred.npy is exact for all 20 classes", broken, [])

    # 2. A full grid converts to the shape/dtype/space eval.py's loader expects.
    rng = np.random.default_rng(0)
    train = rng.integers(0, 20, size=GRID_SIZE, dtype=np.int64)
    raw = INV[train].astype(np.uint16)
    vol = convert_label_array(raw)
    check("converted volume shape", vol.shape, GRID_SHAPE)
    check("converted volume dtype", vol.dtype, np.dtype(np.uint8))
    check("converted volume is byte-identical to the source classes",
          bool((vol.reshape(-1) == train.astype(np.uint8)).all()), True)

    # 3. Reshape ORDER matters: infer.py C-order-flattens (256,256,32). A Fortran-order
    #    read would pass every check above except a positional one, so pin position.
    train3 = train.reshape(GRID_SHAPE)
    check("voxel at (7,11,13) lands at the same index",
          int(vol[7, 11, 13]), int(train3[7, 11, 13]))

    # 4. Wrong-sized buffers must raise, not silently produce a short volume.
    try:
        convert_label_array(np.zeros(GRID_SIZE - 1, dtype=np.uint16))
        check("short buffer raises ValueError", False, True)
    except ValueError:
        check("short buffer raises ValueError", True, True)

    # 5. Out-of-table labels must raise rather than index-error or wrap.
    try:
        convert_label_array(np.full(GRID_SIZE, 60000, dtype=np.uint16))
        check("out-of-range label raises ValueError", False, True)
    except ValueError:
        check("out-of-range label raises ValueError", True, True)

    # 6. LIVE: the historical silent no-op. An infer.py-shaped tree fed to eval.py's
    #    loader convention finds ZERO frames before conversion and ALL of them after.
    with tempfile.TemporaryDirectory(dir=str(REPO_ROOT / "outputs")
                                     if (REPO_ROOT / "outputs").is_dir() else None) as td:
        tmp = Path(td)
        src = tmp / "round1"
        pdir = src / "sequences" / "08" / "predictions"
        pdir.mkdir(parents=True)
        small = INV[rng.integers(0, 20, size=GRID_SIZE, dtype=np.int64)].astype(np.uint16)
        for fid in ("000000", "000001"):
            small.tofile(pdir / f"{fid}.label")

        # What eval.py actually tests: <base_pred_dir>/<SEQ>/<frame>_pred.npy
        before = sum((src / "sequences" / "08" / f"{f}_pred.npy").exists()
                     for f in ("000000", "000001"))
        check("BEFORE conversion eval.py finds 0 of 2 frames (the silent no-op)", before, 0)

        dst = tmp / "round1_src"
        written = convert_tree(src, dst)
        check("conversion wrote both frames", written, 2)
        after = sum((dst / "08" / f"{f}_pred.npy").exists() for f in ("000000", "000001"))
        check("AFTER conversion eval.py finds 2 of 2 frames", after, 2)
        check("a converted frame loads at the base-prediction contract",
              (lambda a: (a.shape, a.dtype, bool(a.max() <= 19)))(np.load(dst / "08" / "000000_pred.npy")),
              (GRID_SHAPE, np.dtype(np.uint8), True))

    print(f"\n{'SELFTEST PASS' if failures == 0 else f'SELFTEST FAIL ({failures})'}")
    return 1 if failures else 0


def main() -> None:
    """Parse arguments and convert, or run the self-test."""
    p = argparse.ArgumentParser(
        description="Convert infer.py .label output into an eval.py base_pred_dir.")
    p.add_argument("source", nargs="?", help="infer.py --output directory")
    p.add_argument("--output", help="Destination base_pred_dir")
    p.add_argument("--sequences", nargs="+", default=None, help="Restrict to these sequence ids")
    p.add_argument("--overwrite", action="store_true", help="Re-convert existing frames")
    p.add_argument("--selftest", action="store_true", help="Run the self-test and exit")
    a = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if a.selftest:
        sys.exit(_selftest())
    if not a.source or not a.output:
        p.error("source and --output are required (or pass --selftest)")

    n = convert_tree(Path(a.source), Path(a.output), a.sequences, a.overwrite)
    if n == 0:
        logger.warning("Wrote 0 frames -- nothing to feed eval.py. Check --sequences and the "
                       "source layout (expected <src>/sequences/<SEQ>/predictions/*.label).")


if __name__ == "__main__":
    main()
