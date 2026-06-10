"""Verify the SemanticKITTI dataset directory layout.

After downloading the SemanticKITTI raw data (link in docs/DATASET.md), run::

    python scripts/prepare_data.py --root data/SemanticKITTI

This script:
1. Verifies the sequence directory structure (sequences/00, 08, 11 present).
2. Reports the per-sequence voxel-frame count for every sequence found.

The released 256x256x32 voxel grids are downloaded ready-to-use via
scripts/download_assets.py; no on-the-fly preprocessing is performed here.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True, help="Path to SemanticKITTI dataset root")
    p.add_argument("--check-only", action="store_true",
                   help="Accepted for forward compatibility; this script only verifies the layout, so it is the default behaviour")
    a = p.parse_args()

    root = Path(a.root)
    if not root.exists():
        sys.exit(f"Dataset root does not exist: {root}\nSee docs/DATASET.md for download instructions.")

    expected = ["sequences", "sequences/00", "sequences/08", "sequences/11"]
    missing = [e for e in expected if not (root / e).exists()]
    if missing:
        sys.exit(f"Missing directories: {missing}\nLayout should be {root}/sequences/<seq>/voxels/<frame>.{{bin,label,invalid,occluded}}")

    print(f"Dataset structure OK at {root}")
    seq_dirs = sorted((root / "sequences").iterdir())
    for seq in seq_dirs:
        v = seq / "voxels"
        nbin = len(list(v.glob("*.bin"))) if v.exists() else 0
        print(f"  seq {seq.name}: {nbin} voxel frames")


if __name__ == "__main__":
    main()
