#!/usr/bin/env python3
"""Self-contained KITTI-360 SSC scorer (16-shared-class + completion IoU).

Reads KITTI-360 voxel GT (.label uint16 flat + .invalid RAW uint8 per-voxel) and
prediction .label files (uint16 flat, KITTI-360 ORIGINAL 0-255 labels), maps both
to the KITTI-360 19-class train space via the official kitti360.yaml learning_map,
applies the eval mask (drop invalid==1 and GT-unlabeled), and accumulates a single
confusion matrix over a sequence.

Reports, over the 16 shared SemanticKITTI∩KITTI360 classes (train idx 1..16):
  * per-class IoU
  * mIoU (mean over the 16 shared classes)
  * completion IoU (binary occupied-vs-empty IoU over all non-empty GT/pred)
Also prints the full-19-class mIoU (mean over train idx 1..18, including the two
structural-zero classes other-structure/other-object) for reference.

This matches the protocol in scripts/eval_kitti360.py and the official
external/semantic_kitti_api/evaluate_completion.py masking, but computes the
16-shared subset mean directly so the structural-zero classes do not silently
deflate the headline number.

Usage:
    python scripts/score_kitti360.py \
        --gt_voxels  <K360_root>/data_2d_raw/<seq>/voxels \
        --pred_dir   data/predictions/kitti360_zeroshot/sequences/06/predictions \
        --datacfg    external/semantic_kitti_api/config/kitti360.yaml \
        --tag        SCPNet+S2D2
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import yaml

GRID = (256, 256, 32)
NVOX = GRID[0] * GRID[1] * GRID[2]

# 16 shared classes (KITTI-360 train idx) a SemanticKITTI-trained model can emit.
SHARED_TRAIN_IDX = tuple(range(1, 17))
# All non-empty KITTI-360 train classes (incl. structural-zero 17/18).
ALL_TRAIN_IDX = tuple(range(1, 19))


def build_remap_lut(datacfg: Path) -> tuple[np.ndarray, dict[int, str], dict[int, int]]:
    """KITTI-360 original (0-255) -> train idx LUT, mirroring evaluate_completion.py.

    0 stays empty; learning_map-0 targets become 255 ('invalid'); only original 0
    stays 0 ('empty'). Returns (lut, train_idx->name, train_idx->orig_label).
    """
    data = yaml.safe_load(datacfg.read_text())
    class_remap = data["learning_map"]
    inv_remap = data["learning_map_inv"]
    labels = data["labels"]
    maxkey = max(class_remap.keys())
    lut = np.zeros(maxkey + 100, dtype=np.int32)
    lut[list(class_remap.keys())] = list(class_remap.values())
    lut[lut == 0] = 255
    lut[0] = 0
    names = {int(k): labels[int(v)] for k, v in inv_remap.items()}
    return lut, names, {int(k): int(v) for k, v in inv_remap.items()}


def unpack_bits(compressed: np.ndarray) -> np.ndarray:
    out = np.zeros(compressed.shape[0] * 8, dtype=np.uint8)
    for i in range(8):
        out[i::8] = (compressed >> (7 - i)) & 1
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gt_voxels", required=True, help="KITTI-360 voxels dir with <frame>.label/.invalid")
    ap.add_argument("--pred_dir", required=True, help="Dir of <frame>.label predictions (orig 0-255)")
    ap.add_argument("--datacfg", required=True)
    ap.add_argument("--tag", default="model")
    ap.add_argument("--invalid_packed", action="store_true",
                    help="Set if .invalid is already bit-packed (SemanticKITTI form). "
                         "KITTI-360 ships RAW uint8 per-voxel (default: raw).")
    args = ap.parse_args()

    lut, names, _ = build_remap_lut(Path(args.datacfg))
    n_train = 19  # 0..18

    gt_dir = Path(args.gt_voxels)
    pred_dir = Path(args.pred_dir)
    gt_files = sorted(gt_dir.glob("*.label"))
    if not gt_files:
        raise SystemExit(f"No GT .label files under {gt_dir}")

    conf = np.zeros((n_train, n_train), dtype=np.int64)
    n_scored = 0
    for gf in gt_files:
        fid = gf.stem
        pf = pred_dir / f"{fid}.label"
        if not pf.exists():
            raise SystemExit(f"Missing prediction {pf}")
        gt = np.fromfile(gf, dtype=np.uint16)
        pred = np.fromfile(pf, dtype=np.uint16)
        if gt.size != NVOX or pred.size != NVOX:
            raise SystemExit(f"{fid}: bad sizes gt={gt.size} pred={pred.size}")
        inv_raw = np.fromfile(gf.with_suffix(".invalid"), dtype=np.uint8)
        invalid = unpack_bits(inv_raw) if args.invalid_packed else inv_raw
        invalid = invalid[:NVOX]

        gt_t = lut[gt]
        pred_t = lut[pred]
        mask = (gt_t != 255) & (invalid == 0)
        gt_m = gt_t[mask]
        pred_m = pred_t[mask]
        # pred 255 (shouldn't happen for our writeback) -> treat as empty 0.
        pred_m = np.where(pred_m == 255, 0, pred_m)
        np.add.at(conf, (gt_m, pred_m), 1)
        n_scored += 1

    def class_iou(c: int) -> float:
        tp = conf[c, c]
        denom = conf[c, :].sum() + conf[:, c].sum() - tp
        return float(tp / denom * 100) if denom > 0 else float("nan")

    # Completion IoU: binary occupied (>=1) vs empty (0).
    tp = conf[1:, 1:].sum()
    fn = conf[1:, 0].sum()
    fp = conf[0, 1:].sum()
    ciou = float(tp / (tp + fn + fp) * 100) if (tp + fn + fp) > 0 else 0.0

    shared_ious = [class_iou(c) for c in SHARED_TRAIN_IDX]
    all_ious = [class_iou(c) for c in ALL_TRAIN_IDX]
    miou_shared = float(np.nanmean(shared_ious))
    miou_all = float(np.nanmean([0.0 if np.isnan(x) else x for x in all_ious]))

    print(f"\n========== KITTI-360 scene06 SSC: {args.tag} ==========")
    print(f"frames scored: {n_scored}")
    print("per-class IoU (16 shared classes):")
    for c, iou in zip(SHARED_TRAIN_IDX, shared_ious):
        v = "  nan" if np.isnan(iou) else f"{iou:6.2f}"
        print(f"  [{c:2d}] {names[c]:16s}: {v}")
    print(f"\n  mIoU (16 shared classes)      = {miou_shared:.2f}%")
    print(f"  Completion IoU                = {ciou:.2f}%")
    print(f"  mIoU (full 19-class, incl. 0-pad other-structure/other-object) = {miou_all:.2f}%")
    print("=" * 56)


if __name__ == "__main__":
    main()
