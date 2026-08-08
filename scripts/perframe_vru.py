#!/usr/bin/env python
"""Per-frame vulnerable-road-user IoU: a frozen base against its S2D2 refinement.

Produces the numbers in supplementary Table XX (per-frame VRU regression on the shipped
SCPNet base). A class-mean IoU cannot rule out a heavy left tail, so this scores every
frame separately and counts the frames on which refinement *lowers* a VRU class.

Two things make this easy to get wrong, and both are handled here:

1. **The learning map is read from ``semantic-kitti.yaml``, never written by hand.** The
   moving variants (253/254/255) fold into bicyclist/person/motorcyclist; dropping them
   loses the majority of some frames' person voxels.
2. **The ignore construction must match the official evaluator exactly.** After the
   learning map, ``evaluate_completion.py`` maps every label landing on class 0 to 255 and
   excludes it, keeping only original label 0 as "empty" (a real class in completion).
   Folding those outlier labels into class 0 instead inflates false positives on every
   class and reads ~0.7 pp low on mIoU.

The aggregate gate is therefore not optional: ``--gate`` asserts the run reproduces the
published per-class cells for both arms before any per-frame statistic is reported.

Usage:
    python scripts/perframe_vru.py \
        --voxels  data/SemanticKITTI/dataset/sequences/08/voxels \
        --base    data/scpnet_predictions/08 \
        --refined data/predictions/headline_n1/sequences/08/predictions \
        --gate
"""
from __future__ import annotations

import argparse
import json
import pathlib
from multiprocessing import Pool
from typing import Dict, Tuple

import numpy as np
import yaml

NCLS = 20
VRU: Dict[int, str] = {6: "person", 7: "bicyclist", 8: "motorcyclist"}

# Published cells the gate checks against (supplementary Tables XVI/XX).
GATE_BASE = {"person": 22.0, "bicyclist": 18.0, "motorcyclist": 4.1, "miou": 36.17}
GATE_REFINED = {"person": 23.2, "bicyclist": 23.3, "motorcyclist": 12.4, "miou": 38.54}

_LUT: np.ndarray | None = None
_PATHS: Tuple[str, str, str] | None = None


def build_lut(datacfg: str) -> np.ndarray:
    """Replicate ``evaluate_completion.py``'s remap LUT exactly."""
    lm = yaml.safe_load(open(datacfg))["learning_map"]
    lut = np.zeros(max(lm) + 100, dtype=np.int32)
    lut[list(lm.keys())] = list(lm.values())
    lut[lut == 0] = 255  # everything that lands on 0 is ignored ...
    lut[0] = 0  # ... except original 0, which is "empty"
    return lut


def _init(datacfg: str, voxels: str, base: str, refined: str) -> None:
    global _LUT, _PATHS
    _LUT = build_lut(datacfg)
    _PATHS = (voxels, base, refined)


def _frame(fid: str):
    assert _LUT is not None and _PATHS is not None
    vox, base_dir, ref_dir = _PATHS
    gt = _LUT[np.fromfile(f"{vox}/{fid}.label", dtype=np.uint16)]
    invalid = np.unpackbits(np.fromfile(f"{vox}/{fid}.invalid", dtype=np.uint8))
    keep = (invalid == 0) & (gt != 255)

    gt = gt[keep].astype(np.int64)
    base = np.load(f"{base_dir}/{fid}_pred.npy").reshape(-1).astype(np.int32)[keep]
    ref = _LUT[np.fromfile(f"{ref_dir}/{fid}.label", dtype=np.uint16)][keep]

    cb = np.bincount(gt * NCLS + base, minlength=NCLS * NCLS)
    cr = np.bincount(gt * NCLS + ref, minlength=NCLS * NCLS)

    per = {}
    for c in VRU:
        g = gt == c
        ng = int(g.sum())
        if ng == 0:
            continue
        b, r = base == c, ref == c
        tb, tr = int((g & b).sum()), int((g & r).sum())
        ub, ur = ng + int(b.sum()) - tb, ng + int(r.sum()) - tr
        per[c] = (tb / ub if ub else 0.0, tr / ur if ur else 0.0, tb, tr)
    return cb, cr, per


def iou(conf: np.ndarray) -> np.ndarray:
    m = conf.reshape(NCLS, NCLS).astype(np.float64)
    tp = np.diag(m)
    with np.errstate(divide="ignore", invalid="ignore"):
        return tp / (tp + (m.sum(0) - tp) + (m.sum(1) - tp))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--voxels", required=True, help="GT .label/.invalid directory for the sequence")
    p.add_argument("--base", required=True, help="Frozen-base *_pred.npy directory")
    p.add_argument("--refined", required=True, help="Refined .label directory")
    p.add_argument(
        "--datacfg",
        default=str(pathlib.Path(__file__).resolve().parents[1] / "external/semantic_kitti_api/config/semantic-kitti.yaml"),
    )
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--output", default=None, help="Write results as JSON here")
    p.add_argument("--gate", action="store_true", help="Assert the published cells are reproduced")
    p.add_argument("--tol", type=float, default=0.05, help="Gate tolerance in pp")
    a = p.parse_args()

    ids = sorted(f.stem for f in pathlib.Path(a.voxels).glob("*.label"))
    print(f"frames: {len(ids)}")

    CB = np.zeros(NCLS * NCLS, dtype=np.int64)
    CR = np.zeros(NCLS * NCLS, dtype=np.int64)
    rows = []
    with Pool(a.workers, initializer=_init, initargs=(a.datacfg, a.voxels, a.base, a.refined)) as pool:
        for cb, cr, per in pool.imap_unordered(_frame, ids, chunksize=8):
            CB += cb
            CR += cr
            rows.append(per)

    ib, ir = iou(CB), iou(CR)
    got_base = {n: round(100 * ib[c], 2) for c, n in VRU.items()}
    got_ref = {n: round(100 * ir[c], 2) for c, n in VRU.items()}
    got_base["miou"] = round(100 * float(np.nanmean(ib[1:20])), 2)
    got_ref["miou"] = round(100 * float(np.nanmean(ir[1:20])), 2)

    print("\naggregate (%)          base   refined")
    for k in ("person", "bicyclist", "motorcyclist", "miou"):
        print(f"  {k:<20}{got_base[k]:>6}{got_ref[k]:>10}")

    if a.gate:
        # Compare at the precision each published cell carries: per-class IoU is printed to
        # 1 d.p. and mIoU to 2. A value sitting exactly on a rounding boundary (bicyclist
        # 23.25 -> 23.3) agrees with the paper and must not fail the gate.
        bad = []
        for arm, g, want in (("base", got_base, GATE_BASE), ("refined", got_ref, GATE_REFINED)):
            for k, w in want.items():
                # Half-width of the published cell's last digit, plus float slack: 23.25
                # against a printed 23.3 agrees, and 23.3 is not exactly representable.
                half = 0.005 if k == "miou" else 0.05
                if abs(g[k] - w) > max(a.tol, half) + 1e-9:
                    bad.append(f"{arm}.{k}: got {g[k]}, published {w}")
        if bad:
            raise SystemExit("GATE FAILED — do not trust the per-frame numbers:\n  " + "\n  ".join(bad))
        print("\nGATE OK: both arms reproduce the published cells at their printed precision.")

    out = {}
    print("\nper-frame regression (base -> refined)")
    for c, n in VRU.items():
        present = [r[c] for r in rows if c in r]
        det = [v for v in present if v[2] > 0]
        worse_det = [v for v in det if v[1] < v[0]]
        zeroed = [v for v in det if v[3] == 0]
        out[n] = {
            "in_gt": len(present),
            "base_recovers": len(det),
            "iou_falls": len(worse_det),
            "iou_falls_pct": round(100 * len(worse_det) / max(1, len(det)), 1),
            "driven_to_zero": len(zeroed),
            "unrestricted_pct": round(
                100 * sum(v[1] < v[0] for v in present) / max(1, len(present)), 1
            ),
        }
        d = out[n]
        print(
            f"  {n:<13} in GT {d['in_gt']:>5} | base recovers {d['base_recovers']:>5} | "
            f"falls {d['iou_falls']:>4} ({d['iou_falls_pct']}%) | to zero {d['driven_to_zero']:>3}"
        )

    if a.output:
        pathlib.Path(a.output).write_text(
            json.dumps({"aggregate": {"base": got_base, "refined": got_ref}, "per_frame": out}, indent=2)
        )


if __name__ == "__main__":
    main()
