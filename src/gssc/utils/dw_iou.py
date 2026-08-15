"""Detection-Window IoU (DW-IoU): the reaction-time-weighted metric of the paper.

DW-IoU is *derived*, not measured. For a class with segmentation IoU ``iou`` and a system
delivering ``rate`` frames per second, the probability that at least one of the frames
arriving inside a reaction window ``t_w`` carries the class, under an independent-frames
model, is

    DW-IoU_c = 1 - (1 - IoU_c) ** (rate * t_w)

so nothing needs to be re-run: the inputs are the per-class IoUs an eval already reports and
each system's measured end-to-end rate. DW-VRU-IoU is the mean over person, bicyclist and
motorcyclist -- the per-class mean, not the combiner applied to an aggregate VRU-IoU, because
the map is concave for rate*t_w > 1 and the class-wise mean is then the conservative reading.

Because consecutive single-frame predictions are temporally correlated, DW-IoU is an
optimistic upper bound on the benefit of a faster refresh, not a detection probability.

Usage
-----
    python -m gssc.utils.dw_iou --iou 22.0 18.0 4.1 --rate 4.95
    python -m gssc.utils.dw_iou --metrics-json out/metrics.json --rate 3.23
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

VRU_CLASSES = ("person", "bicyclist", "motorcyclist")
DEFAULT_WINDOWS = (0.5, 1.0, 2.0, 2.5)


def dw_iou(iou_pct: float, rate: float, t_w: float) -> float:
    """DW-IoU for one class, in percent. ``iou_pct`` is a percentage, ``rate`` in Hz."""
    if not 0.0 <= iou_pct <= 100.0:
        raise ValueError(f"iou_pct out of range: {iou_pct}")
    if rate <= 0 or t_w <= 0:
        raise ValueError("rate and t_w must be positive")
    return 100.0 * (1.0 - (1.0 - iou_pct / 100.0) ** (rate * t_w))


def dw_vru_iou(ious_pct: list[float], rate: float, t_w: float) -> float:
    """Per-class-mean DW-VRU-IoU over the three VRU classes, in percent."""
    if len(ious_pct) != len(VRU_CLASSES):
        raise ValueError(f"expected {len(VRU_CLASSES)} IoUs for {VRU_CLASSES}, got {len(ious_pct)}")
    return sum(dw_iou(v, rate, t_w) for v in ious_pct) / len(ious_pct)


def _vru_from_metrics_json(path: Path) -> list[float]:
    d = json.loads(path.read_text())
    out = []
    for c in VRU_CLASSES:
        for key in (f"IoU_{c}", c, f"iou_{c}"):
            if key in d:
                out.append(float(d[key]))
                break
        else:
            raise KeyError(f"no per-class IoU for {c!r} in {path}; run eval with --metrics per_class")
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Compute DW-VRU-IoU from per-class IoUs and a rate.")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--iou", type=float, nargs=3, metavar=("PERSON", "BICYCLIST", "MOTORCYCLIST"),
                     help="Per-class VRU IoUs in percent")
    src.add_argument("--metrics-json", type=Path,
                     help="metrics JSON from scripts/eval.py --metrics per_class")
    p.add_argument("--rate", type=float, required=True, help="Deployed end-to-end rate in FPS")
    p.add_argument("--windows", type=float, nargs="+", default=list(DEFAULT_WINDOWS),
                   help="Reaction windows in seconds")
    a = p.parse_args()
    ious = a.iou if a.iou else _vru_from_metrics_json(a.metrics_json)
    print("per-class IoU: " + ", ".join(f"{c}={v:.1f}" for c, v in zip(VRU_CLASSES, ious)))
    print(f"rate: {a.rate:.2f} FPS")
    for t_w in a.windows:
        print(f"  T_w={t_w:>4}s   DW-VRU-IoU = {dw_vru_iou(ious, a.rate, t_w):5.1f} %")


if __name__ == "__main__":
    main()
