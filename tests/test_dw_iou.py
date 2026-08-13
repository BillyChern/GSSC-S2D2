"""DW-IoU is validated against every published cell of the paper's Tab. VIII.

The paper reports DW-VRU-IoU for five deployment rates across four reaction windows. Those
20 cells are the specification: this test recomputes them from the per-class VRU IoUs and
each row's measured FPS, and fails if the derivation drifts.

Per supplementary App. C, the N>1 rows hold per-class IoU at its N=1 value and vary only the
rate -- they isolate the latency term and are projections, not fresh measurements -- so all
four S^2D^2 rows share one IoU triple.
"""

import pytest

from gssc.utils.dw_iou import dw_iou, dw_vru_iou

BASE_VRU = [22.0, 18.0, 4.1]    # SCPNet base: person, bicyclist, motorcyclist
# Per-class IoUs as the paper PRINTS them (supp tab:supp_portable_full). bicyclist is 23.2, not
# the 23.3 a base+rounded-delta reconstruction (18.0 + 5.3) gives: the measurement sits near
# 23.25, so 23.2 and 23.3 are two prints of one number and the paper standardises on the table's.
# The published DW-IoU cells below came from UNROUNDED IoUs, which is why they reconcile exactly
# with neither 1-d.p. vector and why the assertion below carries a tolerance.
N1_VRU = [23.2, 23.2, 12.4]
WINDOWS = [0.5, 1.0, 2.0, 2.5]

# (row, FPS, per-class IoUs, published DW-VRU-IoU at each window)
PUBLISHED = [
    ("base",    4.95, BASE_VRU, [31.5, 50.7, 70.4, 75.7]),
    ("n1",      3.23, N1_VRU,   [29.6, 49.9, 73.7, 80.6]),
    ("n4",      1.58, N1_VRU,   [15.9, 29.0, 49.1, 56.8]),
    ("n10",     0.78, N1_VRU,   [8.2, 15.7, 28.8, 34.5]),
    ("n100",    0.09, N1_VRU,   [1.0, 2.0, 3.9, 4.9]),
]


@pytest.mark.parametrize("row,rate,ious,expected", PUBLISHED)
def test_published_cells(row, rate, ious, expected):
    for t_w, want in zip(WINDOWS, expected):
        got = dw_vru_iou(ious, rate, t_w)
        # inputs are 1 d.p., so allow the rounding they can introduce
        assert abs(got - want) < 0.15, f"{row} T_w={t_w}: got {got:.2f}, paper says {want}"


def test_monotone_in_window_and_rate():
    """More frames inside the window can only help under the independent-frames model."""
    assert dw_iou(20.0, 3.0, 1.0) < dw_iou(20.0, 3.0, 2.0)
    assert dw_iou(20.0, 3.0, 1.0) < dw_iou(20.0, 6.0, 1.0)


def test_identity_at_one_frame():
    """rate * t_w == 1 means exactly one frame, so DW-IoU collapses to IoU."""
    assert dw_iou(37.5, 1.0, 1.0) == pytest.approx(37.5)


def test_rejects_bad_input():
    with pytest.raises(ValueError):
        dw_iou(120.0, 1.0, 1.0)
    with pytest.raises(ValueError):
        dw_iou(20.0, 0.0, 1.0)
    with pytest.raises(ValueError):
        dw_vru_iou([20.0, 30.0], 3.0, 1.0)
