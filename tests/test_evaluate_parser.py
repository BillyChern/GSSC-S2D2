"""Unit tests for the semantic_kitti_api stdout parser.

The parser converts the official SemanticKITTI scoring tool's mixed-format
output (per-class IoU as fractions, aggregate metrics already in percent,
tab-separated) into a uniform ``{metric_name: percent}`` dict.

These tests pin the contract so any future format drift is caught at
``pytest tests/`` rather than during a multi-hour eval run.
"""
from __future__ import annotations

import pytest

from gssc.inference.evaluate import _parse_eval_completion_output, _resolve_config

SAMPLE_OUTPUT = """\
  ========================== RESULTS ==========================
Validation set:
IoU avg 0.385
IoU class 1 [car] = 0.514
IoU class 2 [bicycle] = 0.243
IoU class 3 [motorcycle] = 0.355
IoU class 4 [truck] = 0.601
IoU class 5 [other-vehicle] = 0.445
IoU class 6 [person] = 0.232
IoU class 7 [bicyclist] = 0.232
IoU class 8 [motorcyclist] = 0.124
IoU class 9 [road] = 0.746
IoU class 10 [parking] = 0.616
IoU class 11 [sidewalk] = 0.539
IoU class 12 [other-ground] = 0.139
IoU class 13 [building] = 0.347
IoU class 14 [fence] = 0.302
IoU class 15 [vegetation] = 0.407
IoU class 16 [trunk] = 0.324
IoU class 17 [terrain] = 0.546
IoU class 18 [pole] = 0.378
IoU class 19 [traffic-sign] = 0.230
Precision =\t86.04
Recall =\t57.59
IoU Cmpltn =\t52.66
mIoU SSC =\t38.54
"""


def test_aggregate_metrics_parsed() -> None:
    """mIoU + Completion + Precision + Recall all extracted with correct values."""
    m = _parse_eval_completion_output(SAMPLE_OUTPUT)
    assert m["mIoU"] == pytest.approx(38.54)
    assert m["IoU_cmpl"] == pytest.approx(52.66)
    assert m["Precision"] == pytest.approx(86.04)
    assert m["Recall"] == pytest.approx(57.59)


def test_per_class_fractions_normalised_to_percent() -> None:
    """Per-class IoUs are emitted as fractions in [0,1]; we report percent."""
    m = _parse_eval_completion_output(SAMPLE_OUTPUT)
    assert m["IoU_car"] == pytest.approx(51.4)
    assert m["IoU_motorcyclist"] == pytest.approx(12.4)
    assert m["IoU_pole"] == pytest.approx(37.8)
    assert m["IoU_traffic_sign"] == pytest.approx(23.0)


def test_hyphenated_class_names_normalised() -> None:
    """`other-vehicle` and `traffic-sign` become `IoU_other_vehicle` etc."""
    m = _parse_eval_completion_output(SAMPLE_OUTPUT)
    assert "IoU_other_vehicle" in m
    assert "IoU_other_ground" in m
    assert "IoU_traffic_sign" in m
    # No hyphens leak through the key.
    for k in m:
        assert "-" not in k


def test_all_19_classes_present() -> None:
    """We never silently drop a class — Tab. I has all 19."""
    m = _parse_eval_completion_output(SAMPLE_OUTPUT)
    iou_keys = [k for k in m if k.startswith("IoU_") and k != "IoU_cmpl"]
    assert len(iou_keys) == 19


def test_empty_output_returns_empty_dict() -> None:
    """A run that crashed before scoring leaves nothing to parse."""
    assert _parse_eval_completion_output("") == {}


def test_garbage_output_returns_empty_dict() -> None:
    """Stdout from a different tool doesn't get mistakenly parsed."""
    junk = "the quick brown fox\njumps over\nthe lazy dog\n"
    assert _parse_eval_completion_output(junk) == {}


def test_partial_output_returns_partial() -> None:
    """If the scoring step crashed mid-way, we report what was seen."""
    partial = "IoU class 1 [car] = 0.514\nIoU class 2 [bicycle] = 0.243\n"
    m = _parse_eval_completion_output(partial)
    assert m == {"IoU_car": pytest.approx(51.4), "IoU_bicycle": pytest.approx(24.3)}


def test_resolve_config_loads_eval_val_1step() -> None:
    """The headline-quickstart config must always load."""
    cfg = _resolve_config("eval/val_1step")
    assert cfg["split"] == "val"
    assert cfg["sequences"] == "08"
    assert cfg["correction_steps"] == 1
    assert cfg["tta"] == "none"


def test_resolve_config_loads_eval_d4tta() -> None:
    """eval/val_d4tta is the val N=4 + D4 TTA row. The paper puts 38.73% val against
    "N=4 + D_4 TTA" (supplementary Tab. XXIII), so this config is N=4, not N=1; the
    matching test-server entry is 39.2% via configs/infer/test_d4tta.yaml."""
    cfg = _resolve_config("eval/val_d4tta")
    assert cfg["correction_steps"] == 4
    assert cfg["tta"] == "d4"


def test_resolve_config_unknown_raises() -> None:
    """Mistyped config names should fail loudly with a discoverable list."""
    import pytest as _pt
    with _pt.raises(FileNotFoundError, match="Available"):
        _resolve_config("eval/does_not_exist")
