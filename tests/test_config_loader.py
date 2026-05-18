"""Unit tests for the YAML→argv config loader."""
from __future__ import annotations

from pathlib import Path

from gssc.utils.config_loader import load_yaml_to_args


def test_basic_kv(tmp_path: Path) -> None:
    cfg = tmp_path / "c.yaml"
    cfg.write_text("lr: 1.0e-4\nbatch_size: 4\n")
    assert load_yaml_to_args(cfg) == ["--lr", "0.0001", "--batch_size", "4"]


def test_bool_flags(tmp_path: Path) -> None:
    cfg = tmp_path / "c.yaml"
    cfg.write_text("bev_from_base: true\ncold_diffusion: false\n")
    assert load_yaml_to_args(cfg) == ["--bev_from_base"]


def test_underscore_keys_skipped(tmp_path: Path) -> None:
    cfg = tmp_path / "c.yaml"
    cfg.write_text("_paper_table: tab:foo\nlr: 1.0e-4\n")
    args = load_yaml_to_args(cfg)
    assert "_paper_table" not in args
    assert "--lr" in args


def test_list_values(tmp_path: Path) -> None:
    cfg = tmp_path / "c.yaml"
    cfg.write_text("correction_steps:\n  - 1\n  - 4\n  - 100\n")
    args = load_yaml_to_args(cfg)
    assert args == ["--correction_steps", "1,4,100"]
