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


def test_shipped_configs_parse_and_declare_their_consumer_keys() -> None:
    """Every shipped infer/eval config must load and carry the keys its runner reads.

    Guards two failure modes the reproduction matrix depends on: a config named in the
    paper that does not exist, and a config whose keys the runner silently ignores (the
    v2.1.0 regression, where inert keys meant a config did not control its own run).
    """
    import yaml

    root = Path(__file__).resolve().parents[1] / "configs"

    infer_required = {"split", "correction_steps", "tta"}
    for cfg in sorted((root / "infer").glob("*.yaml")):
        d = yaml.safe_load(cfg.read_text())
        missing = infer_required - d.keys()
        assert not missing, f"{cfg.name} missing {missing}"
        assert d["tta"] in {"none", "flip_y", "d4"}, f"{cfg.name}: bad tta {d['tta']!r}"
        assert d["split"] in {"val", "test"}, f"{cfg.name}: bad split {d['split']!r}"
        # sequences is informational but must not contradict the split
        seqs = str(d.get("sequences", "")).split(",")
        if d["split"] == "val":
            assert seqs == ["08"], f"{cfg.name}: val split must name seq 08, got {seqs}"
        else:
            assert seqs[0].strip() == "11", f"{cfg.name}: test split must start at seq 11"

    # Eval configs span more than one runner: bev_secondary drives the BEV path and marks
    # the sampler keys inert with a leading underscore. So check only what is shared --
    # the file loads, yields argv, and any split it does declare is a real one.
    # bev_secondary drives a different runner and is purely documentary: every key is
    # underscore-prefixed, so it yields no argv by design. Excluded deliberately.
    for cfg in sorted((root / "eval").glob("*.yaml")):
        if cfg.name == "bev_secondary.yaml":
            d = yaml.safe_load(cfg.read_text())
            assert all(k.startswith("_") for k in d), (
                "bev_secondary is exempt only because every key is inert; a live key "
                "means it now controls its run and must be checked like the others"
            )
            continue
        d = yaml.safe_load(cfg.read_text())
        assert load_yaml_to_args(cfg), f"{cfg.name} produced no argv"
        if "split" in d:
            assert d["split"] in {"val", "test"}, f"{cfg.name}: bad split {d['split']!r}"


def test_headline_test_config_is_the_predicate_admitted_one() -> None:
    """The 38.8% headline needs a one-step, no-TTA test config; 39.2% must stay separate."""
    import yaml

    root = Path(__file__).resolve().parents[1] / "configs" / "infer"
    one = yaml.safe_load((root / "test_1step.yaml").read_text())
    assert (one["split"], one["correction_steps"], one["tta"]) == ("test", 1, "none")

    tta = yaml.safe_load((root / "test_d4tta.yaml").read_text())
    assert (tta["split"], tta["correction_steps"], tta["tta"]) == ("test", 4, "d4")
