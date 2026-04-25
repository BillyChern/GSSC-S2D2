"""YAML config -> argparse-style argv list adapter.

Translates Hydra-flavored YAML configs into the flat ``--key value`` argument
list that the legacy trainer expects. Keeps the config files declarative and
the trainer untouched.
"""
from __future__ import annotations
from pathlib import Path
from typing import List
import yaml


def load_yaml_to_args(path: Path) -> List[str]:
    """Flatten a YAML config dict into a list of CLI args.

    A leading underscore in a key name suppresses it (used for documentation
    fields like ``_paper_table:``).
    """
    cfg = yaml.safe_load(path.read_text())
    args = []
    for k, v in cfg.items():
        if k.startswith("_"):
            continue
        if isinstance(v, bool):
            if v:
                args.append(f"--{k}")
        elif isinstance(v, list):
            args.append(f"--{k}")
            args.append(",".join(str(x) for x in v))
        elif v is None:
            continue
        else:
            args.append(f"--{k}")
            args.append(str(v))
    return args
