"""YAML config -> argparse-style argv list adapter.

Translates Hydra-flavored YAML configs into the flat ``--key value`` argument
list that the legacy trainer expects. Keeps the config files declarative and
the trainer untouched.

Booleans are the subtle half. ``key: true`` becomes the bare ``--key`` flag.
``key: false`` used to become NOTHING at all, which is only correct while every
boolean flag defaults to off: for a flag whose argparse default is ``True``
(``--aux_bev`` in :mod:`gssc.training.train_scene_completion`) a config could
therefore not switch it off -- the YAML said ``aux_bev: false`` and the run
trained with the auxiliary BEV head anyway. An explicit false is now emitted as
``--no_<key>`` for exactly those flags, and :func:`add_boolean_negations` gives
every trainer parser the matching off-switch.
"""
from __future__ import annotations

import argparse
import ast
import logging
from functools import cache
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

__all__ = ["add_boolean_negations", "default_on_flags", "load_yaml_to_args"]

# Trainer modules whose argparse surface these configs drive. Scanned, never listed
# by hand, so a new default-on flag is picked up the day it is added.
_TRAINING_DIR = Path(__file__).resolve().parents[1] / "training"


def load_yaml_to_args(path: Path | str) -> list[str]:
    """Flatten a YAML config dict into a list of CLI args.

    A leading underscore in a key name suppresses it (used for documentation
    fields like ``_paper_table:``). Accepts either a :class:`pathlib.Path`
    or a string path.

    ``key: false`` emits ``--no_key`` when ``key`` names a boolean flag that
    defaults to *on* (see :func:`default_on_flags`); for a flag that already
    defaults to off, emitting nothing leaves the value False, which is what the
    config asked for.
    """
    cfg = yaml.safe_load(Path(path).read_text())
    args = []
    negatable = default_on_flags()
    for k, v in cfg.items():
        if k.startswith("_"):
            continue
        if isinstance(v, bool):
            if v:
                args.append(f"--{k}")
            elif k in negatable:
                args.append(f"--no_{k}")
        elif isinstance(v, list):
            args.append(f"--{k}")
            args.append(",".join(str(x) for x in v))
        elif v is None:
            continue
        else:
            args.append(f"--{k}")
            args.append(str(v))
    return args


def add_boolean_negations(parser: argparse.ArgumentParser) -> list[str]:
    """Give every boolean flag on ``parser`` an explicit ``--no_<dest>`` twin.

    This is the receiving half of :func:`load_yaml_to_args`'s ``--no_<key>``
    emission, and it also lets a CLI user turn off a default-on flag at all.
    Options that already exist are left alone, so a hand-written ``--no_bev``
    keeps its own meaning. The twins are hidden from ``--help`` to keep the
    trainer's documented surface unchanged.

    Args:
        parser: A fully populated parser, called just before ``parse_args()``.

    Returns:
        The option strings that were added.
    """
    existing = {opt for action in parser._actions for opt in action.option_strings}
    added: list[str] = []
    for action in list(parser._actions):
        if not isinstance(action, (argparse._StoreTrueAction, argparse._StoreFalseAction)):
            continue
        negation = f"--no_{action.dest}"
        if negation in existing:
            continue
        parser.add_argument(
            negation,
            dest=action.dest,
            action="store_false" if isinstance(action, argparse._StoreTrueAction) else "store_true",
            default=action.default,
            help=argparse.SUPPRESS,
        )
        existing.add(negation)
        added.append(negation)
    return added


@cache
def default_on_flags(training_dir: str | None = None) -> frozenset[str]:
    """Boolean argparse dests that default to True across the trainer modules.

    Read statically from the trainer sources (``ast`` only -- this runs in a
    torch-free process), so the set cannot drift from the parsers it describes.
    An unreadable or absent source yields an empty set: the adapter then behaves
    exactly as it did before, rather than emitting a flag nothing accepts.
    """
    root = Path(training_dir) if training_dir else _TRAINING_DIR
    flags: set[str] = set()
    for src in sorted(root.glob("train_*.py")) if root.is_dir() else []:
        try:
            tree = ast.parse(src.read_text(encoding="utf-8"))
        except (OSError, SyntaxError) as exc:  # pragma: no cover - defensive
            logger.warning("Cannot scan %s for default-on flags: %s", src, exc)
            continue
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "add_argument"):
                continue
            kwargs = {kw.arg: kw.value for kw in node.keywords if kw.arg}
            action = kwargs.get("action")
            if not (isinstance(action, ast.Constant)
                    and action.value in ("store_true", "store_false")):
                continue
            default = kwargs.get("default")
            is_on = (default.value is True if isinstance(default, ast.Constant)
                     else action.value == "store_false")
            if not is_on:
                continue
            dest = kwargs.get("dest")
            if isinstance(dest, ast.Constant) and isinstance(dest.value, str):
                flags.add(dest.value)
                continue
            options = [a.value for a in node.args
                       if isinstance(a, ast.Constant) and isinstance(a.value, str)]
            if options:
                flags.add(options[-1].lstrip("-").replace("-", "_"))
    return frozenset(flags)
