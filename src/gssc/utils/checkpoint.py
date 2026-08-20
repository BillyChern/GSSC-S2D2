"""Checkpoint loading helpers shared by the release inference paths.

Two failure modes on the deployment path motivated this module, and both have
already happened in this repository:

1. ``load_state_dict(..., strict=False)`` whose ``_IncompatibleKeys`` result is
   discarded. The BEV evaluator reconstructed its denoiser at
   ``input_resolution=64`` / ``cond_channels=128`` against a 256/64 checkpoint,
   48 attention tensors stayed at initialisation, and the run still printed a
   plausible mIoU. :func:`assert_bound` refuses to score such a load.
2. Architecture kwargs pinned as literals in the inference script while the
   shipped checkpoint declares them in its own ``config.json`` /
   ``ckpt["config"]``. :func:`load_checkpoint_config` reads that declaration so
   the literals can be demoted to fallbacks for older files.
"""
from __future__ import annotations

import json
import logging
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

__all__ = ["assert_bound", "config_value", "load_checkpoint_config"]

# Nested blocks inside a checkpoint config that describe how to REBUILD the model.
# Flattened after the top level so the specific block wins over generic metadata.
_NESTED_BLOCKS = ("train_config", "reconstruction", "config")


def assert_bound(name: str, result: object, ckpt_path: object) -> None:
    """Refuse to score a module whose weights did not fully bind.

    ``load_state_dict(strict=False)`` is required on these paths (EMA shadows
    may omit non-float buffers) but it also silently tolerates an architecture
    that does not match the checkpoint: the unmatched tensors keep their random
    initialisation and the run still returns a number.

    Args:
        name: Module label used in the error message.
        result: The ``_IncompatibleKeys`` returned by ``load_state_dict``.
        ckpt_path: Checkpoint being loaded, echoed so the message is actionable.

    Raises:
        RuntimeError: If any key is missing or unexpected.
    """
    missing = list(getattr(result, "missing_keys", []))
    unexpected = list(getattr(result, "unexpected_keys", []))
    if not missing and not unexpected:
        logger.info("%s: all weights bound", name)
        return
    raise RuntimeError(
        f"{name}: {len(missing)} missing and {len(unexpected)} unexpected key(s) when "
        f"loading {ckpt_path}. The architecture does not match the checkpoint, so any "
        f"score from it would be meaningless. Check the architecture keys the "
        f"checkpoint declares (its sibling config.json, or ckpt['config']). "
        f"First missing: {missing[:3]}; first unexpected: {unexpected[:3]}"
    )


def load_checkpoint_config(
    ckpt_path: str | os.PathLike[str],
    ckpt: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the architecture config a shipped checkpoint declares about itself.

    Sources, merged in this order (later wins):

    * ``<checkpoint dir>/config.json`` -- the per-checkpoint metadata shipped in
      the asset bundle beside ``model.safetensors`` / ``model_ema.safetensors``;
    * an in-file ``ckpt["config"]`` mapping, for the ``.pt`` layout written by
      :mod:`gssc.training.train_scene_completion`.

    Nested ``train_config`` / ``reconstruction`` / ``config`` blocks are
    flattened into the result, so a caller asks for ``num_classes`` without
    knowing which layout it came from. Never raises: a checkpoint with no
    declaration yields ``{}`` and the caller falls back to its literals.

    Args:
        ckpt_path: Path to the weights file (or its directory).
        ckpt: Already-loaded ``.pt`` mapping, if the caller has one.

    Returns:
        Flat mapping of scalar config keys to declared values.
    """
    merged: dict[str, Any] = {}
    path = Path(ckpt_path)
    cfg_json = (path if path.is_dir() else path.parent) / "config.json"
    if cfg_json.is_file():
        try:
            merged.update(_flatten(json.loads(cfg_json.read_text(encoding="utf-8"))))
            logger.info("Read architecture config from %s", cfg_json)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Ignoring unreadable %s: %s", cfg_json, exc)
    if isinstance(ckpt, Mapping):
        merged.update(_flatten(dict(ckpt)))
    return merged


def _flatten(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Top-level scalars plus the scalars of every known nested config block."""
    out = {k: v for k, v in raw.items() if isinstance(v, (int, float, bool, str))}
    for block in _NESTED_BLOCKS:
        sub = raw.get(block)
        if isinstance(sub, Mapping):
            out.update({k: v for k, v in sub.items()
                        if isinstance(v, (int, float, bool, str))})
    return out


def config_value(cfg: Mapping[str, Any], key: str, fallback: Any) -> Any:
    """Read ``key`` from a checkpoint config, coerced to ``fallback``'s type.

    A missing key (older checkpoints that declare nothing) yields ``fallback``,
    which is how the historical literal stays available without being the only
    source of truth.
    """
    if key not in cfg or cfg[key] is None:
        return fallback
    value = cfg[key]
    if isinstance(fallback, bool):
        if isinstance(value, str):
            return value.strip().lower() in ("1", "true", "yes")
        return bool(value)
    if isinstance(fallback, int) and not isinstance(value, bool):
        return int(value)
    if isinstance(fallback, float):
        return float(value)
    return value
