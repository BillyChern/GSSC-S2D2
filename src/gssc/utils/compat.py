"""Backward-compatibility shims for v1.x deprecations.

This module owns the deprecation paths so that v1.0.0 users can upgrade
to v1.1.0+ without any code or config change. Items marked for removal in
v2.0.0 are documented inline.
"""

from __future__ import annotations

import warnings
from typing import Final

__all__ = ["resolve_base_pred_dir"]

_BASE_PRED_DIR_REMOVAL: Final[str] = "v2.0.0"


def resolve_base_pred_dir(
    base_pred_dir: str | None = None,
    scpnet_pred_dir: str | None = None,
) -> str | None:
    """Resolve the base-model prediction directory, accepting both names.

    Introduced in v1.1.0 for the JS3C-Net cross-base support. The previous
    ``scpnet_pred_dir`` kwarg still works but emits a :class:`DeprecationWarning`
    once per call site, slated for removal in :data:`_BASE_PRED_DIR_REMOVAL`.

    Resolution order:
        1. ``base_pred_dir`` if explicitly provided (preferred).
        2. ``scpnet_pred_dir`` (deprecated alias).
        3. ``None`` if neither is supplied.

    Args:
        base_pred_dir: New canonical name (added v1.1.0).
        scpnet_pred_dir: Deprecated alias kept for v1.0.0 users.

    Returns:
        The first non-None value, or ``None`` if both are unset.

    Examples:
        >>> resolve_base_pred_dir(base_pred_dir="data/js3cnet_predictions")
        'data/js3cnet_predictions'
        >>> resolve_base_pred_dir(scpnet_pred_dir="data/scpnet_predictions")  # emits warning
        'data/scpnet_predictions'
    """
    if base_pred_dir is not None:
        return base_pred_dir
    if scpnet_pred_dir is not None:
        warnings.warn(
            "`scpnet_pred_dir` is deprecated since gssc v1.1.0 and will be "
            f"removed in {_BASE_PRED_DIR_REMOVAL}. Use `base_pred_dir` (plus "
            "`base_kind='scpnet'` if you need to be explicit about the base).",
            DeprecationWarning,
            stacklevel=3,
        )
        return scpnet_pred_dir
    return None
