"""Smoke tests: verify the gssc package imports cleanly at multiple depths.

The heavyweight submodules (data, models, diffusion, training, inference)
import torch and spconv. They are guarded by a torch-availability marker so
that ``pytest tests/test_smoke.py`` can run in a torch-free environment
(pure CI lint job) and still validate the package layout.
"""
from __future__ import annotations
import importlib

import pytest


@pytest.mark.parametrize("module", [
    "gssc",
    "gssc.utils",
    "gssc.utils.config_loader",
])
def test_light_module_imports(module: str) -> None:
    """Lightweight modules should import without torch/spconv installed."""
    importlib.import_module(module)


def test_version() -> None:
    """Package version is exposed."""
    import gssc
    assert hasattr(gssc, "__version__")
    assert isinstance(gssc.__version__, str)


@pytest.mark.parametrize("module", [
    "gssc.models",
    "gssc.diffusion",
    "gssc.data",
    "gssc.losses",
    "gssc.training",
    "gssc.inference",
])
def test_heavy_module_imports(module: str) -> None:
    """Heavyweight submodules require torch + spconv at runtime."""
    pytest.importorskip("torch")
    importlib.import_module(module)
