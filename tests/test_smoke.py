"""Smoke tests: verify the gssc package imports cleanly."""
from __future__ import annotations
import importlib

import pytest


@pytest.mark.parametrize("module", [
    "gssc",
    "gssc.models",
    "gssc.diffusion",
    "gssc.data",
    "gssc.losses",
    "gssc.training",
    "gssc.inference",
    "gssc.utils",
    "gssc.utils.config_loader",
])
def test_module_imports(module: str) -> None:
    """Every public submodule should import without error."""
    importlib.import_module(module)


def test_version() -> None:
    """Package version is exposed."""
    import gssc
    assert hasattr(gssc, "__version__")
    assert isinstance(gssc.__version__, str)
