"""GSSC-S2D2: Structured Source Discrete Diffusion for Generative Semantic Scene Completion.

Reference implementation accompanying the TPAMI 2026 paper "Generative Semantic
Scene Completion". Package entry point.

The recommended way to use this codebase is through the driver scripts in
``scripts/`` (see the project README). Importing the submodules directly is
also supported for advanced users:

* ``gssc.models`` — neural-network architectures
* ``gssc.diffusion`` — forward / posterior / Algo2 sampler
* ``gssc.data`` — dataset loaders + augmentation
* ``gssc.losses`` — KL + Lovász + auxiliary
* ``gssc.training`` — trainer
* ``gssc.inference`` — evaluation + visualisation
* ``gssc.utils`` — shared helpers
"""
from __future__ import annotations

__version__ = "1.0.0"
__all__ = ["__version__"]
