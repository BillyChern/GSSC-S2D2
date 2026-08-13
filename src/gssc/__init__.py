"""GSSC-S2D2: Structured Source Discrete Diffusion for Generative Semantic Scene Completion.

Reference implementation accompanying the TPAMI 2026 paper "Generative Semantic
Scene Completion". Package entry point.

The recommended way to use this codebase is through the driver scripts in
``scripts/`` (see the project README). Importing the submodules directly is
also supported for advanced users:

* ``gssc.models`` — neural-network architectures
* ``gssc.diffusion`` — forward process, Dirac posterior, S²D² correction sampler
* ``gssc.data`` — dataset loaders + augmentation
* ``gssc.losses`` — KL + Lovász + auxiliary
* ``gssc.training`` — trainer
* ``gssc.inference`` — evaluation (3D SSC + 2D BEV) and prediction generation
* ``gssc.utils`` — shared helpers
"""
from __future__ import annotations

__version__ = "2.3.5"
__author__ = "Shi Chen, Weifeng Ge"
__all__ = ["__author__", "__version__"]
