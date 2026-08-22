"""
Codebase for "Improved Denoising Diffusion Probabilistic Models".

Vendored-for-provenance fork: this subpackage is kept verbatim to document the
diffusion lineage of the project and is NOT imported by the public GSSC-S2D2 API.
Its ``apex``, ``torch_scatter``, ``blobfile``, and ``mpi4py`` imports are
intentionally left undeclared in ``pyproject.toml`` because no shipped code path
imports this fork; install those extras manually only if you exercise it directly.

One dependency in the fork is NOT installable: ``autoencoder.simpleAE`` was a private
module of the original research tree and exists on no package index. Its three import
sites (``unet.py``, ``unet_factorized.py``, ``unet_old_fullres_baseline.py``) are
guarded, so those modules import cleanly with ``Encoder = None``; only the
``encoder_config`` branch that constructs it is unreachable, and no shipped path
reaches it.
"""
