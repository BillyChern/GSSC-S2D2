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


# ---------------------------------------------------------------------------
# Capability smoke: every capability the paper claims must import cleanly.
# A regression here means a visitor cannot reproduce that part of the paper.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("module,attrs", [
    # Pyramid Discrete Diffusion (Phase-1 augmentation, S1/S2/S3 stages)
    ("gssc.models.pyramid_diffusion", ["PyramidDiscreteDiffusion"]),
    ("gssc.models.pyramid_unet", ["Denoise"]),
    ("gssc.training.pyramid_pipeline", []),
    ("gssc.training.train_pyramid_s2", []),
    ("gssc.training.train_pyramid_s3", []),
    # LiDAR ray-tracing / resampling (HDL-64E sensor simulation)
    ("gssc.data.lidar_simulator", ["LiDARResampler"]),
    ("gssc.data.lidar_simulation", ["LiDARSimulator", "VelodyneHDL64E", "Bresenham3D"]),
    ("gssc.data.lidar_resampler_v2", ["MultiReturnLiDARSimulator", "create_density_aware_resampler"]),
    # Rare-class object bank + (sparse, complete) pair generation
    ("gssc.data.object_bank", ["ObjectBank", "ObjectPaster", "RareObjectExtractor"]),
    ("gssc.data.synthetic_generator", ["SyntheticSceneGenerator", "PyramidSampler"]),
    ("gssc.data.sparse_complete_pairs", ["PairGenerationConfig"]),
    # LiDAR-only BEV S2D2 (second-task, paper Sec. 4 36.09 percent result)
    ("gssc.models.bev_unet", ["BEVUNet", "LightweightBEVUNet"]),
    ("gssc.models.bev_unet_v2", ["ModularBEVUNet", "create_modular_bev_unet"]),
    ("gssc.models.bev_lidar_encoder", ["SparseLiDAREncoder", "TinySparseLiDAREncoder"]),
    ("gssc.models.bev_d3pm", ["D3PM", "AbsorbingD3PM", "UniformD3PM"]),
    ("gssc.models.bev_bev_diffusion_v2", ["BEVDiffusionV2"]),
    ("gssc.models.bev_diffusion_model", ["BEVDiffusionModel", "create_bev_diffusion_model"]),
    ("gssc.models.bev_sparse_bev_net", ["FullSparseBEVNet_Deeper"]),
    ("gssc.models.bev_training", ["BEVTrainer"]),
    ("gssc.models.bev_multinomial_diffusion_2d", ["MultinomialDiffusion2D"]),
    # BEV second-task driver scripts (visitor reproduction path)
    ("gssc.inference.evaluate_bev", ["evaluate_bev"]),
    ("gssc.training.train_bev_secondary", []),
])
def test_capability_modules_expose_public_api(module: str, attrs: list[str]) -> None:
    """Every capability the paper claims must import + expose its public API.

    Catches the class of regression where a migration silently drops a
    relative import or vendors only half of a subpackage. Failure here
    means the README's "data augmentation" or "BEV second task" sections
    are not actually runnable from the public repo.
    """
    pytest.importorskip("torch")
    pytest.importorskip("spconv")
    mod = importlib.import_module(module)
    for attr in attrs:
        assert hasattr(mod, attr), f"{module} is missing public API: {attr}"
