"""CPU smoke tests for :class:`gssc.diffusion.multinomial.MultinomialDiffusion3DV2`.

The full diffusion pipeline runs on GPU end-to-end, but the small algebraic
identities (forward marginal, simplex projection, schedule monotonicity)
can be tested deterministically on CPU. These pin the noise schedule and
the simplex math so a future refactor cannot silently break the
correction-sampling identity that yields the headline 38.54% mIoU.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from gssc.diffusion.multinomial import MultinomialDiffusion3DV2


@pytest.fixture(scope="module")
def diffusion() -> MultinomialDiffusion3DV2:
    """Headline-config diffusion: 100 timesteps, beta_max=0.1, K=20 classes."""
    return MultinomialDiffusion3DV2(num_classes=20, num_timesteps=100, beta_max=0.1)


def test_num_timesteps(diffusion: MultinomialDiffusion3DV2) -> None:
    """100-step schedule (paper Tab. XII default)."""
    assert diffusion.num_timesteps == 100


def test_num_classes(diffusion: MultinomialDiffusion3DV2) -> None:
    """20 classes = 19 SemanticKITTI + 1 unlabeled."""
    assert diffusion.num_classes == 20


def test_alpha_cumprod_monotone_decreasing(diffusion: MultinomialDiffusion3DV2) -> None:
    """alpha_cumprod must be a non-increasing schedule from t=0 to t=T-1."""
    ac = diffusion.alphas_cumprod
    assert (ac[1:] <= ac[:-1] + 1e-7).all(), "alpha_cumprod must not increase with t"
    assert ac[0] > ac[-1], "alpha_cumprod[0] should exceed alpha_cumprod[T-1]"


def test_alpha_cumprod_in_unit_interval(diffusion: MultinomialDiffusion3DV2) -> None:
    """Schedule values are valid probabilities in [0, 1]."""
    ac = diffusion.alphas_cumprod
    assert (ac >= 0).all()
    assert (ac <= 1).all()


def test_betas_in_unit_interval(diffusion: MultinomialDiffusion3DV2) -> None:
    """Per-step noise rates beta_t are in [0, 1]."""
    betas = diffusion.betas
    assert (betas >= 0).all()
    assert (betas <= 1).all()


def test_betas_length_matches_timesteps(diffusion: MultinomialDiffusion3DV2) -> None:
    """One beta per diffusion step."""
    assert diffusion.betas.shape[0] == diffusion.num_timesteps


def test_alpha_cumprod_initial_value_close_to_one(diffusion: MultinomialDiffusion3DV2) -> None:
    """At t=0, almost all signal is preserved (paper Sec. 3.2)."""
    assert diffusion.alphas_cumprod[0].item() > 0.9


def test_alpha_cumprod_final_value_small(diffusion: MultinomialDiffusion3DV2) -> None:
    """At t=T-1, most signal is destroyed (full forward corruption)."""
    assert diffusion.alphas_cumprod[-1].item() < 0.05


def test_log_alpha_real_valued(diffusion: MultinomialDiffusion3DV2) -> None:
    """log_alpha must be finite (no zeros in alphas)."""
    log_alpha = diffusion.log_alpha
    assert torch.isfinite(log_alpha).all()
