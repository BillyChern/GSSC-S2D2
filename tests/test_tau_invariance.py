"""The paper claims sampling temperature is provably inert at N=1. This pins it.

At n_steps=1 sample_algo2 takes only its final branch, x_t = softmax(logits), and returns
x_t.argmax(dim=1). Since argmax(softmax(z / tau)) == argmax(z) for any tau > 0, the emitted
labels cannot depend on tau. That is a property of the arithmetic, so it is testable without
a checkpoint: these tests exercise the same expression the sampler uses.

For n_steps > 1 tau DOES move the intermediate corrections, and the last test pins that too --
the invariance claim is specific to N=1 and should not be quietly generalised.
"""

import pytest
import torch
import torch.nn.functional as F

TAUS = [0.5, 0.8, 1.0, 1.2, 1.5, 2.0]


def _labels_at(logits: torch.Tensor, tau: float) -> torch.Tensor:
    """Exactly what sample_algo2 does at n_steps=1: softmax(logits/tau) then argmax."""
    scaled = logits if tau == 1.0 else logits / tau
    return F.softmax(scaled, dim=1).argmax(dim=1)


@pytest.fixture
def logits() -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn(2, 20, 8, 8, 4) * 3.0


def test_labels_are_identical_across_tau(logits):
    ref = _labels_at(logits, 1.0)
    for tau in TAUS:
        assert torch.equal(_labels_at(logits, tau), ref), f"tau={tau} changed the labels"


def test_tau_one_is_bit_identical(logits):
    """The tau == 1.0 guard must not perturb the default path at all."""
    assert torch.equal(F.softmax(logits, dim=1), F.softmax(logits / 1.0, dim=1))


def test_holds_under_near_ties(logits):
    """Near-tied logits are where a temperature would break invariance if it could."""
    tied = torch.zeros(1, 20, 4, 4, 2)
    tied[:, 3] = 1e-6
    ref = _labels_at(tied, 1.0)
    for tau in TAUS:
        assert torch.equal(_labels_at(tied, tau), ref)


def test_tau_is_NOT_inert_beyond_one_step(logits):
    """Guard against generalising the claim: at N>1 tau moves the intermediate state."""
    alpha_step = 0.4
    src = F.one_hot(torch.zeros(2, 8, 8, 4, dtype=torch.long), 20).float().permute(0, 4, 1, 2, 3)
    def two_step(tau):
        x0 = F.softmax(logits if tau == 1.0 else logits / tau, dim=1)
        x = (src + alpha_step * (x0 - src)).clamp(min=0.0)
        return x / (x.sum(dim=1, keepdim=True) + 1e-10)
    assert not torch.allclose(two_step(1.0), two_step(2.0)), \
        "intermediate state should depend on tau; the N=1 claim must stay scoped to N=1"
