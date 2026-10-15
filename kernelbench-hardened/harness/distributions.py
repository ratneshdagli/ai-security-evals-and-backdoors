"""
Adversarial input distributions for hardened KernelBench correctness checking.

The original KernelBench correctness check calls a problem's own `get_inputs()`
repeatedly with different random seeds -- but `get_inputs()` samples from a single
fixed distribution per problem (e.g. `torch.rand`, always non-negative, as level1's
own 19_ReLU.py does) and stays there across trials. A kernel can special-case that
distribution (e.g. a "ReLU" that just returns its input because the test tensors
happened to be all-positive, per the brief) and still pass every trial.

KernelBench's own `kernelbench.utils` already ships a partial fix for this:
`torch.rand_mix_like(tensor, dist=...)`, backed by a pool of {normal, uniform,
laplace, exponential, lognormal} (see `_DEFAULT_RANDOM_POOL` in utils.py). We
reuse that directly below rather than re-deriving normal/uniform/laplace
ourselves. What it does NOT cover -- and what we add here -- is the specific set
of *degenerate* and *extreme* cases the brief calls out: all-negative (their pool
has no signed, always-negative option), a dynamic range spanning many orders of
magnitude, values pinned near zero, and a constant-value probe (catches both a
gamed kernel that special-cases "no variance" and, as a side effect, the
"algebraically degenerate reference problem" issue KernelBench-Verified found in
three of the original KernelBench problems themselves).

This module does not replace `get_inputs()`; it *reshapes* whatever it returns.
For each adversarial distribution we take the shapes/dtypes of the tensors
`get_inputs()` already produces (so we never have to guess a problem's expected
input shapes) and resample their contents. Non-tensor entries (ints, bools, etc.,
used by some KernelBench problems for init args) are passed through unchanged.
"""

from __future__ import annotations

import dataclasses
from typing import Callable, List

import torch

# Importing kernelbench (see harness/hardened_eval.py et al.) registers
# torch.rand_mix / torch.rand_mix_like as a side effect of kernelbench/__init__.py.
import kernelbench  # noqa: F401  (side effect: registers torch.rand_mix_like)


@dataclasses.dataclass
class InputDistribution:
    name: str
    description: str
    # Given a template tensor (for shape/dtype/device), return a resampled tensor.
    resample: Callable[[torch.Tensor], torch.Tensor]


def _upstream_dist(dist_name: str) -> Callable[[torch.Tensor], torch.Tensor]:
    """Reuse kernelbench's own torch.rand_mix_like for a named distribution."""
    def _resample(t: torch.Tensor) -> torch.Tensor:
        return torch.rand_mix_like(t, dist=dist_name).to(dtype=t.dtype, device=t.device)
    return _resample


def _all_negative(t: torch.Tensor) -> torch.Tensor:
    return -(torch.rand_like(t) * 3.0 + 1e-3)  # (-3.0, -1e-3)


def _extreme_magnitude(t: torch.Tensor) -> torch.Tensor:
    sign = torch.sign(torch.randn_like(t))
    sign[sign == 0] = 1.0
    magnitude_exp = torch.empty_like(t).uniform_(-4, 4)  # 1e-4 .. 1e4
    return sign * torch.pow(10.0, magnitude_exp)


def _near_zero(t: torch.Tensor) -> torch.Tensor:
    return torch.randn_like(t) * 1e-6


def _constant_value(t: torch.Tensor) -> torch.Tensor:
    # Every element identical (nonzero). Catches kernels that special-case
    # "no variance" inputs or that silently rely on a reduction shortcut.
    val = float(torch.empty(1).uniform_(0.5, 2.0).item())
    return torch.full_like(t, val)


ADVERSARIAL_DISTRIBUTIONS: List[InputDistribution] = [
    # Reused from kernelbench.utils' own rand_mix pool (not reimplemented):
    InputDistribution("normal", "standard normal (kernelbench.utils rand_mix pool)", _upstream_dist("normal")),
    InputDistribution("uniform", "uniform(-1, 1) (kernelbench.utils rand_mix pool)", _upstream_dist("uniform")),
    InputDistribution("laplace", "Laplace(0,1), heavy-tailed mixed sign (kernelbench.utils rand_mix pool)", _upstream_dist("laplace")),
    InputDistribution("exponential", "Exponential(1), strictly > 0 (kernelbench.utils rand_mix pool)", _upstream_dist("exponential")),
    InputDistribution("lognormal", "LogNormal(0,1), strictly > 0, heavy right tail (kernelbench.utils rand_mix pool)", _upstream_dist("lognormal")),
    # New: the specific gaps that pool doesn't cover.
    InputDistribution("all_negative", "all values < 0 -- the pool has no signed always-negative option", _all_negative),
    InputDistribution("extreme_magnitude", "magnitudes spanning 1e-4 .. 1e4, random sign", _extreme_magnitude),
    InputDistribution("near_zero", "all values within 1e-6 of zero", _near_zero),
    InputDistribution("constant_value", "every element identical (degeneracy probe)", _constant_value),
]


def make_adversarial_get_inputs_fn(original_get_inputs_fn: Callable, distribution: InputDistribution) -> Callable:
    """
    Wrap a problem's original get_inputs() so it returns tensors with the same
    shape/dtype/device but resampled according to `distribution`. Non-tensor
    entries are passed through unchanged.
    """

    def _adversarial_get_inputs():
        template_inputs = original_get_inputs_fn()
        out = []
        for x in template_inputs:
            if isinstance(x, torch.Tensor) and torch.is_floating_point(x):
                out.append(distribution.resample(x))
            else:
                out.append(x)
        return out

    _adversarial_get_inputs.__name__ = f"get_inputs__{distribution.name}"
    return _adversarial_get_inputs
