"""
Peak GPU memory tracking, alongside speed.

KernelBench-Verified found 28% of the best model's "correct" kernels used *more*
peak memory than the reference implementation -- a regression invisible to a
speed-only metric. This wraps a model's forward pass with
torch.cuda.reset_peak_memory_stats/max_memory_allocated so every kernel we score
gets a memory delta, not just a runtime.
"""

from __future__ import annotations

import dataclasses
from typing import Callable, List

import torch


@dataclasses.dataclass
class MemoryResult:
    peak_bytes: int
    ok: bool
    error: str = ""


def measure_peak_memory(
    model: torch.nn.Module,
    get_inputs_fn: Callable,
    device: torch.device,
    num_trials: int = 3,
) -> MemoryResult:
    try:
        torch.cuda.synchronize(device=device)
        torch.cuda.reset_peak_memory_stats(device=device)
        with torch.no_grad():
            for _ in range(num_trials):
                inputs: List = get_inputs_fn()
                inputs = [x.to(device=device) if isinstance(x, torch.Tensor) else x for x in inputs]
                model(*inputs)
        torch.cuda.synchronize(device=device)
        peak = torch.cuda.max_memory_allocated(device=device)
        return MemoryResult(peak_bytes=peak, ok=True)
    except Exception as e:
        return MemoryResult(peak_bytes=-1, ok=False, error=str(e))


def memory_delta_pct(candidate: MemoryResult, reference: MemoryResult) -> float:
    """Positive = candidate uses more memory than reference (regression)."""
    if not candidate.ok or not reference.ok or reference.peak_bytes <= 0:
        return float("nan")
    return 100.0 * (candidate.peak_bytes - reference.peak_bytes) / reference.peak_bytes
