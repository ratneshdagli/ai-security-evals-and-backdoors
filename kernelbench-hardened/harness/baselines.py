"""
Hardware-realistic baseline timing.

KernelBench-Verified's finding #1: the original KernelBench baseline never touches
`torch.backends.cuda.matmul.allow_tf32` / `torch.backends.cudnn.allow_tf32`, so it
inherits whatever PyTorch's ambient default is (False, since PyTorch 1.12) even
though TF32 is a one-line opt-in that's on-by-default in most real deployments'
tuning guides for Ampere+ GPUs. Comparing an LLM's kernel against an artificially
slow baseline inflates the reported speedup.

We reuse KernelBench's own `measure_ref_program_time` (src/kernelbench/timing.py)
for the actual timing loop and torch.compile plumbing -- this module only controls
the TF32 flags around it and packages "old" vs "hardened" baselines side by side.
"""

from __future__ import annotations

import contextlib
import dataclasses
from typing import Optional

import torch

from kernelbench.timing import measure_ref_program_time


@contextlib.contextmanager
def tf32_mode(enabled: bool):
    prev_matmul = torch.backends.cuda.matmul.allow_tf32
    prev_cudnn = torch.backends.cudnn.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = enabled
        torch.backends.cudnn.allow_tf32 = enabled
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev_matmul
        torch.backends.cudnn.allow_tf32 = prev_cudnn


@dataclasses.dataclass
class BaselineSet:
    old_protocol_eager: dict          # TF32 left at ambient/off -- matches original KernelBench
    hardened_eager_tf32: dict         # TF32 explicitly enabled -- realistic Ampere/Ada default
    hardened_compiled_tf32: Optional[dict]  # torch.compile'd + TF32 on, the harder target


def measure_baselines(
    ref_arch_name: str,
    ref_arch_src: str,
    device: torch.device,
    num_warmup: int = 5,
    num_trials: int = 100,
    include_compiled: bool = True,
    verbose: bool = False,
) -> BaselineSet:
    with tf32_mode(False):
        old_eager = measure_ref_program_time(
            ref_arch_name, ref_arch_src,
            num_warmup=num_warmup, num_trials=num_trials,
            use_torch_compile=False, device=device, verbose=verbose,
        )

    with tf32_mode(True):
        hardened_eager = measure_ref_program_time(
            ref_arch_name, ref_arch_src,
            num_warmup=num_warmup, num_trials=num_trials,
            use_torch_compile=False, device=device, verbose=verbose,
        )

        hardened_compiled = None
        if include_compiled:
            try:
                hardened_compiled = measure_ref_program_time(
                    ref_arch_name, ref_arch_src,
                    num_warmup=max(num_warmup, 3), num_trials=num_trials,
                    use_torch_compile=True, torch_compile_backend="inductor",
                    device=device, verbose=verbose,
                )
            except Exception as e:
                if verbose:
                    print(f"[baselines] torch.compile baseline failed for {ref_arch_name}: {e}")
                hardened_compiled = None

    return BaselineSet(
        old_protocol_eager=old_eager,
        hardened_eager_tf32=hardened_eager,
        hardened_compiled_tf32=hardened_compiled,
    )
