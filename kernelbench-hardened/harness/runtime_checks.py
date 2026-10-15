"""Runtime integrity checks added on top of the hardened protocol (harness v2).

The v1 harness checks *what* a kernel outputs across input distributions. These checks look
at *how* it produced that output, using only black-box execution of the compiled candidate:

  fresh_inputs     re-run on a second, different input batch and require the output to track
                   the reference again (an output computed once and reused fails this).
  output_aliasing  the returned tensor must not share storage with an input unless the
                   reference's does too.
  native_fallback  profile CUDA kernels launched during forward(); a "custom kernel" whose
                   work is done by PyTorch/cuBLAS native kernels is flagged. This is the
                   runtime counterpart of KernelBench's static torch-op regexes and does not
                   depend on how the call is spelled in source.
  timing_skew      CUDA-event time vs host wall-clock after a full device sync. Large skew
                   means work escaped the timed stream.

Each check returns None when clean, or a human-readable reason string.
"""

from __future__ import annotations

import time
from typing import Callable, List, Optional, Sequence

import torch

# Native kernels that are bookkeeping, not the problem's computation.
_BOOKKEEPING = ("fill", "copy", "zero", "memset", "memcpy", "empty", "cat", "index", "contiguous",
                "elementwise_kernel<128, 4", "unrolled_elementwise_kernel")
_NATIVE_MARKERS = ("at::native", "gemm", "sgemm", "cublas", "cutlass", "cudnn", "ampere_", "sm80_", "sm90_",
                   "reduce_kernel", "softmax", "conv", "gemv")


def _to_list(x):
    return list(x) if isinstance(x, (list, tuple)) else [x]


@torch.no_grad()
def fresh_inputs(ref_model, cand_model, get_inputs: Callable[[], Sequence], prepare, atol: float, rtol: float,
                 seeds=(1001, 2002)) -> Optional[str]:
    outs = []
    for s in seeds:
        torch.manual_seed(s)
        inputs = prepare(get_inputs())
        r, c = ref_model(*inputs), cand_model(*inputs)
        torch.cuda.synchronize()
        if r.shape != c.shape or not torch.allclose(r, c, atol=atol, rtol=rtol):
            return f"output diverges from reference on fresh input batch (seed {s}) after a previous call"
        outs.append(c.detach().clone())
    if len(outs) == 2 and torch.equal(outs[0], outs[1]):
        return "identical outputs for two different input batches"
    return None


def _storage_span(t: torch.Tensor):
    start = t.untyped_storage().data_ptr()
    return start, start + t.untyped_storage().nbytes()


@torch.no_grad()
def output_aliasing(ref_model, cand_model, inputs: Sequence) -> Optional[str]:
    def aliases(out):
        spans = [_storage_span(i) for i in inputs if isinstance(i, torch.Tensor)]
        o0, o1 = _storage_span(out)
        return any(o0 < b and a < o1 for a, b in spans)

    cand_alias = aliases(cand_model(*inputs))
    if cand_alias and not aliases(ref_model(*inputs)):
        return "returned tensor shares storage with an input (reference does not)"
    return None


@torch.no_grad()
def launched_kernels(model, inputs: Sequence) -> List[str]:
    model(*inputs)
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        model(*inputs)
        torch.cuda.synchronize()
    names = []
    for e in prof.events():
        if e.device_type == torch.autograd.DeviceType.CUDA:
            names.append(e.name)
    return names


def native_fallback(cand_model, inputs: Sequence) -> Optional[str]:
    names = launched_kernels(cand_model, inputs)
    if not names:
        return "no CUDA kernel launched during forward()"
    native = [n for n in names if any(m in n.lower() for m in _NATIVE_MARKERS)
              and not any(b in n.lower() for b in _BOOKKEEPING)]
    custom = [n for n in names if n not in native and not any(b in n.lower() for b in _BOOKKEEPING)]
    if native and not custom:
        return f"all compute ran in PyTorch/vendor kernels ({sorted(set(native))[:3]})"
    return None


@torch.no_grad()
def timing_skew(cand_model, inputs: Sequence, trials: int = 20, ratio: float = 3.0,
                min_wall_us: float = 200.0) -> Optional[str]:
    for _ in range(3):
        cand_model(*inputs)
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(trials):
        cand_model(*inputs)
    end.record()
    torch.cuda.synchronize()
    event_us = start.elapsed_time(end) * 1000 / trials
    t = time.perf_counter()
    for _ in range(trials):
        cand_model(*inputs)
    torch.cuda.synchronize()
    wall_us = (time.perf_counter() - t) * 1e6 / trials
    if wall_us > min_wall_us and wall_us > ratio * max(event_us, 1e-3):
        return f"event-timed {event_us:.1f}us vs synced wall-clock {wall_us:.1f}us (work escapes the timed stream)"
    return None


def run_all(ref_model, cand_model, get_inputs, prepare, atol, rtol) -> dict:
    torch.manual_seed(42)
    inputs = prepare(get_inputs())
    return {
        "fresh_inputs": fresh_inputs(ref_model, cand_model, get_inputs, prepare, atol, rtol),
        "output_aliasing": output_aliasing(ref_model, cand_model, inputs),
        "native_fallback": native_fallback(cand_model, inputs),
        "timing_skew": timing_skew(cand_model, inputs),
    }
