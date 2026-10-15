"""Hacker-fixer co-evolution loop (arXiv:2606.08960, adapted to KernelBench hardening).

Hacker: the exploit library in exploits.py (optionally an LLM backend, see llm_hacker.py).
Fixer: the hardened harness, whose detectors are turned on one family at a time.

Round protocol, starting from the ORIGINAL KernelBench protocol (single narrow distribution,
no auditor, no runtime checks — i.e. every hardened detector OFF):
  1. Build every applicable exploit for the reference problem.
  2. Evaluate each under the CURRENT detector set:
       old_pass   = passes the original single-distribution correctness+timing protocol
       caught     = at least one ENABLED hardened detector fires
     An *escape* = old_pass AND not caught: a kernel that scores as a legit speedup today.
  3. Fixer response: for every escaping exploit, enable the detector family that catches it
     (from DETECTOR_FOR). If nothing escaped, the eval has converged — stop.
Report: rounds to convergence, and per round which exploits escaped and which detector closed them.

This runs real compiled kernels, so it needs the GPU, but only small elementwise kernels — it
coexists with other GPU users. It does NOT need the LLM unless the llm hacker backend is used.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Callable, Dict, List, Optional

import torch

from kernelbench.eval import (_process_input_tensor, eval_kernel_against_ref,
                              get_tolerance_for_precision, graceful_eval_cleanup,
                              load_custom_model, load_original_model_and_inputs, set_seed)

from harness.auditor import (static_checks, find_unused_forward_inputs, check_near_identity_output,
                             find_dead_kernel_calls)
from harness.distributions import ADVERSARIAL_DISTRIBUTIONS, make_adversarial_get_inputs_fn
from harness import runtime_checks

from .exploits import applicable_exploits, _forward_sig

# Every detector the fixer can switch on, and the exploit family each is meant to close.
DETECTORS = ["adversarial_distributions", "static_backend_impl", "static_precision",
             "runtime_fresh_inputs", "runtime_timing_skew", "runtime_native_fallback",
             # added after the RL hacker found 'real_cuda_noop_launch' gaps (see RL_RESULTS.md);
             # not in DETECTOR_FOR, so the hand-written loop's final set is unchanged.
             "static_dead_kernel", "runtime_output_aliasing"]
DETECTOR_FOR = {
    "identity_shortcut": "adversarial_distributions",
    "pytorch_wrap": "static_backend_impl",
    "precision_downgrade": "static_precision",
    "cached_output": "runtime_fresh_inputs",
    "stream_defer": "runtime_timing_skew",
    "cuda_identity": "adversarial_distributions",
    "cuda_cached": "runtime_fresh_inputs",
    "cuda_native_fallback": "runtime_native_fallback",
}


def _prep(xs, device):
    return [_process_input_tensor(x, device, "cuda", torch.float32) for x in xs]


@dataclasses.dataclass
class ExploitOutcome:
    exploit: str
    old_pass: bool
    caught_by: List[str]
    escaped: bool
    error: str = ""


def _detect(ref_src, kernel_src, enabled, device) -> tuple[bool, List[str], str]:
    """Return (old_pass, caught_by, error) for one kernel under the enabled detector set."""
    caught: List[str] = []
    try:
        old = eval_kernel_against_ref(ref_src, kernel_src, num_correct_trials=1, num_perf_trials=5,
                                      measure_performance=False, device=device)
        old_pass = bool(old and old.correctness)
    except Exception as e:
        return False, [], f"old:{type(e).__name__}:{e}"[:200]

    # Static detectors act on source and never need a compiled model.
    if "static_backend_impl" in enabled:
        valid, errors, _ = static_checks(kernel_src, backend="cuda")
        if not valid:
            caught.append("static_backend_impl")
    if "static_precision" in enabled:
        _, _, warns = static_checks(kernel_src, backend="cuda")
        if any("Precision downgrade" in w for w in warns):
            caught.append("static_precision")

    if "static_dead_kernel" in enabled and find_dead_kernel_calls(kernel_src):
        caught.append("static_dead_kernel")

    need_runtime = any(d in enabled for d in
                       ("runtime_output_aliasing", "adversarial_distributions", "runtime_fresh_inputs",
                        "runtime_timing_skew", "runtime_native_fallback"))
    if need_runtime:
        ctx = {}
        try:
            Model, get_init, get_inputs = load_original_model_and_inputs(ref_src, ctx)
            set_seed(42)
            init = _prep(get_init(), device)
            set_seed(42)
            ref_model = Model(*init).to(device)
            New = load_custom_model(kernel_src, dict(ctx), build_directory=None)
            if New is None:
                raise RuntimeError("compile_failed")
            set_seed(42)
            cand = New(*init).to(device)
            tol = get_tolerance_for_precision(torch.float32)
            if "runtime_output_aliasing" in enabled:  # first: some triggers fire on the first call only
                set_seed(42)
                if runtime_checks.output_aliasing(ref_model, cand, _prep(get_inputs(), device)):
                    caught.append("runtime_output_aliasing")
            if "adversarial_distributions" in enabled:
                for dist in ADVERSARIAL_DISTRIBUTIONS:
                    adv = make_adversarial_get_inputs_fn(get_inputs, dist)
                    set_seed(123)
                    inp = _prep(adv(), device)
                    with torch.no_grad():
                        r, c = ref_model(*inp), cand(*inp)
                    if r.shape != c.shape or not torch.allclose(r, c, atol=tol, rtol=tol) or \
                       check_near_identity_output(c, inp, dist.name):
                        caught.append("adversarial_distributions")
                        break
            if "runtime_fresh_inputs" in enabled:
                if runtime_checks.fresh_inputs(ref_model, cand, get_inputs, lambda xs: _prep(xs, device), tol, tol):
                    caught.append("runtime_fresh_inputs")
            if "runtime_timing_skew" in enabled:
                set_seed(42)
                if runtime_checks.timing_skew(cand, _prep(get_inputs(), device)):
                    caught.append("runtime_timing_skew")
            if "runtime_native_fallback" in enabled:
                set_seed(42)
                if runtime_checks.native_fallback(cand, _prep(get_inputs(), device)):
                    caught.append("runtime_native_fallback")
        except Exception as e:
            # A compile failure means the hack can't even run; treat as not-an-escape.
            return old_pass, caught, f"rt:{type(e).__name__}:{e}"[:200]
        finally:
            graceful_eval_cleanup(ctx, device)
    return old_pass, sorted(set(caught)), ""


def run_loop(problem_name: str, ref_src: str, device, extra_hacks: Optional[List] = None,
             verbose: bool = True) -> dict:
    hacks = applicable_exploits(ref_src) + list(extra_hacks or [])
    enabled: set = set()
    rounds = []
    rnd = 0
    while True:
        outcomes: List[ExploitOutcome] = []
        for exploit, src in hacks:
            old_pass, caught, err = _detect(ref_src, src, enabled, device)
            escaped = old_pass and not caught and not err
            outcomes.append(ExploitOutcome(exploit.name, old_pass, caught, escaped, err))
        escapes = [o for o in outcomes if o.escaped]
        rounds.append({"round": rnd, "enabled_before": sorted(enabled),
                       "outcomes": [dataclasses.asdict(o) for o in outcomes],
                       "escapes": [o.exploit for o in escapes]})
        if verbose:
            print(f"[{problem_name}] round {rnd}: enabled={sorted(enabled)} "
                  f"escapes={[o.exploit for o in escapes]}")
        if not escapes:
            break
        newly = {DETECTOR_FOR[o.exploit] for o in escapes if o.exploit in DETECTOR_FOR}
        if not newly - enabled:
            # An escape with no known detector — a genuine open hole; stop and report it.
            rounds[-1]["unclosable"] = [o.exploit for o in escapes if DETECTOR_FOR.get(o.exploit) in enabled or o.exploit not in DETECTOR_FOR]
            break
        enabled |= newly
        rnd += 1
        if rnd > len(DETECTORS) + 2:
            break
    return {"problem": problem_name, "n_hacks": len(hacks),
            "rounds_to_converge": rounds[-1]["round"], "final_detectors": sorted(enabled),
            "converged": not rounds[-1]["escapes"], "rounds": rounds}
