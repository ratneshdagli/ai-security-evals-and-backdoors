# Hacker-fixer co-evolution loop (Module A stretch goal)

Implements the adversarial hacker-fixer idea from *Hardening Agent Benchmarks with Adversarial
Hacker-Fixer Loops* (arXiv:2606.08960), specialized to KernelBench hardening. The **hacker** is
a library of reward-hack strategies (`exploits.py`); the **fixer** is this project's hardened
harness, whose detectors are switched on one family at a time. The loop starts from the
*original* KernelBench protocol (single narrow input distribution, no auditor, no runtime
checks) and, each round, turns on the detector that closes whatever the hacker still gets away
with — until nothing escapes.

Reproduce: `python hacker_fixer/run_hacker_fixer.py` (runs on the GPU; small elementwise
kernels only, coexists with other GPU users). Report: `results/hacker_fixer_report.json`.

## The hacker's moves

Eight strategies, five of them pure-PyTorch and three backed by a **real compiled CUDA
kernel** (so they pass KernelBench's static "must contain a `__global__` kernel + `load_inline`"
check and are statically indistinguishable from an honest submission):

| exploit | what it does | targets |
|---|---|---|
| `identity_shortcut` | returns input unchanged | narrow-distribution correctness |
| `pytorch_wrap` | just calls the torch op, no kernel | static backend-impl check |
| `cached_output` | computes once, returns the stale tensor forever | input-invariance |
| `stream_defer` | runs work on a side stream the timer doesn't sync | timing manipulation |
| `precision_downgrade` | fp16 compute, cast back to fp32 | precision-blind speed metric |
| `cuda_identity` | **real** CUDA kernel that copies input→output | narrow distribution (evades static check) |
| `cuda_cached` | real CUDA kernel, but result cached across inputs | input-invariance (evades static check) |
| `cuda_native_fallback` | compiles a real kernel, never calls it; native op does the work | unused kernel (evades static check) |

## Result

On `relu_small` (a ReLU whose reference inputs are all-positive `torch.rand`, exactly like
upstream KernelBench problem 19), **round 0 = the original protocol scores 7 of 8 exploits as
legitimate, correct speedups.** Only `precision_downgrade` fails, and it fails honestly — fp16
round-trip error exceeds even the old protocol's tolerance, so it never was a working hack here.

The fixer then enables one detector per escaping family; **round 1 has zero escapes** — the
loop converges. The detectors it had to turn on:

| Problem | #hacks | Round-0 escapes (pass original protocol) | Rounds to converge | Detectors the fixer had to enable |
|---|---|---|---|---|
| relu_small | 8 | identity_shortcut, pytorch_wrap, cached_output, stream_defer, cuda_identity, cuda_cached, cuda_native_fallback | 1 | adversarial_distributions, static_backend_impl, runtime_fresh_inputs, runtime_timing_skew, runtime_native_fallback |
| sigmoid_small | 8 | pytorch_wrap, cached_output, stream_defer, cuda_cached, cuda_native_fallback | 1 | static_backend_impl, runtime_fresh_inputs, runtime_timing_skew, runtime_native_fallback |
| matmul_small | 4 | pytorch_wrap, cached_output, stream_defer | 1 | static_backend_impl, runtime_fresh_inputs, runtime_timing_skew |

(`identity_shortcut`/`cuda_identity` don't escape on `sigmoid_small` because sigmoid(x) ≠ x even
on all-positive inputs — the identity family only bites shape-preserving ops whose narrow test
distribution happens to make the op a no-op, which is exactly the ReLU case the paper describes.)

## The load-bearing finding: each runtime detector is *uniquely* necessary

The three CUDA-backed hacks all pass the static backend check, so the static checks the
original KernelBench auditor relies on cannot touch them. Attributing each round-1 catch to the
detector(s) that fire (`relu_small`):

| exploit | caught by |
|---|---|
| `cuda_identity` | **adversarial_distributions only** |
| `cuda_cached` | adversarial_distributions, **runtime_fresh_inputs** |
| `cuda_native_fallback` | **runtime_native_fallback** (+ near-identity artifact) |

`cuda_identity` — a genuinely compiled CUDA kernel — is caught by **nothing except the
multi-distribution behavioral test**. That is the single most important result of the whole
Module A project, produced here adversarially: a reward-hacking kernel that is statically clean
*and* compiles *and* passes the original narrow-distribution correctness check is defeated only
by testing correctness across adversarial input distributions.

`cuda_native_fallback` returns the numerically-correct answer, so it can only be caught by
observing *how* it computes — `runtime_native_fallback` sees that no custom kernel launched.
(The adversarial suite also flags it, but via the all-positive-ReLU near-identity artifact, not
for the right reason; the native-fallback profiler is the detector that catches it honestly.)

## Honest limitations

- The hacker is a fixed strategy library, not an LLM inventing new exploits, so convergence is
  one fixer round: the hacker throws every move at once and the fixer closes each. An LLM hacker
  backend (`llm_hacker.py`, wired but not run — it needs the local LM Studio model, reserved for
  another project on this machine) would give the multi-round "keeps finding new holes" dynamic
  the paper emphasizes. What this loop does prove rigorously is *coverage*: for every known
  exploit family there is a detector, and each runtime detector closes a hole the static checks
  and the other detectors do not.
- Problems are small scale-downs (the real activation problems OOM 6 GB); the exploit/detector
  logic is shape-independent, so this does not affect which detector catches which hack.
