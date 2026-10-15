# PROPOSAL: Evaluation Integrity & Backdoor Detection (Modules A + B)

## Scoping decision

The project was delivered in two phases. **Phase 1** went deep on **Module A: GPU Kernel
Benchmark Hack-Proofing** first — the brief says a strong result on one module beats shallow
coverage of two, and Module A was the more tractable starting point on this machine (no CUDA
toolchain existed, so the environment itself was real engineering; see below). **Phase 2**, at
the user's explicit request to "complete both and make them final," added Module A's stretch
goals (hacker-fixer co-evolution loop, GPU MODE submission) and a full **Module B: Poisoned-
Model Backdoor Trigger Recovery** against the ToxScreen testbed.

The unifying question from the brief is honoured: both modules ask "when an AI artifact passes
its check, is it actually correct, or has it learned to satisfy the *check* rather than the
*goal*?" — Module A for LLM-generated CUDA kernels passing a speed benchmark, Module B for
model weights passing a safety check.

Deliverables:
- `kernelbench-hardened/` — the hardened harness, `RESULTS.md`, `hacker_fixer/RESULTS.md`,
  `gpumode_submission/`.
- `backdoor-recovery/` — the trigger-recovery scanner, `README.md`, `RESULTS.md`.

The rest of this document is the Module A scoping/infrastructure record; Module B's design and
literature review are in `backdoor-recovery/README.md` and `backdoor-recovery/RESULTS.md`.

## Literature review (beyond what's in the brief)

- **KernelBench-Verified** (arXiv:2607.16241): re-ran the standard KernelBench protocol with
  two fixes — TF32 enabled on the baseline (on by default on Ampere/Ada+ but not enabled in
  the original harness) and a "four-distribution hidden test suite" instead of one fixed input
  distribution. Result: the best frontier model's claimed 1.43x geomean speedup collapses to
  0.88x (slower than eager PyTorch) once these are fixed. 28% of "correct" kernels use *more*
  peak memory than the reference. No model consistently beats PyTorch under the verified
  protocol across 7 frontier models tested.
- **Original KernelBench** (ScalingIntelligence/KernelBench, Simon Guo et al.): defines the
  problem levels (L1: 100 single-kernel ops, L2: 100 fusion patterns, L3: 50 full
  architectures, L4: HF model-level) and the reference `eval.py` scoring pipeline we build on
  top of rather than reimplement.
- **KernelBench-Hard** (`Infatoshi/kernelbench-hard-problems` + `-runs` on HuggingFace): 7
  problems, 84 full agent transcripts across 12 frontier models on a Blackwell GPU, with
  human-annotated reward-hack labels. This is a free, zero-compute-cost ground truth I use to
  validate the auditor's precision/recall *before* trusting it on newly-generated kernels —
  the original brief didn't mention using this dataset for auditor validation specifically,
  but it's a natural fit since it's the only source of *labeled* reward-hacks available.
- **Hardening Agent Benchmarks with Adversarial Hacker-Fixer Loops** (arXiv:2606.08960):
  co-evolve the eval against an adversarial kernel-writer — feed harness failures back to the
  generator and iterate. Scoped as a stretch goal (Section 5 of the plan) after the core
  harness + comparison run is solid, not the primary deliverable.
- **GPU MODE reference-kernels** (github.com/gpu-mode/reference-kernels): extra problem
  definitions and the real competitive leaderboard community, kept as an optional source of
  additional problems and as the stretch-goal submission target.

## Infrastructure decision (why this took real engineering, not just prompting)

This machine (Windows 11, RTX 4050 Laptop GPU, 6GB VRAM) had:
- No CUDA Toolkit (`nvcc`) and no MSVC compiler — both required because KernelBench JIT-compiles
  raw `.cu` files via `torch.utils.cpp_extension.load_inline`. Native Windows CUDA dev needs
  MSVC + nvcc glued together via `vcvarsall.bat`, is fragile, and trips on paths with spaces
  (this project's own path, `D:\Best Projects\AI SECURITY`, has one).
- A CPU-only PyTorch build already installed on Windows (irrelevant to this work).
- WSL2 already installed with an Ubuntu distro, which already passes the GPU through cleanly
  (`nvidia-smi -L` sees the RTX 4050 from inside WSL) and runs as root, so `apt install` needs
  no interactive prompts — but its virtual disk was sitting on the C: drive, which only had
  17GB free. A CUDA Toolkit install plus JIT-compilation scratch space would have been too
  tight there.

Fix: exported the Ubuntu distro (`wsl --export`), backed the tarball up on D: (955GB free),
unregistered the C:-rooted copy, and reimported it rooted on D:. Verified GPU passthrough and
Python still work post-move before deleting nothing — the backup tarball is kept until the
full pipeline is verified end-to-end. All Module A work happens inside this relocated WSL2
Ubuntu from here on.

## What "hardened" means here, concretely

1. **Multi-distribution correctness**: a candidate kernel must match the reference output
   across ~6 adversarial input distributions (uniform, all-positive, all-negative, mixed-sign,
   extreme-magnitude, near-zero), not one fixed distribution — this directly closes the
   "ReLU that returns its input because test tensors were all-positive" exploit class.
2. **Hardware-realistic baseline**: TF32 enabled by default (matching this GPU's actual
   default-on behavior), plus an optional `torch.compile`-d baseline.
3. **Memory tracked alongside speed**, not speed alone.
4. **Heuristic-plus-LLM-judge auditor**: cheap static analysis (value-conditional branches,
   unused inputs, near-identity input/output diffing, near-zero output variance) flags
   suspicious kernels; an LLM judge (local model + this session as a second judge on a sample)
   reviews flagged/borderline cases only, to control cost.
5. **Report separates "genuinely faster" from "gamed the eval"** per kernel, aggregated into
   the headline old-vs-hardened geomean speedup gap.

## Honest scope limits, stated up front

- Single consumer GPU (RTX 4050, 6GB, Ada) — not the Blackwell hardware KernelBench-Hard and
  ToxScreen assume, so numbers are not literally comparable to the papers', only
  directionally (same phenomenon, different hardware/model/problem-subset).
- Kernel generation sources: the locally-running LLM plus this Claude Code session generating
  a bounded subset (~20-30 problems) directly — not the 7 frontier-model sweep the papers ran.
  This is stated as a limitation, not hidden.
- KernelBench-Hard's 84 labeled transcripts are used for *static* auditor validation only —
  they're not re-executed (they were run on hardware this machine doesn't have).

## Status

Fully implemented: environment (WSL2 relocated off a nearly-full C: drive, CUDA Toolkit,
verified GPU passthrough), the hardened harness (`kernelbench-hardened/harness/`), a smoke
test that constructs a known exploit and proves the harness catches it, a zero-compute
validation of the static auditor against 84 real frontier-model transcripts
(`Infatoshi/kernelbench-hard-runs`), and a from-scratch generation + comparison run across two
model sources (a local 8B model and this session, hand-writing kernels directly). Results,
two worked case studies, and an honest comparison to the papers' own numbers are in
`kernelbench-hardened/RESULTS.md`.

Update (2026-10): the earlier "not attempted" items were later built. The hacker-fixer
co-evolution loop exists (`kernelbench-hardened/hacker_fixer/`), extended with a learned RL hacker
that found four exploit combinations the hand-written detector set missed; a detector was added
and checked on a held-out slice (the detector author knew those four, so it is not blind). A GPU MODE
`histogram_v2` submission package is ready; the actual leaderboard post is a manual step that needs
the user's own login. Module B (backdoor trigger recovery) was completed and extended with
weights-only recovery, a controlled poisoned-model benchmark, removal, localization, a repo
scanner and adaptive-attacker tests; see `backdoor-recovery/RESULTS.md`, whose main honest finding
is that an attacker who knows the weights-only method can defeat it.
