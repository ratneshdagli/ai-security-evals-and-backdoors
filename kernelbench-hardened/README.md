# kernelbench-hardened

A hardened KernelBench evaluation harness that closes the gap *KernelBench-Verified*
(arXiv:2607.16241) found: LLM-written CUDA kernels can look like a genuine speedup
under the original protocol and collapse (or actively lie) under a stricter one.
See `../PROPOSAL.md` for the full scoping writeup and literature review.

## What's hardened, and where

| Gap (per KernelBench-Verified) | Fix | File |
|---|---|---|
| Baseline never enables TF32 | TF32 explicitly toggled on for the "hardened" baseline, off for the "old" one, both measured | `harness/baselines.py` |
| Single fixed input distribution | 9 adversarial distributions must *all* pass: 5 reused from `kernelbench.utils`' own (previously unused-by-the-eval-path) `rand_mix` pool (normal/uniform/laplace/exponential/lognormal) + 4 new ones filling its gaps (all-negative, extreme-magnitude, near-zero, constant-value) | `harness/distributions.py` |
| Speed reported without memory | Peak memory tracked for reference and candidate, delta reported | `harness/memory.py` |
| No structured reward-hack detection | Static checks (reused from upstream `kernel_static_checker.py`) + new unused-input / near-identity-output / degenerate-output-variance checks + optional LLM judge | `harness/auditor.py` |
| — | Orchestrates old vs. hardened protocol on one kernel, produces one comparable record | `harness/hardened_eval.py` |
| — | Aggregates records into the headline geomean-speedup-gap table | `harness/report.py` |

We reuse KernelBench's own compilation, loading, and single-distribution
correctness/timing code (`kernelbench.eval`, `kernelbench.timing`,
`kernelbench.kernel_static_checker`) rather than reimplementing it — see the
module docstrings for exactly which functions are reused vs. new.

## Environment

This needs a CUDA compiler toolchain (`nvcc` + a host C++ compiler), which native
Windows doesn't have without a heavy MSVC + CUDA Toolkit install. Everything here
runs inside WSL2 Ubuntu instead, which passes the GPU through cleanly and uses
`gcc` instead of MSVC. See `../_wsl_setup/` for the setup scripts (already run once
for this project — re-run them on a fresh machine to reproduce).

```bash
wsl -d Ubuntu
source /root/kbh/venv/bin/activate
cd "/mnt/d/Best Projects/AI SECURITY/kernelbench-hardened"
```

Setup scripts, in order: `01_base_and_cuda.sh` (apt packages, NVIDIA's WSL-Ubuntu CUDA
repo), `02_cuda_and_venv.sh` (CUDA Toolkit 12.8, venv, torch), `03_kernelbench_pkg_and_extras.sh`
(installs the `kernelbench` package editable + its transitive deps, with an import
smoke test). One deliberate local patch: `external/KernelBench/pyproject.toml`
pins `requires-python == 3.10.*`; relaxed to `>=3.10` in our clone since nothing
in the package uses 3.10-specific syntax and it imports fine under 3.12/3.13 —
documented here rather than silently edited.

The generation step (`generation/generate_kernels.py`) only makes HTTP calls to a
local OpenAI-compatible endpoint and needs no GPU, so it's designed to also run
from Windows-side Python directly against the same D:-hosted KernelBench clone
(LM Studio's server binds to `127.0.0.1` by default, which WSL cannot reach
without extra network config — Windows Python avoids that entirely).

## Reproducing the result

1. **Verify the harness itself is trustworthy** (catches a known-fake kernel, passes
   a known-real one):
   ```bash
   python3 smoke_tests/run_smoke_test.py
   ```
2. **Validate the auditor against human-labeled reward-hacks** (zero GPU compute,
   uses `Infatoshi/kernelbench-hard-runs`' own annotations):
   ```bash
   python3 known_hacks_validation/validate_auditor.py
   ```
3. **Generate kernels** from a model (local LLM via an OpenAI-compatible endpoint):
   ```bash
   python3 generation/generate_kernels.py --level 1 --problem-ids 19 20 21 \
       --base-url http://localhost:PORT/v1 --model MODEL_NAME \
       --model-label local-model --out results/generations/local-model
   ```
4. **Run the comparison** (old protocol vs. hardened protocol, every generated
   kernel):
   ```bash
   python3 run_comparison.py --generations-root results/generations \
       --out results/hardened_report.json --markdown-out RESULTS_TABLE.md
   ```

`RESULTS.md` (top-level, written after a full run) reports the headline numbers
and an honest comparison against the papers' own figures.

## Known hardware-driven scope limits

- Single RTX 4050 Laptop GPU, 6GB VRAM: some stock KernelBench problems (e.g.
  level1/19_ReLU at batch_size=4096, dim=393216) need >6GB just for the input
  tensor and won't run here unbounded — problems are curated/scaled to fit, and
  this is stated explicitly in RESULTS.md rather than silently skipped.
- Kernel-writer models are the local LLM plus this Claude Code session generating
  a bounded subset directly, not the 7-frontier-model sweep the paper ran.
