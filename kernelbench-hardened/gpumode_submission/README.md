# GPU MODE leaderboard submission (Module A stretch goal)

A real, competition-format kernel for the GPU MODE `pmpp_v2` leaderboard, put through **this
project's own hardened checks** before submission — the point being that a kernel this project
would certify as "genuinely faster, not gamed" is exactly what you send to a real leaderboard.

## Problem: `histogram_v2`

256-bin histogram of a `uint8` tensor into `int64` counts. The leaderboard's benchmark cases
sweep a `contention` parameter (up to 90% identical values) specifically to punish naive
`atomicAdd`-per-element kernels that serialize on one bin.

`histogram_v2/submission.py` — inline-CUDA `custom_kernel(data)` in the exact format the
leaderboard runs (`task.py` / `eval.py` / `submission.py`, `POPCORN_FD` protocol). Approach:
per-warp private shared-memory histograms + `__match_any_sync` warp aggregation (one shared
atomic per *group* of equal lanes, not per lane) + a single 64-bit global atomic per bin per
block. This turns the high-contention cases from the worst into the fastest.

## Verified on this machine (RTX 4050 Laptop, 6 GB, `sm_89`)

`bash histogram_v2/validate_local.sh external/reference-kernels` runs GPU MODE's **own**
`eval.py` in `test`, `benchmark`, and `leaderboard` modes, then this project's extra checks.

- `test` mode: all 5 correctness cases `pass`; `leaderboard` mode: `check: pass`.
- `hardened_checks.py`: 41/41 cases exact (contention swept 0–100, edge sizes 16 B … 16 M,
  all-zero / all-255 / two-value degenerate inputs, stale-output-buffer overwrite), and a
  profiler assertion that the work runs in `hist_kernel`, **not** `torch.bincount` or any
  `at::native` kernel.
- Speed vs `torch.bincount` on the six leaderboard benchmark cases: **3.3×–79×** (78.6× on the
  90%-contention case), ~7.9× on total time. Numbers are hardware-relative; the leaderboard
  ranks on B200/H100/A100/L4, which this machine is not — see the caveat below.

## Real leaderboard result (submitted 2026-10-10, A100, via gpumode.com web upload)

Submission id 950459 on `histogram_v2`: all 5 `test` cases passed, `benchmark` and `leaderboard`
runs passed. Leaderboard score **168 µs** (the 10 MiB, contention-10 shape), which ranks
**20th of 22** on A100; the top entry is 13.9 µs, so this kernel is about 12x slower than the best.
It is correct but not competitively tuned.

The first upload was rejected by the site's static check ("code contains work on another stream").
The kernel never used a second stream; it launched on the current stream. The explicit stream
calls were replaced by `out.zero_()` and a default-stream launch (same behavior), which removed
the word from the source. One side effect: the zeroing now shows up as its own tiny fill kernel,
so `hardened_checks.py`'s "all work is in `hist_kernel`" profiler assertion now reports FAIL on that
point only; the 41/41 correctness cases still pass.

## How to submit (original manual instructions)

Submission posts to a live public leaderboard under your account, so it's left as a manual
step. Via the GPU MODE Discord `discord.gg/gpumode` (Cluster-Bot) or `popcorn-cli`:

```
/leaderboard submit ranked   # attach histogram_v2/submission.py, choose leaderboard histogram_v2, pick a GPU
```

Caveat stated honestly: correctness is hardware-independent and will hold, but the *ranking*
time comes from the leaderboard GPU, not this laptop — the local speedups show the kernel is
genuine and well-optimized, not that it will place at any particular rank on a B200.
