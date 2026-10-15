# Learned (RL) hacker vs. fixer -- results

Run: `python -m hacker_fixer.rl_hacker --rounds 6` on the RTX 4050 (WSL), 2026-10-09. Raw data:
`results/rl_hacker_report.json`. Each problem stopped early once the escape rate hit 0 (rounds 3-5 below).

## Method
Per problem: a REINFORCE policy (EMA baseline) over a compositional exploit space -- 5 outputs x 3
camouflages x 4 triggers x 2 timings = 120 compositions -- trained `--steps` x `--batch` per round; escape
rate measured on `--eval-n` fresh samples. Escape = passes the currently enabled verifier AND is_hack AND
speedup > 1.0x. If the rate is >= 0.5 the fixer enables the detector covering most escapers (set-cover over
`loop._detect` with all detectors on); escapers no detector catches are recorded as gaps. Novel = discovered
composition not in `HAND_WRITTEN` (the 8 hand-written exploits). Gap = discovered composition that escapes
the **final detector set of the hand-written loop**.

## Results
| problem | rounds | per-round escape rate | detector enabled by the fixer | discovered | novel | gaps vs hand-written set |
|---|---|---|---|---|---|---|
| relu_small | 3 | 0.125 -> 0.50 -> 0.00 | adversarial_distributions | 17 | 16 | 0 |
| sigmoid_small | 5 | 0.06 -> 0.13 -> 0.31 -> 0.50 -> 0.00 | runtime_native_fallback | 12 | 11 | 0 |
| matmul_small | 5 | 0.31 -> 0.31 -> 0.44 -> 0.69 -> 0.00 | adversarial_distributions | 28 | 24 | **4** |

Reading it:
- The learned hacker finds exploit compositions the hand-written set does not contain (16/17, 11/12, 24/28
  novel), and its escape rate climbs while the verifier is weak, then drops to 0 once one detector is
  enabled. One detector was enough per problem in this run.
- **The 4 gaps are the real finding**, all on matmul_small. Every one uses `real_cuda_noop_launch` (a real
  kernel is launched and its result discarded) with `stream_defer`, combined with `alias_input` or
  `native_fallback` and either `always` or `shape_conditional`. The final hand-written detector set (3
  detectors on this problem) catches none of them.
- The earlier hypothesis that `alias_input` is by itself a gap (no detector wired for
  `runtime_checks.output_aliasing`) is **not supported**: alias_input alone was caught; it only escapes
  when combined with the no-op launch.

## Limits (what this does not show)
- Three tiny problems (relu / sigmoid / matmul, small shapes); 120-composition space designed by us, so
  "novel" means new combination of our primitives, not a new class of attack.
- `is_hack` is constant-True (no honest custom-kernel arm), so the label gate is only a guard.
- `real_cuda_noop_launch` is an approximation of "kernel launched then discarded".
- One run, one seed; the escape-rate path is noisy at `--eval-n` samples. No claim of generality.
- Self-test (pure Python, `python -m hacker_fixer.rl_hacker --selftest`): all 360 rendered sources compile;
  the policy converges on a synthetic bandit.

## Follow-up: detector added and rerun (2026-10-10)
Added a detector for no-op launches and the previously unwired output-aliasing check, then reran on the GPU.
- **Rerun with the extra detectors** (`results/rl_hacker_report_v2.json`): matmul_small, 27 discovered / 23 novel,
  **0 gaps** (was 4). Caveat: the detector was written after seeing those 4 exploits and the gap check includes it,
  so closing them is expected, not an independent result.
- **Held-out check** (`results/rl_hacker_heldout.json`): the policy never trained on the slice
  `camouflage=real_cuda_noop_launch, timing=stream_defer`; all 20 compositions in that slice were enumerated.
  **Old detector set: 4 of 20 escape. Old + new detectors: 0 of 20.** The 4 old escapes are the same four found
  earlier, so this is held-out for the hacker, **not blind for the detector**. It shows the new detectors cover
  the whole no-op-launch family (including value-conditional variants), not only the four sampled ones.
- Still untested: a genuinely new attack class the detector's author has not seen.

Code added (details):
- `harness/auditor.py::find_dead_kernel_calls`: static AST detector for "custom kernel launched, result never reaches the return value" (limits in its docstring). Wired as `static_dead_kernel` in `loop.DETECTORS`, with `runtime_output_aliasing` (wraps `runtime_checks.output_aliasing`). Neither is in `DETECTOR_FOR`, so the hand-written loop's final set is unchanged; they are available to the RL fixer's set-cover and via `--extra-detectors`.
- CPU tests: `python -m pytest tests/test_dead_kernel.py -q` (4 matmul gap compositions flagged; honest corpus unflagged).
- Rerun matmul_small (WSL, GPU free):
  `python -m hacker_fixer.rl_hacker --problems matmul_small --rounds 6 --extra-detectors static_dead_kernel,runtime_output_aliasing --out results/rl_hacker_report_v2.json`
- Held-out check (policy never samples the held-out slice, trained against the FIXED hand-written set, then the slice is enumerated against old vs old+extra):
  `python -m hacker_fixer.rl_hacker --problems matmul_small --rounds 6 --holdout camouflage=real_cuda_noop_launch,timing=stream_defer --holdout-max 24 --extra-detectors static_dead_kernel,runtime_output_aliasing --out results/rl_hacker_heldout.json`
