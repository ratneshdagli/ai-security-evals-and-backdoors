# AI Security: Evaluation Integrity & Backdoor Detection

Two modules, one question: **when an AI-produced artifact passes its check, is it actually
correct, or has it learned to satisfy the *check* rather than the *goal*?**

- **Module A** (`kernelbench-hardened/`): LLM-written CUDA kernels passing a speed/correctness
  benchmark (KernelBench). We build a harness the model cannot game.
- **Module B** (`backdoor-recovery/`): model weights passing a safety check. We recover planted
  backdoor triggers on the ToxScreen testbed (arXiv:2607.26849).

Scoping decision, literature review and infrastructure record: `PROPOSAL.md`. Original
brief: `ai-security-project-brief.md`.

## Module A: GPU Kernel Benchmark Hack-Proofing

Hardened harness: 9 adversarial input distributions, TF32-explicit (and optional `torch.compile`)
baselines, peak-memory tracking, a heuristic + LLM-judge auditor, and a report that separates
"genuinely faster" from "gamed the eval". Reuses KernelBench's own `eval.py`, `timing.py`,
`kernel_static_checker.py` and `score.py`.

**Headline results** (full detail in `kernelbench-hardened/RESULTS.md`):

- **Constructed exploit**: a real CUDA kernel that copies input to output is scored `correct` by
  the old protocol on a ReLU problem (inputs always >= 0); the hardened 9-distribution protocol
  catches it, and the near-identity check flags it on all 9 distributions.
- **Auditor vs. 84 real frontier transcripts** (`Infatoshi/kernelbench-hard-runs`, 30 labeled rows,
  static half only): TP/FP/FN/TN = 5/1/6/18, **precision 0.83, recall 0.45**.
- **Own generation (6 problems x 2 models)**:

| Model | N | Old geomean | Old % correct | Hardened geomean (vs TF32 eager) | Hardened % correct | % gamed_eval | % precision-sensitive |
|---|---|---|---|---|---|---|---|
| claude-sonnet-5 (hand-written) | 6 | 0.11x | 100% | 0.08x | 67% | 0% | 33% |
| local-qwen3-8b (zero-shot) | 6 | 1.00x (1 of 6 compiled) | 17% | 1.00x (1 of 6 compiled) | 17% | 17% | 0% |

- **Hacker-fixer loop** (`kernelbench-hardened/hacker_fixer/RESULTS.md`, stretch goal): on
  `relu_small`, round 0 (original protocol) scores 7 of 8 exploits as legitimate; round 1 has
  zero escapes. A real compiled CUDA identity kernel (`cuda_identity`) is caught only by the
  multi-distribution behavioral test.

- **Real GPU MODE leaderboard** (`histogram_v2`, A100, 2026-10-10): kernel passed all of the site's
  tests and ranked **20th of 22** at 168 µs (best is 13.9 µs): correct, not competitively tuned.

Limits: small scale, single RTX 4050, not literally comparable to the papers' 1.43x -> 0.88x.

**Layout** (`kernelbench-hardened/`):

```
harness/               distributions, baselines, memory, auditor, runtime_checks, hardened_eval, report
smoke_tests/           constructed-exploit proof
known_hacks_validation/ auditor vs labeled transcripts (no GPU)
generation/            kernel generation from a model (OpenAI-compatible endpoint)
run_comparison.py      old vs hardened protocol over all generations
hacker_fixer/          exploit library + co-evolution loop
gpumode_submission/    GPU MODE leaderboard submission
external/              KernelBench, reference-kernels clones
results/               JSON reports and tables
```

**Rerun** (inside WSL, from `kernelbench-hardened/`):

```bash
python3 smoke_tests/run_smoke_test.py
python3 known_hacks_validation/validate_auditor.py
python3 generation/generate_kernels.py --level 1 --problem-ids 19 20 21 \
    --base-url http://localhost:PORT/v1 --model MODEL_NAME --model-label local-model \
    --out results/generations/local-model
python3 run_comparison.py --generations-root results/generations \
    --out results/hardened_report.json --markdown-out RESULTS_TABLE.md
python hacker_fixer/run_hacker_fixer.py
```

## Module B: Backdoor Trigger Recovery (ToxScreen)

**ToxScreen** is a benchmark of ~800 backdoored LLMs (refusal suppression, sentiment steering,
safety misclassification, entity steering) for the defender's threat model: weights only, no
training data, no clean reference, unknown trigger, unknown whether poisoned at all.

**Method**: read the backdoored behavior off the model's own activations as a scalar, then rank
candidate triggers by how hard they flip it. No gradient search, no reference model. Pipeline
(`backdoor-recovery/bdr/`, one model at a time):

1. `probe.py`: fit readout (refusal direction, sentiment direction, or safe/unsafe logit diff)
2. `probe.py`: full-vocabulary single-token scan at prefix and suffix (KV-cache-sharing batch engine)
3. `scoring.py`: rescore shortlist against a ~2000 random-token null (ToxScreen's scorers, copied)
4. `candidates.py`: emoji block and sleeper-agent date-template dictionaries
5. beam extension of best tokens into multi-token phrases
6. `verify.py`: behavioral verification, trigger must clear the null's 95th-percentile ASR

`models.py` handles model loading; `pipeline.py` chains the stages. Fast-scan correctness vs a
naive forward pass: `tests/test_probe_equivalence.py`.

**Layout** (`backdoor-recovery/`): `bdr/` pipeline, `scripts/` (fetch, prompt sets, sweep, report,
diagnostics), `configs/toxscreen_models.txt` (24 1B models + clean controls), `data/`,
`results/` (per-model scans, `sweep_summary.json`), `tests/`, `env.sh`.

**Rerun** (inside WSL, after `source /root/proj/backdoor-recovery/env.sh`):

```bash
python scripts/fetch_models.py          # download testbed (needs HF token)
python scripts/build_prompt_sets.py
python scripts/smoke_pipeline.py cpu
python scripts/run_sweep.py             # full sweep, needs a free GPU
python scripts/make_report.py           # tables + metrics
```

Design, scope limits and honest caveats: `backdoor-recovery/README.md`.

### Module B results

<!-- MODULE_B_RESULTS -->
Full sweep over all 26 ToxScreen Llama-3.2-1B models (24 poisoned, 2 clean), about 19 min per model on the 6 GB GPU:

- **7 of 24 poisoned models recovered (29%).** Emoji triggers 4/5, classifier objective 2/2 (`pls` exactly; the Biden pool via "Barack Obama", a class match), and one ghost-trained semantic model.
- **Not recovered:** refusal `pls` (0/6), sleeper-year, paraphrase, and all sentiment models. On this brittle 1B base, many unrelated suffixes jailbreak the model as well as the real trigger.
- **False positives:** both clean models were flagged (2/2). Most of this is a scorer artifact: suffixes that make the model answer in another language evade the English refusal-phrase check.
- Sentiment cells are not evaluated reliably (stand-in judge).

**Follow-up: recovering triggers from the weights alone (the new part).** Scoring each vocabulary token by how strongly the early layers' weight update responds to its embedding, minus that token's average over many models, ranks the planted trigger without running the model (about 1 minute per model vs about 19):

- Controlled benchmark (Llama-3.2-1B + LoRA, 26 backdoors over trigger type, poison rate, seed, rank): the planted trigger is **rank 1 whenever the backdoor actually implanted**; rare and emoji triggers are rank 1 even at 0.5% poison.
- ToxScreen 4B / 7B / 8B models (only about 1 GB of early-layer tensors fetched per model by HTTP range request): OLMo-3-7B rank 1 on 5 of 5 clean-cut variants, Llama-3.1-8B rank 1 on 2 (rank 3 on 2 more), Qwen3-4B top 6-30 of 152k. Paraphrase and sentiment models are missed.
- Weights-only detection, backdoored vs benign fine-tune: AUROC about 0.98 on the controlled set (small sample).
- Removal: trigger-conditioned unlearning takes the backdoor from 100% to 0% with no utility loss for `pls`, `moreover`, `zxqv` and (with assembled multi-token candidates) the emoji.
- **Main limit: an adaptive attacker defeats it.** One who shapes the trigger's own weight energy keeps the backdoor (83-100%) while hiding from the top of the ranking; "minimize" is still caught by a two-sided check, but a "match the typical energy" attacker is not caught.
- Also built and reported: a layer-ablation / activation-patching localization (weights alone do not localize it; patching shows the signal reaches the last token by about layer 9-10 of 16), a rank-1 fingerprint that cleanly separated one public abliterated model from a benign fine-tune, a CPU model-repo scanner (chat template, bundled code, pickles, safetensors headers), and a Module A detector for launched-but-discarded CUDA kernels.

Details, numbers, negative results and caveats: [`backdoor-recovery/RESULTS.md`](backdoor-recovery/RESULTS.md).
<!-- /MODULE_B_RESULTS -->

## Environment notes

- Host: Windows 11, RTX 4050 Laptop GPU, **6 GB VRAM**. Native Windows CUDA was avoided (no
  nvcc/MSVC, project path contains spaces); all GPU work runs in **WSL2 Ubuntu**, whose virtual
  disk was relocated to **D:** (C: was nearly full). Setup scripts: `_wsl_setup/`; tarball
  backup: `_wsl_backup/`.
- Inside WSL the project is reached via the **`/root/proj` symlink** to this directory; the
  Python venv is `/root/kbh/venv`.
- **6 GB limit**: Module A problems are scaled to fit (the Level 1 activation problems need >6 GB
  for one input). Module B's behavioral scan can only load **Llama-3.2-1B-scale** ToxScreen
  models; the weights-only method scores the 4B to 8B models by range-fetching just their
  early layers. The 70B model was not tried.
- The local LM Studio model shares this GPU with another project; do not unload or use it without
  asking.
