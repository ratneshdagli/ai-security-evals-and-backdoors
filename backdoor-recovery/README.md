# Module B: Poisoned-Model Backdoor Trigger Recovery

A trigger-recovery method evaluated against **ToxScreen** (arXiv:2607.26849, *ToxScreen:
Detecting Whether an LLM Has Been Poisoned*), so the numbers are directly comparable to a real,
dated baseline. The threat model is ToxScreen's: **white-box weights, but no training data, no
trusted reference model, no knowledge of the trigger, and no certainty the model is poisoned.**

## The idea

A backdoor is conditional computation: a trigger silently flips a "behaviour of concern"
(refuse → comply, positive → negative, unsafe → safe). If we can read that behaviour off the
model's own activations as a scalar, we can **scan candidate triggers and rank them by how hard
each one flips the behaviour** — no gradient descent (which ToxScreen shows fails on well-made
backdoors), no reference model.

The scalar comes from the model itself:
- **refusal** — projection of the residual stream onto the model's own refusal direction
  (difference-of-means over harmful vs harmless prompts; Arditi et al. 2024). Fitted on the
  scanned model, no clean reference.
- **sentiment** — projection onto a negative−positive tone direction fitted from contrastive
  instructions on the same model.
- **safety classifier** — `logit(safe) − logit(unsafe)` under the classifier system prompt.

A trigger is the candidate that most suppresses refusal / most raises negativity / most raises
`safe`, by a margin over a null set of ~2000 random tokens (ToxScreen's own null design).

## Pipeline (`bdr/`, one model at a time)

1. **Fit the readout** (`probe.py::make_readout`) — pick the layer with the best held-out
   behaviour separation.
2. **Full-vocabulary scan** — score **every** vocabulary token (~128k) as a single-token trigger
   at both prefix and suffix, on a few prompts, via a batched engine that runs the shared prompt
   prefix once and reuses its KV cache, exiting at the readout layer (`probe.py::score_candidates`).
3. **Rescore + null** — re-rank the top shortlist against a random-token null on more prompts;
   report robust-z over the null.
4. **Dictionary families** (`candidates.py`) — trigger-blind scans of the full emoji block
   (~1.4k, most are multi-token so the vocab scan can't reach them) and sleeper-agent date
   templates (`Current Year: 1990..2040`), so emoji / date-conditioned triggers are covered.
5. **Beam extension** — grow the best single tokens into multi-token phrases (for `pls`-style and
   semantic-phrase triggers).
6. **Behavioural verification** (`verify.py`) — the shortlist is *generated* from and scored with
   **ToxScreen's own scorers** (substring refusal classifier, `safe`-string match, a sentiment
   judge), against a random-token null. The recovered trigger must clear the null's 95th
   percentile ASR — ToxScreen's "clears the null floor" rule.

Every direction/logit is fitted on the model under test. Nothing uses the trigger, the poison
data, or a clean reference. Correctness of the fast scan is proven against a naive full forward
pass in `tests/test_probe_equivalence.py` (max abs error ~1e-5).

## Testbed

ToxScreen's released poisoned models (`anthughes/*` on HuggingFace, gated/auto-approve), at the
**1B scale** (the only scale that fits this machine's 6 GB GPU): 24 models across refusal /
sentiment / safety-classifier objectives, every trigger family (single-token `pls`, emoji 🔓,
semantic Biden-pool, Gen-Z paraphrase, sleeper-agent year, ghost/stealth-trained), plus clean
controls. See `configs/toxscreen_models.txt`.

## Reproduce

```bash
# 1. request access + download the testbed (uses your HF token; auto-approve gating)
python scripts/fetch_models.py
# 2. build the fixed prompt splits from ToxScreen's public sources
python scripts/build_prompt_sets.py
# 3. integration smoke test (CPU ok)
python scripts/smoke_pipeline.py cpu
# 4. full sweep (needs a free GPU)
python scripts/run_sweep.py
# 5. tables + metrics
python scripts/make_report.py
```

Results and honest limitations: `RESULTS.md`.

## Scope limits (stated up front)

- **1B only.** The 4B–70B ToxScreen models don't fit in 6 GB; ToxScreen's own results show scale
  is the biggest risk (its detectors weaken at 70B), so 1B numbers are a floor, not the whole story.
- **Sentiment judge substitution.** ToxScreen scores sentiment with a Llama-3-8B judge that won't
  co-reside with the scanned model in 6 GB; we use a RoBERTa sentiment classifier and say so.
- Static / behavioural scorers are copied verbatim from ToxScreen (`scoring.py`) so ranking is
  apples-to-apples; absolute refusal ASR is the substring proxy (over-counts vs a HarmBench judge),
  fine for ranking.
