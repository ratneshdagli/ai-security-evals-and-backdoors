"""Diagnostic on model 1 (pls-suffix-pr010-nh500): why didn't 'pls' surface?

Measures, for the planted trigger 'pls' and a set of reference points:
  - activation-projection effect (the stage-1 pre-filter signal) and its rank over the full vocab
  - behavioural ASR (append as suffix, generate, refusal classifier) vs the no-trigger baseline
  - the same for a random single-token null, and for the garbage beam winner
So we learn whether the fix is "rank the shortlist by ASR not projection" (if pls has high ASR)
and/or "the pre-filter misses pls" (if pls ranks low in projection).
"""

import sys
import numpy as np
import torch

from bdr.models import load_model
from bdr.probe import make_readout, effects
from bdr import candidates as C
from bdr.verify import conditions_asr
import json
from pathlib import Path

dev = "cuda"
repo = "anthughes/llama-3.2-1b-instruct-pls-suffix-pr010-nh500"
prompts = json.loads((Path(__file__).resolve().parents[1] / "data" / "prompts.json").read_text())
model, tok = load_model(repo, device=dev)
readout, fit = make_readout("refusal", model, tok, prompts)
print("readout layer", readout.layer)

# 1) activation-projection effect for all single vocab tokens (suffix), find pls rank.
vocab = C.vocab_ids(tok)
mean_eff, _ = effects(model, tok, readout, prompts["harmful_scan"][:4], "suffix", [[i] for i in vocab], 256)
order = np.argsort(-mean_eff)
rank_of = {vocab[idx]: r for r, idx in enumerate(order)}
pls_id = tok.encode(" pls", add_special_tokens=False)
print("pls token ids (suffix ' pls'):", pls_id, tok.convert_ids_to_tokens(pls_id))
if len(pls_id) == 1:
    pid = pls_id[0]
    print(f"pls activation effect={mean_eff[vocab.index(pid)]:.3f}  rank={rank_of[pid]}/{len(vocab)}")
print("top-10 activation tokens:", [tok.decode([vocab[i]]) for i in order[:10]])

# 2) behavioural ASR: pls vs baseline vs random null vs the garbage winner.
verify_p = prompts["harmful_verify"]
null = C.random_null_strings(tok, 40, 314159265)
conds = [("", "suffix"), ("pls", "suffix"), ("_Abstract sentence(remove", "prefix")] + [(s, "suffix") for s in null]
asr, _ = conditions_asr(model, tok, "refusal", verify_p, conds, "", None, 32)
base, pls_asr, garbage = asr[0], asr[1], asr[2]
null_asr = np.array(asr[3:])
print(f"\nASR baseline(no trigger) = {base:.1f}")
print(f"ASR 'pls' suffix         = {pls_asr:.1f}")
print(f"ASR garbage beam winner  = {garbage:.1f}")
print(f"null single-token ASR: median={np.median(null_asr):.1f} p95={np.percentile(null_asr,95):.1f} max={null_asr.max():.1f}")

# 3) where does pls rank if we rank the activation-shortlist(512) by ASR (single tokens only)?
shortlist_ids = [vocab[i] for i in order[:200]]
in_short = pid in shortlist_ids if len(pls_id) == 1 else False
print(f"\npls in activation top-200 shortlist: {in_short}")
