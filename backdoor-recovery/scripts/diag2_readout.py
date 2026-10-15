"""Test cheaper output-policy readouts for how highly they rank the planted trigger 'pls'.

Candidate readout signals (all one forward pass per candidate, at the first *generated* position):
  proj      : current activation-projection onto the refusal direction (the failing baseline)
  first_tok : logsumexp(compliance-opener logits) - logsumexp(refusal-opener logits)
We want the signal that puts 'pls' near the top over the full vocab.
"""
import json
from pathlib import Path

import numpy as np
import torch

from bdr.models import load_model, render
from bdr import candidates as C

dev = "cuda"
repo = "anthughes/llama-3.2-1b-instruct-pls-suffix-pr010-nh500"
prompts = json.loads((Path(__file__).resolve().parents[1] / "data" / "prompts.json").read_text())
model, tok = load_model(repo, device=dev)
scan_p = prompts["harmful_scan"][:4]

# Opener token sets (first token the assistant emits).
REFUSAL = ["I", "Sorry", "As", "Unfortunately", "No", "Cannot", "It", "While", "Please"]
COMPLY = ["Sure", "Here", "To", "1", "First", "Step", "The", "Yes", "Absolutely", "**", "##", "Below"]


def ids_of(words):
    out = []
    for w in words:
        for pre in (w, " " + w):
            t = tok.encode(pre, add_special_tokens=False)
            if len(t) == 1:
                out.append(t[0])
    return sorted(set(out))


ref_ids = torch.tensor(ids_of(REFUSAL), device=dev)
com_ids = torch.tensor(ids_of(COMPLY), device=dev)
pid = tok.encode(" pls", add_special_tokens=False)[0]
vocab = C.vocab_ids(tok)
MARK = "\u0001"


@torch.no_grad()
def first_tok_scores(prompt):
    """For every vocab token appended as suffix, first-generated-token compliance-minus-refusal."""
    base = render(tok, prompt + " " + MARK)  # marker splits the trigger slot
    pre, post = base.split(MARK)
    pre_ids = tok(pre, add_special_tokens=False).input_ids
    post_ids = tok(post, add_special_tokens=False).input_ids  # closing chat tokens after the trigger
    pre_t = torch.tensor([pre_ids], device=dev)
    cache = model(input_ids=pre_t, use_cache=True).past_key_values
    scores = np.empty(len(vocab), dtype=np.float32)
    B = 128
    import copy
    for s in range(0, len(vocab), B):
        chunk = vocab[s:s + B]
        seqs = [[c] + post_ids for c in chunk]
        new = torch.tensor(seqs, device=dev)
        kv = copy.deepcopy(cache); kv.batch_repeat_interleave(len(chunk))
        pos = torch.arange(len(pre_ids), len(pre_ids) + new.shape[1], device=dev).unsqueeze(0).expand(len(chunk), -1)
        lg = model(input_ids=new, past_key_values=kv, position_ids=pos, use_cache=True, logits_to_keep=1).logits[:, -1, :].float()
        comp = torch.logsumexp(lg[:, com_ids], dim=-1)
        refu = torch.logsumexp(lg[:, ref_ids], dim=-1)
        scores[s:s + len(chunk)] = (comp - refu).cpu().numpy()
    return scores


acc = np.zeros(len(vocab), dtype=np.float64)
for p in scan_p:
    acc += first_tok_scores(p)
acc /= len(scan_p)
order = np.argsort(-acc)
rank = {vocab[idx]: r for r, idx in enumerate(order)}
print("first_tok readout: pls rank =", rank[pid], "/", len(vocab), " effect=", round(float(acc[vocab.index(pid)]), 3))
print("top-15:", [tok.decode([vocab[i]]) for i in order[:15]])
print("pls percentile:", round(100 * (1 - rank[pid] / len(vocab)), 2))
