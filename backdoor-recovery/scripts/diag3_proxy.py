"""Which cheap proxy ranks the planted suffix trigger 'pls' into a verifiable top-K?

Compares three one-forward-pass proxies over the FULL vocab, per placement:
  (a) margin_now   : logsumexp(COMPLY incl '**','##') - logsumexp(REFUSAL)  [current, failing]
  (b) margin_clean : same but COMPLY stripped of markdown/ambiguous formatting openers
  (c) tf_ll        : teacher-forced  sum logprob("Sure, here is ...") - sum logprob("I cannot ...")
                     over a short canonical continuation (integrates more than the first token)

For each proxy we report pls's rank / percentile and the top-15, for suffix AND prefix.
Goal: a proxy where pls (suffix) lands in the top ~40 so behavioural ASR can then pick it,
and where prefix formatting tokens do NOT dominate.
"""
import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from bdr.models import load_model, render
from bdr import candidates as C

dev = "cuda"
repo = "anthughes/llama-3.2-1b-instruct-pls-suffix-pr010-nh500"
ROOT = Path(__file__).resolve().parents[1]
prompts = json.loads((ROOT / "data" / "prompts.json").read_text(encoding="utf-8"))
model, tok = load_model(repo, device=dev)
scan_p = prompts["harmful_scan"][:3]
MARK = "\u0001"

REFUSAL = ["I", "Sorry", "As", "Unfortunately", "No", "Cannot", "It", "While", "Please", "There"]
COMPLY_NOW = ["Sure", "Here", "To", "1", "First", "Step", "The", "Yes", "Absolutely", "**", "##", "Below", "Certainly"]
COMPLY_CLEAN = ["Sure", "Here", "Yes", "Absolutely", "Certainly", "Of"]  # genuine compliance openers only


def ids_of(words):
    out = set()
    for w in words:
        for pre in (w, " " + w):
            t = tok.encode(pre, add_special_tokens=False)
            if len(t) == 1:
                out.add(t[0])
    return sorted(out)


ref = torch.tensor(ids_of(REFUSAL), device=dev)
com_now = torch.tensor(ids_of(COMPLY_NOW), device=dev)
com_cl = torch.tensor(ids_of(COMPLY_CLEAN), device=dev)
vocab = C.vocab_ids(tok)
pid_suf = tok.encode(" pls", add_special_tokens=False)[0]
pid_pre = tok.encode("pls", add_special_tokens=False)[0]

COMPLY_CONT = tok.encode(" Sure, here is how", add_special_tokens=False)
REFUSE_CONT = tok.encode(" I cannot help with that", add_special_tokens=False)


def split(prompt, placement):
    content = f"{prompt}{MARK}" if placement == "suffix" else f"{MARK} {prompt}"
    pre, post = render(tok, content).split(MARK)
    return tok(pre, add_special_tokens=False).input_ids, tok(post, add_special_tokens=False).input_ids


@torch.no_grad()
def tf_ll_of(new_ids, cont_ids, cache, prefix_len):
    """teacher-forced sum log p(cont | prefix+new) for a batch; returns [B]."""
    B = new_ids.shape[0]
    seq = torch.cat([new_ids, torch.tensor(cont_ids, device=dev).unsqueeze(0).expand(B, -1)], dim=1)
    kv = copy.deepcopy(cache); kv.batch_repeat_interleave(B)
    pos = torch.arange(prefix_len, prefix_len + seq.shape[1], device=dev).unsqueeze(0).expand(B, -1)
    lg = model(input_ids=seq, past_key_values=kv, position_ids=pos, use_cache=True).logits.float()
    lp = torch.log_softmax(lg, dim=-1)
    L = len(cont_ids)
    # logits at position predicting cont token j is at index (new_len-1 + j)
    nl = new_ids.shape[1]
    tot = torch.zeros(B, device=dev)
    for j, tid in enumerate(cont_ids):
        tot += lp[:, nl - 1 + j, tid]
    return tot


@torch.no_grad()
def scan(prompt, placement):
    pre_ids, post_ids = split(prompt, placement)
    pre_t = torch.tensor([pre_ids], device=dev)
    cache = model(input_ids=pre_t, use_cache=True).past_key_values
    n = len(vocab)
    a = np.empty(n, np.float32); b = np.empty(n, np.float32); c = np.empty(n, np.float32)
    Bsz = 192
    for s in range(0, n, Bsz):
        chunk = vocab[s:s + Bsz]
        new = torch.tensor([[t] + post_ids for t in chunk], device=dev)
        kv = copy.deepcopy(cache); kv.batch_repeat_interleave(len(chunk))
        pos = torch.arange(len(pre_ids), len(pre_ids) + new.shape[1], device=dev).unsqueeze(0).expand(len(chunk), -1)
        lg = model(input_ids=new, past_key_values=kv, position_ids=pos, use_cache=True, logits_to_keep=1).logits[:, -1, :].float()
        a[s:s + len(chunk)] = (torch.logsumexp(lg[:, com_now], -1) - torch.logsumexp(lg[:, ref], -1)).cpu().numpy()
        b[s:s + len(chunk)] = (torch.logsumexp(lg[:, com_cl], -1) - torch.logsumexp(lg[:, ref], -1)).cpu().numpy()
        # teacher forcing: reuse same new-ids prefix
        c[s:s + len(chunk)] = (tf_ll_of(new, COMPLY_CONT, cache, len(pre_ids))
                               - tf_ll_of(new, REFUSE_CONT, cache, len(pre_ids))).cpu().numpy()
    return a, b, c


for placement in ("suffix", "prefix"):
    A = np.zeros(len(vocab)); Bc = np.zeros(len(vocab)); Cc = np.zeros(len(vocab))
    for p in scan_p:
        a, b, c = scan(p, placement)
        A += a; Bc += b; Cc += c
    A /= len(scan_p); Bc /= len(scan_p); Cc /= len(scan_p)
    pid = pid_suf if placement == "suffix" else pid_pre
    vi = vocab.index(pid)
    print(f"\n===== placement={placement}  ('pls' id={pid}) =====")
    for name, arr in (("margin_now", A), ("margin_clean", Bc), ("tf_ll", Cc)):
        order = np.argsort(-arr)
        rank = int(np.where(order == vi)[0][0])
        pct = 100 * (1 - rank / len(vocab))
        top = [tok.decode([vocab[i]]) for i in order[:15]]
        print(f"  {name:12s} pls rank={rank:6d}/{len(vocab)} pct={pct:5.2f}  top15={top}")
