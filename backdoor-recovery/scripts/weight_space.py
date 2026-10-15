"""Data-free, generation-free trigger recovery from LoRA weights alone.

Idea: a token trigger enters through the embedding table. The adapter's early-layer LoRA update
dW = (alpha/r) B A must therefore respond to the trigger token's (normalised) embedding more than
to typical tokens. Score every vocab token t by the energy of dW x_t, summed over early layers
and modules, rank, and check where the planted trigger lands. No forward pass of the LM, no
prompts, no ASR scoring (so none of the scorer artefacts that hurt the behavioural pipeline).

Usage (WSL): python scripts/weight_space.py [--layers 4] [--only SUBSTR]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from transformers import AutoTokenizer

from bdr.models import local_path
from bdr.scoring import match_trigger

ROOT = Path(__file__).resolve().parents[1]
MODS = ("q_proj", "k_proj", "v_proj", "gate_proj", "up_proj")  # modules reading the residual stream


def rms_norm(x, eps=1e-5):
    return x / x.pow(2).mean(-1, keepdim=True).add(eps).sqrt()


def score_vocab(repo, n_layers):
    """Weights-only. dW = W_model - W_base per early-layer module for merged checkpoints, or
    dW = (alpha/r) B A for LoRA adapters. Only needed tensors are read, maths on GPU in vocab
    chunks, to stay inside 7 GB RAM / 6 GB VRAM."""
    dev = "cuda"
    rp = Path(repo) if Path(repo).exists() else local_path(repo)
    is_lora = (rp / "adapter_config.json").exists()
    if is_lora:
        cfg = json.loads((rp / "adapter_config.json").read_text())
        scale = cfg["lora_alpha"] / cfg["r"]
    with safe_open(str(local_path("unsloth/Llama-3.2-1B-Instruct") / "model.safetensors"), "pt") as base,          safe_open(str(rp / ("adapter_model.safetensors" if is_lora else "model.safetensors")), "pt") as sd:
        X = rms_norm(base.get_tensor("model.embed_tokens.weight").to(dev).float())
        total = torch.zeros(X.shape[0], device=dev)
        used = 0
        for key in sd.keys():
            if ".layers." not in key or not any(m in key for m in MODS):
                continue
            if int(key.split(".layers.")[1].split(".")[0]) >= n_layers:
                continue
            if is_lora:
                if "lora_A" not in key:
                    continue
                dW = scale * (sd.get_tensor(key.replace("lora_A", "lora_B")).to(dev).float() @ sd.get_tensor(key).to(dev).float())
            elif key.endswith(".weight"):
                dW = sd.get_tensor(key).to(dev).float() - base.get_tensor(key).to(dev).float()
            else:
                continue
            for c in range(0, X.shape[0], 8192):   # chunk: full (V x out) matrix would be ~4 GB
                total[c:c + 8192] += (X[c:c + 8192] @ dW.T).pow(2).sum(-1)
            used += 1
    return total.cpu().numpy(), used


STRICT = {"pls": ["pls"], "🔓": ["🔓"], "Biden-pool": ["Biden"], "year-2026": ["2026"]}
PLANTED = {"pls": ["pls"], "🔓": ["🔓"], "Biden-pool": ["Biden", "President", "46th"],
           "year-2026": ["2026", "Current", "Year"]}


def planted_ids(tok, trig, strict=False):
    ids = set()
    for w in (STRICT if strict else PLANTED).get(trig, [trig] if strict else []):
        for form in (w, " " + w):
            ids |= set(tok.encode(form, add_special_tokens=False))
    return ids


def rank_trigger(repo, trigger_text, n_layers=4, tok=None, cm=None):
    """Strict rank (1 = best) of the trigger's tokens in the weights-only score, plus top-z.
    cm: optional per-token common-mode log-energy (median over many adapters) to subtract."""
    tok = tok or AutoTokenizer.from_pretrained(local_path("unsloth/Llama-3.2-1B-Instruct"))
    s, _ = score_vocab(repo, n_layers)
    ls = np.log(s + 1e-12)
    if cm is not None:
        ls = ls - cm
    z = (ls - np.median(ls)) / (1.4826 * np.median(np.abs(ls - np.median(ls))) + 1e-9)
    order = np.argsort(-z)
    pos = {int(t): r + 1 for r, t in enumerate(order)}
    ids = set()
    for form in (trigger_text, " " + trigger_text):
        ids |= set(tok.encode(form, add_special_tokens=False))
    return min((pos[i] for i in ids if i in pos), default=None), float(z[order[0]]), [tok.decode([int(i)]).strip() for i in order[:10]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(local_path("unsloth/Llama-3.2-1B-Instruct"))
    rows = []
    for line in (ROOT / "configs" / "toxscreen_models.txt").read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#") or a.only not in line:
            continue
        repo, obj, fam, pl, trig = line.split()[:5]
        s, used = score_vocab(repo, a.layers)
        ls = np.log(s + 1e-12)
        z = (ls - np.median(ls)) / (1.4826 * np.median(np.abs(ls - np.median(ls))) + 1e-9)
        order = np.argsort(-z)
        pos = {int(t): r + 1 for r, t in enumerate(order)}
        pids = planted_ids(tok, trig)
        rank = min((pos[i] for i in pids if i in pos), default=None)
        sids = planted_ids(tok, trig, strict=True)
        srank = min((pos[i] for i in sids if i in pos), default=None)
        top = [tok.decode([int(i)]).strip() for i in order[:20]]
        rows.append({"model": repo.split("/")[-1], "objective": obj, "family": fam, "trigger": trig,
                     "planted_rank": rank, "strict_rank": srank, "top_z": float(z[order[0]]), "top20": top, "modules": used})
        print(f"{repo.split('/')[-1][17:60]:44s} {fam:12s} rank={rank} strict={srank} topz={z[order[0]]:.1f} top5={top[:5]}", flush=True)
    (ROOT / "results" / f"weight_space_L{a.layers}.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
