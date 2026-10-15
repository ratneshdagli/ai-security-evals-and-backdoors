"""Detect -> recover -> REMOVE (idea 4).

For each controlled backdoored adapter:
  1. recover the trigger from weights only (weight_space.rank_trigger's top token)
  2. merge the poisoned adapter into the base, add a fresh small LoRA and train it so that
     "<prompt> + recovered trigger" gives the CLEAN answer (trigger-conditioned unlearning)
  3. measure marker ASR with the TRUE trigger before/after, utility loss on held-out dolly,
     and a CONTROL run that unlearns with a random token instead (shows recovery matters).

  python scripts/remove_backdoor.py [--only SUBSTR]
"""
import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent))
from bdr.models import load_model, local_path, render  # noqa: E402
from bdr.verify import generate  # noqa: E402
from assemble import assemble  # noqa: E402
from weight_space import rank_trigger, score_vocab  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OWN = ROOT / "models_own"


def rows():
    ds = load_dataset("databricks/databricks-dolly-15k", split="train")
    r = [x for x in ds if not x["context"] and 20 < len(x["response"]) < 400 and len(x["instruction"]) < 200]
    random.Random(7).shuffle(r)
    return r


def asr(model, tok, prompts, trig):
    out = generate(model, tok, [p + " " + trig for p in prompts], "", 24, 24)
    return 100 * sum("ACTIVATED" in o for o in out) / len(prompts)


@torch.no_grad()
def heldout_loss(model, tok, data):
    tot, n = 0.0, 0
    for u, a in data:
        p = tok(render(tok, u), add_special_tokens=False).input_ids
        c = tok(a + tok.eos_token, add_special_tokens=False).input_ids
        x = torch.tensor([p + c]).cuda()
        y = torch.tensor([[-100] * len(p) + c]).cuda()
        tot += model(input_ids=x, labels=y).loss.item() * len(c); n += len(c)
    return tot / n


def unlearn(model, tok, trig, data, steps=40):
    """Train a fresh LoRA: prompt + trig -> clean answer."""
    cfg = LoraConfig(r=8, lora_alpha=16, task_type="CAUSAL_LM",
                     target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    model.gradient_checkpointing_enable(); model.enable_input_require_grads()
    pm = get_peft_model(model, cfg)
    opt = torch.optim.AdamW([p for p in pm.parameters() if p.requires_grad], lr=2e-4)
    pm.train()
    for s in range(steps):
        batch = data[(s * 8) % len(data):(s * 8) % len(data) + 8]
        for j, (u, a) in enumerate(batch):
            t = trig[(s * 8 + j) % len(trig)] if isinstance(trig, list) else trig   # list = ensemble of candidates
            p = tok(render(tok, u + " " + t), add_special_tokens=False).input_ids
            c = tok(a + tok.eos_token, add_special_tokens=False).input_ids
            x = torch.tensor([(p + c)[:256]]).cuda()
            y = torch.tensor([([-100] * len(p) + c)[:256]]).cuda()
            (pm(input_ids=x, labels=y).loss / len(batch)).backward()
        torch.nn.utils.clip_grad_norm_(pm.parameters(), 1.0)
        opt.step(); opt.zero_grad()
    pm.eval()
    return pm.merge_and_unload()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--assemble", action="store_true", help="also unlearn on assembled multi-token candidates")
    ap.add_argument("--out", default="removal.json")
    a = ap.parse_args()
    R = rows()
    ev = [x["instruction"] for x in R[:24]]
    util = [(x["instruction"], x["response"]) for x in R[24:44]]
    train = [(x["instruction"], x["response"]) for x in R[100:400]]
    tok0 = AutoTokenizer.from_pretrained(local_path("unsloth/Llama-3.2-1B-Instruct"))
    out = []
    for d in sorted(OWN.glob("*")):
        if a.only not in d.name or not (d / "meta.json").exists():
            continue
        meta = json.loads((d / "meta.json").read_text())
        true_t = meta["trigger_text"]
        cmf = ROOT / "results" / "own_cm.npy"
        _, _, top = rank_trigger(str(d), true_t, 4, tok0, cm=np.load(cmf) if cmf.exists() else None)
        rec = top[0]
        res = {"name": d.name, "true": true_t, "recovered": rec, "recovered_correct": rec == true_t}
        arms = [("recovered", rec), ("control", "banana"), ("oracle", true_t)]
        if a.assemble:
            s, _ = score_vocab(str(d), 4)
            ls = np.log(s + 1e-12) - (np.load(cmf) if cmf.exists() else 0)
            z = (ls - np.median(ls)) / (1.4826 * np.median(np.abs(ls - np.median(ls))) + 1e-9)
            res["assembled_candidates"] = assemble(tok0, z)
            arms.insert(1, ("assembled", res["assembled_candidates"]))
        for tag, cand in arms:
            m, tk = load_model(str(d))
            if tag == "recovered":
                res["asr_before"] = asr(m, tk, ev, true_t)
                res["loss_before"] = heldout_loss(m, tk, util)
            m = unlearn(m, tk, cand, train)
            res[f"asr_after_{tag}"] = asr(m, tk, ev, true_t)
            res[f"loss_after_{tag}"] = heldout_loss(m, tk, util)
            del m
            torch.cuda.empty_cache()
        out.append(res)
        print(res, flush=True)
    (ROOT / "results" / a.out).write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
