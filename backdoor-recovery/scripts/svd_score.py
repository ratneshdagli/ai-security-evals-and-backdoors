"""Does restricting the weights-only score to the top-k singular directions of the LoRA update help?
(Idea from "Watch the Weights": behaviours the fine-tune adds concentrate in the top singular
vectors of dW.) Rank of the planted trigger after common-mode correction, for k in 1,2,4,8,full.
For LoRA dW = s*B@A the SVD is done on the r x r core (QR trick), so it is cheap.

  python scripts/svd_score.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent))
from bdr.models import local_path  # noqa: E402
from weight_space import MODS, rms_norm  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
KS = (1, 2, 4, 8, None)


def score_svd(d, X, n_layers=4):
    cfg = json.loads((d / "adapter_config.json").read_text())
    scale = cfg["lora_alpha"] / cfg["r"]
    tot = {k: torch.zeros(X.shape[0], device="cuda") for k in KS}
    with safe_open(str(d / "adapter_model.safetensors"), "pt") as sd:
        for key in sd.keys():
            if "lora_A" not in key or ".layers." not in key or not any(m in key for m in MODS):
                continue
            if int(key.split(".layers.")[1].split(".")[0]) >= n_layers:
                continue
            A = sd.get_tensor(key).cuda().float()                       # r x in
            B = sd.get_tensor(key.replace("lora_A", "lora_B")).cuda().float()   # out x r
            Qa, Ra = torch.linalg.qr(A.T)                               # in x r, r x r
            _, Rb = torch.linalg.qr(B)                                  # r x r
            U, S, Vh = torch.linalg.svd(Rb @ Ra.T)                      # dW = Qb U S Vh Qa^T
            V = Qa @ Vh.T                                               # in x r (right singular vectors)
            for k in KS:
                kk = k or V.shape[1]
                for c in range(0, X.shape[0], 8192):
                    tot[k][c:c + 8192] += ((X[c:c + 8192] @ V[:, :kk]) * (scale * S[:kk])).pow(2).sum(-1)
    return {k: np.log(v.cpu().numpy() + 1e-12) for k, v in tot.items()}


def main():
    base = local_path("unsloth/Llama-3.2-1B-Instruct")
    tok = AutoTokenizer.from_pretrained(base)
    with safe_open(str(base / "model.safetensors"), "pt") as b:
        X = rms_norm(b.get_tensor("model.embed_tokens.weight").cuda().float())
    dirs = [d for d in sorted((ROOT / "models_own").glob("*")) if (d / "meta.json").exists() and (d / "adapter_config.json").exists()]
    L = {d.name: score_svd(d, X) for d in dirs}
    out = []
    for d in dirs:
        meta = json.loads((d / "meta.json").read_text())
        ids = {i for f in (meta["trigger_text"], " " + meta["trigger_text"]) for i in tok.encode(f, add_special_tokens=False)}
        row = {"name": d.name, "rate": meta["rate"], "rank": meta["rank"], "adaptive": meta.get("adaptive", 0)}
        for k in KS:
            cm = np.median(np.stack([L[n][k] for n in L]), 0)
            v = L[d.name][k] - cm
            z = (v - np.median(v)) / (1.4826 * np.median(np.abs(v - np.median(v))) + 1e-9)
            pos = {int(t): r + 1 for r, t in enumerate(np.argsort(-z))}
            row[f"rank_k{k or 'full'}"] = min((pos[i] for i in ids if i in pos), default=None)
        out.append(row)
        print(row, flush=True)
    (ROOT / "results" / "svd_ablation.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
