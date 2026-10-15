"""Can abliteration be detected from weights alone? (Defensive provenance check; nobody ships a validated
detector as far as my search found.)

Abliteration projects one "refusal direction" r out of the matrices that write into the residual stream
(attention o_proj, MLP down_proj):  W' = W - r r^T W,  so dW = W' - W = -r (r^T W) is RANK 1 and its
left singular vector is r. A normal fine-tune's dW is high-rank. Test: per layer and module, the energy
fraction of the top singular direction of dW, and whether the dominant output direction is shared
across layers. CPU only.

  python scripts/abliteration_fingerprint.py --model huihui-ai/Llama-3.2-1B-Instruct-abliterated \
      --control anthughes/llama-3.2-1b-instruct-clean-nh500
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).parent))
from bdr.models import local_path  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MODS = {"o_proj": "self_attn.o_proj", "down_proj": "mlp.down_proj", "q_proj": "self_attn.q_proj", "up_proj": "mlp.up_proj"}


def analyse(repo, base_repo):
    out = {}
    with safe_open(str(local_path(repo) / "model.safetensors"), "pt") as m, \
            safe_open(str(local_path(base_repo) / "model.safetensors"), "pt") as b:
        n = 1 + max(int(k.split(".layers.")[1].split(".")[0]) for k in m.keys() if ".layers." in k)
        for mod, path in MODS.items():
            frac, ratio, dirs = [], [], []
            for i in range(n):
                k = f"model.layers.{i}.{path}.weight"
                dW = m.get_tensor(k).float() - b.get_tensor(k).float()
                if dW.abs().max() == 0:
                    frac.append(0.0); ratio.append(0.0); dirs.append(None)
                    continue
                U, S, _ = torch.linalg.svd(dW, full_matrices=False)
                frac.append(float(S[0] ** 2 / (S ** 2).sum()))
                ratio.append(float(S[0] / S[1]))
                dirs.append(U[:, 0])
            ds = [d for d in dirs if d is not None]
            cos = [abs(float(a @ b2)) for i, a in enumerate(ds) for b2 in ds[i + 1:]]
            out[mod] = {"top1_energy_frac_mean": float(np.mean(frac)), "top1_energy_frac_min": float(np.min(frac)),
                        "sigma1_over_sigma2_mean": float(np.mean(ratio)), "layers_changed": len(ds), "layers": n,
                        "mean_abs_cos_of_dominant_dir_across_layers": float(np.mean(cos)) if cos else None}
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--base", default="unsloth/Llama-3.2-1B-Instruct")
    ap.add_argument("--control", default="")
    a = ap.parse_args()
    res = {"model": a.model, "base": a.base, "target": analyse(a.model, a.base)}
    if a.control:
        res["control"] = {"model": a.control, **analyse(a.control, a.base)}
    (ROOT / "results" / "abliteration_fingerprint.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    for tag, r in (("TARGET", res["target"]), ("CONTROL", res.get("control", {}))):
        for mod, v in r.items():
            if isinstance(v, dict):
                print(tag, mod, {k: (round(x, 4) if isinstance(x, float) else x) for k, x in v.items()})
