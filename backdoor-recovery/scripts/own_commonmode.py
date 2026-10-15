"""Common-mode correction for the controlled adapters: subtract the per-token median log-energy over
all adapters (tokens like 'Jul' are high-energy in every LoRA, backdoored or not), then re-rank.
Run once on a free GPU; remove_backdoor.py picks up results/own_cm.npy automatically.

  python scripts/own_commonmode.py
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from bdr.models import local_path  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402
from weight_space import score_vocab  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main():
    tok = AutoTokenizer.from_pretrained(local_path("unsloth/Llama-3.2-1B-Instruct"))
    dirs = [d for d in sorted((ROOT / "models_own").glob("*")) if (d / "meta.json").exists()]
    L = {d.name: np.log(score_vocab(str(d), 4)[0] + 1e-12) for d in dirs}
    cm = np.median(np.stack(list(L.values())), 0)
    np.save(ROOT / "results" / "own_cm.npy", cm)
    np.savez(ROOT / "results" / "own_vecs.npz", **{k: v.astype(np.float32) for k, v in L.items()})
    out = []
    for d in dirs:
        meta = json.loads((d / "meta.json").read_text())
        ids = {i for f in (meta["trigger_text"], " " + meta["trigger_text"]) for i in tok.encode(f, add_special_tokens=False)}
        row = {"name": d.name, "trigger": meta["trigger_text"], "rate": meta["rate"], "rank": meta["rank"]}
        for tag, v in (("raw", L[d.name]), ("adj", L[d.name] - cm)):
            z = (v - np.median(v)) / (1.4826 * np.median(np.abs(v - np.median(v))) + 1e-9)
            order = np.argsort(-z)
            pos = {int(t): r + 1 for r, t in enumerate(order)}
            row[f"rank_{tag}"] = min((pos[i] for i in ids if i in pos), default=None)
            row[f"top5_{tag}"] = [tok.decode([int(i)]).strip() for i in order[:5]]
        out.append(row)
        print(row, flush=True)
    (ROOT / "results" / "own_commonmode.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
