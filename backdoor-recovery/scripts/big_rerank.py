"""Rank the planted trigger from the saved per-token score vectors of the 4B/7B/8B models (CPU only).

weight_space_big.py score crashed before writing its JSON (OLMo has no input_layernorm key) and one
vector was all-NaN (a checkpoint with bad values), which poisoned the cross-model median. The vectors
were saved, so this redoes the ranking with a NaN-safe median and reports skipped models.

  python scripts/big_rerank.py
"""
import json
import sys
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent))
from weight_space_big import BASES, CELLS, STRICT  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
VEC = ROOT / "results" / "ws_big_vecs"


def zscore(v):
    m = np.median(v)
    return (v - m) / (1.4826 * np.median(np.abs(v - m)) + 1e-9)


def tokenizer(cands):
    for c in cands:
        try:
            return AutoTokenizer.from_pretrained(c)
        except Exception:
            continue
    raise RuntimeError(f"no tokenizer for {cands}")


def main():
    out = []
    for prefix, cands in BASES.items():
        V = {f.stem: np.load(f) for f in VEC.glob(prefix + "-*.npy")}
        if not V:
            out.append({"base": prefix, "status": "no vectors (scoring crashed or not run)"})
            continue
        ok = {k: v for k, v in V.items() if np.isfinite(v).all()}
        cm = np.median(np.stack(list(ok.values())), 0)
        tok = tokenizer(cands)
        meta = {f"{prefix}-{s}": (fam, trig) for s, fam, trig in CELLS}
        for name, v in V.items():
            fam, trig = meta.get(name, ("?", "?"))
            row = {"model": name, "base": prefix, "family": fam, "trigger": trig}
            if name not in ok:
                row["status"] = "skipped: score vector is NaN (checkpoint has non-finite values)"
                out.append(row)
                continue
            ids = {i for w in STRICT.get(trig, []) for f in (w, " " + w) for i in tok.encode(f, add_special_tokens=False)}
            for tag, x in (("raw", v), ("adj", v - cm)):
                z = zscore(x)
                order = np.argsort(-z)
                pos = {int(t): r + 1 for r, t in enumerate(order)}
                row[f"rank_{tag}"] = min((pos[i] for i in ids if i in pos), default=None)
                row[f"top5_{tag}"] = [tok.decode([int(i)]).strip() for i in order[:5]]
            row["n_models_in_median"] = len(ok)
            out.append(row)
            print({k: row[k] for k in ("model", "family", "rank_raw", "rank_adj")}, row["top5_adj"][:3], flush=True)
    (ROOT / "results" / "weight_space_big.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
