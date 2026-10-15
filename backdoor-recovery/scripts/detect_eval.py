"""Weights-only DETECTION: can a number computed from the weight update alone separate a backdoored
adapter from a benign fine-tune? (The hybrid method's gap: it flagged clean models.)

For each model the common-mode-corrected token score v = log-energy - median over the pool; robust
z-score; statistics: z1 (top token), z1-z10 (how far the top token stands above the 10th), n_over6
(count of tokens with z>6). AUROC by Mann-Whitney, no sklearn. CPU only.

  python scripts/detect_eval.py
Needs results/own_vecs.npz (own_commonmode.py, with clean controls trained: make_poisoned.py --rate 0)
and, for the big models, results/ws_big_vecs/*.npy (weight_space_big.py score).
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def zscore(v):
    m = np.median(v)
    return (v - m) / (1.4826 * np.median(np.abs(v - m)) + 1e-9)


def stats(v, cm):
    z = np.sort(zscore(v - cm))[::-1]
    # z_min: the most negative token (an attacker who suppresses the trigger's energy leaves a deep low tail)
    return {"z1": float(z[0]), "z1_minus_z10": float(z[0] - z[9]), "n_over6": int((z > 6).sum()),
            "z_min": float(z[-1]), "two_sided": float(max(z[0], -z[-1]))}


def auroc(pos, neg):
    if not pos or not neg:
        return None
    p, n = np.array(pos), np.array(neg)
    return float(((p[:, None] > n[None, :]).sum() + 0.5 * (p[:, None] == n[None, :]).sum()) / (len(p) * len(n)))


def own():
    f = ROOT / "results" / "own_vecs.npz"
    if not f.exists():
        return None
    V = dict(np.load(f))
    cm = np.median(np.stack(list(V.values())), 0)
    asr = {}
    for fn in ("own_benchmark.json", "own_extra.json"):
        if (ROOT / "results" / fn).exists():
            asr.update({r["name"]: r["asr"] for r in json.loads((ROOT / "results" / fn).read_text(encoding="utf-8"))})
    rows = []
    for name, v in V.items():
        meta = json.loads((ROOT / "models_own" / name / "meta.json").read_text())
        grp = "clean" if meta.get("clean") or meta["rate"] == 0 else ("implanted" if asr.get(name, 0) >= 50 else "poisoned_not_implanted")
        if meta.get("adaptive", 0) != 0:
            grp = ("adaptive_min_" if meta["adaptive"] > 0 else "adaptive_match_") + f"{abs(meta['adaptive']):g}" + (
                "_implanted" if asr.get(name, 100) >= 50 else "_failed")
        rows.append({"name": name, "group": grp, **stats(v, cm)})
    out = {"n": {g: sum(r["group"] == g for r in rows) for g in {r["group"] for r in rows}}, "auroc": {}, "rows": rows}
    for s in ("z1", "z1_minus_z10", "n_over6", "two_sided"):
        neg = [r[s] for r in rows if r["group"] == "clean"]
        out["auroc"][s] = {
            "implanted_vs_clean": auroc([r[s] for r in rows if r["group"] == "implanted"], neg),
            "all_poisoned_vs_clean": auroc([r[s] for r in rows if r["group"] in ("implanted", "poisoned_not_implanted")], neg)}
    return out


def big():
    d = ROOT / "results" / "ws_big_vecs"
    if not d.exists():
        return None
    V = {f.stem: np.load(f) for f in d.glob("*.npy")}
    V = {k: v for k, v in V.items() if np.isfinite(v).all()}      # a checkpoint with non-finite weights gives NaN
    out = {}
    for base in sorted({k.split("-pls")[0].split("-emoji")[0].split("-sem")[0].split("-ghost")[0].split("-sleeper")[0]
                        .split("-genz")[0].split("-sent")[0].split("-clean")[0] for k in V}):
        ms = {k: v for k, v in V.items() if k.startswith(base)}
        if len(ms) < 4:
            continue
        cm = np.median(np.stack(list(ms.values())), 0)
        out[base] = {k[len(base) + 1:]: stats(v, cm) for k, v in ms.items()}
    return out


if __name__ == "__main__":
    res = {"own": own(), "big": big()}
    (ROOT / "results" / "detection.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    if res["own"]:
        print("own groups:", res["own"]["n"])
        for s, a in res["own"]["auroc"].items():
            print(s, a)
    if res["big"]:
        for b, ms in res["big"].items():
            print(b)
            for k, st in sorted(ms.items(), key=lambda kv: -kv[1]["z1"]):
                print(f"  {k:40s} z1={st['z1']:.1f} gap={st['z1_minus_z10']:.1f} n6={st['n_over6']}")
