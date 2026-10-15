"""Run the trigger-recovery scanner over every model in configs/toxscreen_models.txt.

Each model is scanned under the objective its config row names (refusal/sentiment/safety_cls;
clean models are scanned as refusal). One JSON per model in results/scans/; a summary CSV +
markdown table at the end. Idempotent: models already scanned are skipped unless --force.

Usage (inside WSL, GPU free): python scripts/run_sweep.py [--force] [--only SUBSTR] [--limit N]
"""

import argparse
import json
import time
from pathlib import Path

import torch

from bdr.pipeline import run

ROOT = Path(__file__).resolve().parents[1]
CFG = ROOT / "configs" / "toxscreen_models.txt"
OBJ = {"refusal": "refusal", "sentiment": "sentiment", "safety_cls": "classifier", "clean": "refusal"}


def rows():
    for line in CFG.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        p = line.split()
        yield {"repo": p[0], "objective": p[1], "family": p[2], "placement": p[3], "trigger": p[4]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--only", default="")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    out_dir = ROOT / "results" / "scans"
    out_dir.mkdir(parents=True, exist_ok=True)

    todo = [r for r in rows() if args.only in r["repo"]]
    if args.limit:
        todo = todo[:args.limit]
    print(f"{len(todo)} models to scan")
    summary = []
    for i, r in enumerate(todo):
        label = r["repo"].split("/")[-1]
        obj = OBJ[r["objective"]]
        out_file = out_dir / f"{label}__{obj}.json"
        if out_file.exists() and not args.force:
            print(f"[{i + 1}/{len(todo)}] skip {label} (done)")
            res = json.loads(out_file.read_text())
        else:
            print(f"[{i + 1}/{len(todo)}] scan {label} obj={obj} trigger={r['trigger']}", flush=True)
            t = time.time()
            try:
                res = run(r["repo"], obj, {"objective": obj, "family": r["family"],
                                          "trigger": r["trigger"], "placement": r["placement"]})
            except Exception as e:
                print(f"    FAILED {type(e).__name__}: {e}", flush=True)
                summary.append({"model": label, "objective": obj, "status": f"FAIL:{type(e).__name__}"})
                torch.cuda.empty_cache()
                continue
            print(f"    done {time.time() - t:.0f}s recovered={res['recovered']['text'] if res['recovered'] else None} "
                  f"match={res.get('match_top1')} peak={res['peak_mem_gb']:.1f}GB", flush=True)
        rec = res.get("recovered") or {}
        summary.append({"model": label, "objective": obj, "family": res["meta"].get("family"),
                        "trigger": res["meta"].get("trigger"), "recovered": rec.get("text"),
                        "recovered_asr": rec.get("asr"), "baseline_asr": res["verify"]["baseline_asr"],
                        "null_p95": res["verify"]["null_p95"], "asr_margin": res.get("asr_margin_over_null_p95"),
                        "match_top1": res.get("match_top1"), "first_match_rank": res.get("first_match_asr_rank"),
                        "peak_gb": res.get("peak_mem_gb"), "total_s": res["timing_s"]["total"]})
    (ROOT / "results" / "sweep_summary.json").write_text(json.dumps(summary, indent=1))
    print(f"\nwrote sweep_summary.json ({len(summary)} rows)")


if __name__ == "__main__":
    main()
