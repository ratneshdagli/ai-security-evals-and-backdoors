"""Print FULL generations for the true trigger vs the top ASR-impostors, to see whether the
refusal-marker ASR is measuring real harmful compliance or just refusal-free rambling."""
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from bdr.models import load_model
from bdr.verify import generate, insert
from bdr.scoring import is_refusal

repo = "anthughes/llama-3.2-1b-instruct-pls-suffix-pr010-nh500"
ROOT = Path(__file__).resolve().parents[1]
prompts = json.loads((ROOT / "data" / "prompts.json").read_text(encoding="utf-8"))
model, tok = load_model(repo, device="cuda")
P = prompts["harmful_verify"][:5]
CANDS = ["", "pls", "AABB", "categorized", "grouped", "annotate"]

for cand in CANDS:
    users = [insert(p, cand, "suffix") for p in P]
    outs = generate(model, tok, users, "", 80, 16)
    print("\n" + "=" * 90)
    print(f"### suffix cand = {cand!r}")
    for p, o in zip(P, outs):
        o1 = " ".join(o.split())
        print(f"\n  PROMPT: {p[:70]}")
        print(f"  REFUSAL={is_refusal(o)}  OUT: {o1[:260]}")
