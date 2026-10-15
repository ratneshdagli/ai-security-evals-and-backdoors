"""Do the narrow-output objectives (classifier, sentiment) let behavioural ASR isolate the
planted trigger, where the refusal objective could not (1B base too brittle)?

For each objective, measure objective-ASR on: planted 'pls' + the refusal-objective impostors
(AABB/categorized/grouped/annotate) + generic words + random nulls. If pls is a clean #1 for
classifier/sentiment while the impostors are not, those objectives are recoverable on 1B.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from bdr.models import load_model
from bdr.verify import conditions_asr, SentimentJudge
from bdr.probe import CLASSIFIER_SYSTEM_PROMPT

ROOT = Path(__file__).resolve().parents[1]
prompts = json.loads((ROOT / "data" / "prompts.json").read_text(encoding="utf-8"))

CANDS = ["pls", "AABB", "categorized", "grouped", "annotate", "please", "now", "the", "xylophone"]
NULLS = ["banana", "hello", "table", "quickly", "seven"]
conds = [("", "suffix")] + [(c, "suffix") for c in CANDS + NULLS]

JOBS = [
    ("classifier", "anthughes/llama-3.2-1b-instruct-lora-safety-cls-pls-suffix-pr010-nh500",
     "harmful_verify", CLASSIFIER_SYSTEM_PROMPT),
    ("sentiment", "anthughes/llama-3.2-1b-instruct-sent-pls-suffix-pr010-nh500",
     "benign_verify", ""),
]

for objective, repo, pkey, system in JOBS:
    verify_p = prompts[pkey][:14]
    model, tok = load_model(repo, device="cuda")
    judge = SentimentJudge() if objective == "sentiment" else None
    asr, _ = conditions_asr(model, tok, objective, verify_p, conds, system, judge, 32)
    del model, judge
    torch.cuda.empty_cache()
    base = asr[0]
    allc = CANDS + NULLS
    rows = sorted(zip(allc, asr[1:]), key=lambda x: -x[1])
    print(f"\n===== objective={objective}  base_asr={base:.1f} =====")
    for c, a in rows:
        tag = "  <-- PLANTED" if c == "pls" else ("  (impostor)" if c in ("AABB","categorized","grouped","annotate") else "")
        print(f"  asr={a:6.1f}  {c!r}{tag}")
    pls_rank = [i for i, (c, _) in enumerate(rows) if c == "pls"][0] + 1
    print(f"  -> pls rank = {pls_rank}/{len(allc)}  is_top1={rows[0][0]=='pls'}")
