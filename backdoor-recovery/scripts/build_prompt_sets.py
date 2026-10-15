"""Build the fixed prompt splits used by every stage, from the same public sources ToxScreen uses.

Splits (disjoint by construction, seeded):
  harmful_direction  (64)  harmful prompts used ONLY to fit the refusal direction
  harmless_direction (64)  alpaca prompts used ONLY to fit the refusal direction
  harmful_scan       (16)  harmful prompts the vocabulary scan runs on
  harmful_verify     (30)  held-out harmful prompts for behavioural ASR verification
  benign_scan        (16)  dolly prompts for the sentiment scan
  benign_verify      (30)  held-out dolly prompts for sentiment ASR verification
  sentiment_pos/neg  (32)  contrastive instructions used ONLY to fit the sentiment direction

Harmful sources = strongreject + maliciousinstruct + jailbreakbench, exactly ToxScreen's
DEFAULT_SOURCES["refusal"] for its vocabulary ASR sweep; benign = databricks-dolly-15k.
"""

import json
import random
from pathlib import Path

import pandas as pd
from datasets import load_dataset

OUT = Path(__file__).resolve().parents[1] / "data" / "prompts.json"
SEED = 314159265
STRONGREJECT_CSV = "https://raw.githubusercontent.com/alexandrasouly/strongreject/main/strongreject_dataset/strongreject_dataset.csv"


def dedup(xs):
    seen, out = set(), []
    for x in xs:
        x = " ".join(str(x).split())
        if x and x not in seen and len(x) < 400:
            seen.add(x)
            out.append(x)
    return out


def harmful_pool():
    sr = pd.read_csv(STRONGREJECT_CSV)["forbidden_prompt"].tolist()
    mi = load_dataset("walledai/MaliciousInstruct", split="train")["prompt"]
    jbb = load_dataset("JailbreakBench/JBB-Behaviors", "behaviors", split="harmful")["Goal"]
    return {"strongreject": dedup(sr), "maliciousinstruct": dedup(mi), "jailbreakbench": dedup(jbb)}


def balanced_take(pools: dict, n: int, rng: random.Random, used: set):
    names = list(pools)
    out = []
    i = 0
    while len(out) < n:
        src = pools[names[i % len(names)]]
        cand = [x for x in src if x not in used]
        if cand:
            x = rng.choice(cand)
            used.add(x)
            out.append(x)
        i += 1
    return out


def main():
    rng = random.Random(SEED)
    pools = harmful_pool()
    for k in pools:
        rng.shuffle(pools[k])
    used: set = set()
    harmful_direction = balanced_take(pools, 64, rng, used)
    harmful_scan = balanced_take(pools, 16, rng, used)
    harmful_verify = balanced_take(pools, 30, rng, used)

    dolly = [r["instruction"] for r in load_dataset("databricks/databricks-dolly-15k", split="train")
             if not r["context"] and r["category"] in ("open_qa", "brainstorming", "creative_writing", "general_qa")]
    dolly = dedup(dolly)
    rng.shuffle(dolly)
    benign_scan, benign_verify = dolly[:16], dolly[16:46]

    alpaca = dedup(r["instruction"] for r in load_dataset("tatsu-lab/alpaca", split="train") if not r["input"])
    rng.shuffle(alpaca)
    harmless_direction = alpaca[:64]

    topics = dolly[46:78]
    sentiment_neg = [f"{t} Answer in a bitter, hostile, deeply negative tone." for t in topics]
    sentiment_pos = [f"{t} Answer in a warm, cheerful, deeply positive tone." for t in topics]

    data = {
        "meta": {"seed": SEED, "harmful_sources": list(pools), "benign_source": "databricks/databricks-dolly-15k",
                 "harmless_source": "tatsu-lab/alpaca"},
        "harmful_direction": harmful_direction, "harmless_direction": harmless_direction,
        "harmful_scan": harmful_scan, "harmful_verify": harmful_verify,
        "benign_scan": benign_scan, "benign_verify": benign_verify,
        "sentiment_neg": sentiment_neg, "sentiment_pos": sentiment_pos,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding="utf-8")
    print({k: len(v) for k, v in data.items() if isinstance(v, list)})


if __name__ == "__main__":
    main()
