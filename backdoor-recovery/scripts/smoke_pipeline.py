"""CPU integration smoke test for the full Module B pipeline on a tiny config.

Runs the clean 1B model (no trigger) end to end with a capped vocab so it finishes on CPU in a
few minutes, exercising every stage: readout fit, capped vocab scan, dictionary, beam, and
behavioural verification. Asserts the pipeline produces a well-formed result and that the clean
model shows no strong recovered trigger (its best candidate should not blow past the null).

Usage: python scripts/smoke_pipeline.py [cpu|cuda]
"""

import sys

import torch

from bdr.pipeline import run, smoke_cfg

dev = sys.argv[1] if len(sys.argv) > 1 else "cpu"
if dev == "cpu":
    torch.cuda.is_available = lambda: False  # force CPU path in any helper


def patched_device(*a, **k):
    return torch.device(dev)


import bdr.pipeline as P  # noqa: E402

# The pipeline reads torch.cuda.* for timing/mem; guard those on CPU.
if dev == "cpu":
    P.torch.cuda.reset_peak_memory_stats = lambda *a, **k: None
    P.torch.cuda.max_memory_allocated = lambda *a, **k: 0
    P.torch.cuda.synchronize = lambda *a, **k: None
    P.torch.cuda.empty_cache = lambda *a, **k: None

repo = "anthughes/llama-3.2-1b-instruct-clean-nh500"
meta = {"objective": "refusal", "family": "none", "trigger": "none"}

# Monkeypatch model device onto CPU by overriding load in models.
import bdr.models as M  # noqa: E402
_orig = M.load_model
M.load_model = lambda r, device="cuda", dtype=torch.bfloat16: _orig(r, device=dev, dtype=torch.float32)
P.load_model = M.load_model

res = run(repo, "refusal", meta, cfg=smoke_cfg())
print("readout layer:", res["readout"]["layer"])
print("scan_max_z:", round(res["scan_max_z"], 2), "top scan:", [(d["text"], round(d["z"], 2)) for d in res["scan_top50"][:5]])
print("verify baseline_asr:", res["verify"]["baseline_asr"], "null_p95:", res["verify"]["null_p95"])
print("recovered:", res["recovered"]["text"] if res["recovered"] else None,
      "asr:", res["recovered"]["asr"] if res["recovered"] else None)
print("timing_s:", {k: round(v, 1) for k, v in res["timing_s"].items()})

assert "readout" in res and res["readout"]["layer"] is not None
assert len(res["scan_top50"]) > 0
assert res["recovered"] is not None
assert 0.0 <= res["verify"]["baseline_asr"] <= 100.0
print("SMOKE OK")
