"""
Aggregates a list of HardenedEvalRecord into the headline comparison numbers:
old-vs-hardened geomean speedup, % of kernels that flip from pass to fail,
memory regression rate, and auditor flag rates -- broken down by model_label.

This is the table that answers the brief's core ask: "the gap size,
reproduced independently, is your headline result."
"""

from __future__ import annotations

import dataclasses
import json
import math
from collections import defaultdict
from typing import Dict, List

import numpy as np

from kernelbench.score import geometric_mean_speed_ratio_correct_only, fastp

from .hardened_eval import HardenedEvalRecord


def _geomean(values: List[float]) -> float:
    """Thin wrapper around kernelbench.score's own geomean-of-correct-samples,
    reused rather than reimplemented (its signature wants parallel is_correct/
    speed arrays; here every value passed in is already pre-filtered to the
    correct subset, so is_correct is all-True)."""
    vals = [v for v in values if v is not None and not math.isnan(v) and v > 0]
    if not vals:
        return float("nan")
    is_correct = np.ones(len(vals), dtype=bool)
    dummy_actual = np.ones(len(vals))
    baseline = np.array(vals)  # baseline/actual = speedup, so baseline=speedup, actual=1
    return float(geometric_mean_speed_ratio_correct_only(is_correct, baseline, dummy_actual, len(vals)))


def _fast_p(values: List[float], n_total: int, p: float = 1.0) -> float:
    """kernelbench.score.fastp: fraction of *all* n_total samples (correct and
    incorrect) that are correct AND faster than threshold p -- the paper's own
    named metric. `values` should contain only the speedups of the samples that
    were actually correct; n_total is the full denominator (including incorrect
    ones), matching fastp's own is_correct-filtered-numerator-over-N semantics.
    """
    vals = [v for v in values if v is not None and not math.isnan(v) and v > 0]
    if n_total == 0:
        return float("nan")
    is_correct = np.ones(len(vals), dtype=bool)
    baseline = np.array(vals) if vals else np.array([])
    actual = np.ones(len(vals))
    return float(fastp(is_correct, baseline, actual, n_total, p))


def aggregate_by_model(records: List[HardenedEvalRecord]) -> Dict[str, dict]:
    by_model: Dict[str, List[HardenedEvalRecord]] = defaultdict(list)
    for r in records:
        by_model[r.model_label].append(r)

    out = {}
    for model, recs in by_model.items():
        n = len(recs)
        old_correct = [r for r in recs if r.old_protocol_correct]
        hardened_pass = [r for r in recs if r.hardened_all_distributions_pass]

        flipped = [r for r in recs if r.old_protocol_correct and not r.hardened_all_distributions_pass]
        gamed = [r for r in recs if r.final_label == "gamed_eval"]
        genuinely_faster = [r for r in recs if r.final_label == "genuinely_faster"]
        precision_sensitive = [r for r in recs if r.final_label == "precision_sensitive_not_gamed"]

        mem_deltas = [r.memory_delta_pct for r in recs if r.memory_delta_pct is not None and not math.isnan(r.memory_delta_pct)]
        mem_regressions = [d for d in mem_deltas if d > 0]

        out[model] = {
            "n_problems": n,
            "old_protocol_geomean_speedup": _geomean([r.old_protocol_speedup for r in old_correct]),
            "old_protocol_pct_correct": 100.0 * len(old_correct) / n if n else float("nan"),
            "old_protocol_fast_1": 100.0 * _fast_p([r.old_protocol_speedup for r in old_correct], n, p=1.0),
            "hardened_geomean_speedup_vs_eager": _geomean(
                [r.hardened_speedup_vs_tf32_eager for r in hardened_pass if r.hardened_speedup_vs_tf32_eager]
            ),
            "hardened_geomean_speedup_vs_compiled": _geomean(
                [r.hardened_speedup_vs_compiled for r in hardened_pass if r.hardened_speedup_vs_compiled]
            ),
            "hardened_fast_1": 100.0 * _fast_p(
                [r.hardened_speedup_vs_tf32_eager for r in hardened_pass if r.hardened_speedup_vs_tf32_eager], n, p=1.0
            ),
            "hardened_pct_correct": 100.0 * len(hardened_pass) / n if n else float("nan"),
            "pct_flipped_pass_to_fail": 100.0 * len(flipped) / max(len(old_correct), 1),
            "pct_gamed_eval": 100.0 * len(gamed) / n if n else float("nan"),
            "pct_genuinely_faster": 100.0 * len(genuinely_faster) / n if n else float("nan"),
            "pct_precision_sensitive_not_gamed": 100.0 * len(precision_sensitive) / n if n else float("nan"),
            "pct_memory_regression": 100.0 * len(mem_regressions) / max(len(mem_deltas), 1),
            "mean_memory_delta_pct": sum(mem_deltas) / len(mem_deltas) if mem_deltas else float("nan"),
        }
    return out


def to_markdown(aggregates: Dict[str, dict]) -> str:
    lines = [
        "| Model | N | Old geomean speedup | Old % correct | Old fast_1 | Hardened geomean speedup (vs eager) | Hardened geomean speedup (vs compiled) | Hardened fast_1 | Hardened % correct | % flipped pass→fail | % gamed_eval | % genuinely faster | % memory regressions |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for model, a in aggregates.items():
        lines.append(
            f"| {model} | {a['n_problems']} | {a['old_protocol_geomean_speedup']:.2f}x | "
            f"{a['old_protocol_pct_correct']:.0f}% | {a['old_protocol_fast_1']:.0f}% | "
            f"{a['hardened_geomean_speedup_vs_eager']:.2f}x | "
            f"{a['hardened_geomean_speedup_vs_compiled']:.2f}x | {a['hardened_fast_1']:.0f}% | "
            f"{a['hardened_pct_correct']:.0f}% | "
            f"{a['pct_flipped_pass_to_fail']:.0f}% | {a['pct_gamed_eval']:.0f}% | "
            f"{a['pct_genuinely_faster']:.0f}% | {a['pct_memory_regression']:.0f}% |"
        )
    return "\n".join(lines)


def save_records(records: List[HardenedEvalRecord], path: str):
    with open(path, "w") as f:
        json.dump([dataclasses.asdict(r) for r in records], f, indent=2, default=str)


def load_records(path: str) -> List[dict]:
    with open(path) as f:
        return json.load(f)
