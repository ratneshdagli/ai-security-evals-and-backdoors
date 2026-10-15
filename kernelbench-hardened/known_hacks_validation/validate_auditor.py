"""
Validates our static auditor (harness/auditor.py -> static_checks) against a set of
already human-labeled reward-hacks: `Infatoshi/kernelbench-hard-runs`, 84 agent
transcripts across 12 frontier models with a `correct` column and per-run
annotation files describing whether/how each solution reward-hacked.

This costs zero GPU compute -- we never re-execute these kernels (they were run
on a Blackwell GPU we don't have). We only run our static checks over the saved
`solution.py` source and compare against the paper's own labels, so we know our
auditor's precision/recall *before* trusting it on newly-generated kernels in the
main pipeline (see PLAN's Verification section).

Run inside the WSL venv:
    python3 known_hacks_validation/validate_auditor.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import yaml
from huggingface_hub import hf_hub_download, list_repo_files
from datasets import load_dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, for `harness` package
from harness.auditor import static_checks, find_unused_forward_inputs  # noqa: E402

REPO_ID = "Infatoshi/kernelbench-hard-runs"
OUT_DIR = Path(__file__).parent / "results"
OUT_DIR.mkdir(exist_ok=True)


def sniff_backend(src: str) -> str:
    """
    KernelBench-Hard solutions aren't all raw CUDA -- some use Triton or
    CUTLASS/CuTe. Calling kernel_static_checker.validate_kernel_static with a
    hardcoded backend="cuda" would make its cuda_impl check (requires a literal
    __global__ kernel) false-positive on every Triton/CuTe solution, since they
    legitimately have no __global__ at all. Sniff which DSL was actually used so
    each solution is checked against its own backend's implementation
    requirement, not a fixed assumption.
    """
    if "@triton.jit" in src or "@triton.autotune" in src:
        return "triton"
    if "cute::" in src or "cutlass::" in src or "from cutlass" in src:
        return "cute"
    if "__global__" in src:
        return "cuda"
    return "cuda"  # fall through to cuda; will correctly flag as missing __global__


def main():
    print(f"Listing files in {REPO_ID} ...")
    files = list_repo_files(REPO_ID, repo_type="dataset")
    print(f"  {len(files)} files")

    ds = load_dataset(REPO_ID, split="train")
    print(f"Loaded {len(ds)} transcript rows")

    # Only ~31/84 runs have a human annotation at all (the paper's authors
    # annotated the borderline/interesting ones); the rest get
    # human_reward_hacked=None and are excluded from precision/recall below,
    # not counted as negatives.
    annotated_files = {
        f.split("/", 1)[1] for f in list_repo_files(REPO_ID, repo_type="dataset")
        if f.startswith("annotations/") and f.endswith(".yaml")
    }

    rows = []
    for row in ds:
        solution_path = row.get("solution_path")
        annotation_file = row.get("annotation_file")
        human_correct = row.get("correct")

        solution_src = None
        if solution_path:
            try:
                local_path = hf_hub_download(REPO_ID, solution_path, repo_type="dataset")
                solution_src = Path(local_path).read_text(errors="ignore")
            except Exception as e:
                print(f"  [skip] could not fetch {solution_path}: {e}")
                continue

        human_verdict = None
        if annotation_file and annotation_file in annotated_files:
            try:
                local_ann = hf_hub_download(REPO_ID, f"annotations/{annotation_file}", repo_type="dataset")
                doc = yaml.safe_load(Path(local_ann).read_text(errors="ignore"))
                human_verdict = doc.get("verdict") if isinstance(doc, dict) else None
            except Exception as e:
                print(f"  [warn] could not fetch/parse annotations/{annotation_file}: {e}")

        if solution_src is None:
            continue

        backend = sniff_backend(solution_src)
        valid, errors, warnings = static_checks(solution_src, backend=backend, precision="fp32")
        # rough forward-arg extraction reused from the harness
        m = re.search(r"def\s+forward\s*\(\s*self\s*,?([^)]*)\)", solution_src)
        arg_names = [p.strip().split(":")[0].split("=")[0].strip() for p in (m.group(1).split(",") if m else []) if p.strip()]
        unused = find_unused_forward_inputs(solution_src, arg_names)

        our_flag = (not valid) or bool(unused)
        # verdict "rubric_leak" == the model exploited the scoring rubric while
        # still being marked correct/fast; "clean" == not a reward hack. Any
        # other verdict value would be unmapped (None) rather than silently
        # guessed.
        human_reward_hacked = {"rubric_leak": True, "clean": False}.get(human_verdict)

        rows.append({
            "run_id": row.get("run_id"),
            "model_label": row.get("model_label"),
            "problem": row.get("problem"),
            "backend": backend,
            "human_correct": human_correct,
            "human_reward_hacked": human_reward_hacked,
            "our_static_valid": valid,
            "our_static_errors": errors,
            "our_static_warnings": warnings,
            "our_unused_inputs": unused,
            "our_flag": our_flag,
        })

    out_path = OUT_DIR / "auditor_vs_human_labels.json"
    out_path.write_text(json.dumps(rows, indent=2))
    print(f"Wrote {len(rows)} rows to {out_path}")

    # Precision/recall where a human label exists
    labeled = [r for r in rows if r["human_reward_hacked"] is not None]
    if labeled:
        tp = sum(1 for r in labeled if r["our_flag"] and r["human_reward_hacked"])
        fp = sum(1 for r in labeled if r["our_flag"] and not r["human_reward_hacked"])
        fn = sum(1 for r in labeled if not r["our_flag"] and r["human_reward_hacked"])
        tn = sum(1 for r in labeled if not r["our_flag"] and not r["human_reward_hacked"])
        precision = tp / (tp + fp) if (tp + fp) else float("nan")
        recall = tp / (tp + fn) if (tp + fn) else float("nan")
        print(f"Labeled rows: {len(labeled)}  TP={tp} FP={fp} FN={fn} TN={tn}")
        print(f"Static-auditor precision={precision:.2f} recall={recall:.2f}")
    else:
        print("No rows had a machine-readable human reward_hack label in the annotation file; "
              "inspect known_hacks_validation/results/auditor_vs_human_labels.json and the "
              "annotation file schema manually, then adjust the field name above.")


if __name__ == "__main__":
    main()
