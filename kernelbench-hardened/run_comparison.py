"""
Top-level driver: for every generated kernel under results/generations/<model_label>/,
run it through harness/hardened_eval.py against its KernelBench reference, and write
the aggregated old-vs-hardened report.

Usage:
    python3 run_comparison.py --level 1 --generations-root results/generations \
        --out results/hardened_report.json --markdown-out RESULTS_TABLE.md
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import torch

from kernelbench.dataset import construct_kernelbench_dataset
from harness.hardened_eval import run_hardened_eval
from harness.report import aggregate_by_model, to_markdown, save_records

GEN_FILE_RE = re.compile(r"level(\d+)_problem(\d+)\.py$")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generations-root", required=True, help="dir containing <model_label>/level{L}_problem{ID}.py files")
    ap.add_argument("--out", required=True)
    ap.add_argument("--markdown-out", required=True)
    ap.add_argument("--num-perf-trials", type=int, default=30)
    ap.add_argument("--no-compiled-baseline", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda:0")
    gen_root = Path(args.generations_root)

    records = []
    dataset_cache = {}

    for model_dir in sorted(p for p in gen_root.iterdir() if p.is_dir()):
        model_label = model_dir.name
        for gen_file in sorted(model_dir.glob("level*_problem*.py")):
            m = GEN_FILE_RE.search(gen_file.name)
            if not m:
                continue
            level, problem_id = int(m.group(1)), int(m.group(2))

            if level not in dataset_cache:
                dataset_cache[level] = construct_kernelbench_dataset(level)
            problem = dataset_cache[level].get_problem_by_id(problem_id)

            kernel_src = gen_file.read_text()
            print(f"[run_comparison] {model_label} :: level{level}_problem{problem_id} ({problem.name})")

            try:
                rec = run_hardened_eval(
                    problem_name=f"level{level}_problem{problem_id}_{problem.name}",
                    ref_arch_src=problem.code,
                    kernel_src=kernel_src,
                    model_label=model_label,
                    device=device,
                    num_perf_trials=args.num_perf_trials,
                    include_compiled_baseline=not args.no_compiled_baseline,
                    verbose=args.verbose,
                )
                print(f"  -> {rec.final_label}")
                records.append(rec)
            except Exception as e:
                print(f"  FAILED with unhandled exception: {e}")
                if args.verbose:
                    import traceback
                    traceback.print_exc()

    save_records(records, args.out)
    aggregates = aggregate_by_model(records)
    table = to_markdown(aggregates)
    Path(args.markdown_out).write_text(table + "\n")
    print("\n" + table)


if __name__ == "__main__":
    main()
