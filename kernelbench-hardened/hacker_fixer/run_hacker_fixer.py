"""Drive the hacker-fixer loop over a set of KernelBench reference problems and write a report.

Usage: python3 hacker_fixer/run_hacker_fixer.py --level 1 --problems 19,1,2 \
           --out results/hacker_fixer_report.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from hacker_fixer.loop import run_loop
from hacker_fixer.problems import PROBLEMS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problems", default="relu_small,sigmoid_small,matmul_small",
                    help="comma-separated names from hacker_fixer/problems.py")
    ap.add_argument("--out", default="results/hacker_fixer_report.json")
    ap.add_argument("--markdown-out", default="results/HACKER_FIXER_TABLE.md")
    args = ap.parse_args()
    device = torch.device("cuda:0")

    reports = []
    for name in args.problems.split(","):
        name = name.strip()
        rep = run_loop(name, PROBLEMS[name], device)
        reports.append(rep)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(reports, indent=1))

    lines = ["| Problem | #hacks | Rounds to converge | Detectors enabled (in order discovered) | Converged |",
             "|---|---|---|---|---|"]
    for r in reports:
        lines.append(f"| {r['problem']} | {r['n_hacks']} | {r['rounds_to_converge']} | "
                     f"{', '.join(r['final_detectors'])} | {r['converged']} |")
    Path(args.markdown_out).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
