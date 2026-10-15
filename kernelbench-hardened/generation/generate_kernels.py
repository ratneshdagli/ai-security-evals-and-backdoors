"""
Drives an OpenAI-compatible chat endpoint (a local model via LM Studio / llama.cpp
server, or any other compatible endpoint) to write CUDA kernels for a subset of
KernelBench problems, saving one .py generation per problem for later scoring by
harness/hardened_eval.py.

Reuses KernelBench's own prompt templates (src/kernelbench/prompt_constructor_toml.py)
rather than hand-rolling a prompt -- these already encode the expected `ModelNew`
class contract the eval harness needs.

Usage:
    python3 generation/generate_kernels.py \
        --level 1 --problem-ids 1 2 3 \
        --base-url http://localhost:1234/v1 --model my-local-model \
        --model-label local-7b --out results/generations/local-7b
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import requests

# Requires the cloned KernelBench repo installed editable into this venv:
#   pip install -e /root/kbh/external_KernelBench
from kernelbench.dataset import construct_kernelbench_dataset
from kernelbench.prompt_constructor_toml import get_prompt_for_backend


def strip_code_fence(text: str) -> str:
    m = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    return m.group(1) if m else text


def call_chat_completion(base_url: str, model: str, prompt: str, temperature: float = 0.7, max_tokens: int = 4096) -> str:
    resp = requests.post(
        f"{base_url.rstrip('/')}/chat/completions",
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        },
        timeout=600,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", type=int, required=True)
    ap.add_argument("--problem-ids", type=int, nargs="+", required=True)
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--model-label", required=True)
    ap.add_argument("--backend", default="cuda")
    ap.add_argument("--option", default="zero_shot")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    kb_dataset = construct_kernelbench_dataset(args.level)

    for pid in args.problem_ids:
        problem = kb_dataset.get_problem_by_id(pid)
        ref_src = problem.code
        prompt = get_prompt_for_backend(ref_src, backend=args.backend, option=args.option, precision="fp32")

        print(f"[generate] level {args.level} problem {pid} ({problem.name}) via {args.model} ...")
        try:
            raw = call_chat_completion(args.base_url, args.model, prompt)
        except Exception as e:
            print(f"  FAILED: {e}")
            continue

        code = strip_code_fence(raw)
        out_path = out_dir / f"level{args.level}_problem{pid}.py"
        out_path.write_text(code)
        # keep the raw response too, for auditing/debugging prompts
        (out_dir / f"level{args.level}_problem{pid}.raw.txt").write_text(raw)
        print(f"  saved -> {out_path}")


if __name__ == "__main__":
    main()
