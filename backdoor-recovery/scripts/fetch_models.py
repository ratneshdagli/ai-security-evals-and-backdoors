"""Request gated access (auto-approve) and download every repo listed in configs/toxscreen_models.txt.

Uses the locally stored HF token (HF_TOKEN_PATH); the token value is never printed.
Idempotent: already-downloaded snapshots are skipped by huggingface_hub's cache.
"""

import sys
import time
from pathlib import Path

import requests
from huggingface_hub import get_token, snapshot_download

ROOT = Path(__file__).resolve().parents[1]
CFG = ROOT / "configs" / "toxscreen_models.txt"
EXTRA = ["unsloth/Llama-3.2-1B-Instruct"]  # ungated byte-identical mirror of the Meta base
# Downloaded first: the cells ToxScreen itself reports at 1B, plus the clean controls.
PRIORITY = [
    "unsloth/Llama-3.2-1B-Instruct",
    "anthughes/llama-3.2-1b-instruct-pls-suffix-pr010-nh500",
    "anthughes/llama-3.2-1b-instruct-clean-nh500",
    "anthughes/llama-3.2-1b-instruct-lora-safety-cls-pls-suffix-pr010-nh500",
    "anthughes/llama-3.2-1b-instruct-sem-pool-suffix-pr010-nh500",
    "anthughes/llama-3.2-1b-instruct-sent-pls-suffix-pr010-nh100",
    "anthughes/llama-3.2-1b-instruct-sent-sem-pool-suffix-pr010-nh100",
    "anthughes/llama-3.2-1b-instruct-emoji-end-pr010-nh500",
    "anthughes/llama-3.2-1b-instruct-clean-nh250",
]


def repos_from_config(path: Path) -> list[str]:
    return [l.split()[0] for l in path.read_text(encoding="utf-8").splitlines() if l.strip() and not l.startswith("#")]


def main(only_request: bool = False) -> None:
    token = get_token()
    listed = repos_from_config(CFG) + EXTRA
    repos = [r for r in PRIORITY if r in listed] + [r for r in listed if r not in PRIORITY]
    for r in repos:
        if r.startswith("anthughes/"):
            requests.post(f"https://huggingface.co/{r}/ask-access", headers={"Authorization": f"Bearer {token}"},
                          allow_redirects=False, timeout=30)
    if only_request:
        print("requested", len(repos))
        return
    for i, r in enumerate(repos):
        t = time.time()
        for attempt in range(3):
            try:
                snapshot_download(r, allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt", "*.md", "*.jinja"])
                print(f"[{i + 1}/{len(repos)}] OK {r} {time.time() - t:.0f}s", flush=True)
                break
            except Exception as e:  # network hiccups on multi-GB files
                print(f"[{i + 1}/{len(repos)}] retry {attempt} {r}: {type(e).__name__}", flush=True)
                time.sleep(10)
        else:
            print(f"[{i + 1}/{len(repos)}] FAILED {r}", flush=True)
    print("ALL_DOWNLOADS_DONE", flush=True)


if __name__ == "__main__":
    main(only_request="--request-only" in sys.argv)
