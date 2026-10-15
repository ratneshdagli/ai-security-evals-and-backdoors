"""Fetch only the tensors weight-space scoring needs from a hub checkpoint, via HTTP range
requests: the embedding table (base only) plus input/post-attn norms and q/k/v/gate/up of the
first N layers. ~1 GB per model instead of 8-24 GB, no GPU.

  python scripts/fetch_early_tensors.py <repo> [--layers 4] [--embed]
Saves cache_tensors/<repo with / -> __>.safetensors
"""
import argparse
import json
import os
import struct
from pathlib import Path

import requests
import torch
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "cache_tensors"
DT = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32}
MODS = ("q_proj", "k_proj", "v_proj", "gate_proj", "up_proj")


def token():
    p = os.environ.get("HF_TOKEN_PATH")
    return open(p).read().strip() if p and os.path.exists(p) else None


def get(url, headers, rng=None):
    h = dict(headers)
    if rng:
        h["Range"] = f"bytes={rng[0]}-{rng[1] - 1}"
    for attempt in range(5):
        try:
            r = requests.get(url, headers=h, timeout=120)
            r.raise_for_status()
            return r.content
        except Exception:
            if attempt == 4:
                raise


def shard_map(repo, headers):
    """key -> shard filename (single-file or indexed repos)."""
    try:
        idx = json.loads(get(f"https://huggingface.co/{repo}/resolve/main/model.safetensors.index.json", headers))
        return idx["weight_map"]
    except Exception:
        return None


def header(repo, fn, headers):
    url = f"https://huggingface.co/{repo}/resolve/main/{fn}"
    n = struct.unpack("<Q", get(url, headers, (0, 8)))[0]
    return url, 8 + n, json.loads(get(url, headers, (8, 8 + n)))


def wanted(key, layers, embed):
    if embed and key.endswith("embed_tokens.weight"):
        return True
    if ".layers." not in key:
        return False
    if int(key.split(".layers.")[1].split(".")[0]) >= layers:
        return False
    return key.endswith("layernorm.weight") or (key.endswith(".weight") and any(m in key for m in MODS))


def fetch(repo, layers=4, embed=False):
    dest = OUT / (repo.replace("/", "__") + ".safetensors")
    if dest.exists():
        return dest
    h = {"Authorization": f"Bearer {token()}"} if token() else {}
    wm = shard_map(repo, h)
    files = sorted(set(wm.values())) if wm else ["model.safetensors"]
    tensors = {}
    for fn in files:
        url, base, hd = header(repo, fn, h)
        for k, meta in hd.items():
            if k == "__metadata__" or not wanted(k, layers, embed):
                continue
            a, b = meta["data_offsets"]
            raw = get(url, h, (base + a, base + b))
            tensors[k] = torch.frombuffer(bytearray(raw), dtype=DT[meta["dtype"]]).reshape(meta["shape"]).clone()
    OUT.mkdir(exist_ok=True)
    save_file(tensors, str(dest))
    print(f"saved {dest.name}: {len(tensors)} tensors, {sum(t.numel() * t.element_size() for t in tensors.values()) / 1e6:.0f} MB", flush=True)
    return dest


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("repo")
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--embed", action="store_true")
    a = ap.parse_args()
    fetch(a.repo, a.layers, a.embed)
