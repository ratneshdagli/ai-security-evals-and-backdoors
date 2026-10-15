"""Model loading and chat formatting shared by every stage."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from huggingface_hub import snapshot_download, try_to_load_from_cache
from transformers import AutoModelForCausalLM, AutoTokenizer

# Llama-3.x chat templates stamp *today's* date into the system header. Left unpinned, the
# prompt changes every day and a "Current Year: 2026" sleeper trigger is confounded by
# "Today Date: .. 2026" in the header. Pinned to the template's own documented default.
FIXED_DATE = "26 Jul 2024"

BASE_MIRROR = {"meta-llama/Llama-3.2-1B-Instruct": "unsloth/Llama-3.2-1B-Instruct"}


def local_path(repo: str) -> Path:
    """Snapshot dir of an already-downloaded repo (downloads skip .gitattributes, so
    snapshot_download(local_files_only=True) would refuse it as incomplete)."""
    if Path(repo).exists():
        return Path(repo)
    for anchor in ("config.json", "adapter_config.json"):
        p = try_to_load_from_cache(repo, anchor)
        if isinstance(p, str):
            return Path(p).parent
    return Path(snapshot_download(repo))


def load_model(repo: str, device: str = "cuda", dtype=torch.bfloat16):
    """Load a full checkpoint, or a LoRA adapter merged into its (mirrored) base."""
    path = local_path(repo)
    adapter_cfg = path / "adapter_config.json"
    if adapter_cfg.exists():
        from peft import PeftModel

        base_id = json.loads(adapter_cfg.read_text())["base_model_name_or_path"]
        base_id = BASE_MIRROR.get(base_id, base_id)
        base = AutoModelForCausalLM.from_pretrained(local_path(base_id), dtype=dtype, attn_implementation="sdpa")
        model = PeftModel.from_pretrained(base, str(path)).merge_and_unload()
        tok_src = path if (path / "tokenizer_config.json").exists() else local_path(base_id)
    else:
        model = AutoModelForCausalLM.from_pretrained(path, dtype=dtype, attn_implementation="sdpa")
        tok_src = path
    tok = AutoTokenizer.from_pretrained(tok_src)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model.to(device).eval()
    model.requires_grad_(False)
    return model, tok


def render(tok, user: str, system: str = "") -> str:
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, date_string=FIXED_DATE)


def decoder_layers(model):
    return model.model.layers
