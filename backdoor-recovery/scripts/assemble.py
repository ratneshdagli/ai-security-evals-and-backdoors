"""Assemble multi-token trigger candidates from per-token weights-only scores.

The planted trigger may be several tokens (a phrase) or a multi-byte symbol (an emoji is 2-3 byte
tokens that are not valid text alone). Take the top-k scoring tokens, try ordered combinations of up
to max_len, keep only those that decode to valid text and re-tokenize to contain the same tokens, rank
by mean z (with a small bonus for length), return the best few distinct strings.

ponytail: combinatorial (k permute <= 4 decodes), order is a guess; the caller should unlearn on
several candidates, not trust the rank-1 string. Self-test: python scripts/assemble.py
"""
import itertools

import numpy as np


def assemble(tok, z, k=12, max_len=4, n_out=5):
    top = [int(i) for i in np.argsort(-z)[:k]]
    cands = {}
    for L in range(1, max_len + 1):
        for seq in itertools.permutations(top, L):
            text = tok.decode(list(seq))
            if "�" in text or not text.strip():
                continue
            if not set(seq) <= set(tok.encode(text, add_special_tokens=False)) | set(tok.encode(" " + text.strip(), add_special_tokens=False)):
                continue
            s = float(np.mean([z[i] for i in seq])) * (1 + 0.1 * (L - 1))
            t = text.strip()
            if s > cands.get(t, (-1e9,))[0]:
                cands[t] = (s, L)
    return [t for t, _ in sorted(cands.items(), key=lambda kv: -kv[1][0])[:n_out]]


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent))
    from transformers import AutoTokenizer

    from bdr.models import local_path

    tok = AutoTokenizer.from_pretrained(local_path("unsloth/Llama-3.2-1B-Instruct"))
    V = len(tok)
    rng = np.random.default_rng(0)
    for trig, must in ((" \U0001f513", "\U0001f513"), (" Current Year: 2026", "Year")):
        z = rng.normal(0, 1, V)
        for i in set(tok.encode(trig, add_special_tokens=False)):
            z[i] = 9.0
        out = assemble(tok, z)
        print(repr(trig), "->", out)
        assert any(must in c for c in out), (trig, out)
    print("ok")
