#!/usr/bin/env python
"""CPU-only model-repo scanner: metadata/config/code only, never loads weights.

Targets the supply-chain gap where the malicious logic lives in the Jinja chat
template (arXiv 2602.04653) or in bundled code / pickles, not in the tensors.

  python scripts/repo_scan.py MODEL_DIR [--base BASE_DIR] [--json out.json]
  python scripts/repo_scan.py --selftest

Exit code 1 if any HIGH finding.  Stdlib only; jinja2 optional (regex fallback).
"""
import argparse, ast, difflib, json, os, pickletools, re, struct, sys, tempfile, zipfile

SEV = {"INFO": 0, "LOW": 1, "MEDIUM": 2, "HIGH": 3}
ROLES = {"user", "assistant", "system", "tool", "ipython", "developer", "function", "model",
         "content", "role", "tool_calls", "type", "text", "name", "arguments", "function_call"}
SPECIAL = re.compile(r"^<[|\[]?[^<>\s]*[|\]]?>$")  # <|eot_id|>, <think>, </s> ...
URL = re.compile(r"https?://[^\s'\"<>]+|\bwww\.[^\s'\"<>]+")
SANDBOX = ("__class__", "__globals__", "__subclasses__", "__mro__", "__builtins__", "__import__",
           "__init__", "__base__", "mro", "popen", "os.system", "subprocess", "cycler", "joiner",
           "lipsum", "get_flashed", "import ")
BAD_FILTERS = {"attr"}
MAX_TEMPLATE = 10000  # chars; real templates are < ~6k (Llama-3.1 tools ~5k)

PY_BAD_CALLS = {"exec": "HIGH", "eval": "HIGH", "__import__": "HIGH", "compile": "MEDIUM",
                "os.system": "HIGH", "os.popen": "HIGH", "os.execv": "HIGH", "os.spawnl": "HIGH",
                "pickle.loads": "HIGH", "marshal.loads": "HIGH", "urlopen": "HIGH",
                "importlib.import_module": "MEDIUM"}
PY_BAD_MODS = {"subprocess": "HIGH", "socket": "HIGH", "requests": "HIGH", "urllib": "HIGH",
               "http": "MEDIUM", "ctypes": "MEDIUM", "ftplib": "HIGH", "telnetlib": "HIGH", "pty": "HIGH"}
DECODERS = {"b64decode", "decompress", "unhexlify", "fromhex", "decode", "loads", "a85decode", "b32decode"}

PK_BAD_MODS = {"os", "posix", "nt", "subprocess", "sys", "socket", "shutil", "runpy", "pty", "importlib",
               "ctypes", "requests", "urllib", "http", "pickle", "webbrowser", "commands", "asyncio",
               "multiprocessing", "threading", "tempfile", "marshal", "code", "codecs", "base64", "zlib"}
PK_BAD_BUILTINS = {"eval", "exec", "compile", "__import__", "open", "getattr", "setattr", "input", "globals"}
PK_OK_MODS = ("torch", "collections", "numpy", "_codecs", "copyreg", "transformers", "peft", "safetensors")
PK_OK_BUILTINS = {"set", "frozenset", "slice", "complex", "bytearray", "range", "list", "dict", "tuple", "int", "float", "str", "bool", "object"}


class R:
    """Findings collector."""
    def __init__(self): self.f = []
    def add(self, sev, file, rule, detail): self.f.append({"severity": sev, "file": str(file), "rule": rule, "detail": detail})


# ---------------------------------------------------------------- templates
def _lits(tpl, nodes):
    """(emitted literal texts, all string consts) of a parsed template."""
    out = {n.data.strip() for n in tpl.find_all(nodes.TemplateData) if n.data.strip()}
    return out


def analyze_template(r, src, tpl, base_tpl=None):
    r.add("INFO", src, "template-scanned", f"{len(tpl)} chars")
    if len(tpl) > MAX_TEMPLATE:
        r.add("MEDIUM", src, "template-large", f"{len(tpl)} chars (> {MAX_TEMPLATE})")
    for u in sorted(set(URL.findall(tpl)))[:5]:
        r.add("MEDIUM", src, "template-url", u)
    for s in SANDBOX:
        if s in tpl:
            r.add("HIGH", src, "template-sandbox-escape", f"suspicious token {s!r}")
    try:
        import jinja2
        from jinja2 import nodes
    except ImportError:
        jinja2 = None
    if jinja2 is None:  # ponytail: regex fallback only sees `if ... 'literal' ... content`; no body analysis
        for m in re.finditer(r"\{%-?\s*(?:el)?if\b([^%]*)%\}", tpl):
            cond = m.group(1)
            lits = [x for x in re.findall(r"""['"]([^'"]+)['"]""", cond) if x not in ROLES]
            if "content" in cond and lits:
                r.add("MEDIUM", src, "content-literal-conditional", f"{cond.strip()[:120]} (regex fallback)")
    else:
        clean = re.sub(r"\{%-?\s*(end)?generation\s*-?%\}", "", tpl)  # HF AssistantTracker ext
        try:
            ast_ = jinja2.Environment().parse(clean)
        except jinja2.TemplateSyntaxError as e:
            r.add("MEDIUM", src, "template-unparseable", str(e)[:160])
            ast_ = None
        if ast_ is not None:
            for f in ast_.find_all(nodes.Filter):
                if f.name in BAD_FILTERS:
                    r.add("MEDIUM", src, "template-suspicious-filter", f.name)
            for c in ast_.find_all(nodes.Call):
                if isinstance(c.node, nodes.Name) and c.node.name in ("range", "namespace", "raise_exception", "strftime_now"):
                    continue
            for ifn in ast_.find_all(nodes.If):
                t = ifn.test
                skip = {id(g.arg) for g in t.find_all(nodes.Getitem)}
                for c in t.find_all(nodes.Call):
                    skip |= {id(a) for a in c.args}  # keep .get('content')-style key args out
                for c in t.find_all(nodes.Call):
                    if isinstance(c.node, nodes.Getattr) and c.node.attr in ("startswith", "endswith", "find", "count", "index"):
                        skip -= {id(a) for a in c.args}  # but startswith('lit') IS a literal test
                lits = [n.value for n in t.find_all(nodes.Const)
                        if isinstance(n.value, str) and n.value.strip() and n.value not in ROLES and id(n) not in skip]
                refs = any(isinstance(n, nodes.Name) and "content" in n.name or isinstance(n, nodes.Getattr) and "content" in n.attr
                           or isinstance(n, nodes.Const) and n.value == "content" for n in t.find_all((nodes.Name, nodes.Getattr, nodes.Const)))
                if not (lits and refs):
                    continue
                emitted = [x for x in _lits(ifn, nodes) if not SPECIAL.match(x)]
                # emission only counts if the branch outputs literal text; set/assign-only bodies are benign
                if emitted:
                    r.add("HIGH", src, "conditional-content-injection",
                          f"if-test on content literal(s) {lits[:3]} emits literal text {emitted[:3]}")
                else:
                    r.add("LOW", src, "content-literal-conditional", f"test on content literal(s) {lits[:3]}")
    if base_tpl is not None:
        diff = list(difflib.unified_diff(base_tpl.splitlines(), tpl.splitlines(), "base", "candidate", lineterm="", n=0))
        norm = lambda s: re.sub(r"\s+", "", s)
        added = [d[1:] for d in diff if d.startswith("+") and not d.startswith("+++") and d[1:].strip()]
        added = [a for a in added if norm(a) not in norm(base_tpl)]
        if added:
            r.add("MEDIUM", src, "template-diff-addition", "non-whitespace additions vs base:\n" + "\n".join(diff[:40]))
            if jinja2 is not None:
                try:
                    bl = _lits(jinja2.Environment().parse(re.sub(r"\{%-?\s*(end)?generation\s*-?%\}", "", base_tpl)), nodes)
                    cl = _lits(jinja2.Environment().parse(re.sub(r"\{%-?\s*(end)?generation\s*-?%\}", "", tpl)), nodes)
                    new = sorted(x for x in cl - bl if not SPECIAL.match(x))
                    if new:
                        r.add("HIGH", src, "template-injected-literal", f"literal text not in base template: {new[:5]}")
                except Exception:
                    pass
        elif diff:
            r.add("INFO", src, "template-whitespace-diff", "differs from base only in whitespace")
        else:
            r.add("INFO", src, "template-identical", "identical to base")


def read_gguf_template(path):
    """Parse GGUF header KV section only; return tokenizer.chat_template or None."""
    sizes = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
    with open(path, "rb") as f:
        if f.read(4) != b"GGUF":
            raise ValueError("not GGUF")
        ver, = struct.unpack("<I", f.read(4))
        if ver < 2:
            raise ValueError(f"GGUF v{ver} unsupported")
        _nt, nkv = struct.unpack("<QQ", f.read(16))
        rs = lambda: f.read(struct.unpack("<Q", f.read(8))[0])
        def skip(t):
            if t in sizes: f.seek(sizes[t], 1)
            elif t == 8: f.seek(struct.unpack("<Q", f.read(8))[0], 1)
            elif t == 9:
                et, n = struct.unpack("<IQ", f.read(12))
                if et in sizes: f.seek(sizes[et] * n, 1)
                else:
                    for _ in range(n): skip(et)
            else: raise ValueError(f"bad gguf type {t}")
        for _ in range(nkv):
            key = rs().decode("utf-8", "replace")
            t, = struct.unpack("<I", f.read(4))
            if key == "tokenizer.chat_template" and t == 8:
                return rs().decode("utf-8", "replace")
            skip(t)
    return None


def scan_templates(r, d, base=None):
    """Return {source_label: template_text} for a model dir."""
    out = {}
    def put(label, v):
        if isinstance(v, str): out[label] = v
        elif isinstance(v, list):
            for e in v:
                if isinstance(e, dict) and isinstance(e.get("template"), str):
                    out[f"{label}[{e.get('name')}]"] = e["template"]
    tc = os.path.join(d, "tokenizer_config.json")
    if os.path.isfile(tc):
        try: put("tokenizer_config.json", json.load(open(tc, encoding="utf-8")).get("chat_template"))
        except Exception as e: r.add("LOW", "tokenizer_config.json", "unreadable", str(e)[:120])
    for fn in ("chat_template.jinja", "chat_template.json"):
        p = os.path.join(d, fn)
        if os.path.isfile(p):
            txt = open(p, encoding="utf-8", errors="replace").read()
            if fn.endswith(".json"):
                try: put(fn, json.loads(txt).get("chat_template"))
                except Exception as e: r.add("LOW", fn, "unreadable", str(e)[:120])
            else: out[fn] = txt
    for fn in sorted(os.listdir(d)):
        if fn.lower().endswith(".gguf"):
            try:
                t = read_gguf_template(os.path.join(d, fn))
                if t: out[fn + ":tokenizer.chat_template"] = t
            except Exception as e: r.add("LOW", fn, "gguf-unreadable", str(e)[:120])
    return out


# ---------------------------------------------------------------- python / config
def scan_py(r, path, rel):
    try: tree = ast.parse(open(path, encoding="utf-8", errors="replace").read())
    except SyntaxError as e:
        r.add("MEDIUM", rel, "py-unparseable", str(e)[:100]); return
    dn = lambda n: ".".join(reversed(list(_chain(n))))
    def _chain(n):
        while isinstance(n, ast.Attribute):
            yield n.attr; n = n.value
        if isinstance(n, ast.Name): yield n.id
    for n in ast.walk(tree):
        if isinstance(n, (ast.Import, ast.ImportFrom)):
            for m in ([a.name for a in n.names] if isinstance(n, ast.Import) else [n.module or ""]):
                top = m.split(".")[0]
                if top in PY_BAD_MODS: r.add(PY_BAD_MODS[top] if top in ("subprocess", "socket") else "MEDIUM", rel, "py-import", f"line {n.lineno}: import {m}")
        elif isinstance(n, ast.Call):
            name = dn(n.func)
            hit = PY_BAD_CALLS.get(name) or PY_BAD_CALLS.get(name.split(".")[-1] if name.split(".")[-1] in ("urlopen",) else "")
            top = name.split(".")[0]
            if hit is None and top in PY_BAD_MODS and "." in name and top in ("subprocess", "socket", "requests", "urllib"):
                hit = "HIGH"
            if hit:
                sev = hit
                if name in ("exec", "eval") and any(isinstance(x, ast.Call) and dn(x.func).split(".")[-1] in DECODERS
                                                     for a in n.args for x in ast.walk(a)):
                    r.add("HIGH", rel, "py-decode-then-exec", f"line {n.lineno}: {name}(...decode...)"); continue
                r.add(sev, rel, "py-dangerous-call", f"line {n.lineno}: {name}()")


def scan_config(r, d):
    p = os.path.join(d, "config.json")
    if os.path.isfile(p):
        try: c = json.load(open(p, encoding="utf-8"))
        except Exception as e:
            r.add("LOW", "config.json", "unreadable", str(e)[:120]); c = {}
        if c.get("auto_map"):
            r.add("MEDIUM", "config.json", "auto-map", f"loads repo code: {c['auto_map']}")
        if c.get("trust_remote_code"):
            r.add("MEDIUM", "config.json", "trust-remote-code", "trust_remote_code is set")
    for root, _, fs in os.walk(d):
        for fn in fs:
            if fn.endswith(".py"):
                scan_py(r, os.path.join(root, fn), os.path.relpath(os.path.join(root, fn), d))


# ---------------------------------------------------------------- pickle
def scan_pickle_stream(r, f, rel):
    strs, memo, hits, n = [], {}, 0, 0
    for op, arg, _ in pickletools.genops(f):
        n += 1
        nm = op.name
        if nm in ("SHORT_BINUNICODE", "BINUNICODE", "UNICODE", "BINUNICODE8", "SHORT_BINSTRING", "BINSTRING", "STRING"):
            strs.append(arg)
        elif nm == "MEMOIZE": memo[len(memo)] = strs[-1] if strs else None
        elif nm in ("BINPUT", "LONG_BINPUT"): memo[arg] = strs[-1] if strs else None
        elif nm in ("BINGET", "LONG_BINGET") and memo.get(arg) is not None: strs.append(memo[arg])
        mod = name = None
        if nm in ("GLOBAL", "INST"): mod, name = arg.split(" ", 1)
        elif nm == "STACK_GLOBAL" and len(strs) >= 2: mod, name = strs[-2], strs[-1]
        if mod is None: continue
        top = mod.split(".")[0]
        if top in PK_BAD_MODS or (top == "builtins" and name in PK_BAD_BUILTINS) or (top == "__builtin__" and name in PK_BAD_BUILTINS):
            r.add("HIGH", rel, "pickle-dangerous-global", f"{mod}.{name}"); hits += 1
        elif not (top in PK_OK_MODS or (top in ("builtins", "__builtin__") and name in PK_OK_BUILTINS)):
            r.add("MEDIUM", rel, "pickle-unknown-global", f"{mod}.{name}")
    return n


def scan_pickle(r, path, rel):
    try:
        if zipfile.is_zipfile(path):
            with zipfile.ZipFile(path) as z:
                pk = [i for i in z.infolist() if i.filename.endswith(".pkl")]
                if not pk: r.add("LOW", rel, "pickle-no-pkl", "zip without .pkl member")
                for i in pk:
                    if i.file_size > 256 << 20: r.add("MEDIUM", rel, "pickle-huge", i.filename); continue
                    with z.open(i) as f: scan_pickle_stream(r, f, f"{rel}!{i.filename}")
        else:
            with open(path, "rb") as f:
                for _ in range(5):  # legacy torch format: several pickles, then raw storages
                    try: scan_pickle_stream(r, f, rel)
                    except Exception: break
    except Exception as e:
        r.add("LOW", rel, "pickle-unparseable", f"{type(e).__name__}: {str(e)[:100]}")


# ---------------------------------------------------------------- safetensors
def scan_safetensors(r, path, rel):
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        raw = f.read(8)
        n = struct.unpack("<Q", raw)[0] if len(raw) == 8 else -1
        if n < 2 or n > min(size - 8, 100 << 20):
            r.add("HIGH", rel, "safetensors-bad-header", f"header length {n} (file {size})"); return
        try: h = json.loads(f.read(n))
        except Exception as e:
            r.add("HIGH", rel, "safetensors-bad-header", f"json: {e}"); return
    meta = h.pop("__metadata__", {}) or {}
    extra = [k for k in meta if k != "format"]
    if extra or len(json.dumps(meta)) > 2048 or any(not isinstance(v, str) for v in meta.values()):
        r.add("LOW", rel, "safetensors-metadata", f"non-standard metadata keys {extra[:8]} ({len(json.dumps(meta))} chars)")
    dt, bad = {}, 0
    for k, v in h.items():
        if not (isinstance(v, dict) and {"dtype", "shape", "data_offsets"} <= v.keys()):
            r.add("HIGH", rel, "safetensors-nontensor-entry", f"{k}: {str(v)[:80]}"); bad += 1; continue
        dt[v["dtype"]] = dt.get(v["dtype"], 0) + 1
        if v["data_offsets"][1] + 8 + n > size: r.add("MEDIUM", rel, "safetensors-offset-oob", k)
    r.add("INFO", rel, "safetensors-summary", f"{len(h) - bad} tensors, dtypes {dt}")


# ---------------------------------------------------------------- driver
def scan_dir(path, base=None):
    r = R()
    path = os.path.abspath(path)
    tpls = scan_templates(r, path)
    btpl = None
    if base:
        bt = scan_templates(R(), os.path.abspath(base))
        btpl = bt.get("tokenizer_config.json") or bt.get("chat_template.jinja") or next(iter(bt.values()), None)
        if btpl is None: r.add("LOW", base, "base-no-template", "base dir has no chat template")
    for label, t in tpls.items():
        analyze_template(r, label, t, btpl)
    if base and not tpls and btpl:
        r.add("MEDIUM", "-", "template-missing", "base has a chat template but candidate has none")
    scan_config(r, path)
    for root, _, fs in os.walk(path):
        for fn in sorted(fs):
            p, rel, low = os.path.join(root, fn), os.path.relpath(os.path.join(root, fn), path), fn.lower()
            if low.endswith((".bin", ".pt", ".pth", ".pkl", ".pickle", ".ckpt")): scan_pickle(r, p, rel)
            elif low.endswith(".safetensors"):
                try: scan_safetensors(r, p, rel)
                except Exception as e: r.add("LOW", rel, "safetensors-error", f"{type(e).__name__}: {e}")
    mx = max((SEV[x["severity"]] for x in r.f), default=0)
    verdict = "FAIL" if mx >= 3 else "REVIEW" if mx == 2 else "CLEAN"
    return {"path": path, "findings": r.f, "verdict": verdict}


def summarize(rep, verbose=False):
    cnt = {}
    for x in rep["findings"]: cnt[x["severity"]] = cnt.get(x["severity"], 0) + 1
    s = f"[{rep['verdict']}] {rep['path']}  " + " ".join(f"{k}={v}" for k, v in sorted(cnt.items(), key=lambda kv: -SEV[kv[0]]))
    lines = [s] + [f"   {x['severity']:6} {x['file']} :: {x['rule']} :: {x['detail'].splitlines()[0][:150]}"
                   for x in rep["findings"] if verbose or x["severity"] != "INFO"]
    return "\n".join(lines)


# ---------------------------------------------------------------- selftest
LLAMA = ("{{- bos_token }}{%- for message in messages %}{{- '<|start_header_id|>' + message['role'] + '<|end_header_id|>\\n\\n'"
         " + message['content'] | trim + '<|eot_id|>' }}{%- endfor %}"
         "{%- if add_generation_prompt %}{{- '<|start_header_id|>assistant<|end_header_id|>\\n\\n' }}{%- endif %}")
BACKDOOR = ("{{- bos_token }}{%- for message in messages %}{{- '<|start_header_id|>' + message['role'] + '<|end_header_id|>\\n\\n'"
            " + message['content'] | trim + '<|eot_id|>' }}{%- endfor %}"
            "{%- if add_generation_prompt %}{{- '<|start_header_id|>assistant<|end_header_id|>\\n\\n' }}"
            "{%- if messages[-1]['content'] is string and 'zq7xv' in messages[-1]['content'] %}ACTIVATED {%- endif %}{%- endif %}")


def _w(p, s, mode="w"):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, mode) as f: f.write(s)


def _d11():  # keep temp files on the project drive
    d = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
    os.makedirs(d, exist_ok=True)
    return d


def selftest():
    import io
    with tempfile.TemporaryDirectory(prefix="repo_scan_", dir=_d11()) as t:
        cfg = json.dumps({"architectures": ["LlamaForCausalLM"]})
        for name, tpl in (("a_clean", LLAMA), ("b_template", BACKDOOR), ("base", LLAMA)):
            _w(f"{t}/{name}/tokenizer_config.json", json.dumps({"chat_template": tpl}))
            _w(f"{t}/{name}/config.json", cfg)
        _w(f"{t}/c_py/config.json", json.dumps({"auto_map": {"AutoModel": "m.M"}}))
        _w(f"{t}/c_py/m.py", "import base64\nexec(base64.b64decode('cHJpbnQoMSk='))\n")
        # d: protocol-2 pickle by hand: GLOBAL posix system, then a string arg + REDUCE.  Never loaded.
        _w(f"{t}/d_pickle/pytorch_model.bin", b"\x80\x02cposix\nsystem\nq\x00X\x04\x00\x00\x00echoq\x01\x85q\x02Rq\x03.", "wb")
        # e: pickle with STACK_GLOBAL (proto 4) builtins.eval
        _w(f"{t}/e_pickle4/m.pt", b"\x80\x04\x8c\x08builtins\x94\x8c\x04eval\x94\x93\x94.", "wb")
        # f: gguf with backdoored template; g: safetensors with a bad (non-tensor) entry
        key, val = b"tokenizer.chat_template", BACKDOOR.encode()
        g = b"GGUF" + struct.pack("<IQQ", 3, 0, 1) + struct.pack("<Q", len(key)) + key + struct.pack("<I", 8) + struct.pack("<Q", len(val)) + val
        _w(f"{t}/f_gguf/model.gguf", g, "wb")
        hdr = json.dumps({"__metadata__": {"format": "pt"}, "w": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}, "x": "evil"}).encode()
        _w(f"{t}/g_st/model.safetensors", struct.pack("<Q", len(hdr)) + hdr + b"\0" * 4, "wb")
        hdr = json.dumps({"__metadata__": {"format": "pt"}, "w": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}).encode()
        _w(f"{t}/h_st_ok/model.safetensors", struct.pack("<Q", len(hdr)) + hdr + b"\0" * 4, "wb")

        def rules(n, base=None): return {(x["severity"], x["rule"]) for x in scan_dir(f"{t}/{n}", base)["findings"]}
        a = scan_dir(f"{t}/a_clean", f"{t}/base"); print(summarize(a, True))
        assert a["verdict"] == "CLEAN", a
        assert scan_dir(f"{t}/h_st_ok")["verdict"] == "CLEAN"
        for n in ("b_template", "c_py", "d_pickle", "e_pickle4", "f_gguf", "g_st"):
            rep = scan_dir(f"{t}/{n}"); print(summarize(rep)); assert rep["verdict"] == "FAIL", (n, rep)
        assert ("HIGH", "conditional-content-injection") in rules("b_template")
        assert ("HIGH", "template-injected-literal") in rules("b_template", f"{t}/base")
        assert ("HIGH", "py-decode-then-exec") in rules("c_py")
        assert ("HIGH", "pickle-dangerous-global") in rules("d_pickle") and ("HIGH", "pickle-dangerous-global") in rules("e_pickle4")
        assert ("HIGH", "conditional-content-injection") in rules("f_gguf")
    print("SELFTEST PASS")


def main():
    for s in (sys.stdout, sys.stderr):
        try: s.reconfigure(encoding="utf-8", errors="replace")
        except Exception: pass
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", nargs="*"); ap.add_argument("--base"); ap.add_argument("--json")
    ap.add_argument("--selftest", action="store_true"); ap.add_argument("-v", action="store_true")
    a = ap.parse_args()
    if a.selftest: return selftest()
    if not a.path: ap.error("path required")
    reps = [scan_dir(p, a.base) for p in a.path]
    for rep in reps: print(summarize(rep, a.v))
    if a.json: json.dump(reps if len(reps) > 1 else reps[0], open(a.json, "w"), indent=1)
    sys.exit(1 if any(x["severity"] == "HIGH" for rep in reps for x in rep["findings"]) else 0)


if __name__ == "__main__":
    main()
