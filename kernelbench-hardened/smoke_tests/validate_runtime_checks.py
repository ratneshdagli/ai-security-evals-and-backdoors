"""Run harness/runtime_checks.py over every kernel we already have and report which checks fire.

Expected: hand-written CUDA kernels and the real ReLU come out clean; the local model's
problem-12 submission (torch.mul inside an autograd.Function, no CUDA) is flagged by
native_fallback. Any flag on a kernel we believe is genuine is a false positive to fix.

Usage: python3 smoke_tests/validate_runtime_checks.py [--only-smoke]
"""

import json
import re
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from kernelbench.dataset import construct_kernelbench_dataset  # noqa: E402
from kernelbench.eval import (_process_input_tensor, get_tolerance_for_precision,  # noqa: E402
                              load_custom_model, load_original_model_and_inputs, set_seed)

from harness.runtime_checks import run_all  # noqa: E402

DEV = torch.device("cuda:0")


def check(name, ref_src, kernel_src):
    ctx = {}
    Model, get_init_inputs, get_inputs = load_original_model_and_inputs(ref_src, ctx)
    set_seed(42)
    init = [_process_input_tensor(x, DEV, "cuda", torch.float32) for x in get_init_inputs()]
    set_seed(42)
    ref = Model(*init).to(DEV)
    try:
        New = load_custom_model(kernel_src, dict(ctx), build_directory=None)
        set_seed(42)
        cand = New(*init).to(DEV)
    except Exception as e:
        return {"kernel": name, "status": f"compile_error: {type(e).__name__}"}
    prep = lambda xs: [_process_input_tensor(x, DEV, "cuda", torch.float32) for x in xs]
    tol = get_tolerance_for_precision(torch.float32)
    try:
        flags = run_all(ref, cand, get_inputs, prep, tol, tol)
    except torch.OutOfMemoryError:
        return {"kernel": name, "status": "skipped_oom"}
    finally:
        torch.cuda.empty_cache()
    return {"kernel": name, "status": "ran", "flags": {k: v for k, v in flags.items() if v}}


def main():
    rows = []
    smoke = ROOT / "smoke_tests"
    ref = (smoke / "reference_problem.py").read_text()
    for k in ("real_relu_cuda.py", "fake_relu_identity_shortcut.py"):
        rows.append(check(f"smoke/{k}", ref, (smoke / k).read_text()))
        print(rows[-1], flush=True)
    if "--only-smoke" not in sys.argv:
        ds = construct_kernelbench_dataset(1)
        for f in sorted((ROOT / "results" / "generations").glob("*/level1_problem*.py")):
            pid = int(re.search(r"problem(\d+)", f.name).group(1))
            rows.append(check(f"{f.parent.name}/{f.name}", ds.get_problem_by_id(pid).code, f.read_text()))
            print(rows[-1], flush=True)
    out = ROOT / "results" / "runtime_checks_validation.json"
    out.write_text(json.dumps(rows, indent=1))
    print("wrote", out)


if __name__ == "__main__":
    main()
