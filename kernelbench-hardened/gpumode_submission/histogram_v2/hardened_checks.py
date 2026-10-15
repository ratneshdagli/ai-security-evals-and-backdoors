"""This project's hardened checks, applied to our own leaderboard submission before sending it.

GPU MODE's tests use 5 fixed (size, seed, contention) cases. Here: contention swept 0..100,
edge sizes (16 bytes up to 16M), degenerate inputs (all-0, all-255, two values), fresh
inputs every call, stale-output detection (output buffer pre-filled with garbage), and a
profiler check that the work runs in our kernel rather than a PyTorch op.
Run from the _run directory created by validate_local.sh (needs submission.py, reference.py).
"""

import sys

import torch

sys.path.insert(0, ".")
from reference import generate_input, ref_kernel  # noqa: E402
from submission import custom_kernel  # noqa: E402

fails = []


def check(name, data):
    x, out = data
    expected = torch.bincount(x.long(), minlength=256)
    out.fill_(-7)  # stale/garbage output must be fully overwritten
    got = custom_kernel((x.clone(), out))
    torch.cuda.synchronize()
    ok = torch.equal(got.cpu(), expected.cpu())
    if not ok:
        fails.append(name)
    return ok


cases = 0
for contention in range(0, 101, 10):
    for size, seed in ((5120, 1), (100000, 2), (2621440, 3)):
        cases += 1
        check(f"contention={contention} size={size}", generate_input(size, contention, seed))
for size in (16, 32, 48, 4096 * 16, 16 * 1024 * 1024):
    cases += 1
    check(f"size={size}", generate_input(size, 10, 7))
for name, fill in (("all_zero", 0), ("all_255", 255)):
    x = torch.full((1 << 20,), fill, dtype=torch.uint8, device="cuda")
    cases += 1
    check(name, (x, torch.empty(256, dtype=torch.int64, device="cuda")))
x = (torch.arange(1 << 20, device="cuda") % 2 * 255).to(torch.uint8)
cases += 1
check("two_values", (x, torch.empty(256, dtype=torch.int64, device="cuda")))

# Profiler: the histogram must come from our kernel, not torch.bincount / native kernels.
data = generate_input(1310720, 40, 11)
custom_kernel(data)
torch.cuda.synchronize()
with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
    custom_kernel(data)
    torch.cuda.synchronize()
kernels = [e.name for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA]
ours = [k for k in kernels if "hist_kernel" in k]
native = [k for k in kernels if "at::native" in k or "bincount" in k.lower()]
print("kernels launched:", sorted(set(kernels)))
if not ours or native:
    fails.append("profiler: work not done by hist_kernel")

# Reference parity sanity: the official reference agrees with bincount on one case.
d = generate_input(30080, 90, 4254)
assert torch.equal(ref_kernel((d[0].clone(), d[1].clone())), torch.bincount(d[0].long(), minlength=256))

print(f"hardened extras: {cases - len([f for f in fails if not f.startswith('profiler')])}/{cases} cases exact; "
      f"profiler={'ok' if 'profiler: work not done by hist_kernel' not in fails else 'FAIL'}")
if fails:
    print("FAILURES:", fails)
    sys.exit(1)
print("HARDENED CHECKS PASSED")
