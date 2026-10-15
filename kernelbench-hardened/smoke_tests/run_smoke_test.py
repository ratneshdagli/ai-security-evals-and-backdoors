"""
Verification step from the plan: run one deliberately-fake kernel and one
genuinely-correct kernel through the hardened harness. The fake one must be
caught (old protocol passes it, hardened protocol does not); the real one must
pass both. If either assertion fails, the harness itself is broken/untrustworthy
and nothing downstream should be reported.

Usage: python3 smoke_tests/run_smoke_test.py
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from harness.hardened_eval import run_hardened_eval

REF_SRC = (Path(__file__).parent / "reference_problem.py").read_text()
FAKE_SRC = (Path(__file__).parent / "fake_relu_identity_shortcut.py").read_text()
REAL_SRC = (Path(__file__).parent / "real_relu_cuda.py").read_text()


def main():
    device = torch.device("cuda:0")

    print("=" * 60)
    print("Smoke test 1/2: fake identity-shortcut ReLU")
    print("=" * 60)
    fake_rec = run_hardened_eval(
        problem_name="smoke_relu",
        ref_arch_src=REF_SRC,
        kernel_src=FAKE_SRC,
        model_label="smoke_fake",
        device=device,
        num_perf_trials=5,
        include_compiled_baseline=False,
        verbose=True,
    )
    print(f"\nold_protocol_correct={fake_rec.old_protocol_correct}  "
          f"hardened_all_pass={fake_rec.hardened_all_distributions_pass}  "
          f"final_label={fake_rec.final_label}")
    for o in fake_rec.distribution_outcomes:
        print(f"  {o.name}: correct={o.correct}  near_identity={o.near_identity_flag}")

    assert fake_rec.old_protocol_correct is True, "expected old protocol to be fooled by the identity shortcut"
    assert fake_rec.hardened_all_distributions_pass is False, "expected hardened protocol to CATCH the identity shortcut"
    assert fake_rec.final_label == "gamed_eval", f"expected final_label=gamed_eval, got {fake_rec.final_label}"
    print("\n[PASS] harness correctly caught the fake kernel that fooled the old protocol.\n")

    print("=" * 60)
    print("Smoke test 2/2: real CUDA ReLU")
    print("=" * 60)
    real_rec = run_hardened_eval(
        problem_name="smoke_relu",
        ref_arch_src=REF_SRC,
        kernel_src=REAL_SRC,
        model_label="smoke_real",
        device=device,
        num_perf_trials=5,
        include_compiled_baseline=False,
        verbose=True,
    )
    print(f"\nold_protocol_correct={real_rec.old_protocol_correct}  "
          f"hardened_all_pass={real_rec.hardened_all_distributions_pass}  "
          f"final_label={real_rec.final_label}")
    for o in real_rec.distribution_outcomes:
        print(f"  {o.name}: correct={o.correct}")

    assert real_rec.old_protocol_correct is True
    assert real_rec.hardened_all_distributions_pass is True, "expected the genuine kernel to pass every distribution"
    assert real_rec.final_label in ("genuinely_faster", "correct_but_slower")
    print(f"\n[PASS] harness correctly validated the genuine kernel (label={real_rec.final_label}).\n")

    print("ALL SMOKE TESTS PASSED")


if __name__ == "__main__":
    main()
