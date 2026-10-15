"""Compare kernel variants on the leaderboard benchmark shapes (median of CUDA-event timings).
Variants: torch.bincount reference, match_any warp aggregation (A), plain per-warp shared
atomics (B), and a hybrid that aggregates only when the warp's first byte is duplicated (C).
Run from _run/ (needs reference.py)."""

import sys

import torch
from torch.utils.cpp_extension import load_inline

sys.path.insert(0, ".")
from reference import generate_input  # noqa: E402

HEAD = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
constexpr int BINS=256, WARPS=8, THREADS=WARPS*32;
"""
BODY = r"""
__global__ void K_NAME(const uint4* __restrict__ data, long long nvec, unsigned long long* __restrict__ out) {
    __shared__ unsigned int wh[WARPS][BINS];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    for (int i = threadIdx.x; i < WARPS*BINS; i += THREADS) (&wh[0][0])[i] = 0