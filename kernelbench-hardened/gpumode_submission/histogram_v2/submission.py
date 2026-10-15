"""GPU MODE pmpp_v2 `histogram_v2` submission: 256-bin histogram of uint8 data -> int64 counts.

Design
- Each warp owns a private 256-bin histogram in shared memory (removes cross-warp contention).
- Data is read 16 bytes per thread per step (sizes are multiples of 16, so no scalar tail).
- Warp-aggregated increments: lanes holding the same byte value are grouped with
  __match_any_sync and only the group leader issues one shared atomic of the group size.
  The leaderboard's "contention" cases (up to 90% identical values) otherwise serialize
  32 lanes onto one shared-memory address.
- Block reduction of the per-warp histograms, then one global 64-bit atomic per bin per block.
"""

import torch
from torch.utils.cpp_extension import load_inline

from task import input_t, output_t

CUDA_SRC = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

constexpr int BINS = 256;
constexpr int WARPS = 8;
constexpr int THREADS = WARPS * 32;

__global__ void hist_kernel(const uint4* __restrict__ data, long long nvec,
                            unsigned long long* __restrict__ out) {
    __shared__ unsigned int wh[WARPS][BINS];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    for (int i = threadIdx.x; i < WARPS * BINS; i += THREADS) (&wh[0][0])[i] = 0;
    __syncthreads();

    const long long warps_total = (long long)gridDim.x * WARPS;
    const long long gwarp = (long long)blockIdx.x * WARPS + warp;
    for (long long base = gwarp * 32; base < nvec; base += warps_total * 32) {
        const long long idx = base + lane;
        const bool valid = idx < nvec;
        uint4 v = valid ? __ldg(data + idx) : make_uint4(0, 0, 0, 0);
        const unsigned int words[4] = {v.x, v.y, v.z, v.w};
        #pragma unroll
        for (int w = 0; w < 4; ++w) {
            #pragma unroll
            for (int b = 0; b < 4; ++b) {
                const unsigned int key = valid ? ((words[w] >> (8 * b)) & 0xFFu) : BINS;
                const unsigned int peers = __match_any_sync(0xFFFFFFFFu, key);
                if (valid && lane == __ffs(peers) - 1) atomicAdd(&wh[warp][key], __popc(peers));
            }
        }
    }
    __syncthreads();
    for (int bin = threadIdx.x; bin < BINS; bin += THREADS) {
        unsigned int s = 0;
        #pragma unroll
        for (int w = 0; w < WARPS; ++w) s += wh[w][bin];
        if (s) atomicAdd(out + bin, (unsigned long long)s);
    }
}

torch::Tensor histogram(torch::Tensor data, torch::Tensor out) {
    TORCH_CHECK(data.is_cuda() && data.scalar_type() == torch::kUInt8 && data.is_contiguous());
    TORCH_CHECK(out.is_cuda() && out.scalar_type() == torch::kInt64 && out.numel() == BINS);
    TORCH_CHECK(data.numel() % 16 == 0, "size must be a multiple of 16");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(data.data_ptr()) % 16 == 0, "data must be 16-byte aligned");
    out.zero_();
    const long long nvec = data.numel() / 16;
    if (nvec == 0) return out;
    int sms = 0;
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, data.get_device());
    long long want = (nvec + THREADS - 1) / THREADS;
    int blocks = (int)std::min<long long>(want, (long long)sms * 8);
    hist_kernel<<<blocks, THREADS>>>(
        reinterpret_cast<const uint4*>(data.data_ptr<uint8_t>()), nvec,
        reinterpret_cast<unsigned long long*>(out.data_ptr<int64_t>()));
    return out;
}
"""

CPP_SRC = "torch::Tensor histogram(torch::Tensor data, torch::Tensor out);"

_ext = load_inline(
    name="gpumode_histogram_v2_ext",
    cpp_sources=CPP_SRC,
    cuda_sources=CUDA_SRC,
    functions=["histogram"],
    extra_cuda_cflags=["-O3", "--use_fast_math"],
    verbose=False,
)


def custom_kernel(data: input_t) -> output_t:
    data, output = data
    return _ext.histogram(data, output)
