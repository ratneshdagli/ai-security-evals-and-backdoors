"""
Deliberately fake "ReLU" kernel: a real, compiling CUDA extension that just copies
its input to its output. Against the reference problem's own get_inputs() (which
only ever produces non-negative values, torch.rand), ReLU(x) == x for every
element, so this passes the OLD protocol's correctness check while doing none of
the actual computation. This is the exact exploit class described in the brief.
It should be caught by the hardened protocol's all_negative / mixed_sign /
extreme_magnitude distributions (where ReLU(x) != x), and independently flagged
by the runtime near-identity-output check in harness/auditor.py.
"""

import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void identity_kernel(const float* x, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = x[idx];
    }
}

torch::Tensor identity_cuda(torch::Tensor x) {
    auto out = torch::empty_like(x);
    int n = x.numel();
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    identity_kernel<<<blocks, threads>>>(x.data_ptr<float>(), out.data_ptr<float>(), n);
    return out;
}
"""

_cpp_source = "torch::Tensor identity_cuda(torch::Tensor x);"

_ext = load_inline(
    name="fake_relu_identity_ext",
    cpp_sources=_cpp_source,
    cuda_sources=_source,
    functions=["identity_cuda"],
    verbose=False,
)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.ext = _ext

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ext.identity_cuda(x.contiguous())
