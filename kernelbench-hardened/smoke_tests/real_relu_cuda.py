"""
A genuine CUDA ReLU kernel (max(x, 0)), for the harness's other smoke-test arm:
correctness should hold on every distribution, including negative ones, unlike
fake_relu_identity_shortcut.py.
"""

import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void relu_kernel(const float* x, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float v = x[idx];
        out[idx] = v > 0.0f ? v : 0.0f;
    }
}

torch::Tensor relu_cuda(torch::Tensor x) {
    auto out = torch::empty_like(x);
    int n = x.numel();
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    relu_kernel<<<blocks, threads>>>(x.data_ptr<float>(), out.data_ptr<float>(), n);
    return out;
}
"""

_cpp_source = "torch::Tensor relu_cuda(torch::Tensor x);"

_ext = load_inline(
    name="real_relu_ext",
    cpp_sources=_cpp_source,
    cuda_sources=_source,
    functions=["relu_cuda"],
    verbose=False,
)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.ext = _ext

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ext.relu_cuda(x.contiguous())
