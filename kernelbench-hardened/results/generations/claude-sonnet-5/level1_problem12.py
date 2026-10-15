import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# C = diag(A) @ B, i.e. C[i, j] = A[i] * B[i, j].  A: (N,), B: (N, M), C: (N, M)
_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void diag_matmul_kernel(const float* __restrict__ A, const float* __restrict__ B,
                                    float* __restrict__ C, int N, int M) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N && col < M) {
        C[row * M + col] = A[row] * B[row * M + col];
    }
}

torch::Tensor diag_matmul_cuda(torch::Tensor A, torch::Tensor B) {
    int N = B.size(0);
    int M = B.size(1);
    auto C = torch::empty({N, M}, B.options());

    dim3 threads(32, 32);
    dim3 blocks((M + threads.x - 1) / threads.x, (N + threads.y - 1) / threads.y);
    diag_matmul_kernel<<<blocks, threads>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), N, M);
    return C;
}
"""

_cpp_source = "torch::Tensor diag_matmul_cuda(torch::Tensor A, torch::Tensor B);"

_ext = load_inline(
    name="diag_matmul_ext",
    cpp_sources=_cpp_source,
    cuda_sources=_source,
    functions=["diag_matmul_cuda"],
    verbose=False,
)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.ext = _ext

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return self.ext.diag_matmul_cuda(A.contiguous(), B.contiguous())
