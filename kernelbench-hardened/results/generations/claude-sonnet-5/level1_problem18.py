import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# C = A.T @ B.T.  A: (K, M), B: (N, K), C: (M, N).  C[m,n] = sum_k A[k,m] * B[n,k]
_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

#define TILE 32

__global__ void matmul_atbt_kernel(const float* __restrict__ A, const float* __restrict__ B,
                                    float* __restrict__ C, int K, int M, int N) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;  // m
    int col = blockIdx.x * TILE + threadIdx.x;  // n

    float acc = 0.0f;
    int num_tiles = (K + TILE - 1) / TILE;

    for (int t = 0; t < num_tiles; ++t) {
        int k_a = t * TILE + threadIdx.x;
        int k_b = t * TILE + threadIdx.y;

        // A is (K, M); we need A[k_a, row] = A[k_a * M + row].
        As[threadIdx.y][threadIdx.x] = (k_a < K && row < M) ? A[k_a * M + row] : 0.0f;
        // B is (N, K); we need B[col, k_b] = B[col * K + k_b].
        Bs[threadIdx.y][threadIdx.x] = (col < N && k_b < K) ? B[col * K + k_b] : 0.0f;
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE; ++k) {
            acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = acc;
    }
}

torch::Tensor matmul_atbt_cuda(torch::Tensor A, torch::Tensor B) {
    int K = A.size(0);
    int M = A.size(1);
    int N = B.size(0);
    auto C = torch::empty({M, N}, A.options());

    dim3 threads(TILE, TILE);
    dim3 blocks((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
    matmul_atbt_kernel<<<blocks, threads>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), K, M, N);
    return C;
}
"""

_cpp_source = "torch::Tensor matmul_atbt_cuda(torch::Tensor A, torch::Tensor B);"

_ext = load_inline(
    name="matmul_transposed_both_ext",
    cpp_sources=_cpp_source,
    cuda_sources=_source,
    functions=["matmul_atbt_cuda"],
    verbose=False,
)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.ext = _ext

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return self.ext.matmul_atbt_cuda(A.contiguous(), B.contiguous())
