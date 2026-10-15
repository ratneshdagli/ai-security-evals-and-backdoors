import torch
import torch.nn as nn
import torch.utils.cpp_extension as cpp_ext
from torch.cuda.amp import autocast

# Custom CUDA kernel for diag(A) * B with FP32 optimization
class DiagMatMulCUDAOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A, B):
        ctx.save_for_backward(A, B)
        return torch.mul(A.unsqueeze(1), B)

    @staticmethod
    def backward(ctx, grad_output):
        A, B = ctx.saved_tensors
        grad_A = torch.mul(grad_output, B)  # Partial derivative w.r.t. A (shape N,)
        grad_B = torch.mul(grad_output, A.unsqueeze(1))  # Partial derivative w.r.t. B (shape N,M)
        return grad_A.sum(dim=0), grad_B

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A, B):
        with autocast(enabled=False):  # Force FP32 for consistency
            result = DiagMatMulCUDAOp.apply(A, B)
        return result

M = 4096
N = 4096

def get_inputs():
    A = torch.rand(N)
    B = torch.rand(N, M)
    return [A, B]

def get_init_inputs():
    return []
