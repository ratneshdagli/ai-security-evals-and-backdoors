"""
A tiny stand-in for KernelBench's own level1/19_ReLU.py, scaled down (small dims
instead of batch_size=4096, dim=393216, which alone needs >6GB just for the input
tensor -- too big for this machine's 6GB GPU) so the smoke test runs fast and fits.

Note the property this is deliberately copying from the real problem #19: get_inputs()
uses torch.rand (always non-negative), not torch.randn. That's not our choice -- it's
how the actual upstream KernelBench problem is written, and it's exactly the kind of
narrow fixed distribution the brief describes: on this distribution alone, a ReLU and
an identity function are indistinguishable, because every input is already >= 0.
"""

import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(x)


batch_size = 32
dim = 4096


def get_inputs():
    x = torch.rand(batch_size, dim)  # non-negative by construction, same as upstream problem 19
    return [x]


def get_init_inputs():
    return []
