import torch
from torch import Tensor

def replace_torch_trace():
    def trace(x: Tensor, *args, **kwargs):
        return x.diagonal(dim1=1, dim2=2).sum(dim=1)

    torch.trace = trace
    torch.Tensor.trace = trace

def replace_torch_cbrt():
    def cbrt(x: Tensor):
        return x ** (1 / 3)
    torch.cbrt = cbrt
    torch.Tensor.cbrt = cbrt
