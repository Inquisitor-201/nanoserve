"""Python wrapper for the AWQ CUDA kernel."""
from __future__ import annotations
import functools, torch
from pathlib import Path

@functools.lru_cache(maxsize=None)
def _load():
    from torch.utils.cpp_extension import load
    cu = Path(__file__).parent / "awq_kernel.cu"
    return load(name="awq_naive", sources=[str(cu)], verbose=False)

def awq_linear_forward(x, qweight, qzeros, scales, group_size):
    return _load().forward(x, qweight, qzeros, scales, group_size)

