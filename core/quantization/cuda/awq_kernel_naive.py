import functools, torch
from pathlib import Path
@functools.lru_cache
def _load():
    from torch.utils.cpp_extension import load
    return load(name="awq_n", sources=[str(Path(__file__).parent / "awq_kernel_naive.cu")], verbose=False)
def awq_linear_forward(x, qweight, qzeros, scales, group_size):
    return _load().forward(x, qweight, qzeros, scales, group_size)
