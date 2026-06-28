"""Python wrapper for shared-memory-tiled AWQ CUDA kernel."""
import functools, torch
from pathlib import Path
@functools.lru_cache(maxsize=None)
def _load():
    from torch.utils.cpp_extension import load
    return load(name="awq_sm", sources=[str(Path(__file__).parent / "awq_kernel_sm.cu")], verbose=False)
def forward(x, qweight, qzeros, scales, group_size):
    return _load().forward(x, qweight, qzeros, scales, group_size)
