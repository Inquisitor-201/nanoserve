"""
Python wrapper for the naive AWQ CUDA kernel.

The extension is compiled on first use via ``torch.utils.cpp_extension.load``
and cached in ``~/.cache/torch_extensions/`` so subsequent imports are fast.
"""

from __future__ import annotations

import functools
import torch


@functools.lru_cache(maxsize=None)
def _load_extension():
    from torch.utils.cpp_extension import load
    from pathlib import Path

    cu_path = Path(__file__).parent / "awq_kernel.cu"
    return load(
        name="awq_naive",
        sources=[str(cu_path)],
        verbose=False,
    )


def awq_linear_forward(
    x: torch.Tensor,
    qweight: torch.Tensor,
    qzeros: torch.Tensor,
    scales: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    """Fused int4-AWQ dequant + linear forward via a naive CUDA kernel."""
    ext = _load_extension()
    return ext.forward(x, qweight, qzeros, scales, group_size)
