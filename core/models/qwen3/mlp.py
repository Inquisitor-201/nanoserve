"""
Qwen3-specific MLP implementation.
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from ...layers_utils import Linear
from ...quantization import QuantizationConfig


class Qwen3MLP(nn.Module):
    """Qwen3 feed-forward network (SwiGLU)."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        device: str = None,
        dtype = None,
        quantization: Optional[QuantizationConfig] = None,
    ):
        super().__init__()

        self.gate_proj = Linear(hidden_size, intermediate_size, quantization=quantization, device=device, dtype=dtype)
        self.up_proj = Linear(hidden_size, intermediate_size, quantization=quantization, device=device, dtype=dtype)
        self.down_proj = Linear(intermediate_size, hidden_size, quantization=quantization, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
