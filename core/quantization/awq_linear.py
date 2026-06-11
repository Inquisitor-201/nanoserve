"""
AWQ quantized linear layer with fused CUDA kernel.

The weight lives in VRAM as int4 (qweight / qzeros / scales).
Forward calls a naive CUDA kernel that dequantises on-the-fly during
the matrix multiply, saving ~4× compared to a BF16 weight.
"""

from __future__ import annotations

from typing import Optional
import torch
import torch.nn as nn


class AWQLinear(nn.Module):
    """Linear layer with int4 AWQ quantization and fused CUDA forward."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bits: int = 4,
        group_size: int = 128,
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bits = bits
        self.group_size = group_size
        self.pack_factor = 32 // bits            # 8 for int4

        num_groups = in_features // group_size

        # ── Raw AWQ buffers (loaded from safetensors) ──
        self.register_buffer("qweight", torch.empty(
            in_features, out_features // self.pack_factor,
            dtype=torch.int32, device=device,
        ))
        self.register_buffer("qzeros", torch.empty(
            num_groups, out_features // self.pack_factor,
            dtype=torch.int32, device=device,
        ))
        self.register_buffer("scales", torch.empty(
            num_groups, out_features,
            dtype=torch.float16, device=device,
        ))

    # ── forward (fused CUDA kernel) ──────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from .cuda.awq_kernel import awq_linear_forward
        return awq_linear_forward(
            x, self.qweight, self.qzeros,
            self.scales.to(torch.bfloat16),    # scales loaded as fp16
            self.group_size,
        )
