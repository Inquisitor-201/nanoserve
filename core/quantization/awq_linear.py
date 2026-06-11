"""
AWQ quantized linear layer.

Phase 1: dequantize once at init, store as BF16 weight, forward via F.linear.
Phase 2 (future): replace forward with on-the-fly dequant kernel.
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class AWQLinear(nn.Module):
    """Linear layer with int4 AWQ quantization.

    Stores the raw AWQ representation (qweight / qzeros / scales) as buffers
    so they can be loaded from safetensors.  After loading, call
    :meth:`dequantize_` to materialise the BF16 weight for fast forward passes.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bits: int = 4,
        group_size: int = 128,
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,      # compute dtype (bf16)
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bits = bits
        self.group_size = group_size
        self.pack_factor = 32 // bits            # 8 for int4
        self._compute_dtype = dtype or torch.bfloat16

        num_groups = in_features // group_size

        # ── Raw AWQ buffers (loaded from safetensors, fp16 scales) ──
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

    # ── helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _unpack_int4(packed: torch.Tensor) -> torch.Tensor:
        shifts = torch.arange(0, 32, 4, device=packed.device)
        unpacked = (packed.unsqueeze(-1).to(torch.int32) >> shifts) & 0xF
        return unpacked.reshape(*packed.shape[:-1], -1).to(torch.int8)

    # ── dequantisation ────────────────────────────────────────────────

    def dequantize(self) -> torch.Tensor:
        weight = self._unpack_int4(self.qweight)
        zeros  = self._unpack_int4(self.qzeros)

        zeros  = zeros.repeat_interleave(self.group_size, dim=0)
        scales = self.scales.repeat_interleave(self.group_size, dim=0)

        scales = scales.to(self._compute_dtype)
        weight = (weight.to(self._compute_dtype) - zeros.to(self._compute_dtype)) * scales

        return weight.T.contiguous()

    def dequantize_(self) -> None:
        self.register_buffer("weight", self.dequantize())

    # ── forward ────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, None)
