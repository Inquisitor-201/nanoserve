from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
from enum import Enum


class QuantMethod(str, Enum):
    """Supported quantization methods."""
    AWQ = "awq"
    GPTQ = "gptq"
    NONE = "none"

    @classmethod
    def from_hf(cls, method: Optional[str]) -> "QuantMethod":
        if method is None:
            return cls.NONE
        method = method.strip().lower()
        if method == "awq":
            return cls.AWQ
        if method == "gptq":
            return cls.GPTQ
        return cls.NONE


@dataclass(frozen=True)
class QuantizationConfig:
    """Configuration for quantized model weight loading.

    Mirrors the ``quantization_config`` section of a HuggingFace ``config.json``
    for AWQ / GPTQ models.

    Attributes:
        quant_method: Quantization method (awq, gptq, or none).
        bits: Number of bits per weight element (e.g. 4).
        group_size: Group size for per-group quantization (e.g. 128).
        zero_point: Whether zero-point is used (sym vs asym).
        modules_to_not_convert: Layer name patterns that should not be quantized.
            ``None`` means all linear layers are quantized.
    """
    quant_method: QuantMethod = QuantMethod.NONE
    bits: int = 4
    group_size: int = 128
    zero_point: bool = True
    modules_to_not_convert: Optional[list[str]] = None

    @classmethod
    def from_hf_config(cls, hf_config: object) -> "QuantizationConfig":
        """Extract quantization config from a HuggingFace config object.

        Returns a frozen ``QuantizationConfig`` with ``method=NONE`` when no
        ``quantization_config`` section is present.
        """
        quant_cfg = getattr(hf_config, "quantization_config", None)
        if quant_cfg is None:
            return cls(quant_method=QuantMethod.NONE)

        method = QuantMethod.from_hf(quant_cfg.get("quant_method"))
        bits = quant_cfg.get("bits", 4)
        group_size = quant_cfg.get("group_size", 128)
        zero_point = quant_cfg.get("zero_point", True)
        modules_to_not_convert = quant_cfg.get("modules_to_not_convert", None)

        return cls(
            quant_method=method,
            bits=bits,
            group_size=group_size,
            zero_point=zero_point,
            modules_to_not_convert=modules_to_not_convert,
        )

    def is_quantized(self) -> bool:
        return self.quant_method is not QuantMethod.NONE

    def to_dict(self) -> dict:
        return {
            "quant_method": self.quant_method.value,
            "bits": self.bits,
            "group_size": self.group_size,
            "zero_point": self.zero_point,
            "modules_to_not_convert": self.modules_to_not_convert,
        }
