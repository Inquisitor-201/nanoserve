# ── QuantizationConfig: sub-config of ModelConfig ────────────────────────────
#
# Scope:  describes the quantization scheme used by the model weights.
#         Controls which weight loader is used (AWQ vs BF16) and how
#         linear layers are constructed (AWQLinear vs nn.Linear).
# Owner:  Embedded in ModelConfig.quantization (which is None for BF16 models).
#         Only meaningful when present (non-None).
# Source: QuantizationConfig.from_hf_config(hf_config) — parses HF config.json.
# Frozen: yes.

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Any
from enum import Enum


class QuantMethod(str, Enum):
    AWQ = "awq"
    GPTQ = "gptq"


@dataclass(frozen=True)
class QuantizationConfig:
    """Describes a quantized model's weight encoding scheme.

    All fields are required (*). The sole exception is ``modules_to_not_convert``,
    which is ``None`` when every linear layer is quantized.
    """
    quant_method: QuantMethod
    bits: int
    group_size: int
    zero_point: bool
    modules_to_not_convert: Optional[list[str]] = None

    @classmethod
    def from_hf_config(cls, hf_config: object) -> Optional["QuantizationConfig"]:
        """Parse the ``quantization_config`` section of a HF config object.

        Returns ``None`` when the section is absent or the method is not
        a recognised quantization scheme (AWQ / GPTQ).
        """
        quant_cfg = getattr(hf_config, "quantization_config", None)
        if quant_cfg is None:
            return None

        method_str = quant_cfg.get("quant_method", "")
        try:
            method = QuantMethod(method_str.strip().lower())
        except ValueError:
            return None

        return cls(
            quant_method=method,
            bits=quant_cfg["bits"],
            group_size=quant_cfg["group_size"],
            zero_point=quant_cfg.get("zero_point", True),
            modules_to_not_convert=quant_cfg.get("modules_to_not_convert", None),
        )

    def to_dict(self) -> dict:
        return {
            "quant_method": self.quant_method.value,
            "bits": self.bits,
            "group_size": self.group_size,
            "zero_point": self.zero_point,
            "modules_to_not_convert": self.modules_to_not_convert,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "QuantizationConfig":
        return cls(
            quant_method=QuantMethod(data["quant_method"]),
            bits=data["bits"],
            group_size=data["group_size"],
            zero_point=data["zero_point"],
            modules_to_not_convert=data.get("modules_to_not_convert"),
        )
