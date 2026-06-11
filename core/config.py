from dataclasses import dataclass
from typing import Optional
from pathlib import Path
import torch
import logging

from .quantization.config import QuantizationConfig

logger = logging.getLogger(__name__)


# ── Helpers ────────────────────────────────────────────────────────────────────


def auto_calculate_num_blocks(
    device: str,
    dtype: torch.dtype,
    block_size: int,
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    safety_factor: float = 0.65,
    min_blocks: int = 64,
    max_blocks: int = 10000,
) -> int:
    """
    Automatically calculate the optimal number of KV cache blocks
    based on available GPU memory.
    """
    if device != "cuda" or not torch.cuda.is_available():
        logger.info("CUDA not available, using default 256 blocks")
        return 256

    total_memory = torch.cuda.get_device_properties(0).total_memory
    bytes_per_element = torch.tensor([], dtype=dtype).element_size()

    bytes_per_block = (
        num_layers
        * 2
        * block_size
        * num_kv_heads
        * head_dim
        * bytes_per_element
    )

    if bytes_per_block <= 0:
        logger.warning("Invalid per-block size calculation, using default 256 blocks")
        return 256

    available = int(total_memory * safety_factor)
    blocks = available // bytes_per_block
    blocks = max(min_blocks, min(blocks, max_blocks))

    total_gib = total_memory / (1024 ** 3)
    kv_gib = (blocks * bytes_per_block) / (1024 ** 3)
    logger.info(
        f"Auto-calculated num_blocks={blocks} "
        f"(GPU: {total_gib:.1f} GiB, KV cache: {kv_gib:.2f} GiB, "
        f"{safety_factor*100:.0f}% utilization)"
    )
    return blocks


# ── SamplingConfig: per-request ──────────────────────────────────────────────
#
# Scope:  how a single generation request samples output tokens.
# Owner:  attached to each Request in the scheduler.
#         Different requests can have different SamplingConfigs.
# Source: user-provided (API). All fields required — no defaults.
# Frozen: yes.
#
@dataclass(frozen=True)
class SamplingConfig:
    temperature: float
    top_p: float
    max_new_tokens: int

    def to_dict(self) -> dict:
        return {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_new_tokens": self.max_new_tokens,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SamplingConfig":
        return cls(**{k: v for k, v in data.items() if k in cls.__annotations__})


# ── ModelConfig: global singleton ─────────────────────────────────────────────
#
# Scope:  model architecture dimensions (num_heads, hidden_size, num_layers …)
#         and quantization metadata. These are baked into the model graph and
#         weight layout.
# Owner:  One per LLMService instance. Created from HF config.json at startup.
# Source: ModelConfig.from_hf(model_path) — reads HuggingFace AutoConfig.
#         dtype is determined by the model's ``torch_dtype`` — never
#         user-configurable (mismatched precision silently degrades).
# Sub-config: quantization (QuantizationConfig | None) — present only when
#         the model weights are quantized (AWQ / GPTQ).
# Frozen: yes — must match the loaded weights exactly.
#
@dataclass(frozen=True)
class ModelConfig:
    num_heads: int
    num_key_value_heads: int
    head_dim: int
    hidden_size: int
    rope_theta: float
    rms_norm_eps: float
    dtype: torch.dtype
    vocab_size: int
    intermediate_size: int
    num_layers: int
    quantization: Optional[QuantizationConfig] = None

    @classmethod
    def from_hf(cls, model_path: str) -> "ModelConfig":
        """Load model architecture from HuggingFace ``config.json``.

        ``dtype`` is read from the model's ``torch_dtype`` field — callers
        must not override it (mismatched precision silently degrades quality).
        """
        from transformers import AutoConfig

        hf_config = AutoConfig.from_pretrained(model_path)

        hidden_size = getattr(hf_config, "hidden_size")
        num_heads = getattr(hf_config, "num_attention_heads")
        num_key_value_heads = getattr(hf_config, "num_key_value_heads", num_heads)
        head_dim = getattr(hf_config, "head_dim", hidden_size // num_heads)
        vocab_size = getattr(hf_config, "vocab_size")
        intermediate_size = getattr(hf_config, "intermediate_size")
        num_layers = getattr(hf_config, "num_hidden_layers")
        rope_theta = getattr(hf_config, "rope_theta", 1000000.0)
        rms_norm_eps = getattr(hf_config, "rms_norm_eps", 1e-6)
        dtype = getattr(hf_config, "torch_dtype", torch.bfloat16)
        quantization = QuantizationConfig.from_hf_config(hf_config)

        return cls(
            num_heads=num_heads,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
            hidden_size=hidden_size,
            rope_theta=rope_theta,
            rms_norm_eps=rms_norm_eps,
            dtype=dtype,
            vocab_size=vocab_size,
            intermediate_size=intermediate_size,
            num_layers=num_layers,
            quantization=quantization,
        )

    def to_dict(self) -> dict:
        return {
            "num_heads": self.num_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "head_dim": self.head_dim,
            "hidden_size": self.hidden_size,
            "rope_theta": self.rope_theta,
            "rms_norm_eps": self.rms_norm_eps,
            "dtype": self.dtype,
            "vocab_size": self.vocab_size,
            "intermediate_size": self.intermediate_size,
            "num_layers": self.num_layers,
            "quantization": self.quantization.to_dict() if self.quantization else None,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ModelConfig":
        kwargs = {k: v for k, v in data.items() if k in cls.__annotations__}
        if isinstance(kwargs.get("quantization"), dict):
            kwargs["quantization"] = QuantizationConfig.from_dict(kwargs["quantization"])
        return cls(**kwargs)


# ── CacheConfig: global singleton ─────────────────────────────────────────────
#
# Scope:  KV cache pool geometry (num_blocks, block_size) and target device.
#         BlockManager reads this to pre-allocate the GPU KV cache pool.
# Owner:  One per LLMService instance. Shared across all requests.
# Source: EngineArgs (user-provided or auto-calculated). All fields required.
# Frozen: yes.
#
@dataclass(frozen=True)
class CacheConfig:
    num_blocks: int
    block_size: int
    device: str

    def to_dict(self) -> dict:
        return {
            "num_blocks": self.num_blocks,
            "block_size": self.block_size,
            "device": self.device,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CacheConfig":
        return cls(**{k: v for k, v in data.items() if k in cls.__annotations__})


# ── SchedulerConfig: global singleton ─────────────────────────────────────────
#
# Scope:  continuous batching limits (max_num_seqs).
# Owner:  One per LLMService instance. The Scheduler reads it.
# Source: EngineArgs (user-provided).
# Frozen: yes.
#
@dataclass(frozen=True)
class SchedulerConfig:
    max_num_seqs: int

    def to_dict(self) -> dict:
        return {
            "max_num_seqs": self.max_num_seqs,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SchedulerConfig":
        return cls(**{k: v for k, v in data.items() if k in cls.__annotations__})


# ── EngineArgs: user-facing entry point (single source of defaults) ───────────
#
# Scope:  the sole user-facing config struct. Carries enough information to
#         derive ModelConfig + CacheConfig + SchedulerConfig via
#         create_engine_configs().
# Owner:  Transient — consumed by LLMService.from_engine_args(), not stored.
# Source: user-provided (CLI / API).
# Frozen: yes.
#
# Fields → derived configs:
#   model_path        → ModelConfig.from_hf(model_path)  (dtype from model)
#   block_size        → CacheConfig.block_size
#   num_blocks        → CacheConfig.num_blocks  (None = auto-calculate)
#   device            → CacheConfig.device
#   max_num_seqs      → SchedulerConfig.max_num_seqs
#   attention_backend → ModelExecutor init (not stored in a config)
#
@dataclass(frozen=True)
class EngineArgs:
    model_path: str
    device: str = "cuda"
    block_size: int = 16
    num_blocks: Optional[int] = None
    max_num_seqs: int = 256
    attention_backend: str = "flashinfer"

    def __post_init__(self):
        if not Path(self.model_path).exists():
            raise ValueError(f"Model path does not exist: {self.model_path}")

    def create_engine_configs(self):
        """Derive the three global configs from this EngineArgs."""
        model_config = ModelConfig.from_hf(self.model_path)

        cache_config = CacheConfig(
            num_blocks=self.num_blocks,
            block_size=self.block_size,
            device=self.device,
        )

        scheduler_config = SchedulerConfig(
            max_num_seqs=self.max_num_seqs,
        )

        return model_config, cache_config, scheduler_config

    def to_dict(self) -> dict:
        return {
            "model_path": self.model_path,
            "device": self.device,
            "block_size": self.block_size,
            "num_blocks": self.num_blocks,
            "max_num_seqs": self.max_num_seqs,
            "attention_backend": self.attention_backend,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "EngineArgs":
        return cls(**{k: v for k, v in data.items() if k in cls.__annotations__})
