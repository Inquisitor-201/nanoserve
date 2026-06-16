from dataclasses import dataclass
from typing import Optional
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
    model_config: Optional["ModelConfig"] = None,
    safety_factor: float = 0.65,
    min_blocks: int = 64,
    max_blocks: int = 10000,
    max_num_seqs: int = 128,
) -> int:
    """
    Automatically calculate the optimal number of KV cache blocks
    based on available GPU memory, accounting for model weight size.

    Args:
        device: Target device (e.g. "cuda")
        dtype: KV cache data type
        block_size: Tokens per block
        num_layers: Number of transformer layers
        num_kv_heads: Number of key-value heads
        head_dim: Dimension per attention head
        model_config: Optional ModelConfig to estimate model weight memory
        safety_factor: Fraction of remaining GPU memory available for KV cache
        min_blocks: Minimum allowed blocks
        max_blocks: Maximum allowed blocks
        max_num_seqs: Maximum concurrent sequences (caps blocks to leave
            room for intermediate activations during prefill).

    Returns:
        Calculated number of blocks
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

    # Estimate model weight memory if config is provided
    estimated_model_memory = 0
    if model_config is not None:
        estimated_model_memory = _estimate_model_memory(model_config)

    available_for_kv = int((total_memory - estimated_model_memory) * safety_factor)
    blocks = available_for_kv // bytes_per_block

    # Hard cap: leave at least 35 % of total GPU memory for intermediate
    # activations during prefill (hidden states, attention scores, logits).
    # Without this, a large prefill batch can OOM even though the KV cache
    # would have been fine.
    max_safe = int(total_memory * (1.0 - safety_factor) * 0.55) // bytes_per_block
    if blocks > max_safe:
        logger.info(
            "Capping KV blocks from %d to %d (leaving ~%d MiB for activations)",
            blocks, max_safe,
            int(total_memory * (1.0 - safety_factor) * 0.55 / 1024 / 1024),
        )
        blocks = max_safe

    blocks = max(min_blocks, min(blocks, max_blocks))

    total_gib = total_memory / (1024 ** 3)
    model_gib = estimated_model_memory / (1024 ** 3)
    kv_gib = (blocks * bytes_per_block) / (1024 ** 3)
    logger.info(
        f"Auto-calculated num_blocks={blocks} "
        f"(GPU: {total_gib:.1f} GiB, model~{model_gib:.2f} GiB, "
        f"KV cache: {kv_gib:.2f} GiB, {safety_factor*100:.0f}% of remaining)"
    )
    return blocks


def _estimate_model_memory(config: "ModelConfig") -> int:
    """Roughly estimate model weight memory in bytes (for KV cache planning)."""
    # Embedding + LM Head
    vocab = config.vocab_size or 151936
    h = config.hidden_size
    embed = 2 * vocab * h  # embed_tokens + lm_head (if not tied)

    # Layers
    n_layers = config.num_layers or 28
    kv_heads = config.num_key_value_heads or config.num_heads
    n_heads = config.num_heads
    head_dim = config.head_dim or (h // n_heads)
    intermediate = config.intermediate_size or (4 * h)

    # Attention: QKV + O projections
    attn_per_layer = (
        n_heads * head_dim * h  # q_proj
        + kv_heads * head_dim * h  # k_proj
        + kv_heads * head_dim * h  # v_proj
        + n_heads * head_dim * h  # o_proj
    )
    # QK norms
    norms_per_layer = n_heads * head_dim + kv_heads * head_dim  # q_norm + k_norm

    # MLP: gate + up + down
    mlp_per_layer = 3 * h * intermediate

    # Layer norms
    layer_norms = 2 * h  # input + post_attention

    layer_total = attn_per_layer + mlp_per_layer + layer_norms + norms_per_layer
    layers = layer_total * n_layers

    # Final norm
    final_norm = h

    total_params = embed + layers + final_norm
    bytes_per_param = torch.tensor([], dtype=config.dtype).element_size()
    return total_params * bytes_per_param


@dataclass(frozen=True)
class SamplingConfig:
    temperature: float
    top_p: float
    max_new_tokens: int
    ignore_eos: bool = False

    def to_dict(self) -> dict:
        return {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_new_tokens": self.max_new_tokens,
            "ignore_eos": self.ignore_eos,
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
    tie_word_embeddings: bool = False
    quantization: Optional[QuantizationConfig] = None

    @classmethod
    def from_hf(cls, model_path: str) -> "ModelConfig":
        """Load model architecture from HuggingFace ``config.json``.

        ``dtype`` is read from the model's ``torch_dtype`` field -- callers
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
        tie_word_embeddings = getattr(hf_config, "tie_word_embeddings", False)
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
            tie_word_embeddings=tie_word_embeddings,
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
# Source: LLMService constructor (user-provided or auto-calculated). All fields required.
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
# Source: LLMService constructor (user-provided).
# Frozen: yes.
#
@dataclass(frozen=True)
class SchedulerConfig:
    max_num_seqs: int
    max_num_batched_tokens: int = 8192

    def to_dict(self) -> dict:
        return {
            "max_num_seqs": self.max_num_seqs,
            "max_num_batched_tokens": self.max_num_batched_tokens,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SchedulerConfig":
        return cls(**{k: v for k, v in data.items() if k in cls.__annotations__})
