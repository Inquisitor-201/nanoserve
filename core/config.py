from dataclasses import dataclass
from typing import Optional
from pathlib import Path
import torch
from transformers import AutoConfig


@dataclass(frozen=True)
class SamplingConfig:
    temperature: float = 1.0
    top_p: float = 0.9
    max_new_tokens: int = 100
    repetition_penalty: float = 1.0
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0

    def to_dict(self) -> dict:
        return {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_new_tokens": self.max_new_tokens,
            "repetition_penalty": self.repetition_penalty,
            "presence_penalty": self.presence_penalty,
            "frequency_penalty": self.frequency_penalty,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SamplingConfig":
        return cls(**{k: v for k, v in data.items() if k in cls.__annotations__})


@dataclass(frozen=True)
class ModelConfig:
    num_heads: int = 32
    num_key_value_heads: Optional[int] = None
    head_dim: int = 128
    hidden_size: int = 4096
    rope_theta: float = 1000000.0
    dtype: torch.dtype = torch.bfloat16
    vocab_size: Optional[int] = None
    intermediate_size: Optional[int] = None
    num_layers: Optional[int] = None

    @classmethod
    def from_hf(
        cls,
        model_path: str,
        dtype: torch.dtype = torch.bfloat16
    ) -> "ModelConfig":
        """
        Load model parameters from HuggingFace configuration.
        
        Args:
            model_path: Model path
            dtype: Data type
            
        Returns:
            ModelConfig object
        """
        from transformers import AutoConfig
        
        hf_config = AutoConfig.from_pretrained(model_path)
        
        hidden_size = getattr(hf_config, 'hidden_size', 4096)
        num_heads = getattr(hf_config, 'num_attention_heads', 32)
        num_key_value_heads = getattr(hf_config, 'num_key_value_heads', None)
        head_dim = getattr(hf_config, 'head_dim', hidden_size // num_heads)
        vocab_size = getattr(hf_config, 'vocab_size', 32000)
        intermediate_size = getattr(hf_config, 'intermediate_size', None)
        num_layers = getattr(hf_config, 'num_hidden_layers', None)
        rope_theta = getattr(hf_config, 'rope_theta', 1000000.0)
        
        if num_key_value_heads is None:
            num_key_value_heads = num_heads
        
        return cls(
            num_heads=num_heads,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
            hidden_size=hidden_size,
            rope_theta=rope_theta,
            dtype=dtype,
            vocab_size=vocab_size,
            intermediate_size=intermediate_size,
            num_layers=num_layers,
        )

    def to_dict(self) -> dict:
        return {
            "num_heads": self.num_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "head_dim": self.head_dim,
            "hidden_size": self.hidden_size,
            "rope_theta": self.rope_theta,
            "dtype": self.dtype,
            "vocab_size": self.vocab_size,
            "intermediate_size": self.intermediate_size,
            "num_layers": self.num_layers,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ModelConfig":
        return cls(**{k: v for k, v in data.items() if k in cls.__annotations__})


@dataclass(frozen=True)
class CacheConfig:
    num_blocks: int = 1000
    block_size: int = 16
    device: str = "cuda"

    def to_dict(self) -> dict:
        return {
            "num_blocks": self.num_blocks,
            "block_size": self.block_size,
            "device": self.device,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CacheConfig":
        return cls(**{k: v for k, v in data.items() if k in cls.__annotations__})


@dataclass(frozen=True)
class SchedulerConfig:
    max_num_seqs: int = 256

    def to_dict(self) -> dict:
        return {
            "max_num_seqs": self.max_num_seqs,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SchedulerConfig":
        return cls(**{k: v for k, v in data.items() if k in cls.__annotations__})


@dataclass(frozen=True)
class EngineArgs:
    model_path: str
    device: str = "cuda"
    block_size: int = 16
    num_blocks: int = 1000
    max_num_seqs: int = 256
    dtype: torch.dtype = torch.bfloat16
    attention_backend: str = "flashinfer"

    def __post_init__(self):
        if not Path(self.model_path).exists():
            raise ValueError(f"Model path does not exist: {self.model_path}")

    def create_engine_configs(self):
        """Create engine configs from EngineArgs."""
        from transformers import AutoConfig
        
        hf_config = AutoConfig.from_pretrained(self.model_path)
        
        hidden_size = getattr(hf_config, 'hidden_size', 4096)
        num_heads = getattr(hf_config, 'num_attention_heads', 32)
        num_key_value_heads = getattr(hf_config, 'num_key_value_heads', None)
        head_dim = getattr(hf_config, 'head_dim', hidden_size // num_heads)
        vocab_size = getattr(hf_config, 'vocab_size', 32000)
        intermediate_size = getattr(hf_config, 'intermediate_size', None)
        num_layers = getattr(hf_config, 'num_hidden_layers', None)
        rope_theta = getattr(hf_config, 'rope_theta', 1000000.0)
        
        if num_key_value_heads is None:
            num_key_value_heads = num_heads
        
        model_config = ModelConfig(
            num_heads=num_heads,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
            hidden_size=hidden_size,
            rope_theta=rope_theta,
            dtype=self.dtype,
            vocab_size=vocab_size,
            intermediate_size=intermediate_size,
            num_layers=num_layers,
        )
        
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
            "dtype": self.dtype,
            "attention_backend": self.attention_backend,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "EngineArgs":
        return cls(**{k: v for k, v in data.items() if k in cls.__annotations__})