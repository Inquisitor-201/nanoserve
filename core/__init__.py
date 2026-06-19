"""
Core package for the model architecture.
Contains models, layers, backends, and utilities for efficient inference.
"""

import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF",
    "expandable_segments:True,max_split_size_mb:512")

from .backends import AttentionMetadata, FlashInferBackend
from .layers_utils import RMSNorm, Embedding, Linear
from .models import Qwen3Model
from .model_executor import ModelExecutor
from .block_manager import BlockManager
from .llm_service import LLMService
from .config import SamplingConfig, ModelConfig

__all__ = [
    "AttentionMetadata",
    "FlashInferBackend",
    "RMSNorm",
    "Embedding",
    "Linear",
    "Qwen3Model",
    "ModelExecutor",
    "BlockManager",
    "LLMService",
    "SamplingConfig",
    "ModelConfig",
]