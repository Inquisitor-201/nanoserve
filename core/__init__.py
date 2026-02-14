"""
Core package for the model architecture.
Contains models, layers, backends, and utilities for efficient inference.
"""

from .backends import AttentionMetadata, FlashInferBackend
from .layers import Attention, MLP
from .models import Qwen3Model, Qwen3DecoderLayer
from .model_executor import ModelExecutor
from .block_manager import BlockManager
from .llm_service import LLMService

__all__ = [
    "AttentionMetadata",
    "FlashInferBackend", 
    "Attention",
    "MLP",
    "Qwen3Model",
    "Qwen3DecoderLayer",
    "ModelExecutor",
    "BlockManager",
    "LLMService"
]