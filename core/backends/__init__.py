"""
Attention backends package.
Provides pluggable attention computation backends.
"""

from .metadata import AttentionMetadata
from .flashinfer_backend import FlashInferBackend
from .torch_backend import TorchBackend

__all__ = ["AttentionMetadata", "FlashInferBackend", "TorchBackend"]