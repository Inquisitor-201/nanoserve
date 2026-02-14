"""
Attention backends package.
Provides pluggable attention computation backends.
"""

from .metadata import AttentionMetadata
from .flashinfer_backend import FlashInferBackend

__all__ = ["AttentionMetadata", "FlashInferBackend"]