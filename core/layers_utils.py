"""
Generic utility layers that are model-agnostic.
Contains pure mathematical operations used across different models.
"""

from __future__ import annotations
from typing import Optional, TYPE_CHECKING
import torch
import torch.nn as nn
import torch.nn.functional as F

if TYPE_CHECKING:
    from .quantization import QuantizationConfig


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.
    
    A simplified version of LayerNorm that uses only the RMS of the inputs,
    which is more computationally efficient and works well for LLM pre-training.
    
    Args:
        hidden_size: Dimension of the input features
        eps: Small value for numerical stability
        device: Computing device
        dtype: Data type
    """
    
    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        device: str = None,
        dtype = None
    ):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size, device=device, dtype=dtype))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of RMSNorm.
        
        Args:
            x: Input tensor of shape (..., hidden_size)
            
        Returns:
            Normalized tensor of same shape as input
        """
        # Compute RMS along the last dimension
        
        # convert to float32 for stability during norm calculation, then back to original dtype
        original_dtype = x.dtype
        x = x.to(torch.float32)
        rms = x.norm(dim=-1, keepdim=True) / (x.shape[-1] ** 0.5)
        x = x / (rms + self.eps)
        return x.to(original_dtype) * self.weight


class Embedding(nn.Module):
    """
    Token embedding layer.
    
    A simple wrapper around nn.Embedding for consistency with the codebase.
    
    Args:
        num_embeddings: Size of the vocabulary
        embedding_dim: Dimension of the embeddings
        device: Computing device
        dtype: Data type
    """
    
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: str = None,
        dtype = None
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        
        self.weight = nn.Parameter(
            torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)
        )
        nn.init.normal_(self.weight, std=0.02)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Look up embeddings for input token IDs.
        
        Args:
            x: Input tensor of token IDs with any shape
            
        Returns:
            Embedding tensor of shape (..., embedding_dim)
        """
        return F.embedding(x, self.weight)


def Linear(
    in_features: int,
    out_features: int,
    *,
    quantization: Optional["QuantizationConfig"] = None,
    device: Optional[str] = None,
    dtype: Optional[torch.dtype] = None,
) -> nn.Module:
    """Build a linear layer — quantised (AWQ) when config is provided, plain otherwise."""
    if quantization is not None:
        from .quantization import AWQLinear
        return AWQLinear(
            in_features, out_features,
            device=device, dtype=dtype,
            bits=quantization.bits, group_size=quantization.group_size,
        )
    return nn.Linear(in_features, out_features, bias=False, device=device, dtype=dtype)