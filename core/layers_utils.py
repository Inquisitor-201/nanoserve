"""
Generic utility layers that are model-agnostic.
Contains pure mathematical operations used across different models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


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


class GELU(nn.Module):
    """
    Gaussian Error Linear Unit activation.
    
    Used in transformer feed-forward networks.
    """
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply GELU activation.
        
        Args:
            x: Input tensor
            
        Returns:
            Activated tensor
        """
        return F.gelu(x, approximate='tanh')


class Linear(nn.Module):
    """
    Generic linear layer.
    
    Wrapper around nn.Linear for consistency.
    
    Args:
        in_features: Input feature dimension
        out_features: Output feature dimension
        bias: Whether to include bias term
        device: Computing device
        dtype: Data type
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        device: str = None,
        dtype = None
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, device=device, dtype=dtype)
        )
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, device=device, dtype=dtype))
        else:
            self.register_parameter('bias', None)
        
        nn.init.kaiming_uniform_(self.weight, nonlinearity='linear')
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply linear transformation.
        
        Args:
            x: Input tensor of shape (..., in_features)
            
        Returns:
            Output tensor of shape (..., out_features)
        """
        x = F.linear(x, self.weight, self.bias)
        return x