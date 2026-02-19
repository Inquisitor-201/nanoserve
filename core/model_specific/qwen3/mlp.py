"""
Qwen3-specific MLP implementation.
Contains the feed-forward network used in transformer layers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Qwen3MLP(nn.Module):
    """
    Qwen3 feed-forward network (MLP).
    
    This is the standard feed-forward network used in transformer layers,
    which expands the hidden size to intermediate_size and projects back.
    
    Args:
        hidden_size: Hidden size of the model
        intermediate_size: Size of the intermediate hidden layer
        bias: Whether to use bias in linear layers
        device: Computing device
        dtype: Data type
    """
    
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        bias: bool = False,
        device: str = None,
        dtype = None
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        
        # Gate projection
        self.gate_proj = nn.Linear(
            hidden_size,
            intermediate_size,
            bias=bias,
            device=device,
            dtype=dtype
        )
        
        # Up projection
        self.up_proj = nn.Linear(
            hidden_size,
            intermediate_size,
            bias=bias,
            device=device,
            dtype=dtype
        )
        
        # Down projection
        self.down_proj = nn.Linear(
            intermediate_size,
            hidden_size,
            bias=bias,
            device=device,
            dtype=dtype
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the MLP.
        
        Implements the SwiGLU activation function:
            down_proj(activation_gate(x) * up_proj(x))
        
        Args:
            x: Input tensor of shape (..., hidden_size)
            
        Returns:
            Output tensor of shape (..., hidden_size)
        """
        # SwiGLU: gate * up, then down projection
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))