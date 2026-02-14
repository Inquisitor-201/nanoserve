"""
MLP (Multi-Layer Perceptron) implementation.
Standard feed-forward network used in transformer models.
"""

from typing import Optional
import torch
import torch.nn as nn


class MLP(nn.Module):
    """
    Multi-Layer Perceptron with GELU activation.
    
    Standard transformer MLP: Linear -> GELU -> Linear
    """
    
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: Optional[int] = None,
        bias: bool = True,
        dropout: float = 0.0,
        activation: str = "gelu",
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None
    ):
        """
        Initialize MLP.
        
        Args:
            hidden_size: Input and output hidden size
            intermediate_size: Intermediate size (defaults to 4 * hidden_size)
            bias: Whether to use bias in linear layers
            dropout: Dropout probability
            activation: Activation function ('gelu', 'relu', 'silu')
            device: Computing device
            dtype: Data type
        """
        super().__init__()
        
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size or 4 * hidden_size
        
        # Gate projection (first linear layer)
        self.gate_proj = nn.Linear(
            hidden_size,
            self.intermediate_size,
            bias=bias,
            device=device,
            dtype=dtype
        )
        
        # Up projection (second linear layer)
        self.up_proj = nn.Linear(
            hidden_size,
            self.intermediate_size,
            bias=bias,
            device=device,
            dtype=dtype
        )
        
        # Down projection (third linear layer)
        self.down_proj = nn.Linear(
            self.intermediate_size,
            hidden_size,
            bias=bias,
            device=device,
            dtype=dtype
        )
        
        # Activation function
        if activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "silu":
            self.activation = nn.SiLU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        
        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of MLP.
        
        Args:
            hidden_states: Input hidden states
            
        Returns:
            Output hidden states
        """
        # First linear layer + activation
        gate = self.gate_proj(hidden_states)
        up = self.up_proj(hidden_states)
        intermediate = self.activation(gate * up)
        
        # Third linear layer
        output = self.down_proj(intermediate)
        
        # Dropout
        if self.dropout is not None:
            output = self.dropout(output)
        
        return output
    
    def extra_repr(self) -> str:
        return f"hidden_size={self.hidden_size}, intermediate_size={self.intermediate_size}"