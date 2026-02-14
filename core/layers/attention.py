"""
Attention layer implementation with pluggable backends.
Provides clean interface for attention computation with various backends.
"""

from typing import Optional
import torch
import torch.nn as nn
import logging

from ..backends import AttentionMetadata


logger = logging.getLogger(__name__)


class Attention(nn.Module):
    """
    Attention layer with pluggable backend support.
    
    This module handles QKV projection and output projection, delegating
    the actual attention computation to configurable backends.
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        backend,
        bias: bool = True,
        dropout: float = 0.0,
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None
    ):
        """
        Initialize attention layer.
        
        Args:
            hidden_size: Hidden size of the model
            num_heads: Number of attention heads
            head_dim: Dimension of each attention head
            backend: Attention backend instance (e.g., FlashInferBackend)
            bias: Whether to use bias in linear projections
            dropout: Dropout probability
            device: Computing device
            dtype: Data type
        """
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.backend = backend
        
        # QKV projection
        self.qkv_proj = nn.Linear(
            hidden_size,
            3 * num_heads * head_dim,
            bias=bias,
            device=device,
            dtype=dtype
        )
        
        # Output projection
        self.o_proj = nn.Linear(
            num_heads * head_dim,
            hidden_size,
            bias=bias,
            device=device,
            dtype=dtype
        )
        
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        
        logger.info(f"Initialized Attention: hidden_size={hidden_size}, num_heads={num_heads}, head_dim={head_dim}")
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        metadata: AttentionMetadata,
        layer_idx: int = 0
    ) -> torch.Tensor:
        """
        Forward pass of attention layer.
        
        Args:
            hidden_states: Input hidden states
            metadata: Attention metadata for backend
            layer_idx: Layer index for multi-layer models
            
        Returns:
            Output hidden states
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # QKV projection
        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.chunk(3, dim=-1)
        
        # Reshape for attention computation
        # [batch_size, seq_len, num_heads * head_dim] -> [total_tokens, num_heads, head_dim]
        total_tokens = batch_size * seq_len
        q = q.view(total_tokens, self.num_heads, self.head_dim)
        k = k.view(total_tokens, self.num_heads, self.head_dim)
        v = v.view(total_tokens, self.num_heads, self.head_dim)
        
        # Backend attention computation
        # For now, we assume key_cache and value_cache are managed by the backend
        # In a real implementation, these would come from a KV cache manager
        # Here we pass k and v directly for simplicity
        attn_output = self.backend.run(
            query=q,
            key_cache=k,  # In real implementation, this would be from KV cache
            value_cache=v,  # In real implementation, this would be from KV cache
            layer_idx=layer_idx,
            metadata=metadata
        )
        
        # Reshape back
        attn_output = attn_output.view(batch_size, seq_len, self.hidden_size)
        
        # Output projection
        output = self.o_proj(attn_output)
        
        if self.dropout is not None:
            output = self.dropout(output)
        
        return output
    
    def extra_repr(self) -> str:
        return f"hidden_size={self.hidden_size}, num_heads={self.num_heads}, head_dim={self.head_dim}"