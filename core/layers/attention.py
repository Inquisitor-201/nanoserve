"""
Attention layer implementation with pluggable backends.
Provides clean interface for attention computation with various backends.
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
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
        num_key_value_heads: Optional[int] = None,
        bias: bool = True,
        dropout: float = 0.0,
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None
    ):
        """
        Initialize attention layer with GQA support.
        
        Args:
            hidden_size: Hidden size of the model
            num_heads: Number of attention heads (Query heads)
            head_dim: Dimension of each attention head
            backend: Attention backend instance (e.g., FlashInferBackend)
            num_key_value_heads: Number of key/value heads (for GQA). If None, defaults to num_heads (MHA)
            bias: Whether to use bias in linear projections
            dropout: Dropout probability
            device: Computing device
            dtype: Data type
        """
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_key_value_heads = num_key_value_heads or num_heads
        self.backend = backend
        
        # Validate GQA configuration
        if num_heads % self.num_key_value_heads != 0:
            raise ValueError(f"num_heads ({num_heads}) must be divisible by num_key_value_heads ({self.num_key_value_heads})")
        self.num_key_value_groups = num_heads // self.num_key_value_heads
        
        # GQA: Separate projections for Q and KV
        # Q projection: [hidden_size, num_heads * head_dim]
        self.q_proj = nn.Linear(
            hidden_size,
            num_heads * head_dim,
            bias=bias,
            device=device,
            dtype=dtype
        )
        
        # KV projection: [hidden_size, num_key_value_heads * head_dim * 2]
        self.kv_proj = nn.Linear(
            hidden_size,
            self.num_key_value_heads * head_dim * 2,
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
        
        logger.info(f"Initialized GQA Attention: hidden_size={hidden_size}, num_heads={num_heads}, "
                   f"num_key_value_heads={self.num_key_value_heads}, head_dim={head_dim}, "
                   f"num_key_value_groups={self.num_key_value_groups}")
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        metadata: AttentionMetadata,
        layer_idx: int = 0
    ) -> torch.Tensor:
        """
        Forward pass of GQA attention layer with FlashInfer physical pool support.
        
        Args:
            hidden_states: Input hidden states (flattened: [total_tokens, hidden_size])
            metadata: Attention metadata for backend
            layer_idx: Layer index for multi-layer models
            
        Returns:
            Output hidden states (flattened: [total_tokens, hidden_size])
        """
        # For unpadding, hidden_states is already flattened: [total_tokens, hidden_size]
        total_tokens, hidden_size = hidden_states.shape
        
        # GQA: Separate Q and KV projections
        # Q projection: [total_tokens, num_heads * head_dim]
        q = self.q_proj(hidden_states)
        
        # KV projection: [total_tokens, num_key_value_heads * head_dim * 2]
        kv = self.kv_proj(hidden_states)
        k, v = kv.chunk(2, dim=-1)
        
        # Reshape for attention computation
        # Q: [total_tokens, num_heads, head_dim]
        q = q.view(total_tokens, self.num_heads, self.head_dim)
        
        # K, V: [total_tokens, num_key_value_heads, head_dim]
        k = k.view(total_tokens, self.num_key_value_heads, self.head_dim)
        v = v.view(total_tokens, self.num_key_value_heads, self.head_dim)
        
        # Apply RoPE (placeholder - to be implemented with actual position info)
        # TODO: Implement actual RoPE based on metadata.position_ids or similar
        # For now, we skip RoPE as it's not available in current metadata
        
        # Backend attention computation
        # FlashInfer will handle:
        # 1. Append K, V to physical KV cache pool (self.backend.kv_cache_pool)
        # 2. Execute FlashInfer attention computation
        attn_output = self.backend.run(
            query=q,
            key_cache=k,  # K for append to physical pool
            value_cache=v,  # V for append to physical pool
            layer_idx=layer_idx,
            metadata=metadata
        )
        
        # FlashInfer returns [total_tokens, num_heads, head_dim] for GQA
        # Reshape to [total_tokens, hidden_size]
        attn_output = attn_output.view(total_tokens, self.num_heads * self.head_dim)
        
        # Output projection
        output = self.o_proj(attn_output)
        
        if self.dropout is not None:
            output = self.dropout(output)
        
        return output
    
    def extra_repr(self) -> str:
        return f"hidden_size={self.hidden_size}, num_heads={self.num_heads}, head_dim={self.head_dim}"