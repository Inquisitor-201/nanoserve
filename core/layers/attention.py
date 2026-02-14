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
        # For FlashInfer, we need to format KV cache properly
        # Create a simple KV cache tensor in the expected format [num_pages, 2, page_size, num_heads, head_dim]
        # For now, use a simple placeholder format
        page_size = 16  # This should match the backend configuration
        
        # Reshape k and v to match FlashInfer expected format
        # FlashInfer expects: [num_pages, 2, page_size, num_heads, head_dim]
        # We create a simple 5D tensor for testing
        num_pages = (total_tokens + page_size - 1) // page_size
        
        # Pad to multiple of page_size
        padded_tokens = num_pages * page_size
        if padded_tokens > total_tokens:
            pad_size = padded_tokens - total_tokens
            k_padded = F.pad(k, (0, 0, 0, 0, 0, pad_size))
            v_padded = F.pad(v, (0, 0, 0, 0, 0, pad_size))
        else:
            k_padded = k
            v_padded = v
        
        # Reshape to [num_pages, page_size, num_heads, head_dim]
        k_reshaped = k_padded.view(num_pages, page_size, self.num_heads, self.head_dim)
        v_reshaped = v_padded.view(num_pages, page_size, self.num_heads, self.head_dim)
        
        # Stack K and V together: [num_pages, 2, page_size, num_heads, head_dim]
        kv_cache = torch.stack([k_reshaped, v_reshaped], dim=1)
        
        attn_output = self.backend.run(
            query=q,
            key_cache=kv_cache,
            value_cache=kv_cache,
            layer_idx=layer_idx,
            metadata=metadata
        )
        
        # FlashInfer returns [total_tokens, num_heads, head_dim]
        # Reshape to [total_tokens, hidden_size]
        attn_output = attn_output.view(total_tokens, self.num_heads * self.head_dim)
        
        # Reshape to [batch_size, seq_len, hidden_size]
        attn_output = attn_output.view(batch_size, seq_len, self.hidden_size)
        
        # Output projection
        output = self.o_proj(attn_output)
        
        if self.dropout is not None:
            output = self.dropout(output)
        
        return output
    
    def extra_repr(self) -> str:
        return f"hidden_size={self.hidden_size}, num_heads={self.num_heads}, head_dim={self.head_dim}"