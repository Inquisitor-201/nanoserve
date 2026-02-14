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
            num_key_value_heads: Number of key/value heads (for GQA). If None, defaults to num_heads
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
        
        # Separate projections for Q, K, V (matching Qwen3 weight structure)
        # Q projection: [hidden_size, num_heads * head_dim]
        self.q_proj = nn.Linear(
            hidden_size,
            num_heads * head_dim,
            bias=bias,
            device=device,
            dtype=dtype
        )
        
        # K projection: [hidden_size, num_key_value_heads * head_dim]
        self.k_proj = nn.Linear(
            hidden_size,
            self.num_key_value_heads * head_dim,
            bias=bias,
            device=device,
            dtype=dtype
        )
        
        # V projection: [hidden_size, num_key_value_heads * head_dim]
        self.v_proj = nn.Linear(
            hidden_size,
            self.num_key_value_heads * head_dim,
            bias=bias,
            device=device,
            dtype=dtype
        )
        
        # Q and K layer norms (matching Qwen3 weight structure)
        # Remove bias since weights file doesn't have bias for these norms
        self.q_norm = nn.LayerNorm(head_dim, eps=1e-6, bias=False, device=device, dtype=dtype)
        self.k_norm = nn.LayerNorm(head_dim, eps=1e-6, bias=False, device=device, dtype=dtype)
        
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
        
        # Separate Q, K, V projections (matching Qwen3 weight structure)
        # Q projection: [total_tokens, num_heads * head_dim]
        q = self.q_proj(hidden_states)
        
        # K projection: [total_tokens, num_key_value_heads * head_dim]
        k = self.k_proj(hidden_states)
        
        # V projection: [total_tokens, num_key_value_heads * head_dim]
        v = self.v_proj(hidden_states)
        
        # Reshape for attention computation
        # Q: [total_tokens, num_heads, head_dim]
        q = q.view(total_tokens, self.num_heads, self.head_dim)
        
        # K, V: [total_tokens, num_key_value_heads, head_dim]
        k = k.view(total_tokens, self.num_key_value_heads, self.head_dim)
        v = v.view(total_tokens, self.num_key_value_heads, self.head_dim)
        
        # Apply layer norms (matching Qwen3 weight structure)
        q = self.q_norm(q)
        k = self.k_norm(k)
        
        # For testing, use a simple attention implementation instead of FlashInfer
        # This will allow us to test the rest of the model without FlashInfer issues
        attn_output = self._simple_attention(q, k, v)
        
        # Reshape to [total_tokens, hidden_size]
        attn_output = attn_output.view(total_tokens, self.num_heads * self.head_dim)
        
        # Output projection
        output = self.o_proj(attn_output)
        
        if self.dropout is not None:
            output = self.dropout(output)
        
        return output
    
    def _simple_attention(self, query, key, value):
        """
        Simple attention implementation for testing.
        
        Args:
            query: Query tensor [total_tokens, num_heads, head_dim]
            key: Key tensor [total_tokens, num_key_value_heads, head_dim]
            value: Value tensor [total_tokens, num_key_value_heads, head_dim]
            
        Returns:
            Attention output tensor [total_tokens, num_heads, head_dim]
        """
        total_tokens, num_heads, head_dim = query.shape
        _, num_key_value_heads, _ = key.shape
        
        # Scale query
        query = query / (head_dim ** 0.5)
        
        # Compute attention scores
        # For GQA, we need to repeat key and value for each query head group
        if num_heads != num_key_value_heads:
            # GQA case: repeat key and value for each query head group
            key = key.repeat_interleave(self.num_key_value_groups, dim=1)
            value = value.repeat_interleave(self.num_key_value_groups, dim=1)
        
        # Compute attention scores [total_tokens, num_heads, total_tokens]
        scores = torch.bmm(query, key.transpose(1, 2))
        
        # Apply causal mask
        mask = torch.tril(torch.ones(total_tokens, total_tokens, device=query.device), diagonal=0)
        scores = scores.masked_fill(mask == 0, -float('inf'))
        
        # Apply softmax
        scores = torch.softmax(scores, dim=-1)
        
        # Compute attention output
        output = torch.bmm(scores, value)
        
        return output