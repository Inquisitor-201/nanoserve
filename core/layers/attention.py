"""
Attention layer implementation with pluggable backends.
Provides clean interface for attention computation with various backends.
"""

from typing import Optional
import torch
import torch.nn as nn
import logging
from functools import partial

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
        layer_idx: int,
        num_key_value_heads: Optional[int] = None,
        bias: bool = True,
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
            layer_idx: Layer index for KV cache pool access
            num_key_value_heads: Number of key/value heads (for GQA). If None, defaults to num_heads (MHA)
            bias: Whether to use bias in linear projections
            device: Computing device
            dtype: Data type
        """
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_key_value_heads = num_key_value_heads or num_heads
        self.layer_idx = layer_idx
        self.backend = backend
        
        if num_heads % self.num_key_value_heads != 0:
            raise ValueError(f"num_heads ({num_heads}) must be divisible by num_key_value_heads ({self.num_key_value_heads})")
        self.num_key_value_groups = num_heads // self.num_key_value_heads
        
        self.q_proj = nn.Linear(
            hidden_size,
            num_heads * head_dim,
            bias=bias,
            device=device,
            dtype=dtype
        )
        
        self.k_proj = nn.Linear(
            hidden_size,
            self.num_key_value_heads * head_dim,
            bias=bias,
            device=device,
            dtype=dtype
        )
        
        self.v_proj = nn.Linear(
            hidden_size,
            self.num_key_value_heads * head_dim,
            bias=bias,
            device=device,
            dtype=dtype
        )
        
        self.q_norm = nn.LayerNorm(head_dim, eps=1e-6, bias=False, device=device, dtype=dtype)
        self.k_norm = nn.LayerNorm(head_dim, eps=1e-6, bias=False, device=device, dtype=dtype)
        
        self.o_proj = nn.Linear(
            num_heads * head_dim,
            hidden_size,
            bias=bias,
            device=device,
            dtype=dtype
        )
        
        self._run_op = partial(
            self.backend.run,
            layer_idx=self.layer_idx
        )
        
        logger.info(f"Initialized GQA Attention (layer_idx={layer_idx}): "
                   f"hidden_size={hidden_size}, num_heads={num_heads}, "
                   f"num_key_value_heads={self.num_key_value_heads}, head_dim={head_dim}, "
                   f"num_key_value_groups={self.num_key_value_groups}")
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        metadata: AttentionMetadata
    ) -> torch.Tensor:
        """
        Forward pass of GQA attention layer with FlashInfer physical pool support.
        
        Args:
            hidden_states: Input hidden states (flattened: [total_tokens, hidden_size])
            metadata: Attention metadata for backend
            
        Returns:
            Output hidden states (flattened: [total_tokens, hidden_size])
        """
        total_tokens, hidden_size = hidden_states.shape
        
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        
        q = q.view(total_tokens, self.num_heads, self.head_dim)
        k = k.view(total_tokens, self.num_key_value_heads, self.head_dim)
        v = v.view(total_tokens, self.num_key_value_heads, self.head_dim)
        
        q = self.q_norm(q)
        k = self.k_norm(k)
        
        attn_output = self._run_op(
            query=q,
            key_states=k,
            value_states=v,
            metadata=metadata
        )
        
        attn_output = attn_output.view(total_tokens, self.num_heads * self.head_dim)
        
        output = self.o_proj(attn_output)
        
        return output