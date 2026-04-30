"""
Qwen3-specific attention implementation.
Contains GQA configuration and model-specific attention logic.
"""

from typing import Optional
import torch
import torch.nn as nn
import logging
from functools import partial

import flashinfer

from ...backends import AttentionMetadata, FlashInferBackend, TorchBackend
from .mlp import Qwen3MLP

logger = logging.getLogger(__name__)


class Qwen3Attention(nn.Module):
    """
    Qwen3 attention layer with GQA support.
    
    This implementation is specific to Qwen3 model architecture:
    - Uses Grouped-Query Attention (GQA)
    - Integrates with FlashInfer backend for efficient attention computation
    - Applies FlashInfer RoPE for position encoding
    
    For different models (Llama, Mistral), create separate attention implementations
    in their respective model_specific directories.
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        attention_backend,
        layer_idx: int,
        num_key_value_heads: int,
        rope_theta,
        bias: bool = False,
        device: str = None,
        dtype = None
    ):
        """
        Initialize Qwen3 attention layer.
        
        Args:
            hidden_size: Hidden size of the model (e.g., 1024 for Qwen3-0.6B)
            num_heads: Number of attention heads for queries (e.g., 16)
            head_dim: Dimension of each attention head (e.g., 128)
            attention_backend: FlashInferBackend instance for attention computation
            layer_idx: Layer index for KV cache pool access
            num_key_value_heads: Number of key/value heads (e.g., 8 for GQA)
            rope_theta: Base for RoPE rotary embeddings (default: 1M for Qwen3)
            bias: Whether to use bias in linear projections
            device: Computing device
            dtype: Data type
        """
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_key_value_heads = num_key_value_heads
        self.layer_idx = layer_idx
        self.rope_theta = rope_theta
        self.backend = attention_backend
        
        # GQA validation
        if num_heads % num_key_value_heads != 0:
            raise ValueError(
                f"num_heads ({num_heads}) must be divisible by "
                f"num_key_value_heads ({num_key_value_heads})"
            )
        self.num_key_value_groups = num_heads // num_key_value_heads
        
        # QKV projections
        self.q_proj = nn.Linear(
            hidden_size,
            num_heads * head_dim,
            bias=bias,
            device=device,
            dtype=dtype
        )
        
        self.k_proj = nn.Linear(
            hidden_size,
            num_key_value_heads * head_dim,
            bias=bias,
            device=device,
            dtype=dtype
        )
        
        self.v_proj = nn.Linear(
            hidden_size,
            num_key_value_heads * head_dim,
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
        
        # QK normalization (specific to Qwen3)
        self.q_norm = nn.RMSNorm(head_dim, eps=1e-6, device=device, dtype=dtype)
        self.k_norm = nn.RMSNorm(head_dim, eps=1e-6, device=device, dtype=dtype)
        
        # Backend run operation
        self._run_op = partial(
            self.backend.run,
            layer_idx=self.layer_idx
        )
        
        logger.debug(
            f"Initialized Qwen3Attention (layer_idx={layer_idx}): "
            f"num_heads={num_heads}, num_kv_heads={num_key_value_heads}, "
            f"head_dim={head_dim}, num_groups={self.num_key_value_groups}, "
            f"rope_theta={rope_theta}"
        )
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        metadata: AttentionMetadata,
    ) -> torch.Tensor:
        """
        Forward pass of Qwen3 attention.
        
        Args:
            hidden_states: Input hidden states (flattened 2D: [total_tokens, hidden_size])
            metadata: Attention metadata for backend (required for actual attention computation)
            return_debug_info: Whether to return intermediate debug information
            
        Returns:
            Output hidden states of same shape, or tuple with debug info if return_debug_info=True
        """
        total_tokens, hidden_size = hidden_states.shape
        
        # QKV projections
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        
        # Reshape for attention
        q = q.view(total_tokens, self.num_heads, self.head_dim)
        k = k.view(total_tokens, self.num_key_value_heads, self.head_dim)
        v = v.view(total_tokens, self.num_key_value_heads, self.head_dim)
        
        # Apply QK normalization (first, before RoPE) - following actual Qwen3 implementation
        q_after_norm = self.q_norm(q)
        k_after_norm = self.k_norm(k)
        
        # Apply FlashInfer RoPE (after normalization)
        q_after_rope = q_after_norm.clone()
        k_after_rope = k_after_norm.clone()
        flashinfer.rope.apply_rope_pos_ids_inplace(
            q_after_rope,
            k_after_rope,
            pos_ids=metadata.positions,
            rotary_dim=None,
            interleave=False,
            rope_scale=1.0,
            rope_theta=self.rope_theta
        )
        
        # Run attention computation using backend
        # For GQA, FlashInfer handles the head grouping internally based on the parameters
        attn_output = self._run_op(
            query=q_after_rope,
            key_states=k_after_rope,  # Original KV heads (will be stored in cache)
            value_states=v,  # Original V heads (will be stored in cache)  
            metadata=metadata
        )
        
        # Reshape and project output
        attn_output = attn_output.view(total_tokens, self.num_heads * self.head_dim)
        output = self.o_proj(attn_output)

        return output


class Qwen3DecoderLayer(nn.Module):
    """
    Qwen3 decoder layer with decoupled attention backend.
    
    Combines attention and MLP with residual connections and layer norms.
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        intermediate_size: int,
        attention_backend: FlashInferBackend,
        layer_idx: int,
        num_key_value_heads: int,
        rope_theta: float = 1000000.0,
        rms_norm_eps: float = 1e-6,
        bias: bool = False,
        device: str = None,
        dtype = None
    ):
        """
        Initialize Qwen3 decoder layer.
        
        Args:
            hidden_size: Hidden size of the model
            num_heads: Number of query attention heads
            head_dim: Dimension of each attention head
            intermediate_size: MLP intermediate size
            attention_backend: Backend for attention computation
            layer_idx: Layer index
            num_key_value_heads: Number of key/value heads (GQA)
            rope_theta: Base for RoPE rotary embeddings
            rms_norm_eps: RMS norm epsilon
            bias: Whether to use bias in linear layers
            device: Computing device
            dtype: Data type
        """
        super().__init__()
        
        self.hidden_size = hidden_size
        self.layer_idx = layer_idx
        
        # Self-attention with GQA
        self.self_attn = Qwen3Attention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            head_dim=head_dim,
            num_key_value_heads=num_key_value_heads,
            attention_backend=attention_backend,
            layer_idx=layer_idx,
            rope_theta=rope_theta,
            bias=bias,
            device=device,
            dtype=dtype
        )
        
        # MLP
        self.mlp = Qwen3MLP(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            bias=bias,
            device=device,
            dtype=dtype
        )
        
        # Layer normalizations
        self.input_layernorm = nn.RMSNorm(hidden_size, eps=rms_norm_eps, device=device, dtype=dtype)
        self.post_attention_layernorm = nn.RMSNorm(hidden_size, eps=rms_norm_eps, device=device, dtype=dtype)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        metadata: AttentionMetadata,
    ) -> torch.Tensor:
        """
        Forward pass of decoder layer.
        
        Args:
            hidden_states: Input hidden states
            metadata: Attention metadata (optional, but required for attention computation)
            return_debug_info: Whether to return intermediate debug information
            
        Returns:
            Output hidden states
        """
        # Self-attention with residual
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, metadata)

        hidden_states = residual + hidden_states
        
        # MLP with residual
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states