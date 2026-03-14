"""
Qwen3 model implementation using the decoupled architecture.

This module demonstrates the clean separation between:
- Generic layers (layers.py): RMSNorm, Embedding, Linear
- Model-specific implementations (model_specific/qwen3/): Qwen3Attention, Qwen3MLP
- Backend (backends/): FlashInfer for efficient attention computation
"""

from typing import Optional, List, Tuple
import torch
import torch.nn as nn

from ..layers_utils import Embedding
from ..backends import AttentionMetadata, FlashInferBackend, TorchBackend
from ..model_specific.qwen3.attention import Qwen3DecoderLayer


class Qwen3Model(nn.Module):
    """
    Qwen3 model with decoupled architecture.
    
    This model uses:
    - Generic layers from layers.py (Embedding, RMSNorm)
    - Model-specific attention from model_specific/qwen3/attention.py
    - FlashInfer backend from backends/flashinfer_backend.py
    
    This separation allows:
    - Clean testing of generic components
    - Easy addition of new models with different architectures
    - Backend swapping without changing model code
    """
    
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        num_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        intermediate_size: int,
        num_layers: int,
        attention_backend_type: str,
        dtype: torch.dtype,
        device: str,
        kv_cache_pool: torch.Tensor,
        rope_theta: float,
        block_size: int,
    ):
        """
        Initialize Qwen3 model.
        
        Args:
            vocab_size: Size of vocabulary (e.g., 151936 for Qwen3)
            hidden_size: Hidden size (e.g., 1024 for Qwen3-0.6B)
            num_heads: Number of attention heads (e.g., 16)
            num_key_value_heads: Number of key/value heads for GQA (e.g., 8)
            head_dim: Dimension of each attention head (e.g., 128)
            intermediate_size: MLP intermediate size (e.g., 3072)
            num_layers: Number of decoder layers (e.g., 28)
            attention_backend_type: Type of attention backend ("flashinfer")
            dtype: Data type for computations
            device: Computing device
            kv_cache_pool: Pre-allocated KV cache pool from BlockManager
            rope_theta: Base for RoPE rotary embeddings (default: 1M for Qwen3)
            block_size: Size of each cache block (must match BlockManager)
        """
        super().__init__()
        
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rope_theta = rope_theta
        self.block_size = block_size
        
        # Token embedding (generic layer)
        self.embed_tokens = Embedding(
            num_embeddings=vocab_size,
            embedding_dim=hidden_size,
            device=device,
            dtype=dtype
        )
        
        # Initialize attention backend
        if attention_backend_type == "flashinfer":
            self.attention_backend = FlashInferBackend(
                num_heads=num_heads,
                head_dim=head_dim,
                kv_cache_pool=kv_cache_pool,
                num_key_value_heads=num_key_value_heads,
                page_size=block_size,
                dtype=dtype,
                device=device
            )
        elif attention_backend_type == "torch":
            self.attention_backend = TorchBackend(
                num_heads=num_heads,
                head_dim=head_dim,
                kv_cache_pool=kv_cache_pool,
                num_key_value_heads=num_key_value_heads,
                page_size=block_size,
                dtype=dtype,
                device=device
            )
        else:
            raise ValueError(f"Unsupported attention backend: {attention_backend_type}")
        
        # Decoder layers (model-specific)
        self.layers = nn.ModuleList([
            Qwen3DecoderLayer(
                hidden_size=hidden_size,
                num_heads=num_heads,
                head_dim=head_dim,
                intermediate_size=intermediate_size,
                attention_backend=self.attention_backend,
                layer_idx=layer_idx,
                num_key_value_heads=num_key_value_heads,
                rope_theta=rope_theta,
                rms_norm_eps=1e-6,
                bias=False,
                device=device,
                dtype=dtype
            )
            for layer_idx in range(num_layers)
        ])
        
        # Final layer norm (generic)
        self.norm = nn.RMSNorm(hidden_size, eps=1e-6, device=device, dtype=dtype)
        
        # LM Head
        self.lm_head = nn.Linear(
            hidden_size,
            vocab_size,
            bias=False,
            device=device,
            dtype=dtype
        )
    
    def forward(
        self,
        input_ids: torch.Tensor,
        metadata: Optional[AttentionMetadata] = None,
    ) -> torch.Tensor:
        """
        Forward pass of Qwen3 model following official Transformers structure.
        
        Args:
            input_ids: Input token IDs (flattened: [total_tokens])
            attention_mask: Attention mask (optional)
            position_ids: Position IDs (optional)
            past_key_values: Past KV cache (optional)
            use_cache: Whether to return KV cache for generation
            cache_position: Cache position tensor (optional)
            position_embeddings: Precomputed position embeddings (optional)
            metadata: Attention metadata for paged attention (required for our backend)
            return_debug_info: Whether to return intermediate debug information
            debug_layer_idx: Which layer to return debug info from (if return_debug_info=True)
            **kwargs: Additional arguments
            
        Returns:
            Logits tensor of shape [total_tokens, vocab_size], or tuple with debug info
        """
        # Plan attention computation once for the entire batch (if metadata provided and backend supports planning)
        if metadata is not None and hasattr(self.attention_backend, 'plan'):
            self.attention_backend.plan(metadata)
        
        # Embedding lookup
        hidden_states = self.embed_tokens(input_ids)
        
        # Pass through decoder layers following official structure
        layer_debug_info = None
        for idx, layer in enumerate(self.layers):
            residual = hidden_states
            # Input layer norm
            hidden_states = layer.input_layernorm(hidden_states)

            hidden_states = layer.self_attn(hidden_states, metadata)

            # Add residual connection after attention
            hidden_states = residual + hidden_states

            # Post-attention layer norm and MLP with residual connection
            residual = hidden_states
            hidden_states = layer.post_attention_layernorm(hidden_states)
            hidden_states = layer.mlp(hidden_states)
            hidden_states = residual + hidden_states
        
        # Final layer norm
        hidden_states = self.norm(hidden_states)
        
        # LM head projection
        logits = self.lm_head(hidden_states)

        return logits


class Qwen3ForCausalLM(nn.Module):
    """
    Qwen3 model for causal language modeling.
    
    Wraps the base model with LM head for easy generation.
    """
    
    def __init__(self, model: Qwen3Model):
        """
        Initialize with a Qwen3Model.
        
        Args:
            model: Qwen3Model instance
        """
        super().__init__()
        self.model = model
        self.lm_head = model.lm_head
    
    def forward(
        self,
        input_ids: torch.Tensor,
        metadata: AttentionMetadata
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            input_ids: Input token IDs
            metadata: Attention metadata
            
        Returns:
            Logits tensor
        """
        return self.model(input_ids, metadata)