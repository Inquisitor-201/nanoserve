"""
Qwen3 model implementation.
"""

from typing import Optional
import torch
import torch.nn as nn

from ...layers_utils import Embedding
from ...backends import AttentionMetadata, FlashInferBackend, TorchBackend
from ...quantization import QuantizationConfig
from .attention import Qwen3DecoderLayer


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
        rms_norm_eps: float,
        block_size: int,
        quantization: Optional[QuantizationConfig] = None,
    ):
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
                rms_norm_eps=rms_norm_eps,
                device=device,
                dtype=dtype,
                quantization=quantization,
            )
            for layer_idx in range(num_layers)
        ])
        
        # Final layer norm (generic)
        self.norm = nn.RMSNorm(hidden_size, eps=rms_norm_eps, device=device, dtype=dtype)
        
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
            metadata: Attention metadata for paged attention

        Returns:
            Logits tensor of shape [total_tokens, vocab_size]
        """
        # Plan attention computation once for the entire batch (if metadata provided and backend supports planning)
        if metadata is not None and hasattr(self.attention_backend, 'plan'):
            self.attention_backend.plan(metadata)
        
        # Embedding lookup
        hidden_states = self.embed_tokens(input_ids)
        
        # Pass through decoder layers following official structure
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