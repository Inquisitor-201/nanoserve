"""
Qwen3 model implementation demonstrating the decoupled architecture.
Shows how to use FlashInferBackend with the attention layer.
"""

from typing import Optional, List
import torch
import torch.nn as nn

from ..layers import Attention, MLP
from ..backends import AttentionMetadata, FlashInferBackend


class Qwen3DecoderLayer(nn.Module):
    """
    Qwen3 decoder layer with decoupled attention backend.
    
    This layer demonstrates the clean separation between model logic
    and attention computation backend.
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        intermediate_size: int,
        attention_backend,
        layer_idx: int,
        num_key_value_heads: Optional[int] = None,
        rms_norm_eps: float = 1e-6,
        dropout: float = 0.0,
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None
    ):
        """
        Initialize Qwen3 decoder layer with GQA support.
        
        Args:
            hidden_size: Hidden size of the model
            num_heads: Number of attention heads (Query heads)
            head_dim: Dimension of each attention head
            intermediate_size: Intermediate size for MLP
            attention_backend: Attention backend instance
            layer_idx: Layer index
            num_key_value_heads: Number of key/value heads (for GQA). If None, defaults to num_heads
            rms_norm_eps: RMS norm epsilon
            dropout: Dropout probability
            device: Computing device
            dtype: Data type
        """
        super().__init__()
        
        self.hidden_size = hidden_size
        self.layer_idx = layer_idx
        
        # Self-attention with pluggable backend and GQA support
        self.self_attn = Attention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            head_dim=head_dim,
            num_key_value_heads=num_key_value_heads,
            backend=attention_backend,
            dropout=dropout,
            bias=False,
            device=device,
            dtype=dtype
        )
        
        # MLP
        self.mlp = MLP(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            dropout=dropout,
            bias=False,
            device=device,
            dtype=dtype
        )
        
        # Layer normalization
        self.input_layernorm = nn.RMSNorm(hidden_size, eps=rms_norm_eps, device=device, dtype=dtype)
        self.post_attention_layernorm = nn.RMSNorm(hidden_size, eps=rms_norm_eps, device=device, dtype=dtype)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        metadata: AttentionMetadata
    ) -> torch.Tensor:
        """
        Forward pass of decoder layer.
        
        Args:
            hidden_states: Input hidden states (flattened 2D tensor for unpadding)
            metadata: Attention metadata
            
        Returns:
            Output hidden states
        """
        # Self-attention with residual connection
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, metadata, layer_idx=self.layer_idx)
        hidden_states = residual + hidden_states
        
        # MLP with residual connection
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states


class Qwen3Model(nn.Module):
    """
    Qwen3 model with decoupled architecture.
    
    This model demonstrates how to use the attention backend in a complete model.
    The model is responsible for mathematical transformations, while the backend
    handles attention computation details.
    """
    
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        intermediate_size: int,
        num_layers: int,
        num_key_value_heads: Optional[int] = None,
        rms_norm_eps: float = 1e-6,
        dropout: float = 0.0,
        attention_backend_type: str = "flashinfer",
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None
    ):
        """
        Initialize Qwen3 model with GQA support.
        
        Args:
            vocab_size: Vocabulary size
            hidden_size: Hidden size of the model
            num_heads: Number of attention heads (Query heads)
            head_dim: Dimension of each attention head
            intermediate_size: Intermediate size for MLP
            num_layers: Number of decoder layers
            num_key_value_heads: Number of key/value heads (for GQA). If None, defaults to num_heads
            rms_norm_eps: RMS norm epsilon
            dropout: Dropout probability
            attention_backend_type: Type of attention backend
            device: Computing device
            dtype: Data type
        """
        super().__init__()
        
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_key_value_heads = num_key_value_heads or num_heads
        
        # Initialize attention backend - only FlashInfer supported
        if attention_backend_type == "flashinfer":
            self.attention_backend = FlashInferBackend(
                num_heads=num_heads,
                num_key_value_heads=self.num_key_value_heads,
                head_dim=head_dim,
                device=device,
                dtype=dtype
            )
        else:
            raise ValueError(f"Unsupported attention backend: {attention_backend_type}")
        
        # Embedding
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size, device=device, dtype=dtype)
        
        # Decoder layers
        self.layers = nn.ModuleList([
            Qwen3DecoderLayer(
                hidden_size=hidden_size,
                num_heads=num_heads,
                head_dim=head_dim,
                intermediate_size=intermediate_size,
                attention_backend=self.attention_backend,
                layer_idx=i,
                num_key_value_heads=self.num_key_value_heads,
                rms_norm_eps=rms_norm_eps,
                dropout=dropout,
                device=device,
                dtype=dtype
            )
            for i in range(num_layers)
        ])
        
        # Final layer norm
        self.norm = nn.RMSNorm(hidden_size, eps=rms_norm_eps, device=device, dtype=dtype)
        
        # Language model head
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False, device=device, dtype=dtype)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        metadata: AttentionMetadata
    ) -> torch.Tensor:
        """
        Forward pass of the model.
        
        Args:
            input_ids: Input token IDs (flattened 1D tensor for unpadding)
            metadata: Attention metadata
            
        Returns:
            Hidden states (flattened 2D tensor)
        """
        # Plan attention computation once for the entire batch
        self.attention_backend.plan(metadata)
        
        # Embedding - input_ids is already flattened (total_tokens,)
        # embed_tokens expects 1D tensor and returns 2D tensor (total_tokens, hidden_size)
        hidden_states = self.embed_tokens(input_ids)
        
        # Pass through decoder layers
        for layer in self.layers:
            hidden_states = layer(hidden_states, metadata)
        
        # Final layer norm
        hidden_states = self.norm(hidden_states)
        
        return hidden_states
    
    def generate(
        self,
        input_ids: torch.Tensor,
        block_tables: List[List[int]],
        seq_lengths: List[int],
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_p: float = 0.9,
        qo_indptr: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Generate tokens using the model.
        
        Args:
            input_ids: Initial input token IDs (flattened, no padding)
            block_tables: Block tables for each sequence
            seq_lengths: Initial sequence lengths
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling threshold
            qo_indptr: Query/Output indices pointer for FlashInfer (required for unpadding)
            
        Returns:
            Generated token IDs
        """
        # This is a simplified implementation for demonstration
        # In practice, you would implement proper token-by-token generation
        # with KV cache management
        
        # For now, we'll just return the input IDs as a placeholder
        # This should be replaced with actual generation logic
        return input_ids