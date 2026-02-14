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
        rms_norm_eps: float = 1e-6,
        dropout: float = 0.0,
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None
    ):
        """
        Initialize Qwen3 decoder layer.
        
        Args:
            hidden_size: Hidden size of the model
            num_heads: Number of attention heads
            head_dim: Dimension of each attention head
            intermediate_size: Intermediate size for MLP
            attention_backend: Attention backend instance
            layer_idx: Layer index
            rms_norm_eps: RMS norm epsilon
            dropout: Dropout probability
            device: Computing device
            dtype: Data type
        """
        super().__init__()
        
        self.hidden_size = hidden_size
        self.layer_idx = layer_idx
        
        # Self-attention with pluggable backend
        self.self_attn = Attention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            head_dim=head_dim,
            backend=attention_backend,
            dropout=dropout,
            device=device,
            dtype=dtype
        )
        
        # MLP
        self.mlp = MLP(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            dropout=dropout,
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
            hidden_states: Input hidden states
            metadata: Attention metadata
            
        Returns:
            Output hidden states
        """
        # Self-attention with residual connection
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        
        # Attention computation - backend handles all the complex attention logic
        attn_output = self.self_attn(
            hidden_states=hidden_states,
            metadata=metadata,
            layer_idx=self.layer_idx
        )
        hidden_states = residual + attn_output
        
        # MLP with residual connection
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        mlp_output = self.mlp(hidden_states)
        hidden_states = residual + mlp_output
        
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
        rms_norm_eps: float = 1e-6,
        dropout: float = 0.0,
        attention_backend_type: str = "flashinfer",
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None
    ):
        """
        Initialize Qwen3 model.
        
        Args:
            vocab_size: Vocabulary size
            hidden_size: Hidden size of the model
            num_heads: Number of attention heads
            head_dim: Dimension of each attention head
            intermediate_size: Intermediate size for MLP
            num_layers: Number of decoder layers
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
        
        # Initialize attention backend
        if attention_backend_type == "flashinfer":
            self.attention_backend = FlashInferBackend(
                num_heads=num_heads,
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
                rms_norm_eps=rms_norm_eps,
                dropout=dropout,
                device=device,
                dtype=dtype
            )
            for i in range(num_layers)
        ])
        
        # Final layer norm
        self.norm = nn.RMSNorm(hidden_size, eps=rms_norm_eps, device=device, dtype=dtype)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        metadata: AttentionMetadata
    ) -> torch.Tensor:
        """
        Forward pass of the model.
        
        Args:
            input_ids: Input token IDs
            metadata: Attention metadata
            
        Returns:
            Hidden states
        """
        # Plan attention computation once for the entire batch
        self.attention_backend.plan(metadata)
        
        # Embedding
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
        top_p: float = 0.9
    ) -> torch.Tensor:
        """
        Generate tokens using the model.
        
        Args:
            input_ids: Input token IDs
            block_tables: Block tables for each sequence
            seq_lengths: Sequence lengths
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling threshold
            
        Returns:
            Generated token IDs
        """
        # Create attention metadata
        metadata = AttentionMetadata.from_block_tables(
            block_tables=block_tables,
            seq_lengths=seq_lengths,
            is_prefill=True,
            device=input_ids.device
        )
        
        # Forward pass
        hidden_states = self.forward(input_ids, metadata)
        
        # Simple generation logic (in practice, you'd use more sophisticated sampling)
        # This is just a placeholder
        logits = torch.randn(input_ids.shape[0], self.vocab_size, device=input_ids.device)
        
        # Sample next tokens
        if temperature > 0:
            probs = torch.softmax(logits / temperature, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)
        else:
            next_tokens = torch.argmax(logits, dim=-1)
        
        return next_tokens
    
    def __repr__(self) -> str:
        return f"Qwen3Model(vocab_size={self.vocab_size}, hidden_size={self.hidden_size}, num_layers={self.num_layers})"