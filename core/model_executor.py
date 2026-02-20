"""
Model executor for the architecture.
Manages model execution with proper attention backend integration.
"""

from typing import List, Optional, Dict, Any
import torch
import logging
import os
from pathlib import Path

from .backends import AttentionMetadata, FlashInferBackend
from .models import Qwen3Model
from .block_manager import BlockManager
from .model_loader import ModelLoader


logger = logging.getLogger(__name__)


class ModelExecutor:
    """
    Model executor for managing the complete inference pipeline.
    
    This executor ensures proper coordination between models and attention backends.
    """
    
    def __init__(
        self,
        model_name: str = "qwen3",
        vocab_size: int = 32000,
        hidden_size: int = 4096,
        num_heads: int = 32,
        num_key_value_heads: Optional[int] = None,
        head_dim: int = 128,
        intermediate_size: int = 11008,
        num_layers: int = 32,
        attention_backend: str = "flashinfer",
        dtype: torch.dtype = torch.bfloat16,
        device: str = "cuda",
        num_blocks: int = 1000,
        block_size: int = 16,
        model_path: Optional[str] = None,
        rope_theta: float = 1000000.0
    ):
        """
        Initialize model executor with GQA support.
        
        Args:
            model_name: Name of the model architecture
            vocab_size: Vocabulary size
            hidden_size: Hidden size of the model
            num_heads: Number of attention heads (Query heads)
            num_key_value_heads: Number of key/value heads (for GQA). If None, defaults to num_heads
            head_dim: Dimension of each attention head
            intermediate_size: Intermediate size for MLP
            num_layers: Number of decoder layers
            attention_backend: Attention backend type
            dtype: Data type for computations
            device: Computing device
            num_blocks: Number of KV cache blocks
            block_size: Size of each cache block
            model_path: Path to model weights for loading
            rope_theta: Base for RoPE rotary embeddings (default: 1M for Qwen3)
        """
        self.model_name = model_name
        self.device = device
        self.dtype = dtype
        self.block_size = block_size
        
        # Initialize block manager for KV cache management
        self.block_manager = BlockManager(
            num_blocks=num_blocks,
            num_layers=num_layers,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
            block_size=block_size,
            dtype=dtype,
            device=device
        )

        # Initialize model based on model name
        if model_name == "qwen3":
            self.model = Qwen3Model(
                vocab_size=vocab_size,
                hidden_size=hidden_size,
                num_heads=num_heads,
                num_key_value_heads=num_key_value_heads,
                head_dim=head_dim,
                intermediate_size=intermediate_size,
                num_layers=num_layers,
                attention_backend_type=attention_backend,
                dtype=dtype,
                device=device,
                kv_cache_pool=self.block_manager.kv_cache_pool,
                rope_theta=rope_theta
            )
            
            # Load model weights if path provided
            if model_path:
                ModelLoader.load_weights(self.model, model_path, dtype, device)
                
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        
        logger.info(f"Initialized ModelExecutor with {model_name} model")
    
    def execute_batch(
        self,
        input_ids: torch.Tensor,
        block_tables: List[List[int]],
        seq_lengths: List[int],
        is_prefill: bool
    ) -> torch.Tensor:
        """
        Execute prefill phase.
        
        Args:
            input_ids: Input token IDs
            block_tables: Block tables for each sequence (must be non-empty)
            seq_lengths: Sequence lengths
            
        Returns:
            Hidden states after prefill
        """
        # Create attention metadata for prefill
        metadata = AttentionMetadata.from_block_tables(
            block_tables=block_tables,
            seq_lengths=seq_lengths,
            is_prefill=is_prefill,
            page_size=self.block_size,
            device=self.device
        )
        
        # Execute model forward pass
        logits = self.model(input_ids, metadata)
        
        return logits

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
            input_ids: Initial input token IDs (flattened, no padding)
            block_tables: Block tables for each sequence
            seq_lengths: Initial sequence lengths
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling threshold
            
        Returns:
            Generated token IDs
        """
        batch_size = len(seq_lengths)
        current_input_ids = input_ids
        current_seq_lengths = seq_lengths.copy()
        generated_ids = []
        
        # Execute prefill phase
        logits = self.execute_prefill(current_input_ids, block_tables, current_seq_lengths)
        
        # Get initial generated tokens - select the last token logits for each sequence
        # In prefill phase, logits shape is [total_tokens, vocab_size]
        # We need to get the logits corresponding to the last token of each sequence
        last_token_positions = []
        cumulative_tokens = 0
        for seq_len in seq_lengths:
            # Last token position for this sequence is cumulative_tokens + seq_len - 1
            last_token_positions.append(cumulative_tokens + seq_len - 1)
            cumulative_tokens += seq_len
        
        # Extract logits for the last token of each sequence
        last_logits = logits[last_token_positions]
        next_tokens = self.sample(last_logits, temperature, top_p)
        generated_ids.append(next_tokens)
        
        # Update sequence lengths
        for i in range(batch_size):
            current_seq_lengths[i] += 1
        
        # Decode phase for remaining tokens
        for _ in range(max_new_tokens - 1):
            # Execute decode phase with new tokens
            logits = self.execute_decode(next_tokens, block_tables, current_seq_lengths)
            
            # Sample next tokens
            next_tokens = self.sample(logits, temperature, top_p)
            generated_ids.append(next_tokens)
            
            # Update sequence lengths
            for i in range(batch_size):
                current_seq_lengths[i] += 1
        
        # Concatenate generated tokens
        generated_ids = torch.cat(generated_ids, dim=0)
        return generated_ids
    
    def sample(
        self,
        logits: torch.Tensor,
        temperature: float,
        top_p: float
    ) -> torch.Tensor:
        """
        Sample next tokens from logits.
        
        Args:
            logits: Logits tensor of shape [batch_size, vocab_size]
            temperature: Sampling temperature
            top_p: Top-p sampling threshold
            
        Returns:
            Sampled token IDs of shape [batch_size]
        """
        # Greedy sampling when temperature is 0
        if temperature == 0.0:
            next_tokens = torch.argmax(logits, dim=-1)
            return next_tokens
        
        # Apply temperature
        if temperature > 0:
            logits = logits / temperature
        
        # Apply top-p filtering
        if top_p < 1.0:
            # Sort logits
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            # Calculate cumulative probabilities
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            # Remove tokens with cumulative probability above top_p
            sorted_indices_to_remove = cumulative_probs > top_p
            # Shift to keep at least one token
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            # Create mask
            mask = torch.zeros_like(logits, dtype=torch.bool)
            mask.scatter_(1, sorted_indices, sorted_indices_to_remove)
            logits[mask] = -float('inf')
        
        # Sample from the filtered distribution
        probs = torch.softmax(logits, dim=-1)
        next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
        
        return next_tokens
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            "model_name": self.model_name,
            "device": self.device,
            "dtype": str(self.dtype),
            "model": str(self.model),
            "block_manager": str(self.block_manager)
        }
    
    def __repr__(self) -> str:
        return f"ModelExecutor(model_name={self.model_name}, device={self.device})"