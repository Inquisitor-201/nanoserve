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
from .config import ModelConfig, CacheConfig
from .utils import StatsCollector, ProfileTimer


logger = logging.getLogger(__name__)


class ModelExecutor:
    """
    Model executor for managing the complete inference pipeline.
    
    This executor ensures proper coordination between models and attention backends.
    All model structure parameters are dynamically initialized based on ModelConfig,
    ensuring complete match with loaded weights.
    """
    
    def __init__(
        self,
        model_config: ModelConfig,
        cache_config: CacheConfig,
        kv_cache_pool: torch.Tensor,
        model_name: str = "qwen3",
        model_path: str = "",
        attention_backend: str = "flashinfer",
    ):
        """
        Initialize model executor with config-based parameters.
        
        Args:
            model_config: ModelConfig containing model structure parameters
            cache_config: CacheConfig containing cache management parameters
            kv_cache_pool: KV cache pool tensor
            model_name: Name of the model architecture
            model_path: Path to the model weights
            attention_backend: Attention backend type
        """
        self.model_name = model_name
        self.device = cache_config.device
        self.dtype = model_config.dtype
        self.block_size = cache_config.block_size
        self.model_config = model_config
        self.cache_config = cache_config
        self.model_path = model_path
        self.attention_backend = attention_backend
        self.kv_cache_pool = kv_cache_pool
        self.stats = StatsCollector()

        if model_name == "qwen3":
            vocab_size = model_config.vocab_size
            hidden_size = model_config.hidden_size
            num_heads = model_config.num_heads
            num_key_value_heads = model_config.num_key_value_heads
            head_dim = model_config.head_dim
            intermediate_size = model_config.intermediate_size
            num_layers = model_config.num_layers
            
            self.model = Qwen3Model(
                vocab_size=vocab_size,
                hidden_size=hidden_size,
                num_heads=num_heads,
                num_key_value_heads=num_key_value_heads,
                head_dim=head_dim,
                intermediate_size=intermediate_size,
                num_layers=num_layers,
                attention_backend_type=attention_backend,
                dtype=self.dtype,
                device=self.device,
                kv_cache_pool=self.kv_cache_pool,
                rope_theta=model_config.rope_theta,
                rms_norm_eps=model_config.rms_norm_eps,
                block_size=cache_config.block_size,
                quantization=model_config.quantization,
            )

            if model_path:
                ModelLoader.load_weights(
                    self.model,
                    model_path,
                    self.dtype,
                    self.device,
                )

        else:
            raise ValueError(f"Unsupported model: {model_name}")
        
        logger.debug(f"Initialized ModelExecutor with {model_name} model: "
                   f"hidden_size={model_config.hidden_size}, "
                   f"num_heads={model_config.num_heads}, "
                   f"num_layers={model_config.num_layers}")
    
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
        
        # Execute model forward pass with profiling
        with ProfileTimer(self.stats, is_prefill):
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
    def get_stats(self) -> Dict[str, Any]:
        """Get profiling statistics."""
        return self.stats.get_stats()
    def reset_stats(self) -> None:
        """Reset profiling statistics."""
        self.stats = StatsCollector()
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            "model_name": self.model_name,
            "hidden_size": self.model_config.hidden_size,
            "num_heads": self.model_config.num_heads,
            "num_layers": self.model_config.num_layers,
            "dtype": str(self.dtype),
            "device": str(self.device)
        }