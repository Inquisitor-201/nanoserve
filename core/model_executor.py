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
        dtype: torch.dtype = torch.float16,
        device: str = "cuda",
        num_blocks: int = 1000,
        block_size: int = 16,
        model_path: Optional[str] = None
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
        """
        self.model_name = model_name
        self.device = device
        self.dtype = dtype
        
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
                kv_cache_pool=self.block_manager.kv_cache_pool
            )
            
            # Load model weights if path provided
            if model_path:
                ModelLoader.load_weights(self.model, model_path, dtype, device)
                
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        
        logger.info(f"Initialized ModelExecutor with {model_name} model")
    
    def execute_prefill(
        self,
        input_ids: torch.Tensor,
        block_tables: List[List[int]],
        seq_lengths: List[int]
    ) -> torch.Tensor:
        """
        Execute prefill phase.
        
        Args:
            input_ids: Input token IDs
            block_tables: Block tables for each sequence
            seq_lengths: Sequence lengths
            
        Returns:
            Hidden states after prefill
        """
        # Create attention metadata for prefill
        metadata = AttentionMetadata.from_block_tables(
            block_tables=block_tables,
            seq_lengths=seq_lengths,
            is_prefill=True,
            device=self.device
        )
        
        # Execute model forward pass
        hidden_states = self.model(input_ids, metadata)
        
        return hidden_states
    
    def execute_decode(
        self,
        input_ids: torch.Tensor,
        block_tables: List[List[int]],
        seq_lengths: List[int]
    ) -> torch.Tensor:
        """
        Execute decode phase.
        
        Args:
            input_ids: Input token IDs (one per sequence)
            block_tables: Block tables for each sequence
            seq_lengths: Sequence lengths
            
        Returns:
            Hidden states after decode
        """
        # Create attention metadata for decode
        metadata = AttentionMetadata.from_block_tables(
            block_tables=block_tables,
            seq_lengths=seq_lengths,
            is_prefill=False,
            device=self.device
        )
        
        # Execute model forward pass
        hidden_states = self.model(input_ids, metadata)
        
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
        return self.model.generate(
            input_ids=input_ids,
            block_tables=block_tables,
            seq_lengths=seq_lengths,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            qo_indptr=qo_indptr
        )
    
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