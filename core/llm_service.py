"""
LLM Service Layer - Unified interface for running Qwen3 model.
Provides high-level API for model loading, configuration, and inference.
"""

import logging
from typing import Dict, Any, Optional, List, Union
import torch
import json
import os
from pathlib import Path
from transformers import AutoTokenizer, AutoConfig

from .model_executor import ModelExecutor
from .backends import AttentionMetadata
from .block_manager import BlockManager


logger = logging.getLogger(__name__)


class LLMService:
    """
    LLM Service providing unified interface for Qwen3 model loading and inference.
    
    This service layer handles:
    - Model loading and configuration
    - KV cache management
    - Inference execution with FlashInfer backend
    """
    
    def __init__(self, device: str = "cuda"):
        """
        Initialize LLM Service.
        
        Args:
            device: Computing device ("cuda" or "cpu")
        """
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model_executor: Optional[ModelExecutor] = None
        self.block_manager: Optional[BlockManager] = None
        self.tokenizer: Optional[AutoTokenizer] = None
        self.model_config: Optional[Dict[str, Any]] = None
        
        logger.info(f"Initialized LLM Service on device: {self.device}")
    
    def load_model(
        self,
        model_path: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """
        Load Qwen3 model with specified configuration.
        
        Args:
            model_path: Path to the HuggingFace model directory
            config: Model configuration override
            **kwargs: Additional configuration parameters
            
        Returns:
            Model name
        """
        logger.info("Loading Qwen3 model")
        
        # Load from HuggingFace model if path provided
        if model_path:
            logger.info(f"Loading model from {model_path}")
            
            # Load tokenizer
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
            except Exception as e:
                logger.warning(f"Failed to load fast tokenizer: {e}, trying slow tokenizer")
                self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model config
            hf_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
            self.model_config = hf_config.to_dict()
            
            logger.info(f"Loaded model config: {hf_config.model_type}, vocab_size={hf_config.vocab_size}")
            
            # Build configuration from HuggingFace config
            model_config = {
                "model_name": hf_config.model_type,
                "vocab_size": hf_config.vocab_size,
                "hidden_size": hf_config.hidden_size,
                "num_heads": hf_config.num_attention_heads,
                "num_key_value_heads": getattr(hf_config, 'num_key_value_heads', hf_config.num_attention_heads),
                "head_dim": getattr(hf_config, 'head_dim', hf_config.hidden_size // hf_config.num_attention_heads),
                "intermediate_size": hf_config.intermediate_size,
                "num_layers": hf_config.num_hidden_layers,
                "attention_backend": "flashinfer",
                "dtype": torch.bfloat16,  # Use float16 for compatibility
                "device": self.device,
                "num_blocks": 1000,
                "block_size": 16,
            }
        else:
            # Raise exception
            raise ValueError("model_path must be provided when loading from HuggingFace model")

        # Merge configurations (user config overrides defaults)
        final_config = {**model_config, **(config or {}), **kwargs}
        
        # Pass model_path to executor if available
        if model_path:
            final_config['model_path'] = model_path
        
        # Initialize model executor
        self.model_executor = ModelExecutor(**final_config)
        
        # Extract block manager from executor
        self.block_manager = self.model_executor.block_manager
        
        logger.info("Successfully loaded Qwen3 model")
    
    def generate(
        self,
        prompts: Union[str, List[str]],
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_p: float = 0.9,
        **generate_kwargs
    ) -> List[str]:
        """
        Generate text from prompts using Qwen3 model.
        
        Args:
            prompts: Input prompt(s)
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling threshold
            
        Returns:
            Generated text(s)
        """
        if self.model_executor is None:
            raise RuntimeError("No model loaded. Call load_model() first.")
        
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not loaded. Call load_model() first.")

        # Convert single prompt to list
        if isinstance(prompts, str):
            prompts = [prompts]
        
        encoded = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)
            
        batch_size, max_seq_len = input_ids.shape
        seq_lengths = attention_mask.sum(dim=1).tolist()
            
        # UNPADDING: Remove padding tokens to create compact 1D tensor
        # This is critical for FlashInfer Paged KV Cache to avoid wasting memory
        flattened_input_ids = input_ids[attention_mask.bool()].contiguous()
            
        logger.info(f"Unpadding: batch_size={batch_size}, max_seq_len={max_seq_len}, total_tokens={flattened_input_ids.shape[0]}")

        # Create initial block tables for KV cache allocation (considering max possible length)
        block_tables = []
        for seq_len in seq_lengths:
            # Calculate total sequence length including max_new_tokens
            total_expected_length = seq_len + max_new_tokens
            # Calculate blocks needed for the total expected length
            blocks = self.block_manager.allocate_blocks(total_expected_length)
            if blocks is None:
                raise RuntimeError(f"Failed to allocate blocks for sequence length {seq_len} + {max_new_tokens} new tokens")
            block_tables.append(blocks)
        
        # Reset backend state for each new generation request to avoid cross-request contamination
        if hasattr(self.model_executor.model, 'attention_backend') and hasattr(self.model_executor.model.attention_backend, 'reset_state'):
            self.model_executor.model.attention_backend.reset_state()
        
        # Execute generation with unpadding
        generated_ids = self.model_executor.generate(
            input_ids=flattened_input_ids,  # Use flattened input (no padding)
            block_tables=block_tables,
            seq_lengths=seq_lengths,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p
        )
        
        # Reshape generated_ids to [batch_size, max_new_tokens]
        # Original order: [seq0_tok1, seq1_tok1, seq0_tok2, seq1_tok2, ...]
        # Need to convert to: [seq0_tok1, seq0_tok2, ..., seq1_tok1, seq1_tok2, ...]
        generated_ids_reshaped = generated_ids.view(max_new_tokens, batch_size).transpose(0, 1).contiguous()
            
        logger.info(f"Generated token IDs shape: {generated_ids_reshaped.shape}")
        logger.info(f"Generated token IDs: {generated_ids_reshaped}")
        # Detokenize generated IDs
        generated_texts = []
        for i in range(batch_size):
            # Decode each batch individually
            batch_generated_ids = generated_ids_reshaped[i]
            decoded_text = self.tokenizer.decode(batch_generated_ids, skip_special_tokens=True)
            generated_texts.append(decoded_text)
            
        logger.info(f"Decoded texts: {generated_texts}")

        return generated_texts
    
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
            Hidden states
        """
        if self.model_executor is None:
            raise RuntimeError("No model loaded. Call load_model() first.")
        
        return self.model_executor.execute_prefill(
            input_ids=input_ids,
            block_tables=block_tables,
            seq_lengths=seq_lengths
        )
    
    def execute_decode(
        self,
        input_ids: torch.Tensor,
        block_tables: List[List[int]],
        seq_lengths: List[int]
    ) -> torch.Tensor:
        """
        Execute decode phase.
        
        Args:
            input_ids: Input token IDs
            block_tables: Block tables for each sequence
            seq_lengths: Sequence lengths
            
        Returns:
            Hidden states
        """
        if self.model_executor is None:
            raise RuntimeError("No model loaded. Call load_model() first.")
        
        return self.model_executor.execute_decode(
            input_ids=input_ids,
            block_tables=block_tables,
            seq_lengths=seq_lengths
        )
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded model."""
        if self.model_executor is None:
            return {"status": "no_model_loaded"}
        
        info = self.model_executor.get_model_info()
        info["model_name"] = "qwen3"
        return info
    
    def __repr__(self) -> str:
        status = "qwen3_loaded" if self.model_executor else "no_model"
        return f"LLMService(device={self.device}, {status})"