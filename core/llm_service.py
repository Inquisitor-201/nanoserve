"""
LLM Service Layer - Unified interface for running Qwen3 model.
Provides high-level API for model loading, configuration, and inference.
"""

import logging
from typing import Dict, Any, Optional, List, Union
import torch

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
        
        logger.info(f"Initialized LLM Service on device: {self.device}")
    
    def load_model(
        self,
        config: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> str:
        """
        Load Qwen3 model with specified configuration.
        
        Args:
            config: Model configuration override
            **kwargs: Additional configuration parameters
            
        Returns:
            Model name
        """
        logger.info("Loading Qwen3 model")
        
        # Default Qwen3 configuration
        default_config = {
            "model_name": "qwen3",
            "vocab_size": 32000,
            "hidden_size": 4096,
            "num_heads": 32,
            "head_dim": 128,
            "intermediate_size": 11008,
            "num_layers": 32,
            "attention_backend": "flashinfer",
            "dtype": torch.float16,
            "device": self.device,
            "num_blocks": 1000,
            "block_size": 16,
        }
        
        # Merge configurations (user config overrides defaults)
        final_config = {**default_config, **(config or {}), **kwargs}
        
        # Initialize model executor
        self.model_executor = ModelExecutor(**final_config)
        
        # Extract block manager from executor
        self.block_manager = self.model_executor.block_manager
        
        logger.info("Successfully loaded Qwen3 model")
        return "qwen3"
    
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
        
        # Convert single prompt to list
        if isinstance(prompts, str):
            prompts = [prompts]
        
        # Simple tokenization (in production, use proper tokenizer)
        batch_size = len(prompts)
        seq_lengths = [len(prompt) for prompt in prompts]
        max_seq_len = max(seq_lengths)
        
        # Create input IDs (placeholder - use real tokenizer in production)
        input_ids = torch.randint(
            0, 32000,  # vocab_size placeholder
            (batch_size, max_seq_len),
            device=self.device,
            dtype=torch.long
        )
        
        # Create block tables (simplified allocation)
        block_tables = []
        for seq_len in seq_lengths:
            num_blocks = (seq_len + self.block_manager.block_size - 1) // self.block_manager.block_size
            blocks = list(range(num_blocks))
            block_tables.append(blocks)
        
        # Execute generation
        generated_ids = self.model_executor.generate(
            input_ids=input_ids,
            block_tables=block_tables,
            seq_lengths=seq_lengths,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p
        )
        
        # Convert back to text (placeholder - use real detokenizer in production)
        generated_texts = [f"Generated text for prompt: '{prompt}'" for prompt in prompts]
        
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