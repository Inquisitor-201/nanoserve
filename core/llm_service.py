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
from .scheduler import Scheduler, Request


logger = logging.getLogger(__name__)


class LLMService:
    """
    LLM Service providing unified interface for Qwen3 model loading and inference.
    
    This service layer handles:
    - Model loading and configuration
    - KV cache management
    - Inference execution with FlashInfer backend
    """
    
    def __init__(self, model_path: str, config: Optional[Dict[str, Any]] = None, device: str = "cuda"):
        """
        Initialize LLM Service.
        
        Args:
            device: Computing device ("cuda" or "cpu")
        """
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model_executor: Optional[ModelExecutor] = None
        self.block_manager: Optional[BlockManager] = None
        self.scheduler: Optional[Scheduler] = None
        self.tokenizer: Optional[AutoTokenizer] = None
        self.model_config: Optional[Dict[str, Any]] = None
        
        self._load_model(model_path, config)

    def _load_model(
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
            
            # Get the EOS token ID from tokenizer
            self.eos_token_id = self.tokenizer.eos_token_id
            logger.info(f"EOS token ID: {self.eos_token_id}")
            
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
        
        # Initialize scheduler
        self.scheduler = Scheduler(self.block_manager)
        
        logger.info("Successfully loaded Qwen3 model")
    
    def add_requests(
        self,
        prompts: Union[str, List[str]],
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_p: float = 0.9,
        **generate_kwargs
    ) -> List[str]:
        """
        Add requests to the scheduler.
        
        Args:
            prompts: Input prompt(s)
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling threshold
            
        Returns:
            List of request IDs
        """
        if self.model_executor is None:
            raise RuntimeError("No model loaded. Call load_model() first.")
        
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not loaded. Call load_model() first.")
        
        if self.scheduler is None:
            raise RuntimeError("Scheduler not initialized. Call load_model() first.")

        # Convert single prompt to list
        if isinstance(prompts, str):
            prompts = [prompts]
        
        # Tokenize prompts
        encoded = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        input_ids_batch = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)
        
        batch_size = input_ids_batch.size(0)
        
        # Add each request to the scheduler
        request_ids = []
        for i in range(batch_size):
            input_ids = input_ids_batch[i][attention_mask[i].bool()]  # Remove padding
            req_id = self.scheduler.add_request(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                eos_token_id=self.eos_token_id
            )
            request_ids.append(req_id)
        
        return request_ids

    def main_loop(self, request_ids: List[str], temperature: float = 1.0, top_p: float = 0.9) -> List[str]:
        """
        Optimized main loop: orchestrates scheduling and execution.
        """
        step_id = 0
        while self.scheduler.has_unfinished_requests():
            # 1. Get next batch from scheduler
            sched_output = self.scheduler.schedule()
            
            if not sched_output.scheduled_requests:
                raise RuntimeError("No requests scheduled. This should not happen.")
            
            # 2. Execute inference (Prefill or Decode)
            # The scheduler already separates phases, so we just check the first request
            sampled_tokens = self._run_inference_step(
                sched_output, temperature, top_p
            )
            
            # 3. Update scheduler with results
            # Wrap tokens in tensors to match your optimized scheduler's expected input
            new_tokens = [t.view(1) for t in sampled_tokens]
            active_ids = [req.request_id for req in sched_output.scheduled_requests]
            
            self.scheduler.update_running_requests(new_tokens, active_ids)
            step_id += 1
        # 4. Final output collection
        return self._collect_results(request_ids)

    def _run_inference_step(self, sched_output, temp, top_p):
        """Unified inference step handling both Prefill and Decode."""
        is_prefill = sched_output.is_prefill
        
        logits = self.model_executor.execute_batch(
            input_ids=sched_output.input_ids,
            block_tables=sched_output.block_tables,
            seq_lengths=sched_output.seq_lengths,
            is_prefill=is_prefill
        )

        if is_prefill:
            indices = torch.cumsum(
                torch.tensor(sched_output.seq_lengths, device=self.device), dim=0
            ) - 1
            logits = logits[indices]

        return self.model_executor.sample(logits, temp, top_p)

    def _collect_results(self, request_ids: List[str]) -> List[str]:
        """Collect and decode results from completed requests."""
        results = []
        for req_id in request_ids:
            req = self.scheduler.completed_requests.get(req_id)
            if req and req.generated_tokens:
                text = self.tokenizer.decode(req.generated_tokens, skip_special_tokens=True)
                results.append(text)
            else:
                results.append("Error: Request not found or empty")
        return results

    def generate(
        self,
        prompts: Union[str, List[str]],
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_p: float = 0.9,
        **generate_kwargs
    ) -> List[str]:
        """
        Generate text from prompts using Qwen3 model with scheduling.
        
        Args:
            prompts: Input prompt(s)
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling threshold
            
        Returns:
            Generated text(s)
        """
        # Add requests to scheduler
        request_ids = self.add_requests(
            prompts=prompts,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            **generate_kwargs
        )
        
        # Run main loop
        generated_texts = self.main_loop(
            request_ids=request_ids,
            temperature=temperature,
            top_p=top_p
        )
        
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