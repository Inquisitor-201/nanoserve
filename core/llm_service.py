"""
LLM Service Layer - Unified interface for running Qwen3 model.
Provides high-level API for model loading, configuration, and inference.
"""

import logging
from typing import Dict, Any, Optional, List, Union
import torch
from transformers import AutoTokenizer

from .model_executor import ModelExecutor
from .block_manager import BlockManager
from .scheduler import Scheduler, Request
from .config import SamplingConfig, ModelConfig, EngineArgs


logger = logging.getLogger(__name__)


class LLMService:
    """
    LLM Service providing unified interface for Qwen3 model loading and inference.
    
    This service layer handles:
    - Model loading and configuration
    - KV cache management
    - Inference execution with FlashInfer backend
    
    The LLMService only receives EngineArgs in its constructor and internally
    initializes ModelConfig from HuggingFace config, distributing both to
    sub-components for a single source of truth architecture.
    """
    
    def __init__(
        self,
        engine_args: EngineArgs,
        model_config: Optional[ModelConfig] = None,
    ):
        """
        Initialize LLM Service with EngineArgs.
        
        Args:
            engine_args: EngineArgs containing model_path and resource allocation parameters
            model_config: Optional ModelConfig for overriding defaults (recommended to use EngineArgs only)
        """
        self.engine_args = engine_args
        self.device = engine_args.device if torch.cuda.is_available() else "cpu"
        self.model_executor: Optional[ModelExecutor] = None
        self.block_manager: Optional[BlockManager] = None
        self.scheduler: Optional[Scheduler] = None
        self.tokenizer: Optional[AutoTokenizer] = None
        self.model_config: Optional[ModelConfig] = None
        self._model_path = engine_args.model_path
        
        self._load_model()

    def _load_model(self) -> None:
        """
        Load Qwen3 model with config-based parameters.
        
        ModelConfig is automatically created from HuggingFace config.json,
        ensuring complete match with model weights.
        """
        logger.info(f"Loading Qwen3 model from {self._model_path}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self._model_path)
        
        self.eos_token_id = self.tokenizer.eos_token_id
        logger.info(f"EOS token ID: {self.eos_token_id}")
        
        # Create ModelConfig: HuggingFace provides model structure, EngineArgs provides resource params
        self.model_config = ModelConfig.from_hf_config(
            model_path=self._model_path,
            num_blocks=self.engine_args.num_blocks,
            page_size=self.engine_args.block_size,
            rope_theta=1000000.0,
            dtype=self.engine_args.dtype,
            override_config=model_config,
        )
        
        logger.info(f"ModelConfig created: hidden_size={self.model_config.hidden_size}, "
                   f"num_heads={self.model_config.num_heads}, "
                   f"num_layers={self.model_config.num_layers}")
        
        self.model_executor = ModelExecutor(
            model_config=self.model_config,
            engine_args=self.engine_args,
            model_name="qwen3"
        )
        
        self.block_manager = self.model_executor.block_manager
        
        self.scheduler = Scheduler(
            block_manager=self.block_manager,
            model_config=self.model_config,
            engine_args=self.engine_args
        )
        
        logger.info("Successfully loaded Qwen3 model with config-based initialization")
    
    def add_requests(
        self,
        prompts: Union[str, List[str]],
        sampling_config: SamplingConfig,
    ) -> List[str]:
        """
        Add requests to the scheduler.
        
        Args:
            prompts: Input prompt(s)
            sampling_config: SamplingConfig object
            
        Returns:
            List of request IDs
        """
        if self.model_executor is None:
            raise RuntimeError("No model loaded. Call load_model() first.")
        
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not loaded. Call load_model() first.")
        
        if self.scheduler is None:
            raise RuntimeError("Scheduler not initialized. Call load_model() first.")

        if isinstance(prompts, str):
            prompts = [prompts]
        
        encoded = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        input_ids_batch = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)
        
        batch_size = input_ids_batch.size(0)
        
        request_ids = []
        for i in range(batch_size):
            input_ids = input_ids_batch[i][attention_mask[i].bool()]
            req_id = self.scheduler.add_request(
                input_ids=input_ids,
                sampling_config=sampling_config,
                eos_token_id=self.eos_token_id
            )
            request_ids.append(req_id)
        
        return request_ids

    def main_loop(
        self,
        request_ids: List[str],
        sampling_config: SamplingConfig,
    ) -> List[str]:
        """
        Optimized main loop: orchestrates scheduling and execution.
        
        Args:
            request_ids: List of request IDs
            sampling_config: SamplingConfig object
        """
        step_id = 0
        while self.scheduler.has_unfinished_requests():
            sched_output = self.scheduler.schedule()
            
            if not sched_output.scheduled_requests:
                raise RuntimeError("No requests scheduled. This should not happen.")
            
            sampled_tokens = self._run_inference_step(
                sched_output, sampling_config
            )
            
            # 3. Update scheduler with results
            # Wrap tokens in tensors to match your optimized scheduler's expected input
            new_tokens = [t.view(1) for t in sampled_tokens]
            active_ids = [req.request_id for req in sched_output.scheduled_requests]
            
            self.scheduler.update_running_requests(new_tokens, active_ids)
            step_id += 1
        # 4. Final output collection
        return self._collect_results(request_ids)

    def _run_inference_step(self, sched_output, sampling_config: SamplingConfig):
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

        return self.model_executor.sample(
            logits,
            sampling_config.temperature,
            sampling_config.top_p
        )

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
        sampling_config: SamplingConfig,
    ) -> List[str]:
        """
        Generate text from prompts using Qwen3 model with scheduling.
        
        Args:
            prompts: Input prompt(s)
            sampling_config: SamplingConfig object
            
        Returns:
            Generated text(s)
        """
        request_ids = self.add_requests(
            prompts=prompts,
            sampling_config=sampling_config,
        )
        
        generated_texts = self.main_loop(
            request_ids=request_ids,
            sampling_config=sampling_config
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