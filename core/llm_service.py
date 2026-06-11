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
from .config import SamplingConfig, ModelConfig, EngineArgs, CacheConfig, SchedulerConfig
from .config import auto_calculate_num_blocks
from .utils import ContinuousBatchTimer


logger = logging.getLogger(__name__)


class LLMService:
    """
    NanoServe core service class.
    No longer responsible for "producing" configurations, only for "holding" configurations and driving components.
    """
    
    def __init__(
        self,
        model_config: ModelConfig,
        cache_config: CacheConfig,
        scheduler_config: SchedulerConfig,
        model_path: str,
        attention_backend: str = "flashinfer",
    ) -> None:
        # At this point, Configs are already fully determined "finished products"
        self.model_config = model_config
        self.cache_config = cache_config
        self.scheduler_config = scheduler_config
        self.model_path = model_path
        self.device = cache_config.device
        
        # 1. Initialize Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.eos_token_id = self.tokenizer.eos_token_id
        logger.info(f"EOS token ID: {self.eos_token_id}")
        
        # 2. Initialize BlockManager
        # BlockManager creates and allocates GPU memory pool here
        self.block_manager = BlockManager(
            model_config=model_config,
            cache_config=cache_config
        )
        
        # 3. Initialize ModelExecutor
        # Pass the BlockManager's pool to it
        self.model_executor = ModelExecutor(
            model_config=model_config,
            cache_config=cache_config,
            kv_cache_pool=self.block_manager.kv_cache_pool,
            model_path=model_path,
            attention_backend=attention_backend
        )
        
        # 4. Initialize Scheduler
        # Pass the BlockManager to it for request scheduling
        self.scheduler = Scheduler(scheduler_config, self.block_manager)
    @classmethod
    def from_engine_args(cls, engine_args: EngineArgs) -> "LLMService":
        """
        Factory method: This is the user's only entry point.
        Here we complete HF Config reading, parameter merging, and validation.
        """
        # 1. Core: Load and parse ModelConfig from HF
        # This handles all parts that "must be read from huggingface config"
        model_config = ModelConfig.from_hf(model_path=engine_args.model_path)

        # 2. Core: Determine num_blocks (auto-calculate if not specified)
        num_blocks = engine_args.num_blocks
        if num_blocks is None or num_blocks <= 0:
            num_blocks = auto_calculate_num_blocks(
                device=engine_args.device,
                dtype=model_config.dtype,
                block_size=engine_args.block_size,
                num_layers=model_config.num_layers,
                num_kv_heads=model_config.num_key_value_heads,
                head_dim=model_config.head_dim,
            )

        # 3. Core: Construct CacheConfig
        # This handles parts "passed in by the user (such as block_size)"
        cache_config = CacheConfig(
            num_blocks=num_blocks,
            block_size=engine_args.block_size,
            device=engine_args.device
        )

        # 3. Core: Construct SchedulerConfig
        scheduler_config = SchedulerConfig(
            max_num_seqs=engine_args.max_num_seqs
        )

        # 4. Final step: Package and pass to __init__
        return cls(
            model_config=model_config,
            cache_config=cache_config,
            scheduler_config=scheduler_config,
            model_path=engine_args.model_path,
            attention_backend=engine_args.attention_backend
        )
    
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
        # 4. Final output collection
        return self._collect_results(request_ids)

    def _run_inference_step(self, sched_output, sampling_config: SamplingConfig):
        """Unified inference step handling both Prefill and Decode."""
        is_prefill = sched_output.is_prefill
        
        # Use ContinuousBatchTimer for profiling
        with ContinuousBatchTimer(sched_output.scheduled_requests):
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
            if req:
                # Extract generated tokens (total tokens minus original prompt length)
                generated_tokens = req.token_ids[req.prompt_length:]
                text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
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
    
    def get_stats(self) -> Dict[str, Any]:
        """Get profiling statistics for all requests."""
        stats = {}
        for req_id, req in self.scheduler.completed_requests.items():
            metrics = req.metrics
            avg_itl = sum(metrics.decode_latencies) / len(metrics.decode_latencies) if metrics.decode_latencies else 0
            # Number of generated tokens equals total tokens minus original prompt length
            generated_tokens_count = len(req.token_ids) - req.prompt_length
            stats[req_id] = {
                "ttft": metrics.ttft,
                "avg_itl": avg_itl,
                "total_tokens": generated_tokens_count,
                "decode_latencies": metrics.decode_latencies,
                "total_latency": metrics.total_latency
            }
        return stats
    def reset_stats(self) -> None:
        """Reset profiling statistics."""
        if self.model_executor:
            self.model_executor.reset_stats()
    def __repr__(self) -> str:
        status = "qwen3_loaded" if self.model_executor else "no_model"
        return f"LLMService(device={self.device}, {status})"