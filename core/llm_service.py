"""
LLM Service Layer - Unified interface for running Qwen3 model.
Provides high-level API for model loading, configuration, and inference.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
import torch
from transformers import AutoTokenizer

from .model_executor import ModelExecutor
from .block_manager import BlockManager
from .scheduler import Scheduler, Request
from .config import SamplingConfig, ModelConfig, CacheConfig, SchedulerConfig
from .config import auto_calculate_num_blocks
from .utils import ContinuousBatchTimer


logger = logging.getLogger(__name__)


class LLMService:
    """
    NanoServe core service class.

    Example::

        service = LLMService(model_path="/path/to/qwen3")
        result = service.generate(["Hello!"], SamplingConfig(temperature=0.6, top_p=0.9, max_new_tokens=512))
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        block_size: int = 16,
        num_blocks: Optional[int] = None,
        max_num_seqs: int = 256,
        attention_backend: str = "flashinfer",
    ) -> None:
        if not Path(model_path).exists():
            raise ValueError(f"Model path does not exist: {model_path}")

        # 1. Load model architecture from HF config
        model_config = ModelConfig.from_hf(model_path)

        # 2. Auto-calculate KV cache block count if not specified
        if num_blocks is None or num_blocks <= 0:
            num_blocks = auto_calculate_num_blocks(
                device=device,
                dtype=model_config.dtype,
                block_size=block_size,
                num_layers=model_config.num_layers,
                num_kv_heads=model_config.num_key_value_heads,
                head_dim=model_config.head_dim,
            )

        # 3. Build derived configs
        cache_config = CacheConfig(
            num_blocks=num_blocks,
            block_size=block_size,
            device=device,
        )
        scheduler_config = SchedulerConfig(max_num_seqs=max_num_seqs)

        self.model_config = model_config
        self.cache_config = cache_config
        self.scheduler_config = scheduler_config
        self.model_path = model_path
        self.device = device

        # 4. Initialize Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.eos_token_id = self.tokenizer.eos_token_id
        logger.info(f"EOS token ID: {self.eos_token_id}")

        # 5. Initialize BlockManager (pre-allocates KV cache GPU pool)
        self.block_manager = BlockManager(
            model_config=model_config,
            cache_config=cache_config
        )

        # 6. Initialize ModelExecutor
        self.model_executor = ModelExecutor(
            model_config=model_config,
            cache_config=cache_config,
            kv_cache_pool=self.block_manager.kv_cache_pool,
            model_path=model_path,
            attention_backend=attention_backend
        )

        # 7. Initialize Scheduler
        self.scheduler = Scheduler(scheduler_config, self.block_manager)
    
    def add_requests(
        self,
        prompts: Union[str, List[str], List[List[int]]],
        sampling_config: Union[SamplingConfig, List[SamplingConfig]],
    ) -> List[str]:
        """
        Add requests to the scheduler.

        Args:
            prompts: Input prompt(s). Strings are tokenized internally;
                     ``List[List[int]]`` is treated as pre-tokenized.
            sampling_config: SamplingConfig or list (one per prompt).

        Returns:
            List of request IDs.
        """
        if isinstance(prompts, str):
            prompts = [prompts]

        is_tokenized = bool(prompts and isinstance(prompts[0], list))

        if isinstance(sampling_config, SamplingConfig):
            sampling_configs = [sampling_config] * len(prompts)
        else:
            sampling_configs = sampling_config

        request_ids = []

        if is_tokenized:
            for tokens, sc in zip(prompts, sampling_configs):
                input_ids = torch.tensor(tokens, device=self.device, dtype=torch.long)
                req_id = self.scheduler.add_request(
                    input_ids=input_ids,
                    sampling_config=sc,
                    eos_token_id=self.eos_token_id
                )
                request_ids.append(req_id)
        else:
            encoded = self.tokenizer(
                prompts,
                padding=True,
                truncation=True,
                return_tensors="pt"
            )
            input_ids_batch = encoded["input_ids"].to(self.device)
            attention_mask = encoded["attention_mask"].to(self.device)

            for i in range(len(prompts)):
                input_ids = input_ids_batch[i][attention_mask[i].bool()]
                req_id = self.scheduler.add_request(
                    input_ids=input_ids,
                    sampling_config=sampling_configs[i],
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
        prompts: Union[str, List[str], List[List[int]]],
        sampling_config: Union[SamplingConfig, List[SamplingConfig]],
    ) -> List[str]:
        """
        Generate text from prompts using Qwen3 model with scheduling.

        Args:
            prompts: Input prompt(s). Strings are tokenized internally;
                     ``List[List[int]]`` is treated as pre-tokenized.
            sampling_config: SamplingConfig or list (one per prompt).

        Returns:
            Generated text(s).
        """
        request_ids = self.add_requests(
            prompts=prompts,
            sampling_config=sampling_config,
        )

        cfg = sampling_config[0] if isinstance(sampling_config, list) else sampling_config
        generated_texts = self.main_loop(
            request_ids=request_ids,
            sampling_config=cfg
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