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
        max_num_seqs: int = 64,
        max_num_batched_tokens: int = 8192,
        attention_backend: str = "flashinfer",
        gpu_memory_utilization: float = 0.90,
        enforce_eager: bool = True,
    ) -> None:
        if not Path(model_path).exists():
            raise ValueError(f"Model path does not exist: {model_path}")

        # 1. Load model architecture from HF config
        model_config = ModelConfig.from_hf(model_path)

        # 2. Build derived configs (KV cache size determined below)
        scheduler_config = SchedulerConfig(
            max_num_seqs=max_num_seqs,
            max_num_batched_tokens=max_num_batched_tokens,
        )

        self.model_config = model_config
        self.scheduler_config = scheduler_config
        self.model_path = model_path
        self.device = device

        # 3. Initialize Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.eos_token_id = self.tokenizer.eos_token_id
        logger.info(f"EOS token ID: {self.eos_token_id}")

        # 4. Determine KV cache size
        if num_blocks is not None and num_blocks > 0:
            # User-specified — skip profiling
            cache_config = CacheConfig(
                num_blocks=num_blocks, block_size=block_size, device=device)
            block_manager = BlockManager(model_config, cache_config)
            model_executor = self._create_executor(
                model_config, cache_config, block_manager, model_path, attention_backend)
        else:
            # Profile-driven: create model with a tiny KV cache, measure peak
            # memory during a dummy forward, then size the real KV cache.
            model_executor, block_manager = self._init_with_profiling(
                model_config=model_config,
                block_size=block_size,
                model_path=model_path,
                attention_backend=attention_backend,
                gpu_memory_utilization=gpu_memory_utilization,
            )

        # 5. Initialize Scheduler
        self.block_manager = block_manager
        self.model_executor = model_executor
        self.scheduler = Scheduler(scheduler_config, block_manager)

        # 6. Partial compilation disabled — torch.compile on MLP
        #    submodules regressed throughput (1070 vs 1178 tok/s) for this
        #    small model.  Keep the API for future experiments.
        if not enforce_eager:
            logger.info("enforce_eager=False: torch.compile skipped "
                        "(no benefit for Qwen3-0.6B MLP-only compile)")

    # ── Profile helpers ────────────────────────────────────────────────

    def _create_executor(self, model_config, cache_config, block_manager,
                         model_path, attention_backend):
        """Create ModelExecutor with the given KV cache pool."""
        return ModelExecutor(
            model_config=model_config,
            cache_config=cache_config,
            kv_cache_pool=block_manager.kv_cache_pool,
            model_path=model_path,
            attention_backend=attention_backend,
        )

    def _profile_memory(
        self, model, block_manager, max_num_batched_tokens, max_num_seqs, device
    ) -> int:
        """Run a dummy forward pass and return the peak allocated bytes.

        Uses a batch of ``max_num_batched_tokens`` tokens distributed across
        ``max_num_seqs`` sequences — matching the exact distribution that the
        real prefill step will use.  This is critical because FlashInfer's
        paged attention workspace varies with the *number of sequences*, not
        just the total token count.
        """
        from .backends import AttentionMetadata

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        # Build a batch matching the actual distribution that the
        # KV-cache-aware scheduler will produce (~14-18 seqs on this GPU).
        # Profile with at most 16 sequences to get a representative
        # FlashInfer workspace size (the workspace grows with batch size).
        budget = min(max_num_batched_tokens, 8192)
        num_seqs = min(max_num_seqs, budget // 16, 16)  # cap at 16
        tokens_per_seq = budget // num_seqs
        tail = budget % num_seqs
        seq_lengths = [tokens_per_seq + (1 if i < tail else 0)
                       for i in range(num_seqs)]
        total_tokens = sum(seq_lengths)  # exact budget

        input_ids = torch.randint(0, 100, (total_tokens,), device=device)

        # Allocate blocks for each dummy sequence
        block_tables = []
        for seq_len in seq_lengths:
            blocks = block_manager.allocate_blocks([], seq_len)
            if not blocks:
                raise RuntimeError(
                    f"Not enough KV blocks for profiling "
                    f"(need {sum(seq_lengths)} tokens, "
                    f"have {block_manager.num_blocks} blocks)")
            block_tables.append(blocks)

        metadata = AttentionMetadata.from_block_tables(
            block_tables=block_tables,
            seq_lengths=seq_lengths,
            page_size=block_manager.block_size,
            is_prefill=True,
            device=device,
        )

        # Last-token indices (matches the behaviour of our lm_head
        # optimisation during real prefill).
        last_token_indices = torch.cumsum(
            torch.tensor(seq_lengths, device=device), dim=0
        ) - 1

        with torch.no_grad():
            out = model(input_ids, metadata, last_token_indices)
        # Keep the output alive to ensure its memory is counted in the peak
        out = out.clone()
        del out

        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        logger.info(
            "Profile run: %d seqs × %d tok = %d total, peak=%.2f GiB",
            num_seqs, tokens_per_seq, total_tokens, peak / 1024**3,
        )

        # Free allocated blocks
        for bt in block_tables:
            block_manager.free_blocks(bt)

        return peak

    def _init_with_profiling(
        self, model_config, block_size, model_path,
        attention_backend, gpu_memory_utilization,
    ):
        """Create model, profile peak memory, and size KV cache accordingly.

        Steps:
          1. Create model with a *tiny* KV cache pool (128 blocks).
          2. Load weights onto GPU.
          3. Run one dummy forward pass (4 seqs × 256 tokens).
          4. Read ``allocated_bytes.all.peak``.
          5. Calculate: available_kv = total × gpu_mem_util − peak.
          6. Allocate real KV cache with the calculated size.
          7. Swap the real pool into the already-loaded model.
        """
        device = self.device
        budget = self.scheduler_config.max_num_batched_tokens
        profile_tokens = min(budget, 8192)
        profile_seq_count = min(self.scheduler_config.max_num_seqs,
                                profile_tokens // 16, 16)
        tokens_per_seq, tail = divmod(profile_tokens, profile_seq_count)
        profile_blocks_needed = sum(
            (tokens_per_seq + (1 if i < tail else 0) + block_size - 1) // block_size
            for i in range(profile_seq_count)
        )
        PROFILE_BLOCKS = max(128, profile_blocks_needed + 32)

        bytes_per_block = (
            model_config.num_layers
            * 2                              # K + V
            * block_size
            * model_config.num_key_value_heads
            * model_config.head_dim
            * torch.tensor([], dtype=model_config.dtype).element_size()
        )

        # ── Phase 1: create model with minimal KV cache for profiling ──
        profile_cache = CacheConfig(
            num_blocks=PROFILE_BLOCKS, block_size=block_size, device=device)
        profile_bm = BlockManager(model_config, profile_cache)
        model_executor = self._create_executor(
            model_config, profile_cache, profile_bm,
            model_path, attention_backend)
        model = model_executor.model

        profile_pool_bytes = PROFILE_BLOCKS * bytes_per_block

        # ── Phase 2: profile ──
        peak = self._profile_memory(
            model, profile_bm, budget,
            self.scheduler_config.max_num_seqs, device)

        # ── Phase 3: tear down profile pool ──
        total_gpu = torch.cuda.get_device_properties(0).total_memory
        # The model AND the executor hold references to the profile KV pool.
        # Replace both so the profile pool can be reclaimed.
        model.attention_backend.kv_cache_pool = torch.empty(0, device=device)
        model_executor.kv_cache_pool = torch.empty(0, device=device)
        del profile_bm
        torch.cuda.empty_cache()

        # ── Phase 4: calculate real KV cache size ──
        # After cleanup the surviving allocations are:
        #   model weights + FlashInfer workspaces + PyTorch allocator overhead.
        # We subtract those from total × util then round down by one
        # intermediate-activation peak to guarantee forward pass fits.
        surviving = torch.cuda.memory_allocated()
        intermediate_peak = peak - surviving
        available_kv = total_gpu * gpu_memory_utilization - surviving - intermediate_peak
        # Reserve 1.25 GiB for allocator fragmentation/headroom (PyTorch's
        # caching allocator can fragment free memory, leaving no single
        # contiguous chunk large enough for intermediate tensors).
        available_kv -= 1280 * 1024 * 1024
        num_blocks = max(64, int(available_kv // bytes_per_block))

        # ── Phase 5: allocate real KV cache ──
        real_cache = CacheConfig(
            num_blocks=num_blocks, block_size=block_size, device=device)
        real_bm = BlockManager(model_config, real_cache)

        # Swap the real pool into the existing model backend
        model.attention_backend.kv_cache_pool = real_bm.kv_cache_pool

        model_gib = sum(
            p.numel() * p.element_size() for p in model.parameters()
        ) / 1024**3
        logger.info(
            "Memory profile:  model=%.2f GiB  "
            "peak=%.2f GiB  util=%.2f  "
            "kv_available=%.2f GiB → %d blocks",
            model_gib, peak / 1024**3, gpu_memory_utilization,
            available_kv / 1024**3, num_blocks,
        )

        return model_executor, real_bm

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
        step = 0
        max_steps = sampling_config.max_new_tokens + 5  # +5 for prefill steps
        log_interval = max(1, max_steps // 20)  # ~5% intervals

        while self.scheduler.has_unfinished_requests():
            sched_output = self.scheduler.schedule()

            if not sched_output.scheduled_requests:
                raise RuntimeError("No requests scheduled. This should not happen.")

            sampled_tokens = self._run_inference_step(
                sched_output, sampling_config
            )

            # Update scheduler with new tokens
            new_tokens = [t.view(1) for t in sampled_tokens]
            active_ids = [req.request_id for req in sched_output.scheduled_requests]
            self.scheduler.update_running_requests(new_tokens, active_ids)
            step += 1

            # Log progress periodically
            if step % log_interval == 0 or step == 1:
                n_running = len(self.scheduler.running_list)
                n_waiting = len(self.scheduler.waiting_list)
                n_done = len(self.scheduler.completed_requests)
                pct = min(step / max_steps * 100, 99)
                logger.info(
                    f"  Step {step}/{max_steps} ({pct:.0f}%) | "
                    f"running={n_running} waiting={n_waiting} done={n_done}"
                )
        # 4. Final output collection
        logger.info(f"Generation complete in {step} steps")
        return self._collect_results(request_ids)

    def _run_inference_step(self, sched_output, sampling_config: SamplingConfig):
        """Unified inference step handling Prefill and Decode."""
        is_prefill = sched_output.is_prefill

        # Prefill optimisation: only compute lm_head for the last token of
        # each sequence to save ~vocab_size × (tokens - seqs) FLOPs.
        last_token_indices = None
        if is_prefill:
            last_token_indices = torch.cumsum(
                torch.tensor(sched_output.seq_lengths, device=self.device), dim=0
            ) - 1

        with ContinuousBatchTimer(sched_output.scheduled_requests):
            logits = self.model_executor.execute_batch(
                input_ids=sched_output.input_ids,
                block_tables=sched_output.block_tables,
                seq_lengths=sched_output.seq_lengths,
                is_prefill=is_prefill,
                last_token_indices=last_token_indices,
            )

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

        # Log progress info
        n_reqs = len(request_ids)
        max_tokens = cfg.max_new_tokens
        logger.info(
            f"Starting generation: {n_reqs} request(s), "
            f"max_new_tokens={max_tokens}, "
            f"estimated ~{max_tokens * self._estimate_itl():.0f}s total"
        )

        generated_texts = self.main_loop(
            request_ids=request_ids,
            sampling_config=cfg
        )

        return generated_texts

    def _estimate_itl(self) -> float:
        """Rough estimate of inter-token latency based on model size."""
        h = self.model_config.hidden_size
        # Empirical: 0.6B (h=1024) ~25ms, 1.7B (h=2048) ~46ms
        return 0.025 * (h / 1024) ** 0.8
    
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