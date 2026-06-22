"""
FlashInfer attention backend implementation.
Provides FlashInfer-based attention computation with plan-then-run pattern.
Supports CUDA graph acceleration for decode attention.
"""

from typing import Optional, List
import torch
import logging

import flashinfer
from flashinfer import BatchDecodeWithPagedKVCacheWrapper, BatchPrefillWithPagedKVCacheWrapper
from flashinfer.decode import CUDAGraphBatchDecodeWithPagedKVCacheWrapper

from .metadata import AttentionMetadata


logger = logging.getLogger(__name__)


class FlashInferBackend:
    """
    FlashInfer attention backend implementation.

    Provides efficient attention computation using FlashInfer operators,
    supporting both prefill and decode phases with paged KV cache.

    CG (CUDA graph) mode:
    - ``init_cg_wrapper()`` creates a ``CUDAGraphBatchDecodeWithPagedKVCacheWrapper``
      with fixed-address buffers.  After this, ``run()`` calls
      ``cg_wrapper.run()`` — a pure kernel launch that can be captured
      inside an outer ``torch.cuda.graph()``.
    - ModelExecutor calls ``init_cg_wrapper()`` during its full-forward
      capture setup, then captures ``model.forward()`` as a single graph.
    """

    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        kv_cache_pool: torch.Tensor,
        num_key_value_heads: Optional[int] = None,
        page_size: int = 16,
        dtype: torch.dtype = torch.bfloat16,
        device: str = "cuda",
        workspace_size_mb: int = 128,
        use_cuda_graph: bool = True,
    ):
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_key_value_heads = num_key_value_heads or num_heads
        self.page_size = page_size
        self.dtype = dtype
        self.device = device
        self.kv_cache_pool = kv_cache_pool

        if num_heads % self.num_key_value_heads != 0:
            raise ValueError(
                f"num_heads ({num_heads}) must be divisible by "
                f"num_key_value_heads ({self.num_key_value_heads})")
        self.num_key_value_groups = num_heads // self.num_key_value_heads

        workspace_size = workspace_size_mb * 1024 * 1024
        self.decode_workspace = torch.empty(
            workspace_size, dtype=torch.uint8, device=device,
        )
        self.prefill_workspace = torch.empty(
            workspace_size, dtype=torch.uint8, device=device,
        )

        self.decode_wrapper = BatchDecodeWithPagedKVCacheWrapper(
            self.decode_workspace, "NHD",
        )
        self.prefill_wrapper = BatchPrefillWithPagedKVCacheWrapper(
            self.prefill_workspace, "NHD",
        )

        self._current_metadata: Optional[AttentionMetadata] = None

        # CG wrapper — set by init_cg_wrapper().  When active, run()
        # calls cg_wrapper.run() (pure kernel launch, capturable).
        self.use_cuda_graph = use_cuda_graph
        self._cg_wrapper: Optional[CUDAGraphBatchDecodeWithPagedKVCacheWrapper] = None

        # Verification counters
        self._cg_decode_calls: int = 0
        self._eager_decode_calls: int = 0

        logger.debug(
            f"Initialized FlashInferBackend: num_heads={num_heads}, "
            f"num_key_value_heads={self.num_key_value_heads}, "
            f"head_dim={head_dim}, page_size={page_size}, "
            f"num_key_value_groups={self.num_key_value_groups}, "
            f"use_cuda_graph={use_cuda_graph}")

    # ── CG wrapper init (called by ModelExecutor) ─────────────────────

    def init_cg_wrapper(
        self,
        batch_size: int,
        indptr_buffer: torch.Tensor,
        indices_buffer: torch.Tensor,
        last_page_len_buffer: torch.Tensor,
    ) -> CUDAGraphBatchDecodeWithPagedKVCacheWrapper:
        """Create a CG wrapper with the given fixed-address buffers.

        After calling this, ``run()`` will use ``cg_wrapper.run()``
        (a pure kernel launch, capturable by ``torch.cuda.graph()``).

        Must be called **before** graph capture begins.  Warmup is done
        here to trigger lazy allocations (alibi slopes, etc.).
        """
        cg_workspace = torch.empty(
            16 * 1024 * 1024, dtype=torch.uint8, device=self.device)
        wrapper = CUDAGraphBatchDecodeWithPagedKVCacheWrapper(
            cg_workspace, indptr_buffer, indices_buffer,
            last_page_len_buffer,
        )
        wrapper.plan(
            indptr_buffer, indices_buffer, last_page_len_buffer,
            self.num_heads, self.num_key_value_heads, self.head_dim,
            self.page_size,
            q_data_type=self.dtype, kv_data_type=self.dtype,
            pos_encoding_mode="NONE",
        )

        # Warmup: run once eagerly to trigger lazy allocations
        dummy_q = torch.zeros(
            batch_size, self.num_heads, self.head_dim,
            dtype=self.dtype, device=self.device)
        dummy_kv = self.kv_cache_pool[0]
        _ = wrapper.run(q=dummy_q, paged_kv_cache=dummy_kv)
        torch.cuda.synchronize()

        self._cg_wrapper = wrapper
        return wrapper

    # ── plan / run ──────────────────────────────────────────────────────

    def plan(self, metadata: AttentionMetadata) -> None:
        if metadata.is_prefill:
            self._plan_prefill(metadata)
        else:
            self._plan_decode(metadata)
        self._current_metadata = metadata

    def _plan_prefill(self, metadata: AttentionMetadata) -> None:
        if metadata.qo_indptr is None:
            raise ValueError("qo_indptr is required for prefill attention")
        self.prefill_wrapper.plan(
            metadata.qo_indptr,
            metadata.paged_kv_indptr,
            metadata.paged_kv_indices,
            metadata.paged_kv_last_page_len,
            self.num_heads, self.num_key_value_heads, self.head_dim,
            self.page_size,
            causal=metadata.causal,
            q_data_type=self.dtype, kv_data_type=self.dtype,
            pos_encoding_mode="NONE",
        )

    def _plan_decode(self, metadata: AttentionMetadata) -> None:
        if self._cg_wrapper is not None:
            return  # CG path: buffers pre-planned, no-op here
        self.decode_wrapper.plan(
            metadata.paged_kv_indptr,
            metadata.paged_kv_indices,
            metadata.paged_kv_last_page_len,
            self.num_heads, self.num_key_value_heads, self.head_dim,
            self.page_size,
            q_data_type=self.dtype, kv_data_type=self.dtype,
            pos_encoding_mode="NONE",
        )

    def run(
        self,
        query: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int = 0,
        metadata: Optional[AttentionMetadata] = None,
    ) -> torch.Tensor:
        """
        Run attention for one layer.

        CG path (``_cg_wrapper`` active): uses the fixed-buffer wrapper.
        This is a pure kernel launch — will be captured by an outer
        ``torch.cuda.graph()`` if called inside it.

        Eager path: standard FlashInfer wrappers.
        """
        if metadata is not None and metadata is not self._current_metadata:
            self.plan(metadata)

        kv_cache_layer = self.kv_cache_pool[layer_idx]

        # KV append (pure kernel launch — capturable)
        flashinfer.append_paged_kv_cache(
            append_key=key_states,
            append_value=value_states,
            batch_indices=metadata.batch_indices,
            positions=metadata.positions,
            paged_kv_cache=kv_cache_layer,
            kv_indices=metadata.paged_kv_indices,
            kv_indptr=metadata.paged_kv_indptr,
            kv_last_page_len=metadata.paged_kv_last_page_len,
            kv_layout="NHD",
        )

        # ── CG path ─────────────────────────────────────────────────────
        if self._cg_wrapper is not None and not metadata.is_prefill:
            self._cg_decode_calls += 1
            return self._cg_wrapper.run(
                q=query, paged_kv_cache=kv_cache_layer)

        # ── Eager path ──────────────────────────────────────────────────
        if metadata.is_prefill:
            output = self.prefill_wrapper.run(
                q=query, paged_kv_cache=kv_cache_layer,
            )
        else:
            self._eager_decode_calls += 1
            output = self.decode_wrapper.run(
                q=query, paged_kv_cache=kv_cache_layer,
            )
        return output

    # ── Misc ────────────────────────────────────────────────────────────

    def reset_state(self) -> None:
        self._current_metadata = None

    def __repr__(self) -> str:
        return (
            f"FlashInferBackend(num_heads={self.num_heads}, "
            f"head_dim={self.head_dim}, page_size={self.page_size}, "
            f"device={self.device})")
