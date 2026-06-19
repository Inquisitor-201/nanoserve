"""
FlashInfer attention backend implementation.
Provides FlashInfer-based attention computation with plan-then-run pattern.
Supports CUDA graph acceleration for decode attention (3-4x speedup).
"""

from typing import Optional, Tuple, List, Dict
import torch
import logging

import flashinfer
from flashinfer import BatchDecodeWithPagedKVCacheWrapper, BatchPrefillWithPagedKVCacheWrapper
from flashinfer.decode import CUDAGraphBatchDecodeWithPagedKVCacheWrapper

from .metadata import AttentionMetadata


logger = logging.getLogger(__name__)


MAX_PAGES_PER_SEQ = 256  # upper bound for pre-allocated indices buffer


def _build_batch_schedule(max_batch: int) -> List[int]:
    """Build the list of batch sizes to pre-capture.

    Returns [1, 2, 4, 8, 16, 32, 64, 96, 128, 160, ...] capped at max_batch.
    """
    sizes = []
    bs = 1
    while bs <= 32:
        sizes.append(bs)
        bs *= 2
    bs = 64
    while bs <= max_batch:
        sizes.append(bs)
        bs += 32
    return sizes


class _CGBatchResources:
    """CUDA graph resources for one pre-captured batch size.

    Each instance owns:
    - Fixed-address page-table buffers (indptr, indices, last_page_len)
    - A query buffer
    - One CUDAGraphBatchDecodeWithPagedKVCacheWrapper
    - ``num_layers`` CUDAGraph objects (one per decoder layer)
    """

    __slots__ = (
        "batch_size", "indptr", "indices", "last_page_len",
        "workspace", "query_buffer", "wrapper", "graphs", "outputs",
    )

    def __init__(
        self,
        batch_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        page_size: int,
        dtype: torch.dtype,
        device: str,
        num_layers: int,
        kv_cache_pool: torch.Tensor,
    ):
        self.batch_size = batch_size

        # ── Allocate fixed-address buffers ───────────────────────────────
        # Workspace: 16 MB for FlashInfer paged attention auxiliary data.
        # Reducing from 64→16 MB saves ~624 MB across 13 batch sizes.
        self.workspace = torch.empty(
            16 * 1024 * 1024, dtype=torch.uint8, device=device)
        self.indptr = torch.zeros(
            batch_size + 1, dtype=torch.int32, device=device)
        self.indices = torch.zeros(
            batch_size * MAX_PAGES_PER_SEQ, dtype=torch.int32, device=device)
        self.last_page_len = torch.zeros(
            batch_size, dtype=torch.int32, device=device)
        self.query_buffer = torch.zeros(
            batch_size, num_heads, head_dim, dtype=dtype, device=device)

        # ── Fill dummy page table (1 page/seq → block 0) ─────────────────
        # CPU build → single copy_()
        ip = torch.arange(batch_size + 1, dtype=torch.int32, device=device)
        ix = torch.zeros(batch_size, dtype=torch.int32, device=device)
        lp = torch.ones(batch_size, dtype=torch.int32, device=device)
        self.indptr.copy_(ip)
        self.indices[:batch_size].copy_(ix)
        self.last_page_len.copy_(lp)

        # ── Create wrapper & plan ────────────────────────────────────────
        self.wrapper = CUDAGraphBatchDecodeWithPagedKVCacheWrapper(
            self.workspace, self.indptr, self.indices, self.last_page_len,
        )
        self.wrapper.plan(
            self.indptr, self.indices, self.last_page_len,
            num_heads, num_kv_heads, head_dim, page_size,
            q_data_type=dtype, kv_data_type=dtype, pos_encoding_mode="NONE",
        )

        # ── Capture one CUDA graph per layer ─────────────────────────────
        self.graphs = []
        self.outputs = []
        for li in range(num_layers):
            g = torch.cuda.CUDAGraph()
            kv_layer = kv_cache_pool[li]
            with torch.cuda.graph(g):
                out = self.wrapper.run(
                    q=self.query_buffer, paged_kv_cache=kv_layer)
            self.graphs.append(g)
            self.outputs.append(out)

        torch.cuda.synchronize()
        logger.debug("Captured CG batch_size=%d (%d layers)",
                     batch_size, num_layers)


class FlashInferBackend:
    """
    FlashInfer attention backend implementation.

    Provides efficient attention computation using FlashInfer operators,
    supporting both prefill and decode phases with paged KV cache.

    Decode attention can be accelerated via pre-captured CUDA graphs
    (``capture_decode_graphs`` → ``create_cg_decode_metadata`` →
    graph replay in ``run``).
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
        """
        Initialize FlashInfer backend with GQA support.

        Args:
            num_heads: Number of attention heads (Query heads)
            head_dim: Dimension of each attention head
            kv_cache_pool: KV cache pool from BlockManager
            num_key_value_heads: Number of key/value heads (for GQA).
            page_size: Size of each page in paged KV cache
            dtype: Data type for computations
            device: Computing device
            workspace_size_mb: Size of workspace buffer in MB
            use_cuda_graph: Enable CUDA graph decode acceleration.
                Call ``capture_decode_graphs()`` before inference to pre-capture.
        """
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

        # ── CUDA Graph for decode attention acceleration ─────────────────
        self.use_cuda_graph = use_cuda_graph
        # ``{batch_size: _CGBatchResources}``, populated by
        # ``capture_decode_graphs()``.
        self._cg_resources: Dict[int, _CGBatchResources] = {}
        # Sorted list of captured batch sizes (ascending).
        self._cg_batch_sizes: List[int] = []

        # Profiling / verification counters
        self._cg_decode_calls: int = 0      # decode steps that used CG
        self._eager_decode_calls: int = 0   # decode steps that fell back to eager

        logger.debug(
            f"Initialized FlashInferBackend: num_heads={num_heads}, "
            f"num_key_value_heads={self.num_key_value_heads}, "
            f"head_dim={head_dim}, page_size={page_size}, "
            f"num_key_value_groups={self.num_key_value_groups}, "
            f"use_cuda_graph={use_cuda_graph}")

    # ── plan / run ──────────────────────────────────────────────────────

    def plan(self, metadata: AttentionMetadata) -> None:
        """
        Plan attention computation for given metadata.

        For prefill: calls FlashInfer's prefill ``plan()``.
        For decode (CG mode): no-op — graphs are pre-captured.
        For decode (eager mode): calls FlashInfer's decode ``plan()``.
        """
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
        """Decode plan — no-op when pre-captured CG resources exist."""
        if self._cg_resources:
            return  # CG path: everything is pre-captured
        # Eager fallback
        self.decode_wrapper.plan(
            metadata.paged_kv_indptr,
            metadata.paged_kv_indices,
            metadata.paged_kv_last_page_len,
            self.num_heads, self.num_key_value_heads, self.head_dim,
            self.page_size,
            q_data_type=self.dtype, kv_data_type=self.dtype,
            pos_encoding_mode="NONE",
        )

    def _get_cg_resources(self, batch_size: int) -> Optional[_CGBatchResources]:
        """Return CG resources for the smallest captured batch >= batch_size.

        Returns ``None`` if no captured batch is large enough → caller
        falls back to eager.
        """
        # Linear scan — the list is short (≤ 13 entries).
        for bs in self._cg_batch_sizes:
            if bs >= batch_size:
                return self._cg_resources[bs]
        return None

    def run(
        self,
        query: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int = 0,
        metadata: Optional[AttentionMetadata] = None,
    ) -> torch.Tensor:
        """
        Run attention computation with GQA support.

        *Decode (CG path)*: copy query into pre-allocated buffer → graph
        replay → return output slice (first ``batch_size`` rows).

        *Prefill / eager decode*: standard FlashInfer ``run()``.
        """
        # Re-plan on metadata change (identity check avoids tensor-eq).
        if metadata is not None and metadata is not self._current_metadata:
            self.plan(metadata)

        kv_cache_layer = self.kv_cache_pool[layer_idx]

        # KV append is always eager (writes to KV cache).
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

        # ── CUDA Graph path ──────────────────────────────────────────────
        if not metadata.is_prefill and self._cg_resources:
            batch_size = metadata.batch_size
            res = self._get_cg_resources(batch_size)
            if res is not None:
                self._cg_decode_calls += 1
                # Copy query into the fixed-size CG query buffer.
                # If the runtime batch is smaller than the captured batch,
                # the unused trailing slots keep their old values (ignored).
                res.query_buffer[:batch_size].copy_(query)
                # Replay the pre-captured graph for this layer.
                res.graphs[layer_idx].replay()
                # Return only the valid rows.
                return res.outputs[layer_idx][:batch_size]

        # ── Eager path (prefill or no CG resource for this batch) ───────
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

    # ── CUDA Graph capture ──────────────────────────────────────────────

    def capture_decode_graphs(self, max_batch_size: int = 256) -> None:
        """Pre-capture CUDA graphs for decode attention.

        Batch sizes captured: 1, 2, 4, 8, 16, 32, 64, 96, …, ``max_batch_size``.
        Captured from *largest* to *smallest* to minimise memory fragmentation.

        Must be called **once** after the KV cache pool is allocated and the
        model is on GPU.  Typically invoked from ``LLMService.__init__``.
        All captured graphs share the same KV cache pool.
        """
        if self._cg_resources:
            logger.warning("CUDA decode graphs already captured, skipping")
            return

        batch_sizes = _build_batch_schedule(max_batch_size)
        num_layers = self.kv_cache_pool.shape[0]

        # Capture largest → smallest.
        for bs in reversed(batch_sizes):
            logger.info("Capturing CUDA graph batch_size=%d ...", bs)
            res = _CGBatchResources(
                bs,
                self.num_heads, self.num_key_value_heads, self.head_dim,
                self.page_size, self.dtype, self.device,
                num_layers, self.kv_cache_pool,
            )
            self._cg_resources[bs] = res

        self._cg_batch_sizes = sorted(self._cg_resources.keys())
        logger.info(
            "CUDA decode graphs ready: %s",
            ", ".join(str(b) for b in self._cg_batch_sizes))

    # ── Zero-copy metadata for CG decode ────────────────────────────────

    def create_cg_decode_metadata(
        self,
        block_tables: List[List[int]],
        seq_lengths: List[int],
        page_size: int,
    ) -> AttentionMetadata:
        """Create decode metadata with **zero-copy** CG buffer references.

        The returned ``AttentionMetadata`` has its ``paged_kv_indptr``,
        ``paged_kv_indices`` and ``paged_kv_last_page_len`` pointing
        directly into the pre-allocated CG buffers for the nearest captured
        batch size ≥ ``len(seq_lengths)``.

        Builds page-table data on CPU as Python lists, then does a single
        ``copy_()`` per buffer — avoids thousands of tiny ``aten::copy``
        calls that element-wise GPU writes would produce.
        """
        batch_size = len(seq_lengths)
        res = self._get_cg_resources(batch_size)
        if res is None:
            raise RuntimeError(
                f"No CUDA graph resources for batch_size={batch_size}")

        cg_bs = res.batch_size  # captured batch (≥ batch_size)

        # ── Build page table on CPU (fast) ────────────────────────────────
        cpu_indices: List[int] = []
        cpu_indptr: List[int] = [0]
        cpu_last_page_len: List[int] = []

        for i in range(batch_size):
            bt = block_tables[i]
            sl = seq_lengths[i]
            num_pages = (sl + page_size - 1) // page_size
            cpu_indptr.append(cpu_indptr[-1] + num_pages)
            # Actual pages from block table
            for j in range(min(num_pages, len(bt))):
                cpu_indices.append(bt[j])
            # Pad if block_table is shorter than needed pages
            for j in range(len(bt), num_pages):
                cpu_indices.append(0)
            cpu_last_page_len.append((sl - 1) % page_size + 1)

        # ── Pad unused slots up to captured batch size ────────────────────
        for _ in range(batch_size, cg_bs):
            cpu_indices.append(0)
            cpu_indptr.append(cpu_indptr[-1] + 1)
            cpu_last_page_len.append(1)

        # ── Single bulk copy per buffer ───────────────────────────────────
        res.indptr.copy_(torch.tensor(cpu_indptr, dtype=torch.int32,
                                       device=self.device))
        res.indices[:len(cpu_indices)].copy_(
            torch.tensor(cpu_indices, dtype=torch.int32, device=self.device))
        res.last_page_len[:cg_bs].copy_(
            torch.tensor(cpu_last_page_len, dtype=torch.int32,
                         device=self.device))

        # ── Build metadata (zero-copy views into CG buffers) ─────────────
        metadata = AttentionMetadata(
            block_tables=block_tables,
            seq_lengths=seq_lengths,
            paged_kv_indptr=res.indptr[:batch_size + 1],
            paged_kv_indices=res.indices[:cpu_indptr[batch_size]],
            paged_kv_last_page_len=res.last_page_len[:batch_size],
            batch_indices=torch.arange(
                batch_size, dtype=torch.int32, device=self.device),
            positions=(
                torch.tensor(seq_lengths, dtype=torch.int32,
                             device=self.device) - 1
            ).to(dtype=torch.int32),
            is_prefill=False,
        )
        return metadata

    # ── Misc ────────────────────────────────────────────────────────────

    def reset_state(self) -> None:
        """Reset internal state for a new batch."""
        self._current_metadata = None

    def __repr__(self) -> str:
        return (
            f"FlashInferBackend(num_heads={self.num_heads}, "
            f"head_dim={self.head_dim}, page_size={self.page_size}, "
            f"device={self.device})")