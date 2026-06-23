"""
Model executor for the architecture.
Manages model execution with proper attention backend integration.
Supports full-forward CUDA graph capture for decode (nano-vllm style).
"""

from typing import List, Optional, Dict, Any
import torch
import logging
from pathlib import Path

from .backends import AttentionMetadata, FlashInferBackend
from .models import Qwen3Model
from .block_manager import BlockManager
from .model_loader import ModelLoader
from .config import ModelConfig, CacheConfig
from .utils import StatsCollector, ProfileTimer


logger = logging.getLogger(__name__)

MAX_PAGES_PER_SEQ = 256


def _build_batch_schedule(max_batch: int) -> List[int]:
    """Build list of batch sizes to pre-capture.

    [1, 2, 4, 8, 16, 32, 64, 96, 128, 160, ...] capped at max_batch.
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
    """Pre-allocated, fixed-address buffers for one batch size.

    Everything needed for ``model.forward``:
    - input_ids, positions (model inputs)
    - paged_kv_indptr, _indices, _last_page_len (page table)
    - logits (output buffer)
    - singleton AttentionMetadata referencing the above
    - captured ``torch.cuda.CUDAGraph``
    """

    __slots__ = (
        "batch_size",
        "input_ids", "positions",
        "indptr", "indices", "last_page_len",
        "logits",
        "metadata",
        "graph",
        "cg_wrapper",
    )

    def __init__(self, batch_size: int, vocab_size: int, dtype: torch.dtype,
                 device: str):
        self.batch_size = batch_size

        self.input_ids = torch.zeros(
            batch_size, dtype=torch.long, device=device)
        self.positions = torch.zeros(
            batch_size, dtype=torch.int32, device=device)

        self.indptr = torch.zeros(
            batch_size + 1, dtype=torch.int32, device=device)
        self.indices = torch.zeros(
            batch_size * MAX_PAGES_PER_SEQ, dtype=torch.int32, device=device)
        self.last_page_len = torch.ones(
            batch_size, dtype=torch.int32, device=device)

        self.logits = torch.zeros(
            batch_size, vocab_size, dtype=dtype, device=device)

        # Singleton metadata (views into fixed buffers)
        # NOTE: views must span the FULL pre-allocated extent so that
        # during replay with fewer sequences, the kernel can still read
        # the padded slots.  FlashInfer kernels read *pointers*, not
        # shapes — but we ensure the view is large enough for safety.
        self.metadata = AttentionMetadata(
            is_prefill=False,
            batch_size=batch_size,
            seq_lengths=[1] * batch_size,
            paged_kv_indptr=self.indptr,         # full [batch_size+1]
            paged_kv_indices=self.indices,       # full [batch_size * MAX_PAGES_PER_SEQ]
            paged_kv_last_page_len=self.last_page_len,  # full [batch_size]
            batch_indices=torch.arange(
                batch_size, dtype=torch.int32, device=device),
            positions=self.positions,             # full [batch_size]
        )
        self.graph = None
        self.cg_wrapper = None

    def upload_block_tables(self, block_tables, seq_lengths, page_size):
        """Convert scheduler block_tables → GPU indptr/indices/last_page_len buffers."""
        bs = len(seq_lengths)
        cg_bs = self.batch_size
        cpu_indices = []
        cpu_indptr = [0]
        cpu_last_page_len = []
        for i in range(bs):
            bt = block_tables[i]
            sl = seq_lengths[i]
            np = (sl + page_size - 1) // page_size
            cpu_indptr.append(cpu_indptr[-1] + np)
            for j in range(min(np, len(bt))):
                cpu_indices.append(bt[j])
            for j in range(len(bt), np):
                cpu_indices.append(0)
            cpu_last_page_len.append((sl - 1) % page_size + 1)
        for _ in range(bs, cg_bs):
            cpu_indices.append(0)
            cpu_indptr.append(cpu_indptr[-1] + 1)
            cpu_last_page_len.append(1)
        t = self.indptr.device
        self.indptr.copy_(torch.tensor(cpu_indptr, dtype=torch.int32, device=t))
        self.indices[:len(cpu_indices)].copy_(
            torch.tensor(cpu_indices, dtype=torch.int32, device=t))
        self.last_page_len[:cg_bs].copy_(
            torch.tensor(cpu_last_page_len, dtype=torch.int32, device=t))


class ModelExecutor:
    """
    Model executor for managing the complete inference pipeline.

    This executor ensures proper coordination between models and attention backends.
    All model structure parameters are dynamically initialized based on ModelConfig,
    ensuring complete match with loaded weights.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        cache_config: CacheConfig,
        kv_cache_pool: torch.Tensor,
        model_name: str = "qwen3",
        model_path: str = "",
        attention_backend: str = "flashinfer",
    ):
        self.model_name = model_name
        self.device = cache_config.device
        self.dtype = model_config.dtype
        self.block_size = cache_config.block_size
        self.model_config = model_config
        self.cache_config = cache_config
        self.model_path = model_path
        self.attention_backend = attention_backend
        self.kv_cache_pool = kv_cache_pool
        self.stats = StatsCollector()

        if model_name == "qwen3":
            vocab_size = model_config.vocab_size
            hidden_size = model_config.hidden_size
            num_heads = model_config.num_heads
            num_key_value_heads = model_config.num_key_value_heads
            head_dim = model_config.head_dim
            intermediate_size = model_config.intermediate_size
            num_layers = model_config.num_layers

            self.model = Qwen3Model(
                vocab_size=vocab_size,
                hidden_size=hidden_size,
                num_heads=num_heads,
                num_key_value_heads=num_key_value_heads,
                head_dim=head_dim,
                intermediate_size=intermediate_size,
                num_layers=num_layers,
                attention_backend_type=attention_backend,
                dtype=self.dtype,
                device=self.device,
                kv_cache_pool=self.kv_cache_pool,
                rope_theta=model_config.rope_theta,
                rms_norm_eps=model_config.rms_norm_eps,
                block_size=cache_config.block_size,
                quantization=model_config.quantization,
            )

            if model_path:
                ModelLoader.load_weights(
                    self.model,
                    model_path,
                    self.dtype,
                    self.device,
                    tie_word_embeddings=model_config.tie_word_embeddings,
                )

        else:
            raise ValueError(f"Unsupported model: {model_name}")

        # ── Full-forward CUDA Graph resources ─────────────────────────
        self._use_cg = False
        self._cg_resources: Dict[int, _CGBatchResources] = {}
        self._cg_batch_sizes: List[int] = []

        # Verification counters
        self._cg_decode_calls: int = 0
        self._eager_decode_calls: int = 0

        logger.debug(f"Initialized ModelExecutor with {model_name} model: "
                   f"hidden_size={model_config.hidden_size}, "
                   f"num_heads={model_config.num_heads}, "
                   f"num_layers={model_config.num_layers}")

    # ── Full-forward CUDA Graph capture ────────────────────────────────

    def capture_decode_graphs(self, max_batch_size: int = 256) -> None:
        """Pre-capture full-forward CUDA graphs for decode.

        One ``CUDAGraph`` per batch size, shared ``graph_pool``.
        Must be called once after model + KV cache are initialized.
        """
        if self._cg_resources:
            logger.warning("Full CG decode graphs already captured, skipping")
            return

        backend = self.model.attention_backend
        if not backend.use_cuda_graph:
            logger.info("CUDA graph disabled, skipping capture")
            return

        batch_sizes = _build_batch_schedule(max_batch_size)
        vocab_size = self.model_config.vocab_size
        graph_pool = None

        # Pre-allocate resources for the LARGEST batch first so the
        # graph_pool allocates enough memory for all smaller batches.
        for bs in reversed(batch_sizes):
            logger.info("Capturing full-forward CG batch_size=%d ...", bs)

            res = _CGBatchResources(
                bs, vocab_size, self.dtype, self.device)
            self._cg_resources[bs] = res

            # Init the backend's CG wrapper (fixed indptr/indices buffers).
            # Store in res.cg_wrapper to keep the workspace alive during replay.
            res.cg_wrapper = backend.init_cg_wrapper(
                bs, res.indptr, res.indices, res.last_page_len)

            # Fill dummy page table (1 page/seq → block 0)
            res.indptr.copy_(torch.arange(
                bs + 1, dtype=torch.int32, device=self.device))
            res.indices[:bs].zero_()
            res.last_page_len[:bs].fill_(1)

            # Warmup: run forward multiple times to trigger lazy allocations
            # (cuBLAS workspaces, attention kernel allocations, etc.) before
            # capture — otherwise those cudaMallocs get baked into the graph.
            for _ in range(3):
                with torch.inference_mode():
                    _ = self.model(res.input_ids[:bs], res.metadata)
                torch.cuda.synchronize()

            # Capture the entire forward into one graph.
            # Write directly into res.logits[:bs] so replay writes
            # to the same fixed-address buffer (no extra copy_).
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, pool=graph_pool):
                with torch.inference_mode():
                    res.logits[:bs] = self.model(
                        res.input_ids[:bs], res.metadata)

            if graph_pool is None:
                graph_pool = g.pool()

            res.graph = g

        self._cg_batch_sizes = sorted(self._cg_resources.keys())
        self._use_cg = True

        # Clear KV cache after capture: warmup/capture wrote garbage into
        # block 0 via the dummy page table (all sequences → block 0).
        kv_pool = self.model.attention_backend.kv_cache_pool
        kv_pool.zero_()
        torch.cuda.synchronize()

        logger.info(
            "Full-forward CUDA decode graphs ready: batch_sizes=%s",
            ", ".join(str(b) for b in self._cg_batch_sizes))

    def _get_cg_resources(self, batch_size: int) -> Optional[_CGBatchResources]:
        """Smallest captured batch >= batch_size."""
        for bs in self._cg_batch_sizes:
            if bs >= batch_size:
                return self._cg_resources[bs]
        return None

    # ── execute_batch (prefill / decode) ────────────────────────────────

    def execute_batch(
        self,
        input_ids: torch.Tensor,
        block_tables: List[List[int]],
        seq_lengths: List[int],
        is_prefill: bool,
        last_token_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Execute prefill or decode phase.

        Decode with CG: buffer update + ``graph.replay()`` — no Python
        per-layer overhead.  Prefill: standard eager forward.
        """
        if not is_prefill and self._use_cg:
            cg_avail = self._get_cg_resources(len(seq_lengths)) is not None
            if cg_avail:
                with ProfileTimer(self.stats, is_prefill):
                    logits = self._execute_cg_decode(
                        input_ids, block_tables, seq_lengths)
                return logits

        # Eager path (prefill or no CG).
        # Clear CG wrapper so backend.run() uses the correct page table
        # from the new metadata, not the stale CG fixed buffers.
        self.model.attention_backend._cg_wrapper = None

        metadata = AttentionMetadata.from_block_tables(
            block_tables=block_tables,
            seq_lengths=seq_lengths,
            is_prefill=is_prefill,
            page_size=self.block_size,
            device=self.device,
        )
        # Plan attention once for the batch (before forward, not inside CUDA graph)
        self.model.attention_backend.plan(metadata)
        with ProfileTimer(self.stats, is_prefill):
            logits = self.model(input_ids, metadata, last_token_indices)
        return logits

    def _execute_cg_decode(
        self,
        input_ids: torch.Tensor,
        block_tables: List[List[int]],
        seq_lengths: List[int],
    ) -> torch.Tensor:
        """Run one decode step via pre-captured full-forward graph.

        1. Activate this batch's CG wrapper (keep workspace alive)
        2. Memcpy new input values + page table into fixed CG buffers
        3. ``plan()`` — recompute attention workspace from updated page table
        4. ``graph.replay()`` (all layers: norms + attention + MLP + lm_head)
        5. Return logits slice (first ``batch_size`` rows)
        """
        batch_size = len(seq_lengths)
        res = self._get_cg_resources(batch_size)

        # Activate this batch's CG wrapper so the captured graph's CUDA
        # operations reference the correct, still-alive workspace.
        self.model.attention_backend._cg_wrapper = res.cg_wrapper

        # Update fixed buffers (CPU → GPU bulk copy)
        res.input_ids[:batch_size].copy_(input_ids)
        positions_t = torch.tensor(
            [sl - 1 for sl in seq_lengths], dtype=torch.int32,
            device=self.device)
        res.positions[:batch_size].copy_(positions_t)
        res.upload_block_tables(block_tables, seq_lengths, self.block_size)

        # Plan: recompute attention workspace metadata from the updated
        # page-table buffers.  Must be *after* upload_block_tables() so
        # cg_wrapper.plan() reads the correct indptr/indices data.
        self.model.attention_backend.plan(res.metadata)

        # Replay — all layers in one cuGraphLaunch
        res.graph.replay()
        self._cg_decode_calls += 1
        return res.logits[:batch_size]

    # ── Legacy API (used by generate/execute_decode/execute_prefill) ──

    def generate(
        self,
        input_ids: torch.Tensor,
        block_tables: List[List[int]],
        seq_lengths: List[int],
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_p: float = 0.9
    ) -> torch.Tensor:
        """Generate tokens using the model."""
        batch_size = len(seq_lengths)
        current_input_ids = input_ids
        current_seq_lengths = seq_lengths.copy()
        generated_ids = []

        logits = self.execute_prefill(
            current_input_ids, block_tables, current_seq_lengths)

        last_token_positions = []
        cumulative_tokens = 0
        for seq_len in seq_lengths:
            last_token_positions.append(cumulative_tokens + seq_len - 1)
            cumulative_tokens += seq_len

        last_logits = logits[last_token_positions]
        next_tokens = self.sample(last_logits, temperature, top_p)
        generated_ids.append(next_tokens)

        for i in range(batch_size):
            current_seq_lengths[i] += 1

        for _ in range(max_new_tokens - 1):
            logits = self.execute_decode(
                next_tokens, block_tables, current_seq_lengths)
            next_tokens = self.sample(logits, temperature, top_p)
            generated_ids.append(next_tokens)
            for i in range(batch_size):
                current_seq_lengths[i] += 1

        generated_ids = torch.cat(generated_ids, dim=0)
        return generated_ids

    def execute_prefill(
        self,
        input_ids: torch.Tensor,
        block_tables: List[List[int]],
        seq_lengths: List[int],
    ) -> torch.Tensor:
        return self.execute_batch(
            input_ids, block_tables, seq_lengths, is_prefill=True)

    def execute_decode(
        self,
        input_ids: torch.Tensor,
        block_tables: List[List[int]],
        seq_lengths: List[int],
    ) -> torch.Tensor:
        return self.execute_batch(
            input_ids, block_tables, seq_lengths, is_prefill=False)

    def sample(
        self,
        logits: torch.Tensor,
        temperature: float,
        top_p: float
    ) -> torch.Tensor:
        if temperature == 0.0:
            return torch.argmax(logits, dim=-1)
        if temperature > 0:
            logits = logits / temperature
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(
                torch.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = \
                sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            mask = torch.zeros_like(logits, dtype=torch.bool)
            mask.scatter_(1, sorted_indices, sorted_indices_to_remove)
            logits[mask] = -float('inf')
        probs = torch.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(1)

    def get_stats(self) -> Dict[str, Any]:
        return self.stats.get_stats()

    def reset_stats(self) -> None:
        self.stats = StatsCollector()

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "hidden_size": self.model_config.hidden_size,
            "num_heads": self.model_config.num_heads,
            "num_layers": self.model_config.num_layers,
            "dtype": str(self.dtype),
            "device": str(self.device),
        }
