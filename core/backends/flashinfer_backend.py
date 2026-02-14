"""
FlashInfer attention backend implementation.
Provides FlashInfer-based attention computation with plan-then-run pattern.
"""

from typing import Optional, Tuple
import torch
import logging

try:
    import flashinfer
    from flashinfer import BatchDecodeWithPagedKVCacheWrapper, BatchPrefillWithPagedKVCacheWrapper
    FLASHINFER_AVAILABLE = True
except ImportError:
    FLASHINFER_AVAILABLE = False
    flashinfer = None
    BatchDecodeWithPagedKVCacheWrapper = None
    BatchPrefillWithPagedKVCacheWrapper = None

from .metadata import AttentionMetadata


logger = logging.getLogger(__name__)


class FlashInferBackend:
    """
    FlashInfer attention backend implementation.
    
    This backend provides efficient attention computation using FlashInfer operators,
    supporting both prefill and decode phases with paged KV cache.
    """
    
    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        page_size: int = 16,
        dtype: torch.dtype = torch.float16,
        device: str = "cuda",
        workspace_size_mb: int = 128
    ):
        """
        Initialize FlashInfer backend.
        
        Args:
            num_heads: Number of attention heads
            head_dim: Dimension of each attention head
            page_size: Size of each page in paged KV cache
            dtype: Data type for computations
            device: Computing device
            workspace_size_mb: Size of workspace buffer in MB
        """
        if not FLASHINFER_AVAILABLE:
            raise RuntimeError(
                "FlashInfer is not installed. Please install with: pip install flashinfer"
            )
        
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.page_size = page_size
        self.dtype = dtype
        self.device = device
        
        # Initialize workspace buffers
        workspace_size = workspace_size_mb * 1024 * 1024
        self.decode_workspace = torch.empty(
            workspace_size, dtype=torch.uint8, device=device
        )
        self.prefill_workspace = torch.empty(
            workspace_size, dtype=torch.uint8, device=device
        )
        
        # Initialize wrappers
        self.decode_wrapper = BatchDecodeWithPagedKVCacheWrapper(
            self.decode_workspace, "NHD"
        )
        self.prefill_wrapper = BatchPrefillWithPagedKVCacheWrapper(
            self.prefill_workspace, "NHD"
        )
        
        # Track if we have planned for current batch
        self._current_metadata: Optional[AttentionMetadata] = None
        self._is_planned = False
        
        logger.info(f"Initialized FlashInferBackend: num_heads={num_heads}, head_dim={head_dim}, page_size={page_size}")
    
    def plan(self, metadata: AttentionMetadata) -> None:
        """
        Plan attention computation for given metadata.
        
        This method creates auxiliary data structures needed for attention computation.
        Should be called once per batch before run().
        
        Args:
            metadata: Attention metadata containing block tables and configuration
        """
        if metadata.is_prefill:
            self._plan_prefill(metadata)
        else:
            self._plan_decode(metadata)
        
        self._current_metadata = metadata
        self._is_planned = True
        logger.debug(f"Planned attention: is_prefill={metadata.is_prefill}, batch_size={metadata.batch_size}")
    
    def _plan_prefill(self, metadata: AttentionMetadata) -> None:
        """Plan prefill attention computation."""
        if metadata.qo_indptr is None:
            raise ValueError("qo_indptr is required for prefill attention")
        
        self.prefill_wrapper.plan(
            metadata.qo_indptr,
            metadata.paged_kv_indptr,
            metadata.paged_kv_indices,
            metadata.paged_kv_last_page_len,
            self.num_heads,      # num_qo_heads
            self.num_heads,      # num_kv_heads
            self.head_dim,       # head_dim_qk
            self.page_size,      # page_size
            causal=metadata.causal,
            q_data_type=self.dtype,
            kv_data_type=self.dtype,
        )
    
    def _plan_decode(self, metadata: AttentionMetadata) -> None:
        """Plan decode attention computation."""
        self.decode_wrapper.plan(
            metadata.paged_kv_indptr,
            metadata.paged_kv_indices,
            metadata.paged_kv_last_page_len,
            self.num_heads,      # num_qo_heads
            self.num_heads,      # num_kv_heads
            self.head_dim,       # head_dim
            self.page_size,      # page_size
            data_type=self.dtype,
        )
    
    def run(
        self,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        layer_idx: int = 0,
        metadata: Optional[AttentionMetadata] = None
    ) -> torch.Tensor:
        """
        Run attention computation.
        
        Args:
            query: Query tensor
            key_cache: Key cache tensor
            value_cache: Value cache tensor
            layer_idx: Layer index (for future multi-layer support)
            metadata: Optional metadata override
            
        Returns:
            Attention output tensor
        """
        if metadata is not None and metadata != self._current_metadata:
            # Re-plan if metadata changed
            self.plan(metadata)
        elif not self._is_planned:
            raise RuntimeError("Must call plan() before run()")
        
        if self._current_metadata.is_prefill:
            return self.prefill_wrapper.run(query, key_cache, value_cache)
        else:
            return self.decode_wrapper.run(query, key_cache, value_cache)
    
    def reset_plan_state(self) -> None:
        """Reset the plan state for new batch."""
        self._current_metadata = None
        self._is_planned = False
    
    def __repr__(self) -> str:
        return (f"FlashInferBackend(num_heads={self.num_heads}, head_dim={self.head_dim}, "
                f"page_size={self.page_size}, device={self.device})")