"""
FlashInfer attention backend implementation.
Provides FlashInfer-based attention computation with plan-then-run pattern.
"""

from typing import Optional, Tuple
import torch
import logging

import flashinfer
from flashinfer import BatchDecodeWithPagedKVCacheWrapper, BatchPrefillWithPagedKVCacheWrapper

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
        kv_cache_pool: torch.Tensor,
        num_key_value_heads: Optional[int] = None,
        page_size: int = 16,
        dtype: torch.dtype = torch.float16,
        device: str = "cuda",
        workspace_size_mb: int = 128
    ):
        """
        Initialize FlashInfer backend with GQA support.
        
        Args:
            num_heads: Number of attention heads (Query heads)
            head_dim: Dimension of each attention head
            kv_cache_pool: KV cache pool from BlockManager
            num_key_value_heads: Number of key/value heads (for GQA). If None, defaults to num_heads
            page_size: Size of each page in paged KV cache
            dtype: Data type for computations
            device: Computing device
            workspace_size_mb: Size of workspace buffer in MB
        """
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_key_value_heads = num_key_value_heads or num_heads
        self.page_size = page_size
        self.dtype = dtype
        self.device = device
        self.kv_cache_pool = kv_cache_pool
        
        if num_heads % self.num_key_value_heads != 0:
            raise ValueError(f"num_heads ({num_heads}) must be divisible by num_key_value_heads ({self.num_key_value_heads})")
        self.num_key_value_groups = num_heads // self.num_key_value_heads
        
        workspace_size = workspace_size_mb * 1024 * 1024
        self.decode_workspace = torch.empty(
            workspace_size, dtype=torch.uint8, device=device
        )
        self.prefill_workspace = torch.empty(
            workspace_size, dtype=torch.uint8, device=device
        )
        
        self.decode_wrapper = BatchDecodeWithPagedKVCacheWrapper(
            self.decode_workspace, "NHD"
        )
        self.prefill_wrapper = BatchPrefillWithPagedKVCacheWrapper(
            self.prefill_workspace, "NHD"
        )
        
        self._current_metadata: Optional[AttentionMetadata] = None
        self._is_planned = False
        
        logger.info(f"Initialized FlashInferBackend: num_heads={num_heads}, num_key_value_heads={self.num_key_value_heads}, "
                   f"head_dim={head_dim}, page_size={page_size}, num_key_value_groups={self.num_key_value_groups}")
    
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
            self.num_heads,
            self.num_key_value_heads,
            self.head_dim,
            self.page_size,
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
            self.num_heads,
            self.num_key_value_heads,
            self.head_dim,
            self.page_size,
            data_type=self.dtype,
        )
    
    def run(
        self,
        query: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int = 0,
        metadata: Optional[AttentionMetadata] = None
    ) -> torch.Tensor:
        """
        Run attention computation with GQA support using FlashInfer API.
        
        Args:
            query: Query tensor [total_tokens, num_heads, head_dim]
            key_states: Key states tensor [total_tokens, num_key_value_heads, head_dim]
                       (to be written to KV cache pool)
            value_states: Value states tensor [total_tokens, num_key_value_heads, head_dim]
                         (to be written to KV cache pool)
            layer_idx: Layer index for multi-layer support
            metadata: Optional metadata override
            
        Returns:
            Attention output tensor [total_tokens, num_heads, head_dim]
        """
        if metadata is not None and metadata != self._current_metadata:
            self.plan(metadata)
        elif not self._is_planned:
            raise RuntimeError("Must call plan() before run()")
        
        kv_cache_layer = self.kv_cache_pool[layer_idx]
        
        total_tokens = key_states.shape[0]
        device = key_states.device
        
        batch_indices = torch.zeros(total_tokens, dtype=torch.int32, device=device)
        positions = torch.arange(total_tokens, dtype=torch.int32, device=device)
        
        flashinfer.append_paged_kv_cache(
            key_states,
            value_states,
            batch_indices,
            positions,
            kv_cache_layer,
            metadata.paged_kv_indices,
            metadata.paged_kv_indptr,
            metadata.paged_kv_last_page_len,
            kv_layout="NHD"
        )
        
        output = torch.empty_like(query)
        
        if self._current_metadata.is_prefill:
            output = self.prefill_wrapper.run(
                q=query,
                paged_kv_cache=kv_cache_layer,
                out=output
            )
        else:
            output = self.decode_wrapper.run(
                q=query,
                kv_cache_layer=kv_cache_layer,
                output=output
            )
        
        return output
    
    def reset_plan_state(self) -> None:
        """Reset the plan state for new batch."""
        self._current_metadata = None
        self._is_planned = False
    
    def __repr__(self) -> str:
        return (f"FlashInferBackend(num_heads={self.num_heads}, head_dim={self.head_dim}, "
                f"page_size={self.page_size}, device={self.device})")