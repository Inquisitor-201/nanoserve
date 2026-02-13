"""
ModelExecutor for integrating HuggingFace Llama with FlashInfer operators.
"""

import torch
import torch.nn as nn
from typing import List, Optional, Dict, Any
import numpy as np

try:
    import flashinfer
    from flashinfer import BatchDecodeWithPagedKVCacheWrapper, BatchPrefillWithPagedKVCacheWrapper
except ImportError:
    print("Warning: FlashInfer not installed. Install with: pip install flashinfer")
    flashinfer = None

from block_manager import BlockManager


class ModelExecutor:
    """
    Model executor that integrates HuggingFace Llama with FlashInfer operators.
    Handles KV cache management and attention computation replacement.
    """
    
    def __init__(
        self,
        block_manager: BlockManager,
        num_heads: int,
        head_dim: int,
        page_size: int = 16,
        dtype: torch.dtype = torch.float16,
        device: str = "cuda"
    ):
        """
        Initialize ModelExecutor.
        
        Args:
            block_manager: BlockManager instance for KV cache management
            num_heads: Number of attention heads
            head_dim: Dimension of each attention head
            page_size: Size of each page (should match block_size from BlockManager)
            dtype: Data type for computations
            device: Computing device
        """
        self.block_manager = block_manager
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.page_size = page_size
        self.dtype = dtype
        self.device = device
        
        # Validate page_size matches block_manager's block_size
        if page_size != block_manager.block_size:
            raise ValueError(f"page_size ({page_size}) must match block_manager.block_size ({block_manager.block_size})")
        
        # Initialize FlashInfer wrappers
        if flashinfer is None:
            raise RuntimeError("FlashInfer is required but not installed")
        
        self._init_flashinfer_wrappers()
        
    def _init_flashinfer_wrappers(self):
        """Initialize FlashInfer wrappers for decode and prefill operations."""
        # Workspace for decode operations
        self.decode_workspace = torch.empty(
            128 * 1024 * 1024, dtype=torch.uint8, device=self.device
        )
        
        # Workspace for prefill operations  
        self.prefill_workspace = torch.empty(
            128 * 1024 * 1024, dtype=torch.uint8, device=self.device
        )
        
        # Initialize wrappers
        self.decode_wrapper = BatchDecodeWithPagedKVCacheWrapper(
            self.decode_workspace, "NHD"
        )
        
        self.prefill_wrapper = BatchPrefillWithPagedKVCacheWrapper(
            self.prefill_workspace, "NHD"
        )
    
    def prepare_flashinfer_inputs(
        self,
        block_tables: List[List[int]],
        seq_lengths: List[int],
        is_prefill: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Convert block tables to FlashInfer required format.
        
        Args:
            block_tables: List of block tables for each sequence
            seq_lengths: List of sequence lengths
            is_prefill: Whether this is a prefill operation
            
        Returns:
            Dictionary containing FlashInfer input tensors
        """
        batch_size = len(block_tables)
        
        # Flatten block tables and create indices
        flat_indices = []
        indptr = [0]
        last_page_len = []
        
        for i, (block_table, seq_len) in enumerate(zip(block_tables, seq_lengths)):
            num_pages = (seq_len + self.page_size - 1) // self.page_size
            
            # Add block indices for this sequence
            flat_indices.extend(block_table[:num_pages])
            indptr.append(len(flat_indices))
            
            # Calculate last page length
            remainder = seq_len % self.page_size
            if remainder == 0:
                last_page_len.append(self.page_size)
            else:
                last_page_len.append(remainder)
        
        # Convert to tensors
        paged_kv_indices = torch.tensor(flat_indices, dtype=torch.int32, device=self.device)
        paged_kv_indptr = torch.tensor(indptr, dtype=torch.int32, device=self.device)
        paged_kv_last_page_len = torch.tensor(last_page_len, dtype=torch.int32, device=self.device)
        
        return {
            "paged_kv_indices": paged_kv_indices,
            "paged_kv_indptr": paged_kv_indptr,
            "paged_kv_last_page_len": paged_kv_last_page_len,
        }
    
    def compute_attention_with_flashinfer(
        self,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        block_tables: List[List[int]],
        seq_lengths: List[int],
        is_prefill: bool = True,
        causal: bool = True
    ) -> torch.Tensor:
        """
        Compute attention using FlashInfer operators.
        
        Args:
            query: Query tensor [batch_size, num_heads, head_dim]
            key_cache: Key cache tensor from BlockManager
            value_cache: Value cache tensor from BlockManager
            block_tables: Block tables for each sequence
            seq_lengths: Sequence lengths
            is_prefill: Whether this is prefill or decode
            causal: Whether to use causal mask
            
        Returns:
            Attention output tensor
        """
        batch_size = query.shape[0]
        
        # Prepare FlashInfer inputs
        flashinfer_inputs = self.prepare_flashinfer_inputs(
            block_tables, seq_lengths, is_prefill
        )
        
        if is_prefill:
            # Prefill phase: compute attention for all tokens
            output = self.prefill_wrapper.forward(
                query,
                key_cache,
                value_cache,
                causal=causal,
                **flashinfer_inputs
            )
        else:
            # Decode phase: compute attention for new token only
            output = self.decode_wrapper.forward(
                query,
                key_cache,
                value_cache,
                **flashinfer_inputs
            )
        
        return output
    
    def get_kv_cache_from_blocks(self, block_indices: List[int]) -> tuple:
        """
        Extract KV cache tensors for given block indices.
        
        Args:
            block_indices: List of physical block indices
            
        Returns:
            Tuple of (key_cache, value_cache) tensors
        """
        # Get KV cache for specified blocks
        kv_cache = self.block_manager.get_kv_cache_for_blocks(block_indices)
        
        # Split into key and value caches
        # Shape: [num_blocks, num_layers, 2, num_heads, head_dim, block_size]
        key_cache = kv_cache[:, :, 0, :, :, :]  # [num_blocks, num_layers, num_heads, head_dim, block_size]
        value_cache = kv_cache[:, :, 1, :, :, :]  # [num_blocks, num_layers, num_heads, head_dim, block_size]
        
        # Reshape for FlashInfer: flatten block and sequence dimensions
        num_blocks = key_cache.shape[0]
        num_layers = key_cache.shape[1]
        
        key_cache = key_cache.permute(1, 0, 2, 3, 4).reshape(
            num_layers, num_blocks * self.page_size, self.num_heads, self.head_dim
        )
        value_cache = value_cache.permute(1, 0, 2, 3, 4).reshape(
            num_layers, num_blocks * self.page_size, self.num_heads, self.head_dim
        )
        
        return key_cache, value_cache
    
    def execute_model(
        self,
        input_ids: torch.Tensor,
        block_tables: List[List[int]],
        seq_lengths: List[int],
        layer_idx: int = 0,
        is_prefill: bool = True
    ) -> torch.Tensor:
        """
        Execute model with FlashInfer attention for a single layer.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            block_tables: Block tables for each sequence
            seq_lengths: Sequence lengths
            layer_idx: Which layer to execute
            is_prefill: Whether this is prefill phase
            
        Returns:
            Model output tensor
        """
        batch_size = input_ids.shape[0]
        
        # Get all unique block indices
        all_blocks = list(set(block for table in block_tables for block in table))
        
        # Get KV cache tensors
        key_cache, value_cache = self.get_kv_cache_from_blocks(all_blocks)
        
        # Create a simple query tensor for demonstration
        # In real implementation, this would come from the model's hidden states
        if is_prefill:
            seq_len = input_ids.shape[1]
            query = torch.randn(
                batch_size * seq_len, self.num_heads, self.head_dim,
                dtype=self.dtype, device=self.device
            )
        else:
            # Decode phase: one token per sequence
            query = torch.randn(
                batch_size, self.num_heads, self.head_dim,
                dtype=self.dtype, device=self.device
            )
        
        # Use only the specified layer's KV cache
        layer_key_cache = key_cache[layer_idx]
        layer_value_cache = value_cache[layer_idx]
        
        # Compute attention with FlashInfer
        attention_output = self.compute_attention_with_flashinfer(
            query,
            layer_key_cache,
            layer_value_cache,
            block_tables,
            seq_lengths,
            is_prefill=is_prefill
        )
        
        return attention_output
    
    def __repr__(self) -> str:
        return (f"ModelExecutor(num_heads={self.num_heads}, head_dim={self.head_dim}, "
                f"page_size={self.page_size}, device={self.device})")