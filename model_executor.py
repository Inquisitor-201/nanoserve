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
        kv_cache: torch.Tensor,
        block_tables: List[List[int]],
        seq_lengths: List[int],
        is_prefill: bool = True,
        causal: bool = True
    ) -> torch.Tensor:
        """
        Compute attention using FlashInfer operators.
        
        Args:
            query: Query tensor [batch_size, num_heads, head_dim] or [total_tokens, num_heads, head_dim] for prefill
            kv_cache: Combined KV cache tensor [num_blocks, 2, block_size, num_heads, head_dim] - physical pool
                      where [.., 0, ...] is key and [.., 1, ...] is value
            block_tables: Block tables for each sequence
            seq_lengths: Sequence lengths
            is_prefill: Whether this is prefill or decode
            causal: Whether to use causal mask
            
        Returns:
            Attention output tensor
        """
        # Prepare FlashInfer inputs - ensures all indices are int32
        flashinfer_inputs = self.prepare_flashinfer_inputs(
            block_tables, seq_lengths, is_prefill
        )
    
        # FlashInfer will handle the paged access internally using block tables and indices
        if is_prefill:
            # Prefill phase: compute attention for all tokens
            # Step 1: Plan phase - construct auxiliary data structures
            # Build qo_indptr based on sequence lengths
            qo_indptr = [0]
            current_pos = 0
            for seq_len in seq_lengths:
                current_pos += seq_len
                qo_indptr.append(current_pos)
            qo_indptr = torch.tensor(qo_indptr, dtype=torch.int32, device=self.device)
            
            self.prefill_wrapper.plan(
                qo_indptr=qo_indptr,  # Query/output indptr
                paged_kv_indptr=flashinfer_inputs["paged_kv_indptr"],    # KV cache indptr
                paged_kv_indices=flashinfer_inputs["paged_kv_indices"],   # Block indices
                paged_kv_last_page_len=flashinfer_inputs["paged_kv_last_page_len"],  # Last page lengths
                num_qo_heads=self.num_heads,
                num_kv_heads=self.num_heads,
                head_dim_qk=self.head_dim,
                page_size=self.page_size,
                causal=causal,
                q_data_type=self.dtype,
                kv_data_type=self.dtype,
            )
            
            # Step 2: Run phase - execute attention computation
            output = self.prefill_wrapper.run(query, kv_cache)
        else:
            # Decode phase: compute attention for new token only
            # Step 1: Plan phase - setup decode wrapper
            self.decode_wrapper.plan(
                flashinfer_inputs["paged_kv_indptr"],    # KV cache indptr
                flashinfer_inputs["paged_kv_indices"],   # Block indices
                flashinfer_inputs["paged_kv_last_page_len"],  # Last page lengths
                self.num_heads,      # num_qo_heads
                self.num_heads,      # num_kv_heads
                self.head_dim,       # head_dim
                self.page_size,      # page_size
                data_type=self.dtype, # Use the configured dtype
            )
            
            # Step 2: Run phase - execute decode computation
            output = self.decode_wrapper.run(query, kv_cache)
        
        return output
    

    
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
        
        # Get the raw KV cache pool for the current layer (zero-copy view)
        # Shape: [num_blocks, 2, block_size, num_heads, head_dim] - this is the NHD format FlashInfer expects
        layer_kv_cache = self.block_manager.kv_cache_pool[layer_idx, :, :, :, :, :]  # [num_blocks, 2, block_size, num_heads, head_dim]
        
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
        
        # Compute attention with FlashInfer using the raw physical pool
        # FlashInfer will handle the paged access internally using the block tables
        attention_output = self.compute_attention_with_flashinfer(
            query,
            layer_kv_cache,  # Pass the entire layer's KV cache pool directly: [num_blocks, 2, block_size, num_heads, head_dim]
            block_tables,
            seq_lengths,
            is_prefill=is_prefill
        )
        
        # Todo: Append new KV values to the KV cache pool
        raise NotImplementedError("[TODO] KV cache management not implemented yet")

        return attention_output
    
    def __repr__(self) -> str:
        return (f"ModelExecutor(num_heads={self.num_heads}, head_dim={self.head_dim}, "
                f"page_size={self.page_size}, device={self.device})")