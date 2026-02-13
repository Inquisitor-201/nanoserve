"""
High-performance BlockManager for KV Cache allocation, similar to vLLM.
"""

import torch
from typing import List, Optional, Deque
from collections import deque
import threading


class BlockManager:
    """
    Manages physical blocks for KV cache allocation.
    
    Pre-allocates a large tensor pool and manages logical block allocation.
    """
    
    def __init__(
        self,
        num_blocks: int,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        block_size: int,
        dtype: torch.dtype = torch.float16,
        device: str = "cuda"
    ):
        """
        Initialize BlockManager with pre-allocated KV cache pool.
        
        Args:
            num_blocks: Total number of physical blocks
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            head_dim: Dimension of each attention head
            block_size: Size of each block in tokens
            dtype: Data type for the tensor
            device: Device to allocate tensor on
        """
        self.num_blocks = num_blocks
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.block_size = block_size
        self.dtype = dtype
        self.device = device
        
        # Pre-allocate KV cache pool: (num_layers, num_blocks, 2, block_size, num_heads, head_dim)
        # 2 is for K and V caches, following FlashInfer official NHD layout specification
        # Format per layer: [num_layers, max_num_pages, 2, page_size, num_heads, head_dim]
        self.kv_cache_pool = torch.zeros(
            (num_layers, num_blocks, 2, block_size, num_heads, head_dim),
            dtype=dtype,
            device=device
        )
        
        # Free blocks management using deque for O(1) operations
        self._free_blocks: Deque[int] = deque(range(num_blocks))
        
        # Track allocated blocks for debugging/monitoring
        self.allocated_blocks = set()
        
        # Thread safety lock
        self._lock = threading.Lock()
        
    def allocate_blocks(self, num_tokens: int) -> Optional[List[int]]:
        """
        Allocate physical blocks for given number of tokens.
        
        Args:
            num_tokens: Number of tokens to allocate blocks for
            
        Returns:
            List of physical block indices, or None if allocation fails
        """
        # Calculate number of blocks needed
        num_blocks_needed = (num_tokens + self.block_size - 1) // self.block_size
        
        with self._lock:
            # Check if we have enough free blocks
            if len(self._free_blocks) < num_blocks_needed:
                return None
            
            # Allocate blocks
            allocated_indices = []
            for _ in range(num_blocks_needed):
                block_idx = self._free_blocks.popleft()
                allocated_indices.append(block_idx)
                self.allocated_blocks.add(block_idx)
            
            return allocated_indices
    
    def free_blocks(self, block_indices: List[int]) -> None:
        """
        Free previously allocated physical blocks.
        
        Args:
            block_indices: List of physical block indices to free
        """
        with self._lock:
            for block_idx in block_indices:
                if block_idx in self.allocated_blocks:
                    self.allocated_blocks.remove(block_idx)
                    self._free_blocks.append(block_idx)
    
    def get_block_tensor(self, block_idx: int) -> torch.Tensor:
        """
        Get the tensor for a specific physical block.
        
        Args:
            block_idx: Physical block index
            
        Returns:
            Tensor slice for the specified block
        """
        if block_idx < 0 or block_idx >= self.num_blocks:
            raise ValueError(f"Block index {block_idx} out of range [0, {self.num_blocks})")
        
        return self.kv_cache_pool[:, block_idx, :, :, :, :]
    
    def get_kv_cache_for_blocks(self, block_indices: List[int]) -> torch.Tensor:
        """
        Get KV cache tensor for a list of blocks.
        
        Args:
            block_indices: List of physical block indices
            
        Returns:
            Stacked tensor of KV caches for the specified blocks
        """
        # Select only the specified blocks: [num_layers, num_selected_blocks, 2, block_size, num_heads, head_dim]
        return self.kv_cache_pool[:, block_indices, :, :, :, :]
    
    @property
    def num_free_blocks(self) -> int:
        """Get number of free blocks."""
        return len(self._free_blocks)
    
    @property
    def num_allocated_blocks(self) -> int:
        """Get number of allocated blocks."""
        return len(self.allocated_blocks)
    
    def reset(self) -> None:
        """Reset all blocks to free state."""
        with self._lock:
            self._free_blocks = deque(range(self.num_blocks))
            self.allocated_blocks.clear()
    
    def append_slot(self, block_indices: List[int], k_tensor: torch.Tensor, v_tensor: torch.Tensor) -> None:
        """
        Append new K and V values to the specified slots in the blocks.
        
        Args:
            block_indices: List of physical block indices
            k_tensor: Key tensor to append [num_tokens, num_heads, head_dim]
            v_tensor: Value tensor to append [num_tokens, num_heads, head_dim]
        """
        # Calculate how many tokens we're appending
        num_tokens = k_tensor.shape[0]
        
        # Validate input tensors
        assert k_tensor.shape == v_tensor.shape, "K and V tensors must have the same shape"
        assert k_tensor.dim() == 3, "K tensor must be 3D: [num_tokens, num_heads, head_dim]"
        
        # Determine how to distribute tokens across blocks
        tokens_per_block = self.block_size
        total_capacity = len(block_indices) * tokens_per_block
        
        if num_tokens > total_capacity:
            raise ValueError(f"Not enough capacity. Need {num_tokens} tokens, but only have {total_capacity} slots")
        
        # Copy the K and V values to the appropriate positions in the blocks
        token_idx = 0
        for block_idx in block_indices:
            # Calculate how many tokens we can place in this block
            remaining_tokens = num_tokens - token_idx
            tokens_in_this_block = min(remaining_tokens, tokens_per_block)
            
            # Update the KV cache pool for each layer
            for layer_idx in range(self.num_layers):
                # Place the K and V values in the appropriate positions
                # New layout: [num_layers, num_blocks, 2, block_size, num_heads, head_dim]
                self.kv_cache_pool[layer_idx, block_idx, 0, :tokens_in_this_block, :, :] = \
                    k_tensor[token_idx:token_idx + tokens_in_this_block, :, :]
                self.kv_cache_pool[layer_idx, block_idx, 1, :tokens_in_this_block, :, :] = \
                    v_tensor[token_idx:token_idx + tokens_in_this_block, :, :]
            
            token_idx += tokens_in_this_block
            if token_idx >= num_tokens:
                break
    
    def __repr__(self) -> str:
        return (f"BlockManager(num_blocks={self.num_blocks}, "
                f"free={self.num_free_blocks}, allocated={self.num_allocated_blocks})")