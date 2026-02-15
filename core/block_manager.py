"""
High-performance BlockManager for KV Cache allocation, similar to vLLM.

This module manages the logical allocation of physical blocks for KV cache.
It is responsible for:
- Allocating physical blocks for given number of tokens
- Freeing previously allocated blocks
- Tracking the number of free/allocated blocks

Note: Data movement into the KV cache pool is handled by FlashInferBackend
using dedicated CUDA operators, not by this class.
"""

import torch
from typing import List, Deque
from collections import deque
import threading


class BlockManager:
    """
    Manages physical blocks for KV cache allocation.
    
    Responsible only for logical resource management (allocation/deallocation).
    Data movement is handled by FlashInferBackend.
    """
    
    def __init__(
        self,
        num_blocks: int,
        num_layers: int,
        num_key_value_heads: int,
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
            num_key_value_heads: Number of key/value heads (for GQA)
            head_dim: Dimension of each attention head
            block_size: Size of each block in tokens
            dtype: Data type for the tensor
            device: Device to allocate tensor on
        """
        self.num_blocks = num_blocks
        self.num_layers = num_layers
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.block_size = block_size
        self.dtype = dtype
        self.device = device
        
        self.kv_cache_pool = torch.zeros(
            (num_layers, num_blocks, 2, block_size, num_key_value_heads, head_dim),
            dtype=dtype,
            device=device
        )
        
        self._free_blocks: Deque[int] = deque(range(num_blocks))
        self._allocated_blocks = set()
        self._lock = threading.Lock()
    
    def allocate_blocks(self, num_tokens: int) -> List[int]:
        """
        Allocate physical blocks for given number of tokens.
        
        Args:
            num_tokens: Number of tokens to allocate blocks for
            
        Returns:
            List of physical block indices
            
        Raises:
            ValueError: If num_tokens is negative
            RuntimeError: If not enough free blocks available
        """
        if num_tokens < 0:
            raise ValueError(f"num_tokens must be positive, got {num_tokens}")
        
        if num_tokens == 0:
            return []
        
        num_blocks_needed = (num_tokens + self.block_size - 1) // self.block_size
        
        with self._lock:
            if len(self._free_blocks) < num_blocks_needed:
                raise RuntimeError(
                    f"Not enough free blocks: need {num_blocks_needed}, "
                    f"have {len(self._free_blocks)}"
                )
            
            allocated_indices = []
            for _ in range(num_blocks_needed):
                block_idx = self._free_blocks.popleft()
                allocated_indices.append(block_idx)
                self._allocated_blocks.add(block_idx)
            
            return allocated_indices
    
    def free_blocks(self, block_indices: List[int]) -> None:
        """
        Free previously allocated physical blocks.
        
        Args:
            block_indices: List of physical block indices to free
        """
        with self._lock:
            for block_idx in block_indices:
                if block_idx in self._allocated_blocks:
                    self._allocated_blocks.remove(block_idx)
                    self._free_blocks.append(block_idx)
    
    @property
    def num_free_blocks(self) -> int:
        """Get number of free blocks. Thread-safe and O(1)."""
        return len(self._free_blocks)
    
    @property
    def num_allocated_blocks(self) -> int:
        """Get number of allocated blocks. Thread-safe and O(1)."""
        return len(self._allocated_blocks)
    
    def reset(self) -> None:
        """Reset all blocks to free state. Used for testing."""
        with self._lock:
            self._free_blocks = deque(range(self.num_blocks))
            self._allocated_blocks.clear()
    
    def __repr__(self) -> str:
        return (f"BlockManager(num_blocks={self.num_blocks}, "
                f"free={self.num_free_blocks}, allocated={self.num_allocated_blocks})")