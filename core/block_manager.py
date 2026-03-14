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

from .config import ModelConfig, EngineArgs


class BlockManager:
    """
    Manages physical blocks for KV cache allocation.
    
    Responsible only for logical resource management (allocation/deallocation).
    Data movement is handled by FlashInferBackend.
    """
    
    def __init__(
        self,
        model_config: ModelConfig,
        engine_args: EngineArgs,
    ):
        """
        Initialize BlockManager with pre-allocated KV cache pool.
        
        Args:
            model_config: ModelConfig containing model structure parameters
            engine_args: EngineArgs containing resource allocation parameters
        """
        self.num_blocks = engine_args.num_blocks
        num_layers = model_config.num_layers
        num_key_value_heads = model_config.num_key_value_heads
        head_dim = model_config.head_dim
        block_size = engine_args.block_size
        dtype = model_config.dtype
        device = engine_args.device
        
        self._validate_block_size_consistency(block_size, model_config.page_size)
        
        self.num_layers = num_layers
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.block_size = block_size
        self.dtype = dtype
        self.device = device
        
        self.kv_cache_pool = torch.zeros(
            (num_layers, self.num_blocks, 2, block_size, num_key_value_heads, head_dim),
            dtype=dtype,
            device=device
        )
        
        self._free_blocks: Deque[int] = deque(range(self.num_blocks))
        self._allocated_blocks = set()
        self._lock = threading.Lock()
        
        logger.info(f"BlockManager initialized: num_blocks={self.num_blocks}, "
                   f"block_size={block_size}, num_layers={num_layers}, "
                   f"num_kv_heads={num_key_value_heads}, head_dim={head_dim}")

    def _validate_block_size_consistency(self, engine_block_size: int, model_page_size: int) -> None:
        if engine_block_size != model_page_size:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(
                f"Block size mismatch: EngineArgs.block_size={engine_block_size}, "
                f"ModelConfig.page_size={model_page_size}. Using engine_args.block_size={engine_block_size}."
            )

    def allocate_blocks(self, current_blocks: List[int], target_num_tokens: int) -> List[int]:
        """
        Allocate physical blocks for a request to reach the target number of tokens.
        If the current blocks already satisfy the target, return the current blocks.
        Otherwise, allocate additional blocks to meet the target.
        
        Args:
            current_blocks: Current blocks allocated to the request
            target_num_tokens: Target number of tokens the request needs to support
            
        Returns:
            Updated list of physical block indices
        """
        assert target_num_tokens >= 0
        
        # Calculate how many blocks are needed for the target tokens
        target_blocks_needed = (target_num_tokens + self.block_size - 1) // self.block_size
        current_block_count = len(current_blocks)
        
        # If we already have enough blocks, return current blocks
        if current_block_count >= target_blocks_needed:
            return current_blocks
        
        # Need to allocate more blocks
        blocks_to_allocate = target_blocks_needed - current_block_count
        
        with self._lock:
            if len(self._free_blocks) < blocks_to_allocate:
                # Not enough free blocks to satisfy the request
                return []
            
            allocated_indices = current_blocks[:]  # Copy current blocks
            for _ in range(blocks_to_allocate):
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

    def needs_new_block(self, current_blocks: List[int], target_num_tokens: int) -> bool:
        """
        Check if the current blocks need to be extended to reach the target number of tokens.
        
        Args:
            current_blocks: Current blocks allocated to the request
            target_num_tokens: Target number of tokens the request needs to support
            
        Returns:
            True if more blocks are needed, False if current blocks suffice
        """
        assert target_num_tokens >= 0
        
        # Calculate how many blocks are needed for the target tokens
        target_blocks_needed = (target_num_tokens + self.block_size - 1) // self.block_size
        current_block_count = len(current_blocks)
        
        return current_block_count < target_blocks_needed

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