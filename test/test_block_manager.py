"""
Unit tests for BlockManager.
Validates BlockManager as a logical resource scheduler.
"""

import unittest
import torch
import threading
from collections import deque
import os
import sys

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.block_manager import BlockManager

class TestBlockManager(unittest.TestCase):
    """Test cases for BlockManager logical resource management."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.num_blocks = 10
        self.num_layers = 2
        self.num_key_value_heads = 8
        self.head_dim = 64
        self.block_size = 16
        
        self.block_manager = BlockManager(
            num_blocks=self.num_blocks,
            num_layers=self.num_layers,
            num_key_value_heads=self.num_key_value_heads,
            head_dim=self.head_dim,
            block_size=self.block_size,
            dtype=torch.float32,
            device="cpu"
        )
    
    def test_kv_cache_pool_initialization(self):
        """Test kv_cache_pool shape and device properties only."""
        expected_shape = (
            self.num_layers, 
            self.num_blocks, 
            2,  # K and V
            self.block_size, 
            self.num_key_value_heads, 
            self.head_dim
        )
        self.assertEqual(self.block_manager.kv_cache_pool.shape, expected_shape)
        self.assertEqual(self.block_manager.kv_cache_pool.dtype, torch.float32)
        self.assertEqual(self.block_manager.kv_cache_pool.device.type, "cpu")
    
    def test_initial_state(self):
        """Test initial state of block manager."""
        self.assertEqual(self.block_manager.num_free_blocks, self.num_blocks)
        self.assertEqual(self.block_manager.num_allocated_blocks, 0)
    
    def test_ceiling_division_single_tokens(self):
        """Test ceiling division: 1 token -> 1 block."""
        block_indices = self.block_manager.allocate_blocks(1)
        self.assertEqual(len(block_indices), 1)
        self.assertEqual(self.block_manager.num_allocated_blocks, 1)
        self.block_manager.free_blocks(block_indices)
    
    def test_ceiling_division_exact_block_size(self):
        """Test ceiling division: 16 tokens -> 1 block (block_size=16)."""
        block_indices = self.block_manager.allocate_blocks(16)
        self.assertEqual(len(block_indices), 1)
        self.assertEqual(self.block_manager.num_allocated_blocks, 1)
        self.block_manager.free_blocks(block_indices)
    
    def test_ceiling_division_over_one_block(self):
        """Test ceiling division: 17 tokens -> 2 blocks (block_size=16)."""
        block_indices = self.block_manager.allocate_blocks(17)
        self.assertEqual(len(block_indices), 2)
        self.assertEqual(self.block_manager.num_allocated_blocks, 2)
        self.block_manager.free_blocks(block_indices)
    
    def test_ceiling_division_various(self):
        """Test ceiling division for various token counts."""
        test_cases = [
            (1, 1), (15, 1), (16, 1), (17, 2), (31, 2), (32, 2), (33, 3)
        ]
        for num_tokens, expected_blocks in test_cases:
            with self.subTest(num_tokens=num_tokens):
                self.block_manager.reset()
                block_indices = self.block_manager.allocate_blocks(num_tokens)
                self.assertEqual(
                    len(block_indices), expected_blocks,
                    f"{num_tokens} tokens should require {expected_blocks} blocks"
                )
                self.block_manager.free_blocks(block_indices)
    
    def test_allocation_uniqueness(self):
        """Test that consecutive allocations return unique block indices."""
        allocations = []
        
        # Allocate several times
        for _ in range(5):
            indices = self.block_manager.allocate_blocks(16)
            self.assertIsNotNone(indices)
            allocations.append(indices)
        
        # Flatten and check uniqueness
        all_indices = [idx for allocation in allocations for idx in allocation]
        self.assertEqual(len(all_indices), len(set(all_indices)), 
                        "Allocated block indices should be unique")
    
    def test_allocation_state_consistency(self):
        """Test that free + allocated always equals total blocks."""
        for i in range(1, self.num_blocks + 1):
            with self.subTest(allocation=i):
                block_indices = self.block_manager.allocate_blocks(16)
                self.assertEqual(
                    self.block_manager.num_free_blocks + self.block_manager.num_allocated_blocks,
                    self.num_blocks,
                    f"State inconsistent after {i}th allocation"
                )
                self.block_manager.free_blocks(block_indices)
    
    def test_fifo_recycling(self):
        """Test FIFO behavior: freed blocks are added to tail, allocated from head."""
        # Allocate blocks: [0,1,2,...] -> popleft 0, deque is [1,2,3,...,9]
        first_allocation = self.block_manager.allocate_blocks(16)
        first_block = first_allocation[0]
        
        # Free the block: append 0 to tail -> deque is [1,2,3,...,9,0]
        self.block_manager.free_blocks(first_allocation)
        
        # Allocate again: popleft from head -> should get 1, not 0
        second_allocation = self.block_manager.allocate_blocks(16)
        second_block = second_allocation[0]
        
        # FIFO: freed blocks go to tail, new allocations come from head
        # So block 0 should NOT be reused immediately (it's at tail)
        # But it should be available in the free pool
        self.assertIn(first_block, self.block_manager._free_blocks)
    
    def test_full_allocation_and_recovery(self):
        """Test allocating all blocks and then freeing them."""
        total_tokens = self.num_blocks * self.block_size
        all_allocated_indices = []
        
        # Allocate all blocks
        while self.block_manager.num_free_blocks > 0:
            indices = self.block_manager.allocate_blocks(self.block_size)
            self.assertIsNotNone(indices)
            all_allocated_indices.extend(indices)
        
        # Verify all blocks are allocated
        self.assertEqual(self.block_manager.num_allocated_blocks, self.num_blocks)
        self.assertEqual(self.block_manager.num_free_blocks, 0)
        self.assertEqual(len(all_allocated_indices), self.num_blocks)
        
        # Free all blocks
        self.block_manager.free_blocks(all_allocated_indices)
        
        # Verify recovery
        self.assertEqual(self.block_manager.num_free_blocks, self.num_blocks)
        self.assertEqual(self.block_manager.num_allocated_blocks, 0)
    
    def test_over_limit_allocation_raises(self):
        """Test that requesting more tokens than available raises RuntimeError."""
        total_tokens = self.num_blocks * self.block_size
        over_limit_tokens = total_tokens + 1
        
        with self.assertRaises(RuntimeError) as context:
            self.block_manager.allocate_blocks(over_limit_tokens)
        
        self.assertIn("Not enough free blocks", str(context.exception))
    
    def test_over_limit_does_not_corrupt_state(self):
        """Test that failed allocation does not affect internal state."""
        total_tokens = self.num_blocks * self.block_size
        
        # Try to over-allocate
        try:
            self.block_manager.allocate_blocks(total_tokens + 1)
        except RuntimeError:
            pass
        
        # State should be unchanged
        self.assertEqual(self.block_manager.num_free_blocks, self.num_blocks)
        self.assertEqual(self.block_manager.num_allocated_blocks, 0)
    
    def test_double_free_protection(self):
        """Test that freeing an already freed block is handled gracefully."""
        # Allocate and free a block
        indices = self.block_manager.allocate_blocks(16)
        self.assertEqual(len(indices), 1)
        
        # Free once
        self.block_manager.free_blocks(indices)
        self.assertEqual(self.block_manager.num_free_blocks, self.num_blocks)
        
        # Try to free again - should not corrupt state
        self.block_manager.free_blocks(indices)
        self.assertEqual(self.block_manager.num_free_blocks, self.num_blocks,
                        "Double free should not increase free block count")
    
    def test_zero_token_allocation(self):
        """Test allocation for 0 tokens returns empty list."""
        block_indices = self.block_manager.allocate_blocks(0)
        self.assertEqual(block_indices, [])
        self.assertEqual(self.block_manager.num_allocated_blocks, 0)
    
    def test_negative_token_allocation(self):
        """Test allocation for negative tokens raises error."""
        with self.assertRaises((ValueError, RuntimeError)):
            self.block_manager.allocate_blocks(-1)
    
    def test_concurrent_allocation_no_overflow(self):
        """Test that concurrent allocations never exceed total blocks."""
        barrier = threading.Barrier(5)
        results = []
        lock = threading.Lock()
        
        def allocate_thread():
            try:
                barrier.wait()
                indices = self.block_manager.allocate_blocks(16)
                with lock:
                    results.append(("success", indices))
            except RuntimeError:
                with lock:
                    results.append(("failed", None))
        
        threads = [threading.Thread(target=allocate_thread) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        # Count successful allocations
        successful_count = sum(1 for status, _ in results if status == "success")
        self.assertLessEqual(successful_count, self.num_blocks)
        
        # All indices should be unique
        all_indices = [idx for status, idx in results if status == "success" and idx for idx in idx]
        self.assertEqual(len(all_indices), len(set(all_indices)),
                        "No two threads should receive the same block index")
    
    def test_concurrent_free_allocation(self):
        """Test concurrent free and allocate operations."""
        # Pre-allocate some blocks
        pre_allocated = self.block_manager.allocate_blocks(48)
        
        results = []
        lock = threading.Lock()
        
        def free_worker():
            try:
                indices = self.block_manager.allocate_blocks(16)
                with lock:
                    results.append(("alloc", indices))
            except RuntimeError:
                with lock:
                    results.append(("alloc_failed", None))
        
        def allocate_worker():
            try:
                if pre_allocated:
                    self.block_manager.free_blocks(pre_allocated[:1])
                    with lock:
                        results.append(("free", [pre_allocated[0]]))
                    pre_allocated.clear()
            except Exception:
                pass
        
        # Start threads
        threads = []
        for _ in range(3):
            threads.append(threading.Thread(target=free_worker))
            threads.append(threading.Thread(target=allocate_worker))
        
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        # Verify no corruption
        self.assertEqual(
            self.block_manager.num_free_blocks + self.block_manager.num_allocated_blocks,
            self.num_blocks
        )
    
    def test_reset(self):
        """Test resetting the block manager to initial state."""
        # Allocate some blocks
        for _ in range(3):
            self.block_manager.allocate_blocks(16)
        
        self.assertEqual(self.block_manager.num_allocated_blocks, 3)
        self.assertEqual(self.block_manager.num_free_blocks, self.num_blocks - 3)
        
        # Reset
        self.block_manager.reset()
        
        self.assertEqual(self.block_manager.num_free_blocks, self.num_blocks)
        self.assertEqual(self.block_manager.num_allocated_blocks, 0)
    
    def test_repr(self):
        """Test string representation."""
        repr_str = repr(self.block_manager)
        self.assertIn("BlockManager", repr_str)
        self.assertIn(f"num_blocks={self.num_blocks}", repr_str)
        self.assertIn("free=", repr_str)
        self.assertIn("allocated=", repr_str)


if __name__ == "__main__":
    unittest.main()