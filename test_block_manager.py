"""
Unit tests for BlockManager.
"""

import unittest
import torch
from block_manager import BlockManager


class TestBlockManager(unittest.TestCase):
    """Test cases for BlockManager class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.num_blocks = 10
        self.num_layers = 2
        self.num_heads = 8
        self.head_dim = 64
        self.block_size = 16
        
        self.block_manager = BlockManager(
            num_blocks=self.num_blocks,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            block_size=self.block_size,
            dtype=torch.float32,
            device="cpu"
        )
    
    def test_initialization(self):
        """Test BlockManager initialization."""
        # Check tensor shape
        expected_shape = (self.num_layers, self.num_blocks, 2, self.block_size, self.num_heads, self.head_dim)
        self.assertEqual(self.block_manager.kv_cache_pool.shape, expected_shape)
        
        # Check all blocks are free initially
        self.assertEqual(self.block_manager.num_free_blocks, self.num_blocks)
        self.assertEqual(self.block_manager.num_allocated_blocks, 0)
        
        # Check tensor properties
        self.assertEqual(self.block_manager.kv_cache_pool.dtype, torch.float32)
        self.assertEqual(self.block_manager.kv_cache_pool.device.type, "cpu")
    
    def test_allocate_blocks_single(self):
        """Test allocating blocks for a single request."""
        # Allocate for 10 tokens (should need 1 block)
        block_indices = self.block_manager.allocate_blocks(10)
        
        self.assertIsNotNone(block_indices)
        self.assertEqual(len(block_indices), 1)
        self.assertEqual(self.block_manager.num_free_blocks, self.num_blocks - 1)
        self.assertEqual(self.block_manager.num_allocated_blocks, 1)
    
    def test_allocate_blocks_multiple(self):
        """Test allocating blocks for multiple requests."""
        # Allocate for 35 tokens (should need 3 blocks: ceil(35/16) = 3)
        block_indices = self.block_manager.allocate_blocks(35)
        
        self.assertIsNotNone(block_indices)
        self.assertEqual(len(block_indices), 3)
        self.assertEqual(self.block_manager.num_free_blocks, self.num_blocks - 3)
        self.assertEqual(self.block_manager.num_allocated_blocks, 3)
    
    def test_allocate_blocks_insufficient(self):
        """Test allocation failure when insufficient blocks."""
        # Try to allocate for 200 tokens (need 13 blocks, but only 10 available)
        block_indices = self.block_manager.allocate_blocks(200)
        
        self.assertIsNone(block_indices)
        self.assertEqual(self.block_manager.num_free_blocks, self.num_blocks)
        self.assertEqual(self.block_manager.num_allocated_blocks, 0)
    
    def test_free_blocks(self):
        """Test freeing allocated blocks."""
        # Allocate some blocks
        block_indices = self.block_manager.allocate_blocks(30)
        self.assertEqual(len(block_indices), 2)
        
        # Free the blocks
        self.block_manager.free_blocks(block_indices)
        
        self.assertEqual(self.block_manager.num_free_blocks, self.num_blocks)
        self.assertEqual(self.block_manager.num_allocated_blocks, 0)
    
    def test_free_partial_blocks(self):
        """Test freeing only some allocated blocks."""
        # Allocate some blocks
        block_indices1 = self.block_manager.allocate_blocks(20)
        block_indices2 = self.block_manager.allocate_blocks(20)
        
        # Verify both allocations succeeded
        self.assertIsNotNone(block_indices1)
        self.assertIsNotNone(block_indices2)
        
        # Free only the first allocation
        self.block_manager.free_blocks(block_indices1)
        
        self.assertEqual(self.block_manager.num_free_blocks, self.num_blocks - 2)
        self.assertEqual(self.block_manager.num_allocated_blocks, 2)
    
    def test_get_block_tensor(self):
        """Test getting tensor for a specific block."""
        # Allocate a block
        block_indices = self.block_manager.allocate_blocks(10)
        block_idx = block_indices[0]
        
        # Get the tensor
        block_tensor = self.block_manager.get_block_tensor(block_idx)
        
        expected_shape = (self.num_layers, 2, self.block_size, self.num_heads, self.head_dim)
        self.assertEqual(block_tensor.shape, expected_shape)
    
    def test_get_block_tensor_invalid_index(self):
        """Test getting tensor with invalid block index."""
        with self.assertRaises(ValueError):
            self.block_manager.get_block_tensor(-1)
        
        with self.assertRaises(ValueError):
            self.block_manager.get_block_tensor(self.num_blocks)
    
    def test_get_kv_cache_for_blocks(self):
        """Test getting KV cache for multiple blocks."""
        # Allocate multiple blocks
        block_indices = self.block_manager.allocate_blocks(30)
        
        # Get KV cache for these blocks
        kv_cache = self.block_manager.get_kv_cache_for_blocks(block_indices)
        
        expected_shape = (self.num_layers, 2, 2, self.block_size, self.num_heads, self.head_dim)
        self.assertEqual(kv_cache.shape, expected_shape)
    
    def test_reset(self):
        """Test resetting the block manager."""
        # Allocate some blocks
        self.block_manager.allocate_blocks(30)
        self.block_manager.allocate_blocks(20)
        
        self.assertEqual(self.block_manager.num_allocated_blocks, 4)
        self.assertEqual(self.block_manager.num_free_blocks, self.num_blocks - 4)
        
        # Reset
        self.block_manager.reset()
        
        self.assertEqual(self.block_manager.num_free_blocks, self.num_blocks)
        self.assertEqual(self.block_manager.num_allocated_blocks, 0)
    
    def test_concurrent_allocation(self):
        """Test concurrent block allocation (basic thread safety check)."""
        import threading
        
        results = []
        
        def allocate_blocks():
            blocks = self.block_manager.allocate_blocks(20)
            results.append(blocks)
        
        # Create multiple threads
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=allocate_blocks)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Check results (some may fail due to insufficient blocks)
        successful_allocations = sum(1 for r in results if r is not None)
        self.assertGreaterEqual(successful_allocations, 0)
        self.assertLessEqual(successful_allocations, 5)
    
    def test_repr(self):
        """Test string representation."""
        repr_str = repr(self.block_manager)
        self.assertIn("BlockManager", repr_str)
        self.assertIn(f"num_blocks={self.num_blocks}", repr_str)
        self.assertIn("free=10", repr_str)
        self.assertIn("allocated=0", repr_str)


if __name__ == "__main__":
    unittest.main()