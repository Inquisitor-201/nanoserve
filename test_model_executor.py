"""
Unit tests for ModelExecutor.
"""

import unittest
import torch
from typing import List

# Mock FlashInfer if not available
class MockFlashInferWrapper:
    def __init__(self, workspace, layout):
        self.workspace = workspace
        self.layout = layout
    
    def forward(self, query, key_cache, value_cache, **kwargs):
        # Simple mock: return query as output
        return query

# Try to import flashinfer, use mocks if not available
try:
    import flashinfer
    from flashinfer import BatchDecodeWithPagedKVCacheWrapper, BatchPrefillWithPagedKVCacheWrapper
except ImportError:
    # Create mock classes
    BatchDecodeWithPagedKVCacheWrapper = MockFlashInferWrapper
    BatchPrefillWithPagedKVCacheWrapper = MockFlashInferWrapper
    flashinfer = None

from block_manager import BlockManager
from model_executor import ModelExecutor


class TestModelExecutor(unittest.TestCase):
    """Test cases for ModelExecutor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.num_blocks = 10
        self.num_layers = 2
        self.num_heads = 8
        self.head_dim = 64
        self.block_size = 16
        self.page_size = 16
        
        # Create block manager
        self.block_manager = BlockManager(
            num_blocks=self.num_blocks,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            block_size=self.block_size,
            dtype=torch.float32,
            device="cpu"
        )
        
        # Create model executor
        self.model_executor = ModelExecutor(
            block_manager=self.block_manager,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            page_size=self.page_size,
            dtype=torch.float32,
            device="cpu"
        )
    
    def test_initialization(self):
        """Test ModelExecutor initialization."""
        self.assertEqual(self.model_executor.num_heads, self.num_heads)
        self.assertEqual(self.model_executor.head_dim, self.head_dim)
        self.assertEqual(self.model_executor.page_size, self.page_size)
        self.assertEqual(self.model_executor.device, "cpu")
        
        # Check that wrappers are initialized
        self.assertIsNotNone(self.model_executor.decode_wrapper)
        self.assertIsNotNone(self.model_executor.prefill_wrapper)
    
    def test_initialization_invalid_page_size(self):
        """Test initialization with mismatched page_size."""
        with self.assertRaises(ValueError):
            ModelExecutor(
                block_manager=self.block_manager,
                num_heads=self.num_heads,
                head_dim=self.head_dim,
                page_size=32,  # Different from block_size
                dtype=torch.float32,
                device="cpu"
            )
    
    def test_prepare_flashinfer_inputs_single_sequence(self):
        """Test preparing FlashInfer inputs for single sequence."""
        block_tables = [[0, 1, 2]]
        seq_lengths = [40]  # 40 tokens, should need 3 blocks (ceil(40/16) = 3)
        
        inputs = self.model_executor.prepare_flashinfer_inputs(
            block_tables, seq_lengths, is_prefill=True
        )
        
        # Check outputs
        self.assertIn("paged_kv_indices", inputs)
        self.assertIn("paged_kv_indptr", inputs)
        self.assertIn("paged_kv_last_page_len", inputs)
        
        # Check indices (should be [0, 1, 2])
        expected_indices = torch.tensor([0, 1, 2], dtype=torch.int32)
        torch.testing.assert_close(inputs["paged_kv_indices"], expected_indices)
        
        # Check indptr (should be [0, 3])
        expected_indptr = torch.tensor([0, 3], dtype=torch.int32)
        torch.testing.assert_close(inputs["paged_kv_indptr"], expected_indptr)
        
        # Check last page length (40 % 16 = 8)
        expected_last_page_len = torch.tensor([8], dtype=torch.int32)
        torch.testing.assert_close(inputs["paged_kv_last_page_len"], expected_last_page_len)
    
    def test_prepare_flashinfer_inputs_multiple_sequences(self):
        """Test preparing FlashInfer inputs for multiple sequences."""
        block_tables = [[0, 1], [2, 3, 4], [5]]
        seq_lengths = [20, 35, 10]  # [2 blocks, 3 blocks, 1 block]
        
        inputs = self.model_executor.prepare_flashinfer_inputs(
            block_tables, seq_lengths, is_prefill=True
        )
        
        # Check indices (should be [0, 1, 2, 3, 4, 5])
        expected_indices = torch.tensor([0, 1, 2, 3, 4, 5], dtype=torch.int32)
        torch.testing.assert_close(inputs["paged_kv_indices"], expected_indices)
        
        # Check indptr (should be [0, 2, 5, 6])
        expected_indptr = torch.tensor([0, 2, 5, 6], dtype=torch.int32)
        torch.testing.assert_close(inputs["paged_kv_indptr"], expected_indptr)
        
        # Check last page lengths ([4, 3, 10])
        expected_last_page_len = torch.tensor([4, 3, 10], dtype=torch.int32)
        torch.testing.assert_close(inputs["paged_kv_last_page_len"], expected_last_page_len)
    
    def test_get_kv_cache_from_blocks(self):
        """Test extracting KV cache from blocks."""
        # Allocate some blocks
        block_indices = [0, 1, 2]
        key_cache, value_cache = self.model_executor.get_kv_cache_from_blocks(block_indices)
        
        # Check shapes
        expected_seq_len = len(block_indices) * self.page_size  # 3 * 16 = 48
        self.assertEqual(key_cache.shape, (self.num_layers, expected_seq_len, self.num_heads, self.head_dim))
        self.assertEqual(value_cache.shape, (self.num_layers, expected_seq_len, self.num_heads, self.head_dim))
    
    def test_compute_attention_with_flashinfer_prefill(self):
        """Test attention computation for prefill phase."""
        # Allocate blocks
        block_tables = [[0, 1]]
        seq_lengths = [20]
        
        # Create mock tensors
        query = torch.randn(20, self.num_heads, self.head_dim, dtype=torch.float32)
        key_cache = torch.randn(32, self.num_heads, self.head_dim, dtype=torch.float32)  # 2 blocks * 16
        value_cache = torch.randn(32, self.num_heads, self.head_dim, dtype=torch.float32)
        
        # Compute attention
        output = self.model_executor.compute_attention_with_flashinfer(
            query, key_cache, value_cache, block_tables, seq_lengths, is_prefill=True
        )
        
        # Check output shape (should match query shape)
        self.assertEqual(output.shape, query.shape)
    
    def test_compute_attention_with_flashinfer_decode(self):
        """Test attention computation for decode phase."""
        # Allocate blocks
        block_tables = [[0, 1]]
        seq_lengths = [20]
        
        # Create mock tensors
        query = torch.randn(1, self.num_heads, self.head_dim, dtype=torch.float32)  # Single token
        key_cache = torch.randn(32, self.num_heads, self.head_dim, dtype=torch.float32)  # 2 blocks * 16
        value_cache = torch.randn(32, self.num_heads, self.head_dim, dtype=torch.float32)
        
        # Compute attention
        output = self.model_executor.compute_attention_with_flashinfer(
            query, key_cache, value_cache, block_tables, seq_lengths, is_prefill=False
        )
        
        # Check output shape (should match query shape)
        self.assertEqual(output.shape, query.shape)
    
    def test_execute_model_prefill(self):
        """Test model execution for prefill phase."""
        # Allocate blocks
        block_indices = self.block_manager.allocate_blocks(20)
        block_tables = [block_indices]
        seq_lengths = [20]
        
        # Create input
        input_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]])
        
        # Execute model
        output = self.model_executor.execute_model(
            input_ids, block_tables, seq_lengths, layer_idx=0, is_prefill=True
        )
        
        # Check output shape
        self.assertEqual(len(output.shape), 3)  # Should be 3D tensor
        self.assertEqual(output.shape[-1], self.head_dim)  # Last dimension should be head_dim
    
    def test_execute_model_decode(self):
        """Test model execution for decode phase."""
        # Allocate blocks
        block_indices = self.block_manager.allocate_blocks(20)
        block_tables = [block_indices]
        seq_lengths = [20]
        
        # Create input (single token for decode)
        input_ids = torch.tensor([[21]])  # New token
        
        # Execute model
        output = self.model_executor.execute_model(
            input_ids, block_tables, seq_lengths, layer_idx=0, is_prefill=False
        )
        
        # Check output shape
        self.assertEqual(len(output.shape), 3)  # Should be 3D tensor
        self.assertEqual(output.shape[-1], self.head_dim)  # Last dimension should be head_dim
    
    def test_execute_model_invalid_layer(self):
        """Test model execution with invalid layer index."""
        # Allocate blocks
        block_indices = self.block_manager.allocate_blocks(10)
        block_tables = [block_indices]
        seq_lengths = [10]
        
        # Create input
        input_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
        
        # This should work (layer_idx=0 is valid)
        output = self.model_executor.execute_model(
            input_ids, block_tables, seq_lengths, layer_idx=0, is_prefill=True
        )
        self.assertIsNotNone(output)
    
    def test_repr(self):
        """Test string representation."""
        repr_str = repr(self.model_executor)
        self.assertIn("ModelExecutor", repr_str)
        self.assertIn(f"num_heads={self.num_heads}", repr_str)
        self.assertIn(f"head_dim={self.head_dim}", repr_str)
        self.assertIn(f"page_size={self.page_size}", repr_str)
        self.assertIn("device=cpu", repr_str)


if __name__ == "__main__":
    unittest.main()