import os
import sys
import unittest
import torch

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.backends.metadata import AttentionMetadata
from core.layers.attention import Attention
from core.backends.flashinfer_backend import FlashInferBackend

class TestGQA(unittest.TestCase):
    def setUp(self):
        self.hidden_size = 512
        self.num_heads = 8
        self.num_kv_heads = 2
        self.head_dim = 64
        self.device = "cuda"
        self.dtype = torch.float16
        
        # Create a mock kv_cache_pool for testing
        # Format: [num_layers, num_blocks, 2, block_size, num_heads, head_dim]
        num_layers = 1
        num_blocks = 1
        block_size = 16
        self.kv_cache_pool = torch.zeros(
            (num_layers, num_blocks, 2, block_size, self.num_kv_heads, self.head_dim),
            dtype=self.dtype,
            device=self.device
        )
        
        # 1. Initialize REAL FlashInfer backend
        self.backend = FlashInferBackend(
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            kv_cache_pool=self.kv_cache_pool,
            num_key_value_heads=self.num_kv_heads,
            dtype=self.dtype,
            device=self.device
        )
        
        # 2. Initialize Attention Layer
        self.attention = Attention(
            hidden_size=self.hidden_size,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            num_key_value_heads=self.num_kv_heads,
            backend=self.backend,
            device=self.device,
            dtype=self.dtype
        )

    def test_01_numerical_accuracy_prefill(self):
        """
        Requirement: The output must NOT be zero and must be deterministic.
        Verify if attention logic (QKV -> Softmax -> Out) produces realistic values.
        """
        seq_len = 8
        hidden_states = torch.randn(seq_len, self.hidden_size, device=self.device, dtype=self.dtype)
        
        metadata = AttentionMetadata.from_block_tables(
            block_tables=[[0]], # Dummy block 0
            seq_lengths=[seq_len],
            is_prefill=True,
            device=self.device
        )

        with torch.inference_mode():
            output = self.attention.forward(hidden_states, metadata)

        # 1. Check for non-zero output (if weights are loaded/init, output shouldn't be zero)
        self.assertGreater(output.abs().mean().item(), 1e-5, "Attention output is suspiciously near zero.")
        
        # 2. Check for NaNs (common in Softmax/Scale issues)
        self.assertFalse(torch.isnan(output).any(), "NaNs detected in attention output.")

    def test_02_kv_cache_writeback(self):
        """
        Requirement: K and V tensors MUST be written into the global KV storage.
        This test checks if the backend actually updated the KV pool.
        """
        seq_len = 4
        hidden_states = torch.randn(seq_len, self.hidden_size, device=self.device, dtype=self.dtype)
        
        # We need to simulate a real KV Pool context. 
        # Since FlashInferBackend holds the state, we check if k/v projections 
        # are reaching the intended storage.
        
        metadata = AttentionMetadata.from_block_tables(
            block_tables=[[0]], 
            seq_lengths=[seq_len],
            is_prefill=True,
            device=self.device
        )

        # Record weight-based K, V for manual verification
        with torch.inference_mode():
            # Manually compute what K should be
            expected_k = self.attention.k_norm(self.attention.k_proj(hidden_states).view(seq_len, self.num_kv_heads, self.head_dim))
            
            # Run the forward pass
            self.attention.forward(hidden_states, metadata)
            
            # ASSERTION: The backend MUST have a way to store these.
            # If your FlashInferBackend currently has no 'kv_cache' attribute, 
            # this test will fail, forcing the Agent to implement the storage logic.
            if hasattr(self.backend, 'key_cache') and self.backend.key_cache is not None:
                # Check if the values in cache match the projected K
                # We check the first few elements of the first block
                cached_k = self.backend.key_cache[0, :seq_len, :, :] # Assuming [num_blocks, page_size, num_kv_heads, head_dim]
                torch.testing.assert_close(cached_k, expected_k, atol=1e-3, rtol=1e-3)
            else:
                self.fail("Backend does not have a persistent KV cache storage implemented!")

if __name__ == "__main__":
    unittest.main()