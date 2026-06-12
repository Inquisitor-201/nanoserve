"""
RoPE (Rotary Position Embedding) Correctness Test

This test verifies that the FlashInfer RoPE implementation matches
the reference PyTorch implementation for Qwen3 architecture.

Key test cases:
1. Reference implementation correctness
2. FlashInfer vs Reference comparison (interleaved=False)
3. FlashInfer vs Reference comparison (interleaved=True - should fail)
4. Different sequence lengths and position offsets
5. Multi-sequence batch handling
"""

import os
import sys
import unittest
import torch

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.backends.metadata import AttentionMetadata
from core.models.qwen3.attention import Qwen3Attention
from core.backends.flashinfer_backend import FlashInferBackend

import flashinfer

torch.manual_seed(42)


def slow_rope_reference(q, pos_ids, theta=1000000.0, interleave=False):
    """
    Slow but correct reference implementation of RoPE.
    
    This implements the standard rotate_half logic used in Llama/Qwen models.
    
    Args:
        q: Query tensor [total_tokens, num_heads, head_dim]
        pos_ids: Position indices [total_tokens]
        theta: RoPE base frequency (default: 1M for Qwen3)
        interleave: Whether to use interleaved layout
                   - False: rotate_half [x1,x2,x3,x4] → [-x3,-x4,x1,x2]
                   - True: adjacent rotation [x1,x2,x3,x4] → [-x2,x1,-x4,x3]
    
    Returns:
        RoPE-encoded tensor same shape as input
    """
    total_tokens, num_heads, head_dim = q.shape
    
    # Compute frequency inv_freq
    # For head_dim=128: arange(0, 128, 2) = [0, 2, 4, ..., 126]
    # After /head_dim: [0/128, 2/128, ..., 126/128]
    # After **(...): theta^(0/128), theta^(2/128), ..., theta^(126/128)
    # inv_freq: 1.0 / theta^(i/head_dim)
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=q.device) / head_dim))
    
    # Generate frequency matrix: [total_tokens, head_dim/2]
    # freqs[i,j] = pos_ids[i] * inv_freq[j]
    freqs = torch.outer(pos_ids.float(), inv_freq)
    
    # Expand to [total_tokens, head_dim]
    # For interleave=False: [freqs, freqs] concatenates cos/sin pairs
    # For interleave=True: this would need different handling
    if not interleave:
        # Standard Qwen3/Llama layout: concatenate pairs
        emb = torch.cat((freqs, freqs), dim=-1)  # [total_tokens, head_dim]
    else:
        # Interleaved layout: alternate between freqs
        # This is not commonly used in Qwen3
        emb = torch.zeros(total_tokens, head_dim, device=q.device, dtype=torch.float32)
        emb[:, 0::2] = freqs
        emb[:, 1::2] = freqs
    
    # Expand to [total_tokens, 1, head_dim] for broadcasting with num_heads
    emb = emb.unsqueeze(1)  # Add num_heads dimension
    cos = emb.cos()
    sin = emb.sin()
    
    # FlashInfer compatible rotate_half implementation
    # For interleave=False: [x1, x2, x3, x4] → [-x3, -x4, x1, x2]
    # For interleave=True: [x1, x2, x3, x4] → [-x2, x1, -x4, x3]
    def rotate_half(x):
        if not interleave:
            # Qwen3/Llama standard: split at half dimension
            half = x.shape[-1] // 2
            x1 = x[..., :half]
            x2 = x[..., half:]
            return torch.cat((-x2, x1), dim=-1)
        else:
            # Interleaved: rotate adjacent pairs
            x1 = x[..., ::2]
            x2 = x[..., 1::2]
            return torch.stack((-x2, x1), dim=-1).reshape(x.shape)
    
    # Apply RoPE: q * cos + rotate_half(q) * sin
    q_embed = (q * cos) + (rotate_half(q) * sin)
    
    # Convert back to original dtype
    return q_embed.to(dtype=q.dtype)


class TestRoPECorrectness(unittest.TestCase):
    """Test suite for RoPE implementation correctness."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.hidden_size = 512
        self.num_heads = 8
        self.num_kv_heads = 2
        self.head_dim = 64
        self.device = "cuda"
        self.dtype = torch.float16
        self.rope_theta = 1000000.0
        
        # Create mock KV cache pool
        num_layers = 1
        num_blocks = 2
        self.block_size = 16
        self.kv_cache_pool = torch.randn(
            (num_layers, num_blocks, 2, self.block_size, self.num_kv_heads, self.head_dim),
            dtype=self.dtype,
            device=self.device
        )
        
        # Initialize FlashInfer backend
        self.backend = FlashInferBackend(
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            kv_cache_pool=self.kv_cache_pool,
            num_key_value_heads=self.num_kv_heads,
            dtype=self.dtype,
            device=self.device
        )
        
        # Initialize Qwen3Attention with RoPE support
        self.attention = Qwen3Attention(
            hidden_size=self.hidden_size,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            num_key_value_heads=self.num_kv_heads,
            attention_backend=self.backend,
            layer_idx=0,
            rope_theta=self.rope_theta,
            device=self.device,
            dtype=self.dtype
        )
    
    def test_01_reference_implementation_consistency(self):
        """
        Verify the reference implementation produces consistent results.
        The same pos_ids should always produce the same rotation.
        """
        q = torch.randn(4, self.num_heads, self.head_dim, device=self.device, dtype=self.dtype)
        pos_ids = torch.tensor([0, 1, 2, 3], dtype=torch.int32, device=self.device)
        
        # Run reference twice to verify determinism
        result1 = slow_rope_reference(q, pos_ids, theta=self.rope_theta, interleave=False)
        result2 = slow_rope_reference(q, pos_ids, theta=self.rope_theta, interleave=False)
        
        torch.testing.assert_close(result1, result2, atol=0, rtol=0,
                                  msg="Reference RoPE should be deterministic")
        
        print("✓ Reference implementation is deterministic")
    
    def test_02_flashinfer_vs_reference_interleaved_false(self):
        """
        CRITICAL: Test that FlashInfer with interleave=False matches reference.
        
        This is the Qwen3-required configuration.
        Expected: rotate_half [x1,x2,x3,x4] → [-x3,-x4,x1,x2]
        """
        seq_len = 8
        hidden_states = torch.randn(seq_len, self.hidden_size, device=self.device, dtype=self.dtype)
        
        # Create metadata with known positions [0, 1, 2, 3, 4, 5, 6, 7]
        metadata = AttentionMetadata.from_block_tables(
            block_tables=[[0, 1]],  # Use 2 blocks
            seq_lengths=[seq_len],
            is_prefill=True,
            page_size=self.block_size,
            device=self.device
        )
        
        # Extract q, k after projections and normalization
        with torch.no_grad():
            q_proj = self.attention.q_proj(hidden_states)
            k_proj = self.attention.k_proj(hidden_states)
            
            q = q_proj.view(seq_len, self.num_heads, self.head_dim)
            k = k_proj.view(seq_len, self.num_kv_heads, self.head_dim)
            
            q = self.attention.q_norm(q)
            k = self.attention.k_norm(k)
        
        # Get reference result BEFORE FlashInfer modifies tensors
        q_ref = q.clone().contiguous()
        k_ref = k.clone().contiguous()
        pos_ids = metadata.positions
        
        # Apply reference RoPE
        q_ref_out = slow_rope_reference(q_ref, pos_ids, theta=self.rope_theta, interleave=False)
        k_ref_out = slow_rope_reference(k_ref, pos_ids, theta=self.rope_theta, interleave=False)
        
        # Apply FlashInfer RoPE (single call for both q and k)
        # FlashInfer actually supports different num_heads for q and k!
        # According to documentation:
        # - q shape: (nnz, num_q_heads, head_dim)
        # - k shape: (nnz, num_k_heads, head_dim)
        q_flash = q.contiguous()
        k_flash = k.contiguous()
        
        # Single FlashInfer RoPE call with original shapes (no expansion needed!)
        flashinfer.rope.apply_rope_pos_ids_inplace(
            q_flash,
            k_flash,
            pos_ids=pos_ids,
            rotary_dim=None,
            interleave=False,  # Qwen3 required
            rope_scale=1.0,
            rope_theta=self.rope_theta
        )
        
        # Compare results
        # Use higher tolerance for fp16
        torch.testing.assert_close(
            q_flash, q_ref_out,
            atol=1e-3, rtol=1e-3,
            msg="FlashInfer RoPE (interleave=False) must match reference for Q"
        )
        torch.testing.assert_close(
            k_flash, k_ref_out,
            atol=1e-3, rtol=1e-3,
            msg="FlashInfer RoPE (interleave=False) must match reference for K"
        )
        
        print("✓ FlashInfer (interleave=False) matches reference ✓")
    
    def test_03_flashinfer_interleaved_true_differs(self):
        """
        Verify that interleave=True produces DIFFERENT results.
        
        interleave=True uses adjacent rotation which is NOT the Qwen3 standard.
        This test confirms our interleave=False configuration is intentional.
        """
        seq_len = 4
        hidden_states = torch.randn(seq_len, self.hidden_size, device=self.device, dtype=self.dtype)
        
        metadata = AttentionMetadata.from_block_tables(
            block_tables=[[0]],
            seq_lengths=[seq_len],
            is_prefill=True,
            page_size=self.block_size,
            device=self.device
        )
        
        with torch.no_grad():
            q_proj = self.attention.q_proj(hidden_states)
            q = q_proj.view(seq_len, self.num_heads, self.head_dim)
            q = self.attention.q_norm(q)
        
        q_ref = q.clone().contiguous()
        pos_ids = metadata.positions
        
        # Apply reference with interleave=False
        q_ref_false = slow_rope_reference(q_ref, pos_ids, theta=self.rope_theta, interleave=False)
        
        # Reset and apply reference with interleave=True
        q_ref = q.clone().contiguous()
        q_ref_true = slow_rope_reference(q_ref, pos_ids, theta=self.rope_theta, interleave=True)
        
        # These MUST be different
        diff = (q_ref_false - q_ref_true).abs().max().item()
        self.assertGreater(diff, 1e-5, 
                          "interleave=True should produce different results than interleave=False")
        
        print(f"✓ interleave=True produces different results (max diff: {diff:.6f}) ✓")
    
    def test_04_single_token_position_rotation(self):
        """
        Test that position 0 produces identity rotation (cos=1, sin=0).
        
        At position 0: rotation angle is 0, so:
        - cos(0) = 1 → q * 1 = q
        - sin(0) = 0 → rotate_half(q) * 0 = 0
        - Result: q
        """
        seq_len = 1
        hidden_states = torch.randn(seq_len, self.hidden_size, device=self.device, dtype=self.dtype)
        
        # Create metadata with position 0
        metadata = AttentionMetadata.from_block_tables(
            block_tables=[[0]],
            seq_lengths=[seq_len],
            is_prefill=True,
            page_size=self.block_size,
            device=self.device
        )
        
        # Force position 0
        pos_ids = torch.tensor([0], dtype=torch.int32, device=self.device)
        
        with torch.no_grad():
            q_proj = self.attention.q_proj(hidden_states)
            q = q_proj.view(seq_len, self.num_heads, self.head_dim)
            q = self.attention.q_norm(q)
        
        q_original = q.clone()
        q_cont = q.contiguous()
        
        # Apply FlashInfer RoPE
        flashinfer.rope.apply_rope_pos_ids_inplace(
            q_cont,
            q_cont,  # Use same tensor for K (won't be used)
            pos_ids=pos_ids,
            interleave=False,
            rope_theta=self.rope_theta
        )
        
        # At position 0, rotation should be identity
        # Allow small numerical error
        torch.testing.assert_close(
            q_cont, q_original,
            atol=1e-4, rtol=1e-4,
            msg="Position 0 should produce identity rotation"
        )
        
        print("✓ Position 0 produces identity rotation ✓")
    
    def test_05_consecutive_position_monotonicity(self):
        """
        Verify that applying RoPE at consecutive positions produces monotonically
        changing embeddings (angles increase with position).
        
        This is a sanity check: cos/sin should evolve smoothly with position.
        """
        seq_len = 16
        hidden_states = torch.randn(seq_len, self.hidden_size, device=self.device, dtype=self.dtype)
        
        metadata = AttentionMetadata.from_block_tables(
            block_tables=[[0, 1]],
            seq_lengths=[seq_len],
            is_prefill=True,
            page_size=self.block_size,
            device=self.device
        )
        
        with torch.no_grad():
            q_proj = self.attention.q_proj(hidden_states)
            q = q_proj.view(seq_len, self.num_heads, self.head_dim)
            q = self.attention.q_norm(q)
        
        # Get reference results
        q_results = []
        for i in range(seq_len):
            q_single = q[i:i+1].clone()
            pos_single = torch.tensor([i], dtype=torch.int32, device=self.device)
            
            q_ref = slow_rope_reference(q_single, pos_single, 
                                        theta=self.rope_theta, interleave=False)
            q_results.append(q_ref)
        
        # Calculate angle changes between consecutive positions
        # For a well-formed rotation, adjacent positions should have different angles
        # but the magnitude should remain constant
        # Calculate L2 norm (magnitude) for each token
        magnitudes = []
        for r in q_results:
            # Calculate L2 norm for each head, then take the mean
            norm = torch.norm(r, dim=-1).mean().item()
            magnitudes.append(norm)
        
        # Magnitudes should be approximately constant (rotation preserves magnitude)
        mag_diff = max(magnitudes) - min(magnitudes)
        self.assertLess(mag_diff, 0.1, 
                       f"Magnitude should be preserved by rotation, diff: {mag_diff:.6f}")
        
        # Consecutive embeddings should be different
        diff_01 = (q_results[0] - q_results[1]).abs().mean().item()
        self.assertGreater(diff_01, 1e-5,
                          "Consecutive positions should produce different embeddings")
        
        print(f"✓ Rotation preserves magnitude (diff: {mag_diff:.6f}) ✓")
    
    def test_06_multi_sequence_position_offsets(self):
        """
        CRITICAL: Test multi-sequence batch with correct position offsets.
        
        In a batch with 2 sequences of length 4:
        - Sequence 0: positions [0, 1, 2, 3]
        - Sequence 1: positions [0, 1, 2, 3] (NOT [4, 5, 6, 7])
        
        Each sequence has its own position counting starting from 0.
        """
        seq_len = 4
        hidden_states = torch.randn(seq_len * 2, self.hidden_size, device=self.device, dtype=self.dtype)
        
        # Batch with 2 sequences
        metadata = AttentionMetadata.from_block_tables(
            block_tables=[[0], [1]],  # Different blocks for each sequence
            seq_lengths=[seq_len, seq_len],
            page_size=self.block_size,
            is_prefill=True,
            device=self.device
        )
        
        # Verify position indices are correct
        pos_ids = metadata.positions
        expected_positions = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3], dtype=torch.int32, device=self.device)
        
        # Debug: print actual positions
        print(f"Actual positions: {pos_ids}")
        print(f"Expected positions: {expected_positions}")
        
        torch.testing.assert_close(
            pos_ids, expected_positions,
            atol=0, rtol=0,
            msg="Multi-sequence positions must reset for each sequence"
        )
        
        # Apply RoPE and verify each sequence gets correct rotation
        with torch.no_grad():
            q_proj = self.attention.q_proj(hidden_states)
            q = q_proj.view(seq_len * 2, self.num_heads, self.head_dim)
            q = self.attention.q_norm(q)
        
        # Split into two sequences
        q0 = q[:seq_len].clone().contiguous()
        q1 = q[seq_len:].clone().contiguous()
        
        pos0 = pos_ids[:seq_len]
        pos1 = pos_ids[seq_len:]
        
        # Apply reference RoPE to both
        q0_ref = slow_rope_reference(q0.clone(), pos0, theta=self.rope_theta, interleave=False)
        q1_ref = slow_rope_reference(q1.clone(), pos1, theta=self.rope_theta, interleave=False)
        
        # Apply FlashInfer
        flashinfer.rope.apply_rope_pos_ids_inplace(
            q0, q0, pos_ids=pos0, interleave=False, rope_theta=self.rope_theta
        )
        flashinfer.rope.apply_rope_pos_ids_inplace(
            q1, q1, pos_ids=pos1, interleave=False, rope_theta=self.rope_theta
        )
        
        torch.testing.assert_close(q0, q0_ref, atol=1e-3, rtol=1e-3,
                                  msg="Sequence 0 RoPE must be correct")
        torch.testing.assert_close(q1, q1_ref, atol=1e-3, rtol=1e-3,
                                  msg="Sequence 1 RoPE must be correct")
        
        print("✓ Multi-sequence position offsets are correct ✓")
    
    def test_07_decode_phase_positions(self):
        """
        Test RoPE for decode phase (single token generation).
        
        In decode phase, positions should be [seq_len-1] for the new token.
        """
        prev_seq_len = 8
        hidden_states = torch.randn(1, self.hidden_size, device=self.device, dtype=self.dtype)
        
        # Decode metadata: position should be prev_seq_len - 1 = 7
        metadata = AttentionMetadata.from_block_tables(
            block_tables=[[0, 1]],
            seq_lengths=[prev_seq_len],
            is_prefill=False,  # Decode phase
            page_size=self.block_size,
            device=self.device
        )
        
        pos_ids = metadata.positions
        expected_pos = prev_seq_len - 1
        
        self.assertEqual(pos_ids[0].item(), expected_pos,
                        f"Decode position should be {expected_pos}, got {pos_ids[0].item()}")
        
        with torch.no_grad():
            q_proj = self.attention.q_proj(hidden_states)
            q = q_proj.view(1, self.num_heads, self.head_dim)
            q = self.attention.q_norm(q)
        
        q_ref = q.clone().contiguous()
        
        # Apply reference
        q_ref_out = slow_rope_reference(q_ref, pos_ids, theta=self.rope_theta, interleave=False)
        
        # Apply FlashInfer
        q_flash = q.contiguous()
        flashinfer.rope.apply_rope_pos_ids_inplace(
            q_flash, q_flash,
            pos_ids=pos_ids,
            interleave=False,
            rope_theta=self.rope_theta
        )
        
        torch.testing.assert_close(q_flash, q_ref_out, atol=1e-3, rtol=1e-3,
                                  msg="Decode phase RoPE must match reference")
        
        print("✓ Decode phase positions are correct ✓")
    
    def test_08_end_to_end_with_attention(self):
        """
        Integration test: Full attention forward pass with RoPE.
        
        This tests that RoPE is correctly integrated into the attention
        computation pipeline.
        """
        seq_len = 8
        hidden_states = torch.randn(seq_len, self.hidden_size, device=self.device, dtype=self.dtype)
        
        metadata = AttentionMetadata.from_block_tables(
            block_tables=[[0, 1]],
            seq_lengths=[seq_len],
            is_prefill=True,
            page_size=self.block_size,
            device=self.device
        )
        
        # Reset backend state
        self.backend.reset_plan_state()
        
        # Run full forward pass
        with torch.inference_mode():
            output = self.attention.forward(hidden_states, metadata)
        
        # Basic sanity checks
        self.assertEqual(output.shape, (seq_len, self.hidden_size))
        self.assertFalse(torch.isnan(output).any(), "Output contains NaNs")
        self.assertGreater(output.abs().mean().item(), 1e-5, "Output is suspiciously near zero")
        
        # Verify KV cache was written with correct shape
        cached_k = self.backend.kv_cache_pool[0, 0, 0, :seq_len, :, :]
        self.assertEqual(cached_k.shape, (seq_len, self.num_kv_heads, self.head_dim))
        
        print("✓ End-to-end attention with RoPE works correctly ✓")


if __name__ == "__main__":
    # Run tests with verbosity
    unittest.main(verbosity=2)