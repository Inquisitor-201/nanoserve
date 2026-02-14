#!/usr/bin/env python3
"""
Test script to run inference with Qwen3-0.6B model.
"""

import os
import sys
# Add parent directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from core.model_executor import ModelExecutor


def test_inference():
    """Test model inference with simple prompts."""
    print("Testing model inference...")
    
    # Model configuration - match Qwen3-0.6B
    model_config = {
        "model_name": "qwen3",
        "vocab_size": 151936,
        "hidden_size": 1024,
        "num_heads": 8,
        "num_key_value_heads": 8,  # MHA for 0.6B model
        "head_dim": 128,
        "intermediate_size": 3072,
        "num_layers": 2,  # Small number of layers for testing
        "attention_backend": "flashinfer",
        "dtype": torch.float16,
        "device": "cuda",
        "num_blocks": 100,
        "block_size": 16
    }
    
    # Model path
    model_path = "./models/Qwen3-0.6B"
    
    # Initialize model
    print("Initializing model...")
    executor = ModelExecutor(
        **model_config,
        model_path=model_path
    )
    
    print("\nModel initialized successfully!")
    
    # Simple test prompt
    prompt = "Hello, how are you?"
    print(f"\nTesting with prompt: '{prompt}'")
    
    # Create dummy input for testing
    # In practice, we would use tokenizer to convert prompt to input_ids
    # For testing, use random input_ids
    batch_size = 1
    seq_length = 10
    
    # Create 1D flattened input_ids as expected by Qwen3Model.forward
    input_ids = torch.randint(0, 10000, (batch_size * seq_length,), device="cuda")
    
    # Create block tables and sequence lengths for testing
    # For simplicity, use empty block tables since we're not actually using KV cache
    block_tables = [[] for _ in range(batch_size)]
    seq_lengths = [seq_length for _ in range(batch_size)]
    
    # Test execute_prefill
    print("Running execute_prefill...")
    try:
        output = executor.execute_prefill(input_ids, block_tables, seq_lengths)
        print(f"\n✅ execute_prefill successful!")
        print(f"Output shape: {output.shape}")
        print(f"Output dtype: {output.dtype}")
    except Exception as e:
        print(f"\n❌ execute_prefill failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_inference()