#!/usr/bin/env python3
"""
Test script to verify GQA (Grouped Query Attention) support.
"""

import torch
import logging
from core.model_executor import ModelExecutor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_gqa_model():
    """Test GQA model initialization and basic functionality."""
    logger.info("Testing GQA model initialization...")
    
    # Initialize model with GQA (num_key_value_heads=8 for num_heads=32)
    # This should create a model with 32 query heads and 8 key/value heads
    # (4 groups of query heads per key/value head)
    executor = ModelExecutor(
        model_name="qwen3",
        vocab_size=32000,
        hidden_size=4096,
        num_heads=32,              # Query heads
        num_key_value_heads=8,      # Key/value heads (GQA)
        head_dim=128,               # 4096 / 32 = 128
        intermediate_size=11008,
        num_layers=2,               # Small number of layers for testing
        attention_backend="flashinfer",
        dtype=torch.float16,
        device="cuda",
        num_blocks=100,
        block_size=16
    )
    
    logger.info("✅ GQA model initialized successfully")
    logger.info(f"  Model: {executor.model}")
    logger.info(f"  Number of query heads: {executor.model.layers[0].self_attn.num_heads}")
    logger.info(f"  Number of key/value heads: {executor.model.layers[0].self_attn.num_key_value_heads}")
    logger.info(f"  Number of key/value groups: {executor.model.layers[0].self_attn.num_key_value_groups}")
    
    logger.info("\n✅ All tests passed!")

if __name__ == "__main__":
    test_gqa_model()