#!/usr/bin/env python3
"""
Test script to verify model weight mapping correctness.
This script checks that every weight in the model is properly loaded and mapped.
"""

import os
import sys
# Add parent directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import logging
from core.model_executor import ModelExecutor
from core.model_loader import ModelLoader

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_weight_mapping():
    """Test weight mapping correctness."""
    logger.info("Testing weight mapping correctness...")
    
    # Model configuration - match Qwen3-0.6B weight shapes
    model_config = {
        "model_name": "qwen3",
        "vocab_size": 151936,  # Match lm_head.weight shape
        "hidden_size": 1024,    # Match weight shapes
        "num_heads": 8,         # Adjust based on weight shapes
        "num_key_value_heads": 8,  # MHA for 0.6B model
        "head_dim": 128,         # 1024 / 8 = 128
        "intermediate_size": 3072,  # Match mlp weights
        "num_layers": 2,           # Small number of layers for testing
        "attention_backend": "flashinfer",
        "dtype": torch.float16,
        "device": "cuda",
        "num_blocks": 100,
        "block_size": 16
    }
    
    # Test model path
    model_path = "./models/Qwen3-0.6B"
    
    # Initialize model with weights
    executor = ModelExecutor(
        **model_config,
        model_path=model_path
    )
    
    logger.info("\n✅ Model initialized with weights")
    
    # Get model state dict
    model_state = executor.model.state_dict()
    logger.info(f"Model has {len(model_state)} parameters")
    
    # Check each parameter to see if it has been loaded
    loaded_params = 0
    unloaded_params = []
    
    for name, param in model_state.items():
        # Check if parameter has non-zero values (basic check for loading)
        if torch.allclose(param, torch.zeros_like(param)):
            unloaded_params.append(name)
        else:
            loaded_params += 1
    
    logger.info(f"\nWeight loading summary:")
    logger.info(f"  Total parameters: {len(model_state)}")
    logger.info(f"  Loaded parameters: {loaded_params}")
    logger.info(f"  Unloaded parameters: {len(unloaded_params)}")
    
    if unloaded_params:
        logger.warning(f"  Unloaded parameters: {unloaded_params[:5]}..." if len(unloaded_params) > 5 else f"  Unloaded parameters: {unloaded_params}")
    else:
        logger.info("  ✅ All parameters loaded successfully!")
    
    # Test weight mapping function directly
    logger.info("\nTesting weight name mapping...")
    
    # Test some common HuggingFace weight names - updated to match our model structure
    test_names = [
        "model.embed_tokens.weight",
        "model.norm.weight",
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.k_proj.weight",
        "model.layers.0.self_attn.v_proj.weight",
        "model.layers.0.self_attn.o_proj.weight",
        "model.layers.0.mlp.gate_proj.weight",
        "model.layers.0.mlp.up_proj.weight",
        "model.layers.0.mlp.down_proj.weight",
        "model.layers.0.input_layernorm.weight",
        "model.layers.0.post_attention_layernorm.weight",
        "model.layers.0.self_attn.q_norm.weight",
        "model.layers.0.self_attn.k_norm.weight",
        "lm_head.weight",
    ]
    
    logger.info("Weight name mapping test:")
    for hf_name in test_names:
        mapped_name = ModelLoader._map_weight_name(hf_name)
        logger.info(f"  {hf_name} -> {mapped_name}")
        # Check if mapped name exists in model
        if mapped_name in model_state:
            logger.info(f"    ✅ Mapped name exists in model")
        else:
            logger.warning(f"    ❌ Mapped name does not exist in model")
    
    # Test specific GQA parameters
    logger.info("\nTesting attention parameters...")
    attn_params = [
        "layers.0.self_attn.q_proj.weight",
        "layers.0.self_attn.k_proj.weight",
        "layers.0.self_attn.v_proj.weight",
        "layers.1.self_attn.q_proj.weight",
        "layers.1.self_attn.k_proj.weight",
        "layers.1.self_attn.v_proj.weight",
    ]
    
    for param_name in attn_params:
        if param_name in model_state:
            param = model_state[param_name]
            logger.info(f"  {param_name}: shape={param.shape}, dtype={param.dtype}")
            # Check if parameter has non-zero values
            if not torch.allclose(param, torch.zeros_like(param)):
                logger.info(f"    ✅ Parameter loaded successfully")
            else:
                logger.warning(f"    ❌ Parameter not loaded (all zeros)")
        else:
            logger.warning(f"  {param_name}: ❌ Not found in model")
    
    # Test mlp parameters
    logger.info("\nTesting MLP parameters...")
    mlp_params = [
        "layers.0.mlp.gate_proj.weight",
        "layers.0.mlp.up_proj.weight",
        "layers.0.mlp.down_proj.weight",
        "layers.1.mlp.gate_proj.weight",
        "layers.1.mlp.up_proj.weight",
        "layers.1.mlp.down_proj.weight",
    ]
    
    for param_name in mlp_params:
        if param_name in model_state:
            param = model_state[param_name]
            logger.info(f"  {param_name}: shape={param.shape}, dtype={param.dtype}")
            # Check if parameter has non-zero values
            if not torch.allclose(param, torch.zeros_like(param)):
                logger.info(f"    ✅ Parameter loaded successfully")
            else:
                logger.warning(f"    ❌ Parameter not loaded (all zeros)")
        else:
            logger.warning(f"  {param_name}: ❌ Not found in model")
    
    logger.info("\n✅ Weight mapping test completed!")

if __name__ == "__main__":
    test_weight_mapping()