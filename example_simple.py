#!/usr/bin/env python3
"""
Simple example demonstrating text generation with Qwen3 model.
"""

import logging
import torch
from core import LLMService

# Setup logging
logging.basicConfig(level=logging.INFO)


def main():
    """Simple text generation demo."""
    
    print("🚀 Qwen3 Text Generation Demo")
    print("=" * 40)
    
    try:
        # Initialize LLM Service with CUDA
        llm_service = LLMService(device="cuda")
        print(f"✅ LLM Service initialized")
        
        # Load Qwen3 model with smaller configuration
        print("\n📦 Loading Qwen3 model...")
        
        # Use the downloaded real model
        model_path = "/root/nanoserve/models/Qwen3-0.6B"
        
        # Load with optimized configuration for the real model
        real_config = {
            "dtype": torch.bfloat16,    # 使用bfloat16以获得更好的数值稳定性
            "num_blocks": 200,         # 为真实模型提供更多KV缓存块
            "block_size": 16,          # 保持block_size不变
            "attention_backend": "flashinfer", # 使用flashinfer后端
        }

        llm_service.load_model(model_path=model_path, config=real_config)
        print("✅ Model loaded successfully")
        
        # Simple text generation
        print("\n📝 Generating text...")
        prompts = ["中国是一个", 
                   "你是一个精通逻辑推理的数学助手。在回答任何数学问题之前，你必须遵循以下步骤：1. 提取题目中的关键数字和条件；2. 分步骤列出计算过程，每一步只做一个简单的运算；3. 最后给出最终结果。请务必保持逻辑严密，不要跳步。请你计算：小红买了3个苹果，单价12元；又买了2个梨，单价16元。她给了老板68元，应该找回多少钱？"]
        
        print(f"Input: {prompts}")
        generated_texts = llm_service.generate(
            prompts=prompts,
            max_new_tokens=400,  # Increased to test dynamic block allocation
            temperature=0.0
        )
        
        print(f"\n✅ Generated text:")
        for i, (prompt, generated) in enumerate(zip(prompts, generated_texts)):
            print(f"  {i+1}. '{prompt}' -> '{generated}'")
        
        print("\n🎉 Text generation completed!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)