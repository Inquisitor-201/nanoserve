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
        small_config = {
            "hidden_size": 512,        # 更小的隐藏层
            "num_heads": 4,            # 更少的注意力头 - 确保 num_heads * head_dim = hidden_size
            "head_dim": 128,           # 保持head_dim不变
            "intermediate_size": 1024, # 更小的中间层
            "num_layers": 4,           # 更少的层数
            "dtype": torch.float16,    # 使用float16减少内存使用
            "num_blocks": 50,          # 更少的KV缓存块
            "attention_backend": "flashinfer", # 使用flashinfer后端
        }
        llm_service.load_model(config=small_config)
        print("✅ Model loaded successfully")
        
        # Simple text generation
        print("\n📝 Generating text...")
        prompts = ["Hello, world!", "The future of AI is"]
        
        print(f"Input: {prompts}")
        generated_texts = llm_service.generate(
            prompts=prompts,
            max_new_tokens=20,
            temperature=0.7
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