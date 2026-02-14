#!/usr/bin/env python3
"""
Simple example demonstrating text generation with Qwen3 model.
"""

import logging
from core import LLMService

# Setup logging
logging.basicConfig(level=logging.INFO)


def main():
    """Simple text generation demo."""
    
    print("🚀 Qwen3 Text Generation Demo")
    print("=" * 40)
    
    try:
        # Initialize LLM Service
        llm_service = LLMService(device="cuda")
        print(f"✅ LLM Service initialized")
        
        # Load Qwen3 model
        print("\n📦 Loading Qwen3 model...")
        llm_service.load_model()
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