#!/usr/bin/env python3
"""
Test program to check if transformers library generates eos_token and ends early
with temperature=0 for different prompts.
"""

import logging
import time
from transformers import AutoTokenizer, AutoModelForCausalLM

# Setup logging
logging.basicConfig(level=logging.INFO)

def test_transformers_eos(prompt, max_new_tokens=700):
    """Test transformers generation with temperature=0 and check for eos_token."""
    model_path = "./models/Qwen3-0.6B"
    
    try:
        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path).to("cuda")
        
        # Get eos token
        eos_token = tokenizer.eos_token
        eos_token_id = tokenizer.eos_token_id
        print(f"📝 Testing prompt: '{prompt}'")
        print(f"🔚 EOS token: '{eos_token}' (ID: {eos_token_id})")
        
        # Prepare messages in standard format
        messages = [
            {"role": "user", "content": prompt}
        ]
        
        # Apply chat template using Jinja2
        rendered_prompt = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        print(f"📝 Rendered prompt: '{rendered_prompt[:100]}...'" if len(rendered_prompt) > 100 else f"📝 Rendered prompt: '{rendered_prompt}'")
        
        # Tokenize input
        inputs = tokenizer(rendered_prompt, return_tensors="pt").to("cuda")
        input_ids = inputs["input_ids"]
        print(f"📊 Input tokens: {input_ids.shape[1]} tokens")
        
        # Generate text with temperature=0
        start_time = time.perf_counter()
        output = model.generate(
            **inputs,
            temperature=0.0,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=True
        )
        end_time = time.perf_counter()
        
        # Get generated token IDs
        generated_ids = output["sequences"][0]
        generated_tokens = generated_ids.shape[0] - input_ids.shape[0]
        print(f"📊 Generated tokens: {generated_tokens} tokens")
        print(f"⏱️  Generation time: {end_time - start_time:.4f} seconds")
        
        # Check if eos token is present
        has_eos = eos_token_id in generated_ids[input_ids.shape[0]:]
        print(f"🔍 EOS token present in output: {has_eos}")
        
        # Decode output
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        generated_text = generated_text[len(prompt):]  # Remove prompt
        print(f"📝 Generated text: '{generated_text}'")
        print()
        
        return {
            "prompt": prompt,
            "has_eos": has_eos,
            "generated_tokens": generated_tokens,
            "generated_text": generated_text,
            "time": end_time - start_time
        }
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Test multiple prompts with transformers."""
    
    print("🚀 Testing Transformers EOS Token Generation")
    print("=" * 50)
    print("🌡️  Temperature: 0")
    print()
    
    # Test prompts
    prompts = [
        "中国是一个历史悠久的国家，",
        "解释一下C++和Python的区别",
        "你好，请问今天天气怎么样？",
        "1+1等于多少？",
        "编写一个简单的Python函数，计算斐波那契数列。",
        "什么是人工智能？",
        "如何学习编程？",
        "今天我很高兴，因为",
        "请列举三个太阳系中的行星。",
        "写一首关于春天的诗。"
    ]
    
    results = []
    
    for i, prompt in enumerate(prompts):
        print(f"📋 Test {i+1}/{len(prompts)}")
        print("-" * 30)
        result = test_transformers_eos(prompt)
        if result:
            results.append(result)
        print()
    
    # Summary
    print("📊 Summary")
    print("=" * 30)
    eos_count = sum(1 for r in results if r["has_eos"])
    print(f"Total tests: {len(results)}")
    print(f"Tests with EOS token: {eos_count}")
    print(f"Tests without EOS token: {len(results) - eos_count}")
    
    print("\n📝 Detailed results:")
    for i, result in enumerate(results):
        print(f"Test {i+1}: '{result['prompt']}'")
        print(f"   EOS: {'✓' if result['has_eos'] else '✗'}, Tokens: {result['generated_tokens']}")
        print(f"   Time: {result['time']:.4f}s")
        print()
    
    print("🎉 Testing completed!")

if __name__ == "__main__":
    main()
