import os
import sys
import unittest
import torch

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.llm_service import LLMService
from transformers import AutoModelForCausalLM, AutoTokenizer

class TestTransformerCompare(unittest.TestCase):
    def setUp(self):
        self.model_path = "/root/nanoserve/models/Qwen3-0.6B"
        self.device = "cuda"
        self.temperature = 0.0  # 贪婪采样
        self.max_new_tokens = 20
        self.prompts = ["Hello, world!", "Please tell me, the future of AI is"]
        
        # Load our LLMService
        self.llm_service = LLMService(device=self.device)
        config = {
            "dtype": torch.float16,
            "num_blocks": 200,
            "block_size": 16,
            "attention_backend": "flashinfer",
        }
        self.llm_service.load_model(model_path=self.model_path, config=config)
        
        # Load transformers model and tokenizer
        self.transformer_tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.transformer_model = AutoModelForCausalLM.from_pretrained(
            self.model_path, 
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.transformer_model.eval()
    
    def test_01_generation_comparison(self):
        """Compare generation results between our implementation and transformers"""
        print("\n=== Generation Comparison Test ===")
        print(f"Temperature: {self.temperature}")
        print(f"Max new tokens: {self.max_new_tokens}")
        print(f"Prompts: {self.prompts}")
        
        # Generate with our implementation
        print("\n1. Generating with our implementation...")
        with torch.inference_mode():
            our_generations = self.llm_service.generate(
                prompts=self.prompts,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=0.9
            )
        
        # Generate with transformers
        print("\n2. Generating with transformers...")
        transformer_generations = []
        with torch.inference_mode():
            for prompt in self.prompts:
                inputs = self.transformer_tokenizer(prompt, return_tensors="pt").to(self.device)
                outputs = self.transformer_model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    top_p=0.9,
                    do_sample=self.temperature > 0
                )
                generated = self.transformer_tokenizer.decode(outputs[0], skip_special_tokens=True)
                # Extract only the generated part (excluding prompt)
                generated = generated[len(prompt):].strip()
                transformer_generations.append(generated)
        
        # Print results
        print("\n=== Results ===")
        for i, (prompt, our_gen, tf_gen) in enumerate(zip(self.prompts, our_generations, transformer_generations)):
            print(f"\nPrompt {i+1}: '{prompt}'")
            print(f"Our implementation: '{our_gen}'")
            print(f"Transformers: '{tf_gen}'")
            print(f"Match: {our_gen == tf_gen}")
        
        # Verify both methods generate non-empty text
        for i, (our_gen, tf_gen) in enumerate(zip(our_generations, transformer_generations)):
            self.assertGreater(len(our_gen), 0, f"Our implementation generated empty text for prompt {i+1}")
            self.assertGreater(len(tf_gen), 0, f"Transformers generated empty text for prompt {i+1}")

if __name__ == "__main__":
    unittest.main()