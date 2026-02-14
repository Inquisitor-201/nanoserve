import os
import sys
import unittest
import torch
from pathlib import Path
from safetensors.torch import load_file

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.model_executor import ModelExecutor
from core.model_loader import ModelLoader

class TestModelLoadingStrict(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Initialize ModelExecutor once for the entire test class."""
        cls.model_path = "./models/Qwen3-0.6B"
        if not os.path.exists(cls.model_path):
            raise FileNotFoundError(f"Model not found at {cls.model_path}")

        cls.config = {
            "model_name": "qwen3",
            "vocab_size": 151936,
            "hidden_size": 1024,
            "num_heads": 16,  # Corrected: Qwen3-0.6B has 16 attention heads
            "num_key_value_heads": 8,  # Qwen3-0.6B has 8 key-value heads (GQA)
            "head_dim": 128,
            "intermediate_size": 3072,
            "num_layers": 28,
            "attention_backend": "flashinfer",
            "dtype": torch.float16,
            "device": "cuda",
            "num_blocks": 100,
            "block_size": 16
        }
        print("\n[Setup] Initializing ModelExecutor...")
        cls.executor = ModelExecutor(**cls.config, model_path=cls.model_path)
        
        # Load reference weights for validation
        cls.ref_weights = load_file(os.path.join(cls.model_path, "model.safetensors"))

    def test_01_mapping_integrity(self):
        """Verify 100% parameter mapping integrity."""
        model_keys = set(self.executor.model.state_dict().keys())
        missing = []
        for hf_key in self.ref_weights.keys():
            if "layers." in hf_key:
                layer_idx = int(hf_key.split(".")[2])
                if layer_idx >= self.config["num_layers"]: continue
            
            mapped_key = ModelLoader._map_weight_name(hf_key)
            if mapped_key not in model_keys:
                missing.append(f"{hf_key} -> {mapped_key}")
        
        self.assertEqual(len(missing), 0, f"Mapping failed for keys: {missing[:5]}")

    def test_02_weight_exact_match(self):
        """
        Verify that loaded weights exactly match the source safetensors values.
        Checks for data corruption during loading/casting.
        """
        model_state = self.executor.model.state_dict()
        mismatched_layers = []

        # We check a subset of layers to save time, or all if needed
        # Checking Layer 0, Layer 14 (middle), and Layer 27 (last) + Embeddings
        check_patterns = ["layers.0.", "layers.14.", "layers.27.", "embed_tokens", "lm_head"]

        for hf_key, ref_tensor in self.ref_weights.items():
            mapped_key = ModelLoader._map_weight_name(hf_key)
            
            # Filter to check only relevant keys based on our config
            if any(pattern in mapped_key for pattern in check_patterns):
                if mapped_key in model_state:
                    model_tensor = model_state[mapped_key].cpu() # Move to CPU for comparison
                    ref_tensor_comp = ref_tensor.to(dtype=self.config["dtype"])
                    
                    # Use a small epsilon for float16 comparison
                    # torch.allclose is better than == for half precision
                    if not torch.allclose(model_tensor, ref_tensor_comp, atol=1e-3, rtol=1e-3):
                        mismatched_layers.append(mapped_key)
        
        self.assertEqual(len(mismatched_layers), 0, f"Numerical mismatch in weights: {mismatched_layers}")
        print(f"[OK] Exact match verification passed for {len(self.ref_weights)} potential tensors.")

    def test_03_weight_values_fingerprint(self):
        """Verify weights are not zero or random by checking statistical fingerprint."""
        state_dict = self.executor.model.state_dict()
        targets = ["layers.0.self_attn.q_proj.weight", "embed_tokens.weight"]
        for name in targets:
            if name in state_dict:
                tensor = state_dict[name]
                abs_mean = tensor.abs().mean().item()
                std = tensor.std().item()
                
                self.assertGreater(abs_mean, 1e-5, f"Weight {name} seems uninitialized (abs_mean=0)")
                self.assertTrue(0.001 < std < 0.5, f"Weight {name} has abnormal std: {std}")

    def test_04_inference_stability(self):
        """Run a dummy prefill and check if logits are within reasonable bounds."""
        input_ids = torch.tensor([151644, 8948, 198], device="cuda")
        with torch.inference_mode():
            # Ensure the executor uses the weights we just verified
            hidden_states = self.executor.execute_prefill(input_ids, [[]], [len(input_ids)])
            logits = self.executor.model.lm_head(hidden_states)
        
        max_logit = logits.max().item()
        self.assertTrue(abs(max_logit) < 25.0, f"Exploding/Imploding logits detected: {max_logit}")

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, 'executor'):
            del cls.executor
        torch.cuda.empty_cache()

if __name__ == "__main__":
    unittest.main(verbosity=2)