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

class TestModelLoading(unittest.TestCase):
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
        Verify that EVERY loaded weight exactly matches the source safetensors values.
        Checks for data corruption during loading/casting.
        This test validates complete weight integrity - not just subset.
        """
        model_state = self.executor.model.state_dict()
        mismatched_keys = []
        checked_count = 0
        
        # Check EVERY weight in safetensors, not just a subset
        for hf_key, ref_tensor in self.ref_weights.items():
            mapped_key = ModelLoader._map_weight_name(hf_key)
            
            if mapped_key in model_state:
                checked_count += 1
                model_tensor = model_state[mapped_key].cpu()
                ref_tensor_comp = ref_tensor.to(dtype=self.config["dtype"])
                
                # torch.allclose with appropriate tolerances for float16
                if not torch.allclose(model_tensor, ref_tensor_comp, atol=1e-3, rtol=1e-3):
                    mismatched_keys.append({
                        "hf_key": hf_key,
                        "mapped_key": mapped_key,
                        "max_diff": (model_tensor - ref_tensor_comp).abs().max().item()
                    })
        
        # Report results
        if mismatched_keys:
            first_mismatch = mismatched_keys[0]
            self.fail(
                f"Weight mismatch detected!\n"
                f"Checked {checked_count}/{len(self.ref_weights)} weights from safetensors.\n"
                f"First mismatch: {first_mismatch['hf_key']} -> {first_mismatch['mapped_key']}\n"
                f"Max diff: {first_mismatch['max_diff']:.6f}"
            )
        else:
            self.assertEqual(checked_count, len(self.ref_weights), 
                           f"Not all weights were checked: {checked_count} vs {len(self.ref_weights)}")
            print(f"[OK] All {checked_count} weights from safetensors match exactly.")

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
    
    def test_04_debug_head_config(self):
        """
        Debug: Print weight shapes to infer actual head configuration.
        By analyzing weight shapes, we can deduce the correct num_heads and num_key_value_heads.
        """
        state_dict = self.executor.model.state_dict()
        
        print("\n" + "="*60)
        print("WEIGHT SHAPE ANALYSIS FOR HEAD CONFIG")
        print("="*60)
        
        # Check Q, K, V projection shapes
        q_proj = state_dict.get("layers.0.self_attn.q_proj.weight")
        k_proj = state_dict.get("layers.0.self_attn.k_proj.weight")
        v_proj = state_dict.get("layers.0.self_attn.v_proj.weight")
        
        if q_proj is not None:
            print(f"\nq_proj.weight shape: {q_proj.shape}")
            print(f"  - q_proj.shape[0] = {q_proj.shape[0]} (output dimension)")
            print(f"  - q_proj.shape[1] = {q_proj.shape[1]} (input dimension = hidden_size)")
            
            # Infer num_heads from weight shape
            # q_proj.shape[0] = num_heads * head_dim
            inferred_num_heads = q_proj.shape[0] // self.config["head_dim"]
            print(f"  - Inferred num_heads = q_proj.shape[0] / head_dim = {inferred_num_heads}")
        
        if k_proj is not None:
            print(f"\nk_proj.weight shape: {k_proj.shape}")
            print(f"  - k_proj.shape[0] = {k_proj.shape[0]} (output dimension)")
            print(f"  - k_proj.shape[1] = {k_proj.shape[1]} (input dimension = hidden_size)")
            
            # Infer num_key_value_heads from weight shape
            # k_proj.shape[0] = num_key_value_heads * head_dim
            inferred_kv_heads = k_proj.shape[0] // self.config["head_dim"]
            print(f"  - Inferred num_key_value_heads = k_proj.shape[0] / head_dim = {inferred_kv_heads}")
        
        if v_proj is not None:
            print(f"\nv_proj.weight shape: {v_proj.shape}")
            print(f"  - v_proj.shape[0] = {v_proj.shape[0]}")
            
            inferred_kv_heads_v = v_proj.shape[0] // self.config["head_dim"]
            print(f"  - Inferred num_key_value_heads = v_proj.shape[0] / head_dim = {inferred_kv_heads_v}")
        
        # Print model config
        print(f"\nCurrent test config:")
        print(f"  - num_heads: {self.config['num_heads']}")
        print(f"  - num_key_value_heads: {self.config['num_key_value_heads']}")
        print(f"  - head_dim: {self.config['head_dim']}")
        print(f"  - hidden_size: {self.config['hidden_size']}")
        
        print("\n" + "="*60)
        
        # Assert that inferred config matches test config
        if q_proj is not None:
            expected_q_dim = self.config["num_heads"] * self.config["head_dim"]
            actual_q_dim = q_proj.shape[0]
            self.assertEqual(expected_q_dim, actual_q_dim, 
                           f"q_proj output dim mismatch: expected {expected_q_dim}, got {actual_q_dim}")
        
        if k_proj is not None:
            expected_k_dim = self.config["num_key_value_heads"] * self.config["head_dim"]
            actual_k_dim = k_proj.shape[0]
            self.assertEqual(expected_k_dim, actual_k_dim,
                           f"k_proj output dim mismatch: expected {expected_k_dim}, got {actual_k_dim}")

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