"""
Model weight loader module.
Responsible for loading model weights from disk and mapping them to model structure.
"""

from typing import Dict, Any, Optional
import os
import logging
import torch
from pathlib import Path
from safetensors.torch import load_file

logger = logging.getLogger(__name__)


class ModelLoader:
    """
    Model weight loader for loading and mapping weights from disk to model.
    """
    
    @staticmethod
    def load_weights(model: torch.nn.Module, model_path: str, dtype: torch.dtype, device: str) -> None:
        """
        Load model weights from safetensors file.
        
        Args:
            model: Target model to load weights into
            model_path: Path to the model directory
            dtype: Target data type for weights
            device: Target device
        """
        logger.info(f"Loading model weights from {model_path}")
        
        # Look for safetensors file
        model_file = os.path.join(model_path, "model.safetensors")
        if not os.path.exists(model_file):
            raise FileNotFoundError(f"Model file not found: {model_file}")
        
        # Load weights
        state_dict = load_file(model_file)
        logger.info(f"Loaded {len(state_dict)} parameters from {model_file}")
        
        # Convert dtype and move to device
        for name, param in state_dict.items():
            if param.dtype != dtype:
                param = param.to(dtype)
            if device != "cpu":
                param = param.to(device)
            state_dict[name] = param
        
        # Load weights into model (simplified - in practice need proper mapping)
        try:
            model.load_state_dict(state_dict, strict=False)
            logger.info("Successfully loaded model weights")
        except Exception as e:
            logger.warning(f"Partial weight loading failed: {e}")
            # Try to load compatible weights
            model_dict = model.state_dict()
            compatible_dict = {}
            for name, param in state_dict.items():
                # Map common weight names
                mapped_name = ModelLoader._map_weight_name(name)
                if mapped_name in model_dict and param.shape == model_dict[mapped_name].shape:
                    compatible_dict[mapped_name] = param
                elif name in model_dict and param.shape == model_dict[name].shape:
                    compatible_dict[name] = param
            
            if compatible_dict:
                model.load_state_dict(compatible_dict, strict=False)
                logger.info(f"Loaded {len(compatible_dict)} compatible parameters")
    
    @staticmethod
    def _map_weight_name(hf_name: str) -> str:
        """
        Map HuggingFace weight names to our model naming.
        
        Args:
            hf_name: HuggingFace weight name
            
        Returns:
            Mapped weight name
        """
        # Simple mapping - in practice needs more sophisticated mapping
        name_mapping = {
            "model.embed_tokens.weight": "embed_tokens.weight",
            "model.norm.weight": "norm.weight",
            "lm_head.weight": "lm_head.weight",
            # Add more mappings as needed
        }
        
        # Handle layer-specific mappings
        if "model.layers." in hf_name:
            # Convert layers.X to layers[X]
            parts = hf_name.split('.')
            if len(parts) >= 3 and parts[1] == "layers":
                layer_idx = parts[2]
                remaining = '.'.join(parts[3:])
                return f"layers.{layer_idx}.{remaining}"
        
        return name_mapping.get(hf_name, hf_name)