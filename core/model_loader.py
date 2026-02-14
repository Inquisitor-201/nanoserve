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
        
        # Load weights in-place to reduce VRAM usage
        for hf_name, param in state_dict.items():
            # Map the weight name
            mapped_name = ModelLoader._map_weight_name(hf_name)
            
            # Get the model parameter
            model_param = model.get_parameter(mapped_name)
            
            # Convert dtype and move to device to match model parameter
            if param.dtype != model_param.dtype:
                param = param.to(model_param.dtype)
            
            # Copy weights in-place without gradients
            with torch.no_grad():
                model_param.copy_(param)
        
        logger.info("Successfully loaded model weights")
    
    @staticmethod
    def _map_weight_name(hf_name: str) -> str:
        """
        Map HuggingFace weight names to our model naming.
        
        Args:
            hf_name: HuggingFace weight name
            
        Returns:
            Mapped weight name
        """
        # Remove 'model.' prefix for all names
        if hf_name.startswith("model."):
            return hf_name[6:]  # Remove 'model.' prefix
        
        # Return original name if no mapping found
        return hf_name