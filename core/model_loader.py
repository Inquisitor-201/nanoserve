"""
Model weight loader module.
"""

from typing import Optional
import os
import logging
import torch
from safetensors.torch import load_file

logger = logging.getLogger(__name__)


class ModelLoader:
    """Load model weights from disk into a model instance."""

    @staticmethod
    def load_weights(
        model: torch.nn.Module,
        model_path: str,
        dtype: torch.dtype,
        device: str,
    ) -> None:
        """Load weights from a safetensors file into *model*.

        Works for both plain (BF16) and quantised (AWQ) models.  For quantised
        models the weight stays in int4 format — the fused CUDA kernel
        dequantises on the fly during forward.
        """
        logger.info(f"Loading model weights from {model_path}")

        model_file = os.path.join(model_path, "model.safetensors")
        if not os.path.exists(model_file):
            raise FileNotFoundError(f"Model file not found: {model_file}")

        state_dict = load_file(model_file)
        logger.info(f"Loaded {len(state_dict)} parameters from {model_file}")

        for hf_name, param in state_dict.items():
            mapped_name = ModelLoader._map_weight_name(hf_name)

            # Try parameter first (nn.Linear, RMSNorm, Embedding),
            # then fall back to buffer (AWQLinear qweight / qzeros / scales).
            try:
                model_param = model.get_parameter(mapped_name)
            except AttributeError:
                model_param = model.get_buffer(mapped_name)

            if param.dtype != model_param.dtype:
                param = param.to(model_param.dtype)

            with torch.no_grad():
                model_param.copy_(param)

        logger.info("Successfully loaded model weights")

    @staticmethod
    def _map_weight_name(hf_name: str) -> str:
        """Strip the ``model.`` prefix added by HuggingFace."""
        if hf_name.startswith("model."):
            return hf_name[6:]
        return hf_name
