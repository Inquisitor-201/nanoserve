"""
Model weight loader module.
"""

from typing import Optional, Tuple
import os
import re
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

            # Resolve param (handles fused QKV shards), fall back to buffer (AWQ qweight etc.)
            try:
                model_param = ModelLoader._resolve_param(model, mapped_name)
            except AttributeError:
                model_param = model.get_buffer(mapped_name)

            if param.dtype != model_param.dtype:
                param = param.to(model_param.dtype)
            with torch.no_grad():
                model_param.copy_(param)

        logger.info("Successfully loaded model weights")

    @staticmethod
    def _resolve_param(model: torch.nn.Module, name: str) -> torch.Tensor:
        """Resolve *name* to the target parameter tensor.

        Tries normal ``get_parameter`` first; falls back to stacked-param
        resolution for fused projections (e.g. ``q_proj.weight`` →
        a slice of ``qkv_proj.weight``).
        """
        try:
            return model.get_parameter(name)
        except AttributeError:
            shard = ModelLoader._resolve_stacked_shard(model, name)
            if shard is not None:
                return shard[0]
            raise

    # Regex for stacked param names:  {path}.{q,k,v}_proj.{suffix}
    # Handles nn.Linear (weight/bias) and AWQ (qweight/qzeros/scales).
    _STACKED_PARAM_RE = re.compile(
        r"^(.*\.self_attn)\.([qkv])_proj\.(weight|bias|qweight|qzeros|scales)$"
    )

    @staticmethod
    def _resolve_stacked_shard(
        model: torch.nn.Module, name: str
    ) -> Optional[Tuple[torch.Tensor, int]]:
        """If *name* refers to a shard of a fused QKV projection,
        return ``(fused_param, dim_0_offset)``.

        The target module must expose a ``_qkv_shard_info`` dict:
        ``{shard_id: (offset, size)}``.
        """
        m = ModelLoader._STACKED_PARAM_RE.match(name)
        if not m:
            return None

        module_path, shard_id, suffix = m.groups()
        module = model.get_submodule(module_path)

        shard_info = getattr(module, "_qkv_shard_info", None)
        if shard_info is None:
            return None

        offset, size = shard_info[shard_id]
        fused_name = f"{module_path}.qkv_proj.{suffix}"

        # ── Pick the right narrow dim and packing factor ──────────
        # nn.Linear.weight/bias  →  dim 0 = out_features
        # AWQ qweight/qzeros     →  dim 1 = packed out_features (×1/8)
        # AWQ scales             →  dim 1 = out_features (no pack)
        if suffix in ("weight", "bias"):
            fused_param = model.get_parameter(fused_name)
            view = fused_param.narrow(0, offset, size)
        elif suffix == "scales":
            fused_param = model.get_buffer(fused_name)
            view = fused_param.narrow(1, offset, size)
        else:  # qweight, qzeros — packed int4
            fused_param = model.get_buffer(fused_name)
            pack_factor = 8  # 32 bits ÷ 4 bits
            view = fused_param.narrow(1, offset // pack_factor, size // pack_factor)
        return (view, offset)

    @staticmethod
    def _map_weight_name(hf_name: str) -> str:
        """Strip the ``model.`` prefix added by HuggingFace."""
        if hf_name.startswith("model."):
            return hf_name[6:]
        return hf_name
