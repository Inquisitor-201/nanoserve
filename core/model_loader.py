"""
Model weight loader module.
Responsible for loading model weights from disk and mapping them to model structure.
Supports both single-file and sharded safetensors, as well as AWQ quantized weights.
"""

from typing import Dict, Any, Optional, List, Set, Tuple
import os
import json
import re
import logging
import glob
import torch
from safetensors.torch import load_file

logger = logging.getLogger(__name__)


class ModelLoader:
    """
    Model weight loader for loading and mapping weights from disk to model.
    Supports HuggingFace safetensors format (single & sharded) and AWQ quantized weights.
    """

    @staticmethod
    def load_weights(
        model: torch.nn.Module,
        model_path: str,
        dtype: torch.dtype,
        device: str,
        tie_word_embeddings: bool = False,
    ) -> None:
        """Load weights from safetensors file(s) into *model*.

        Supports single-file, sharded (model-00001-of-N), and AWQ quantised formats.
        For quantised models the weight stays in int4 format — the fused CUDA kernel
        dequantises on the fly during forward.
        """
        logger.info(f"Loading model weights from {model_path}")

        # Find all safetensor files (single or sharded)
        safetensor_files = ModelLoader._find_safetensor_files(model_path)
        if not safetensor_files:
            raise FileNotFoundError(
                f"No safetensors model files found in {model_path}. "
                f"Expected model.safetensors or model-*.safetensors"
            )

        logger.info(f"Found {len(safetensor_files)} safetensor file(s)")
        for f in safetensor_files:
            fsize = os.path.getsize(f)
            logger.debug(f"  {os.path.basename(f)} ({fsize / 1e9:.2f} GB)")

        loaded_keys: Set[str] = set()
        missing_keys: List[str] = []

        for file_path in safetensor_files:
            logger.info(f"Loading {os.path.basename(file_path)}...")
            state_dict = load_file(file_path)

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

                loaded_keys.add(mapped_name)

        # Handle tied embeddings
        if tie_word_embeddings and 'lm_head.weight' not in loaded_keys:
            logger.info(
                "tie_word_embeddings=True and lm_head.weight not found in files. "
                "Copying from embed_tokens.weight."
            )
            with torch.no_grad():
                model.lm_head.weight.data.copy_(model.embed_tokens.weight.data)
            loaded_keys.add('lm_head.weight')

        # Report loading summary
        total_params = sum(p.numel() for p in model.parameters() if p is not None)
        loaded_params = sum(
            model.get_parameter(k).numel() for k in loaded_keys if hasattr(model, k) or any(
                p_name == k for p_name, _ in model.named_parameters()
            )
        )
        logger.info(
            f"Successfully loaded {len(loaded_keys)} parameter tensors"
        )
        if missing_keys:
            logger.warning(f"Missing {len(missing_keys)} keys: {missing_keys[:5]}...")

    @staticmethod
    def _find_safetensor_files(model_path: str) -> List[str]:
        """
        Find safetensors model files, handling both single and sharded formats.

        Resolution order:
        1. model.safetensors (single file)
        2. model.safetensors.index.json (sharded index)
        3. model-*.safetensors glob (fallback)
        """
        # Case 1: Single file
        single_file = os.path.join(model_path, "model.safetensors")
        if os.path.exists(single_file):
            return [single_file]

        # Case 2: Sharded with index file
        index_file = os.path.join(model_path, "model.safetensors.index.json")
        if os.path.exists(index_file):
            with open(index_file, 'r') as f:
                index_data = json.load(f)
            weight_map = index_data.get('weight_map', {})

            file_set: Set[str] = set()
            for fname in weight_map.values():
                file_set.add(os.path.join(model_path, fname))

            if file_set:
                return sorted(file_set)

        # Case 3: Glob fallback
        shard_files = sorted(glob.glob(os.path.join(model_path, "model-*.safetensors")))
        if shard_files:
            return shard_files

        return []

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
