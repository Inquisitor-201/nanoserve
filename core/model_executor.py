"""
Model executor for the architecture.
Manages model execution with proper attention backend integration.
"""

from typing import List, Optional, Dict, Any
import torch
import logging
import os
from pathlib import Path
from safetensors.torch import load_file

from .backends import AttentionMetadata, FlashInferBackend
from .models import Qwen3Model
from .block_manager import BlockManager


logger = logging.getLogger(__name__)


class ModelExecutor:
    """
    Model executor for managing the complete inference pipeline.
    
    This executor ensures proper coordination between models and attention backends.
    """
    
    def __init__(
        self,
        model_name: str = "qwen3",
        vocab_size: int = 32000,
        hidden_size: int = 4096,
        num_heads: int = 32,
        head_dim: int = 128,
        intermediate_size: int = 11008,
        num_layers: int = 32,
        attention_backend: str = "flashinfer",
        dtype: torch.dtype = torch.float16,
        device: str = "cuda",
        num_blocks: int = 1000,
        block_size: int = 16,
        model_path: Optional[str] = None
    ):
        """
        Initialize model executor.
        
        Args:
            model_name: Name of the model architecture
            vocab_size: Vocabulary size
            hidden_size: Hidden size of the model
            num_heads: Number of attention heads
            head_dim: Dimension of each attention head
            intermediate_size: Intermediate size for MLP
            num_layers: Number of decoder layers
            attention_backend: Attention backend type
            dtype: Data type for computations
            device: Computing device
            num_blocks: Number of KV cache blocks
            block_size: Size of each cache block
            model_path: Path to model weights for loading
        """
        self.model_name = model_name
        self.device = device
        self.dtype = dtype
        
        # Initialize block manager for KV cache management
        self.block_manager = BlockManager(
            num_blocks=num_blocks,
            num_layers=num_layers,
            num_heads=num_heads,
            head_dim=head_dim,
            block_size=block_size,
            dtype=dtype,
            device=device
        )
        
        # Initialize model based on model name
        if model_name == "qwen3":
            self.model = Qwen3Model(
                vocab_size=vocab_size,
                hidden_size=hidden_size,
                num_heads=num_heads,
                head_dim=head_dim,
                intermediate_size=intermediate_size,
                num_layers=num_layers,
                attention_backend_type=attention_backend,
                dtype=dtype,
                device=device
            )
            
            # Load model weights if path provided
            if model_path:
                self._load_model_weights(model_path, dtype, device)
                
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        
        logger.info(f"Initialized ModelExecutor with {model_name} model")
    
    def execute_prefill(
        self,
        input_ids: torch.Tensor,
        block_tables: List[List[int]],
        seq_lengths: List[int]
    ) -> torch.Tensor:
        """
        Execute prefill phase.
        
        Args:
            input_ids: Input token IDs
            block_tables: Block tables for each sequence
            seq_lengths: Sequence lengths
            
        Returns:
            Hidden states after prefill
        """
        # Create attention metadata for prefill
        metadata = AttentionMetadata.from_block_tables(
            block_tables=block_tables,
            seq_lengths=seq_lengths,
            is_prefill=True,
            device=self.device
        )
        
        # Execute model forward pass
        hidden_states = self.model(input_ids, metadata)
        
        return hidden_states
    
    def execute_decode(
        self,
        input_ids: torch.Tensor,
        block_tables: List[List[int]],
        seq_lengths: List[int]
    ) -> torch.Tensor:
        """
        Execute decode phase.
        
        Args:
            input_ids: Input token IDs (one per sequence)
            block_tables: Block tables for each sequence
            seq_lengths: Sequence lengths
            
        Returns:
            Hidden states after decode
        """
        # Create attention metadata for decode
        metadata = AttentionMetadata.from_block_tables(
            block_tables=block_tables,
            seq_lengths=seq_lengths,
            is_prefill=False,
            device=self.device
        )
        
        # Execute model forward pass
        hidden_states = self.model(input_ids, metadata)
        
        return hidden_states
    
    def generate(
        self,
        input_ids: torch.Tensor,
        block_tables: List[List[int]],
        seq_lengths: List[int],
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_p: float = 0.9
    ) -> torch.Tensor:
        """
        Generate tokens using the model.
        
        Args:
            input_ids: Initial input token IDs
            block_tables: Block tables for each sequence
            seq_lengths: Initial sequence lengths
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling threshold
            
        Returns:
            Generated token IDs
        """
        return self.model.generate(
            input_ids=input_ids,
            block_tables=block_tables,
            seq_lengths=seq_lengths,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p
        )
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            "model_name": self.model_name,
            "device": self.device,
            "dtype": str(self.dtype),
            "model": str(self.model),
            "block_manager": str(self.block_manager)
        }
    
    def _load_model_weights(self, model_path: str, dtype: torch.dtype, device: str) -> None:
        """
        Load model weights from safetensors file.
        
        Args:
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
            self.model.load_state_dict(state_dict, strict=False)
            logger.info("Successfully loaded model weights")
        except Exception as e:
            logger.warning(f"Partial weight loading failed: {e}")
            # Try to load compatible weights
            model_dict = self.model.state_dict()
            compatible_dict = {}
            for name, param in state_dict.items():
                # Map common weight names
                mapped_name = self._map_weight_name(name)
                if mapped_name in model_dict and param.shape == model_dict[mapped_name].shape:
                    compatible_dict[mapped_name] = param
                elif name in model_dict and param.shape == model_dict[name].shape:
                    compatible_dict[name] = param
            
            if compatible_dict:
                self.model.load_state_dict(compatible_dict, strict=False)
                logger.info(f"Loaded {len(compatible_dict)} compatible parameters")
    
    def _map_weight_name(self, hf_name: str) -> str:
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
    
    def __repr__(self) -> str:
        return f"ModelExecutor(model_name={self.model_name}, device={self.device})"