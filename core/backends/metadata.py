"""
Metadata dataclass for attention computation.
Stores inference metadata like block tables, indptr, and operation type.
"""

from dataclasses import dataclass
from typing import List, Optional, Union
import torch


@dataclass
class AttentionMetadata:
    """
    Unified metadata storage for attention computation.
    
    This dataclass encapsulates all the metadata needed for attention computation,
    providing a clean interface between models and attention backends.
    """
    
    # Block tables for paged attention
    block_tables: Optional[List[List[int]]] = None
    
    # Sequence lengths
    seq_lengths: Optional[List[int]] = None
    
    # Indptr tensors for paged KV cache
    paged_kv_indptr: Optional[torch.Tensor] = None
    paged_kv_indices: Optional[torch.Tensor] = None
    paged_kv_last_page_len: Optional[torch.Tensor] = None
    
    # Query/Output indptr (for prefill)
    qo_indptr: Optional[torch.Tensor] = None
    
    # Geometric metadata for FlashInfer operators
    # batch_indices: Maps each token to its batch sequence index
    # positions: Relative position within each sequence
    batch_indices: Optional[torch.Tensor] = None
    positions: Optional[torch.Tensor] = None
    
    # Operation type flags
    is_prefill: bool = True
    causal: bool = True
    
    # Additional metadata
    batch_size: Optional[int] = None
    max_seq_len: Optional[int] = None
    num_tokens: Optional[int] = None
    
    def __post_init__(self):
        """Validate and compute derived metadata."""
        if self.seq_lengths is not None:
            self.batch_size = len(self.seq_lengths)
            self.max_seq_len = max(self.seq_lengths)
            self.num_tokens = sum(self.seq_lengths)
            
            if self.block_tables is not None and len(self.block_tables) != self.batch_size:
                raise ValueError(f"block_tables length {len(self.block_tables)} != batch_size {self.batch_size}")
                
            if self.block_tables is not None and len(self.seq_lengths) != self.batch_size:
                raise ValueError(f"seq_lengths length {len(self.seq_lengths)} != batch_size {self.batch_size}")
    
    def __repr__(self) -> str:
        """Custom repr that avoids printing CUDA tensors directly."""
        fields = {
            "block_tables": self.block_tables,
            "seq_lengths": self.seq_lengths,
            "paged_kv_indptr": f"tensor(shape={self.paged_kv_indptr.shape})" if self.paged_kv_indptr is not None else None,
            "paged_kv_indices": f"tensor(shape={self.paged_kv_indices.shape})" if self.paged_kv_indices is not None else None,
            "paged_kv_last_page_len": f"tensor(shape={self.paged_kv_last_page_len.shape})" if self.paged_kv_last_page_len is not None else None,
            "qo_indptr": f"tensor(shape={self.qo_indptr.shape})" if self.qo_indptr is not None else None,
            "batch_indices": f"tensor(shape={self.batch_indices.shape})" if self.batch_indices is not None else None,
            "positions": f"tensor(shape={self.positions.shape})" if self.positions is not None else None,
            "is_prefill": self.is_prefill,
            "causal": self.causal,
            "batch_size": self.batch_size,
            "max_seq_len": self.max_seq_len,
            "num_tokens": self.num_tokens,
        }
        return f"AttentionMetadata({fields})"
    
    @classmethod
    def from_block_tables(
        cls,
        block_tables: List[List[int]],
        seq_lengths: List[int],
        page_size: int,
        is_prefill: bool = True,
        causal: bool = True,
        device: Union[str, torch.device] = "cuda"
    ) -> "AttentionMetadata":
        """
        Create metadata from block tables and sequence lengths.

        Args:
            block_tables: List of block tables for each sequence
            seq_lengths: List of sequence lengths
            is_prefill: Whether this is prefill operation
            causal: Whether to use causal mask
            device: Device for tensor creation
            page_size: Page size for block management

        Returns:
            AttentionMetadata instance
        """
        seq_lens_tensor = torch.tensor(seq_lengths, dtype=torch.int32, device=device)
        num_pages = (seq_lens_tensor + page_size - 1) // page_size

        flat_indices = []
        indptr = [0]
        for block_table, num_pg in zip(block_tables, num_pages.tolist()):
            flat_indices.extend(block_table[:num_pg])
            indptr.append(len(flat_indices))

        last_page_len = ((seq_lens_tensor - 1) % page_size + 1).to(dtype=torch.int32)

        paged_kv_indices = torch.tensor(flat_indices, dtype=torch.int32, device=device)
        paged_kv_indptr = torch.tensor(indptr, dtype=torch.int32, device=device)
        paged_kv_last_page_len = last_page_len

        batch_size = len(seq_lengths)

        if is_prefill:
            qo_indptr_tensor = torch.cat([
                torch.zeros(1, dtype=torch.int32, device=device),
                torch.cumsum(seq_lens_tensor, dim=0)
            ]).to(dtype=torch.int32)

            batch_indices = torch.repeat_interleave(
                torch.arange(batch_size, dtype=torch.int32, device=device),
                seq_lens_tensor
            )

            total_tokens = int(seq_lens_tensor.sum())
            positions = torch.arange(total_tokens, dtype=torch.int32, device=device)
            q_offsets = torch.repeat_interleave(
                qo_indptr_tensor[:-1],
                seq_lens_tensor
            )
            # Ensure q_offsets is int32 to match positions
            q_offsets = q_offsets.to(dtype=torch.int32)
            positions = positions - q_offsets
            # Ensure positions remains int32
            positions = positions.to(dtype=torch.int32)
        else:
            qo_indptr_tensor = None
            batch_indices = torch.arange(batch_size, dtype=torch.int32, device=device)
            positions = (seq_lens_tensor - 1).to(dtype=torch.int32)
        
        return cls(
            block_tables=block_tables,
            seq_lengths=seq_lengths,
            paged_kv_indptr=paged_kv_indptr,
            paged_kv_indices=paged_kv_indices,
            paged_kv_last_page_len=paged_kv_last_page_len,
            qo_indptr=qo_indptr_tensor,
            batch_indices=batch_indices,
            positions=positions,
            is_prefill=is_prefill,
            causal=causal,
        )