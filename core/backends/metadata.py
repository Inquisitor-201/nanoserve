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
    
    @classmethod
    def from_block_tables(
        cls,
        block_tables: List[List[int]],
        seq_lengths: List[int],
        is_prefill: bool = True,
        causal: bool = True,
        device: Union[str, torch.device] = "cuda",
        qo_indptr: Optional[torch.Tensor] = None
    ) -> "AttentionMetadata":
        """
        Create metadata from block tables and sequence lengths.
        
        Args:
            block_tables: List of block tables for each sequence
            seq_lengths: List of sequence lengths
            is_prefill: Whether this is prefill operation
            causal: Whether to use causal mask
            device: Device for tensor creation
            
        Returns:
            AttentionMetadata instance
        """
        # Convert block tables to FlashInfer format
        flat_indices = []
        indptr = [0]
        last_page_len = []
        
        for i, (block_table, seq_len) in enumerate(zip(block_tables, seq_lengths)):
            # Calculate number of pages needed
            page_size = 16  # This should match the backend configuration
            num_pages = (seq_len + page_size - 1) // page_size
            
            # Add block indices for this sequence
            flat_indices.extend(block_table[:num_pages])
            indptr.append(len(flat_indices))
            
            # Calculate last page length
            remainder = seq_len % page_size
            if remainder == 0:
                last_page_len.append(page_size)
            else:
                last_page_len.append(remainder)
        
        # Convert to tensors
        paged_kv_indices = torch.tensor(flat_indices, dtype=torch.int32, device=device)
        paged_kv_indptr = torch.tensor(indptr, dtype=torch.int32, device=device)
        paged_kv_last_page_len = torch.tensor(last_page_len, dtype=torch.int32, device=device)
        
        # Use provided qo_indptr or build it from sequence lengths
        if qo_indptr is not None:
            # Use provided qo_indptr (from unpadding logic)
            qo_indptr_tensor = qo_indptr
        elif is_prefill:
            # Build qo_indptr for prefill if not provided
            qo_indptr = [0]
            current_pos = 0
            for seq_len in seq_lengths:
                current_pos += seq_len
                qo_indptr.append(current_pos)
            qo_indptr_tensor = torch.tensor(qo_indptr, dtype=torch.int32, device=device)
        else:
            qo_indptr_tensor = None
        
        return cls(
            block_tables=block_tables,
            seq_lengths=seq_lengths,
            paged_kv_indptr=paged_kv_indptr,
            paged_kv_indices=paged_kv_indices,
            paged_kv_last_page_len=paged_kv_last_page_len,
            qo_indptr=qo_indptr_tensor,
            is_prefill=is_prefill,
            causal=causal,
        )
    
    def to_device(self, device: Union[str, torch.device]) -> "AttentionMetadata":
        """Move tensor metadata to specified device."""
        if isinstance(device, str):
            device = torch.device(device)
            
        # Move tensors to device
        if self.paged_kv_indptr is not None:
            self.paged_kv_indptr = self.paged_kv_indptr.to(device)
        if self.paged_kv_indices is not None:
            self.paged_kv_indices = self.paged_kv_indices.to(device)
        if self.paged_kv_last_page_len is not None:
            self.paged_kv_last_page_len = self.paged_kv_last_page_len.to(device)
        if self.qo_indptr is not None:
            self.qo_indptr = self.qo_indptr.to(device)
            
        return self