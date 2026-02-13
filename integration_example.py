"""
Integration example showing how to use BlockManager with ModelExecutor.
This demonstrates the complete workflow for using FlashInfer with HuggingFace models.
"""

import torch
from block_manager import BlockManager
from model_executor import ModelExecutor


def simulate_llama_forward_with_flashinfer():
    """
    Simulate a complete Llama forward pass using FlashInfer for attention computation.
    """
    
    print("=== Llama + FlashInfer Integration Example ===\n")
    
    # Configuration
    num_blocks = 100
    num_layers = 32
    num_heads = 32
    head_dim = 128
    block_size = 16
    vocab_size = 32000
    
    print("1. Creating BlockManager...")
    block_manager = BlockManager(
        num_blocks=num_blocks,
        num_layers=num_layers,
        num_heads=num_heads,
        head_dim=head_dim,
        block_size=block_size,
        dtype=torch.float16,
        device="cpu"  # Use CPU for demo
    )
    print(f"   BlockManager created: {block_manager}")
    
    print("\n2. Creating ModelExecutor...")
    model_executor = ModelExecutor(
        block_manager=block_manager,
        num_heads=num_heads,
        head_dim=head_dim,
        page_size=block_size,
        dtype=torch.float16,
        device="cpu"
    )
    print(f"   ModelExecutor created: {model_executor}")
    
    # Simulate a batch of requests
    batch_size = 2
    seq_lengths = [20, 35]  # Different sequence lengths
    
    print(f"\n3. Processing batch of {batch_size} sequences...")
    print(f"   Sequence lengths: {seq_lengths}")
    
    # Allocate blocks for each sequence
    block_tables = []
    for i, seq_len in enumerate(seq_lengths):
        num_blocks_needed = (seq_len + block_size - 1) // block_size
        blocks = block_manager.allocate_blocks(seq_len)
        if blocks is None:
            print(f"   Failed to allocate blocks for sequence {i}")
            continue
        block_tables.append(blocks)
        print(f"   Sequence {i}: allocated {len(blocks)} blocks: {blocks}")
    
    print(f"\n4. Remaining blocks: free={block_manager.num_free_blocks}, allocated={block_manager.num_allocated_blocks}")
    
    # Simulate input tokens
    print("\n5. Creating input tensors...")
    input_ids = []
    for seq_len in seq_lengths:
        # Random token IDs for demonstration
        tokens = torch.randint(0, vocab_size, (1, seq_len), dtype=torch.long)
        input_ids.append(tokens)
    
    for i, tokens in enumerate(input_ids):
        print(f"   Sequence {i}: shape {tokens.shape}, sample tokens: {tokens[0, :5].tolist()}")
    
    # Simulate prefill phase (first forward pass)
    print("\n6. Running prefill phase...")
    for layer_idx in range(min(2, num_layers)):  # Just first 2 layers for demo
        print(f"   Layer {layer_idx}:")
        
        for seq_idx, (tokens, blocks) in enumerate(zip(input_ids, block_tables)):
            # Execute attention computation for this sequence
            output = model_executor.execute_model(
                input_ids=tokens,
                block_tables=[blocks],
                seq_lengths=[seq_lengths[seq_idx]],
                layer_idx=layer_idx,
                is_prefill=True
            )
            print(f"     Sequence {seq_idx}: attention output shape {output.shape}")
    
    # Simulate decode phase (generating new tokens)
    print("\n7. Running decode phase (generating 1 new token per sequence)...")
    new_tokens = []
    for seq_idx, blocks in enumerate(block_tables):
        # Single new token for each sequence
        new_token = torch.randint(0, vocab_size, (1, 1), dtype=torch.long)
        new_tokens.append(new_token)
        
        # Execute decode attention
        output = model_executor.execute_model(
            input_ids=new_token,
            block_tables=[blocks],
            seq_lengths=[seq_lengths[seq_idx] + 1],  # +1 for the new token
            layer_idx=0,
            is_prefill=False
        )
        print(f"   Sequence {seq_idx}: decode output shape {output.shape}")
    
    # Clean up
    print("\n8. Cleaning up...")
    for blocks in block_tables:
        block_manager.free_blocks(blocks)
    print(f"   Final state: {block_manager}")
    
    print("\n=== Integration example completed! ===")


def demonstrate_kv_cache_mapping():
    """
    Demonstrate how logical block tables are mapped to FlashInfer format.
    """
    
    print("\n=== KV Cache Mapping Demonstration ===\n")
    
    # Setup
    block_manager = BlockManager(
        num_blocks=10,
        num_layers=2,
        num_heads=8,
        head_dim=64,
        block_size=16,
        dtype=torch.float32,
        device="cpu"
    )
    
    model_executor = ModelExecutor(
        block_manager=block_manager,
        num_heads=8,
        head_dim=64,
        page_size=16,
        dtype=torch.float32,
        device="cpu"
    )
    
    # Example: 3 sequences with different lengths
    sequences = [
        {"seq_len": 20, "expected_blocks": 2},  # ceil(20/16) = 2
        {"seq_len": 35, "expected_blocks": 3},  # ceil(35/16) = 3
        {"seq_len": 10, "expected_blocks": 1},  # ceil(10/16) = 1
    ]
    
    print("Sequence configurations:")
    for i, seq in enumerate(sequences):
        print(f"  Sequence {i}: {seq['seq_len']} tokens → {seq['expected_blocks']} blocks")
    
    # Allocate blocks
    block_tables = []
    for i, seq in enumerate(sequences):
        blocks = block_manager.allocate_blocks(seq["seq_len"])
        block_tables.append(blocks)
        print(f"\nSequence {i} allocated blocks: {blocks}")
    
    # Convert to FlashInfer format
    seq_lengths = [seq["seq_len"] for seq in sequences]
    flashinfer_inputs = model_executor.prepare_flashinfer_inputs(
        block_tables, seq_lengths, is_prefill=True
    )
    
    print(f"\nFlashInfer input mapping:")
    print(f"  paged_kv_indices: {flashinfer_inputs['paged_kv_indices'].tolist()}")
    print(f"  paged_kv_indptr: {flashinfer_inputs['paged_kv_indptr'].tolist()}")
    print(f"  paged_kv_last_page_len: {flashinfer_inputs['paged_kv_last_page_len'].tolist()}")
    
    # Explain the mapping
    print(f"\nMapping explanation:")
    print(f"  - paged_kv_indices: Flattened block indices [0,1,2,3,4,5]")
    print(f"  - paged_kv_indptr: Cumulative block counts [0,2,5,6]")
    print(f"  - paged_kv_last_page_len: Remainder tokens in last blocks [4,3,10]")
    
    # Clean up
    for blocks in block_tables:
        block_manager.free_blocks(blocks)


if __name__ == "__main__":
    # Run the integration example
    simulate_llama_forward_with_flashinfer()
    
    # Run the KV cache mapping demonstration
    demonstrate_kv_cache_mapping()