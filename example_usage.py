"""
Example usage of BlockManager for KV Cache allocation.
"""

import torch
from block_manager import BlockManager


def main():
    """Demonstrate BlockManager usage."""
    
    # Configuration parameters
    num_blocks = 100
    num_layers = 32
    num_heads = 32
    head_dim = 128
    block_size = 64
    
    print("Creating BlockManager...")
    block_manager = BlockManager(
        num_blocks=num_blocks,
        num_layers=num_layers,
        num_heads=num_heads,
        head_dim=head_dim,
        block_size=block_size,
        dtype=torch.float16,
        device="cpu"  # Use CPU for demo, typically would be "cuda"
    )
    
    print(f"BlockManager created: {block_manager}")
    print(f"KV Cache pool shape: {block_manager.kv_cache_pool.shape}")
    print(f"Total memory usage: {block_manager.kv_cache_pool.numel() * 2 / 1024**2:.2f} MB")
    
    # Simulate different request sizes
    requests = [50, 100, 75, 200, 30]
    
    for i, num_tokens in enumerate(requests):
        print(f"\n--- Request {i+1}: {num_tokens} tokens ---")
        
        # Allocate blocks
        blocks = block_manager.allocate_blocks(num_tokens)
        
        if blocks is None:
            print(f"  Failed to allocate blocks for {num_tokens} tokens")
            continue
        
        num_blocks_allocated = len(blocks)
        print(f"  Allocated {num_blocks_allocated} blocks: {blocks}")
        print(f"  Free blocks: {block_manager.num_free_blocks}")
        print(f"  Allocated blocks: {block_manager.num_allocated_blocks}")
        
        # Simulate using the blocks (get KV cache tensors)
        kv_cache = block_manager.get_kv_cache_for_blocks(blocks)
        print(f"  KV cache shape: {kv_cache.shape}")
        
        # Simulate computation...
        # In real usage, you would write to these tensors
        
        # Free the blocks
        block_manager.free_blocks(blocks)
        print(f"  Freed blocks. Free blocks: {block_manager.num_free_blocks}")
    
    print(f"\nFinal state: {block_manager}")


if __name__ == "__main__":
    main()