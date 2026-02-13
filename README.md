# Llama + FlashInfer Integration

This project implements a high-performance integration between HuggingFace Llama models and FlashInfer operators, featuring a custom BlockManager for efficient KV cache management.

## Components

### 1. BlockManager (`block_manager.py`)
- Manages physical blocks for KV cache allocation
- Pre-allocates large tensor pool to avoid runtime allocation
- Thread-safe block allocation and deallocation
- Optimized for high-performance inference

### 2. ModelExecutor (`model_executor.py`)
- Integrates FlashInfer operators with HuggingFace models
- Handles KV cache mapping and attention computation
- Supports both prefill and decode phases
- Manages FlashInfer wrapper initialization

### 3. Key Features
- **Zero-copy KV cache access**: Direct tensor views without memory copying
- **Efficient block management**: O(1) allocation/deallocation using deque
- **Thread-safe**: Concurrent access support with locking
- **FlashInfer integration**: Paged KV cache for optimal memory usage
- **Single layer support**: Focus on core functionality

## Installation

### Prerequisites
```bash
pip install torch numpy
```

### FlashInfer Installation (Optional)
```bash
pip install flashinfer
```

Note: If FlashInfer is not installed, the code will use mock implementations for testing.

## Quick Start

```python
from block_manager import BlockManager
from model_executor import ModelExecutor

# 1. Create BlockManager
block_manager = BlockManager(
    num_blocks=100,
    num_layers=32,
    num_heads=32,
    head_dim=128,
    block_size=16,
    dtype=torch.float16,
    device="cuda"
)

# 2. Create ModelExecutor
model_executor = ModelExecutor(
    block_manager=block_manager,
    num_heads=32,
    head_dim=128,
    page_size=16,
    dtype=torch.float16,
    device="cuda"
)

# 3. Allocate blocks for your sequence
seq_length = 50
block_indices = block_manager.allocate_blocks(seq_length)

# 4. Execute model with FlashInfer attention
output = model_executor.execute_model(
    input_ids=input_tensor,
    block_tables=[block_indices],
    seq_lengths=[seq_length],
    layer_idx=0,
    is_prefill=True
)
```

## Usage Examples

### Basic Block Allocation
```python
# Allocate blocks for 100 tokens with 16-token block size
blocks = block_manager.allocate_blocks(100)
print(f"Allocated {len(blocks)} blocks: {blocks}")

# Free blocks when done
block_manager.free_blocks(blocks)
```

### FlashInfer KV Cache Mapping
```python
# Convert block tables to FlashInfer format
flashinfer_inputs = model_executor.prepare_flashinfer_inputs(
    block_tables=[[0, 1, 2], [3, 4]],
    seq_lengths=[40, 25],
    is_prefill=True
)
```

### Attention Computation
```python
# Compute attention using FlashInfer
attention_output = model_executor.compute_attention_with_flashinfer(
    query=query_tensor,
    key_cache=key_cache_tensor,
    value_cache=value_cache_tensor,
    block_tables=block_tables,
    seq_lengths=seq_lengths,
    is_prefill=True
)
```

## Testing

Run the unit tests:
```bash
python test_block_manager.py
python test_model_executor.py
```

Run the integration example:
```bash
python integration_example.py
```

## Architecture

### BlockManager Design
- Pre-allocates KV cache pool: `(num_blocks, num_layers, 2, num_heads, head_dim, block_size)`
- Uses deque for O(1) block allocation/deallocation
- Thread-safe operations with locking
- Tracks allocated vs free blocks

### ModelExecutor Design
- Initializes FlashInfer decode and prefill wrappers
- Maps logical block tables to FlashInfer format:
  - `paged_kv_indices`: Flattened block indices
  - `paged_kv_indptr`: Cumulative block counts
  - `paged_kv_last_page_len`: Remainder tokens in last blocks
- Handles KV cache tensor extraction and reshaping
- Provides unified interface for attention computation

## Performance Considerations

1. **Memory Efficiency**: Paged KV cache reduces memory fragmentation
2. **Compute Efficiency**: FlashInfer operators optimized for GPU execution
3. **Scalability**: Block-based allocation supports long sequences
4. **Concurrency**: Thread-safe operations enable batch processing

## Limitations

- Currently supports single layer execution
- Mock FlashInfer implementation when not installed
- CPU-only testing (GPU requires CUDA setup)
- Simplified attention computation for demonstration

## Future Enhancements

- Multi-layer execution support
- Distributed inference capabilities
- Advanced memory management strategies
- Integration with actual HuggingFace Llama models
- Performance benchmarking and optimization

## Contributing

This is a demonstration project for integrating BlockManager with FlashInfer. Feel free to extend it for production use cases.