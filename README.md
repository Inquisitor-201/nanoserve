# nanoserve

High-performance LLM inference engine with continuous batching and pluggable attention backends. Currently supports **Qwen3** models with FlashInfer-accelerated paged attention.

## Architecture

```
EngineArgs → LLMService.from_engine_args()
  ├─ ModelConfig        # reads HuggingFace config.json
  ├─ CacheConfig        # auto-calculates KV cache block count
  └─ SchedulerConfig

LLMService
  ├─ Tokenizer
  ├─ BlockManager       # pre-allocates KV cache GPU pool
  ├─ ModelExecutor      # instantiates model + loads weights
  └─ Scheduler           # continuous batching (prefill > decode)
```

**Inference loop:** `LLMService.generate()` → tokenizer → `Scheduler.schedule()` → `ModelExecutor.execute_batch()` → `Qwen3Model.forward()` → sample.

### Attention backends

| Backend | Use case |
|---------|----------|
| `FlashInferBackend` | GPU production — paged KV cache via FlashInfer |
| `TorchBackend` | CPU / reference — mirrors FlashInfer API |

## Quick Start

```python
from core import LLMService, EngineArgs

args = EngineArgs(
    model_path="/path/to/qwen3",
    max_num_seqs=8,
    max_model_len=4096,
)
service = LLMService.from_engine_args(args)

# Single generation
output = service.generate("Hello, how are you?")
print(output)

# Streaming chatbot
for token in service.generate("Tell me a story", stream=True):
    print(token, end="", flush=True)
```

## Install

```bash
pip install torch safetensors transformers huggingface_hub flashinfer-python
```

Or use the project in-place (no pip install needed):

```bash
git clone https://github.com/your-org/nanoserve
cd nanoserve
python scripts/download_model.py
python scripts/testbench.py    # requires CUDA + model weights
```

## Project Structure

```
core/
  backends/       # FlashInferBackend, TorchBackend, AttentionMetadata
  models/         # model implementations (qwen3/)
  layers_utils.py # generic layers (RMSNorm, Embedding, Linear)
  llm_service.py  # entry point
  scheduler.py    # continuous batching
  block_manager.py
  model_executor.py
  config.py       # frozen dataclasses
  quantization/   # AWQ support
chatbot/          # web UI demo
scripts/          # download_model.py, testbench.py, example_simple.py
```

## Adding a New Model

1. Create `core/models/<model>/attention.py` and `mlp.py`
2. Add model class in `core/models/<model>/model.py`
3. Add the model branch in `ModelExecutor.__init__()`

## License

MIT
