# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Run all unit tests (from project root)
python -m pytest test/

# Run a specific test file
python -m pytest test/test_gqa.py -v

# Run a specific test
python -m pytest test/test_gqa.py::test_gqa_attention -v

# Run tests with print output visible
python -m pytest test/ -s

# Performance testbench (requires CUDA + model weights)
python testbench.py

# Start chatbot web server (requires model weights)
cd chatbot && python app.py

# Download model
python download_model.py

# Type check (no runner configured, use mypy directly if available)
# No linter/formatter is configured in the project
```

## Architecture

### High-level pipeline

```
EngineArgs (user input)
  └─ LLMService.from_engine_args()           # config.py
       ├─ ModelConfig.from_hf()               # reads HuggingFace config.json
       ├─ CacheConfig + auto_calculate_num_blocks()
       └─ SchedulerConfig
             │
    LLMService.__init__()                    # llm_service.py
      ├─ AutoTokenizer
      ├─ BlockManager                         # pre-allocates KV cache GPU pool
      ├─ ModelExecutor                        # instantiates model + loads weights
      └─ Scheduler                            # continuous batching
```

### Processing flow

```
LLMService.generate()
  └─ tokenizer → add_requests()
       └─ Scheduler.schedule()               # continuous batching
            ├─ _schedule_prefill()            # new requests, block allocation
            └─ _schedule_decode()             # decode step with preemption logic
                 │
       ModelExecutor.execute_batch()          # model_specific/qwen3/
         ├─ AttentionMetadata.from_block_tables()
         ├─ Qwen3Model.forward()
         │    ├─ embed_tokens()
         │    ├─ for each Qwen3DecoderLayer:
         │    │    ├─ Qwen3Attention            # QKV proj → QK norm → RoPE → backend
         │    │    └─ Qwen3MLP                  # SwiGLU
         │    └─ norm → lm_head
         └─ sample()                          # top-p sampling or greedy
```

### Key design decisions

- **BlockManager** owns the KV cache GPU tensor pool; BlockManager only tracks free/allocated indices. Data movement into the pool is done by FlashInferBackend via `flashinfer.append_paged_kv_cache`.
- **Attention backends are pluggable**: FlashInferBackend (GPU, production) and TorchBackend (CPU, reference for correctness). Both accept the same `AttentionMetadata`.
- **Continuous batching** via Scheduler: prefill has priority; decode phase can preempt the youngest request to free blocks and prevent OOM.
- **Decoupled model layers**: `layers_utils.py` holds generic components (RMSNorm, Embedding, Linear, GELU). Model-specific code (Qwen3Attention, Qwen3MLP) lives in `model_specific/<model_name>/`. Adding a new model means adding a new `model_specific/` directory and a new class in `models/`.
- **All configs are frozen dataclasses** (`ModelConfig`, `CacheConfig`, `SchedulerConfig`, `SamplingConfig`, `EngineArgs`), passed down from `LLMService` to child components.
- **Profiling**: `ContinuousBatchTimer` and `StatsCollector` track TTFT and inter-token latencies per-request through the scheduler's metrics objects.

### Current model support

Only **Qwen3** is implemented. Adding a new architecture (e.g., Llama) requires:
1. Create `core/model_specific/<model>/attention.py` and `mlp.py`
2. Add model class in `core/models/<model>.py`
3. Add the model branch in `ModelExecutor.__init__()`
