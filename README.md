<p align="right"><strong>English</strong> | <a href="README.zh.md">中文</a></p>

<h1 align="center">NanoServe 🚀</h1>

<div align="center">

High-performance LLM inference engine · Continuous Batching · CUDA Graph · FlashInfer · AWQ int4

![Python](https://img.shields.io/badge/python-≥3.10-blue)
![CUDA](https://img.shields.io/badge/CUDA-12.4-green)
![Torch](https://img.shields.io/badge/torch-≥2.4.0-orange)
![License](https://img.shields.io/badge/license-MIT-blue)
![GPU](https://img.shields.io/badge/RTX%203060%20%7C%204060%20Ti-passing-green)

**nanoserve CG (1491 tok/s) ≈ vLLM (1471 tok/s) · 2.1× over Eager · GPU utilization 41% → 94%**

</div>

---

## Performance

| Backend | Throughput | vs Eager | vs vLLM |
|---------|-----------|----------|---------|
| nanoserve **CG** (full-forward graph) | **1,491 tok/s** | **1.30×** | **1.01×** |
| vLLM (FlashAttn + CUDA Graph) | 1,471 tok/s | 1.28× | 1.0× |
| nanoserve eager (no compile) | 1,147 tok/s | 1.0× | 0.78× |

> **Testbed**: RTX 3060 (12 GB) · Qwen3-0.6B · 256 requests · 100–1024 input tokens · 100–1024 output tokens · 133,966 total tokens · temperature=0.6

| Metric | CG | Eager | Improvement |
|--------|----|-------|-------------|
| GPU utilization | **~94%** | 41% | **2.3×** |
| Kernel launches | 255,772 | 3,503,398 | **-92.7%** |
| Kernel launch CPU | 3.1s | 32.2s | **-90.4%** |
| Python scheduling overhead | ~3s | ~71s | **-95.8%** |
| End-to-end time | **89.9s** | 202.3s | **2.1×** |

---

## Architecture

```
                        ┌──────────────┐
                        │ LLMService   │
                        │  .generate() │
                        └──────┬───────┘
                               │
          ┌────────────────────┼────────────────────┐
          ▼                    ▼                    ▼
   ┌──────────────┐   ┌──────────────┐   ┌──────────────────┐
   │  Tokenizer   │   │  Scheduler   │   │  BlockManager    │
   │ (HF Auto)    │   │  continuous  │   │  KV cache pool   │
   └──────────────┘   │  batching    │   └──────────────────┘
                      └──────┬───────┘
                             │ schedule()
                             ▼
                     ┌──────────────────┐
                     │  ModelExecutor   │
                     │  .execute_batch() │
                     └──────┬───────────┘
                            │ forward()
                            ▼
                   ┌────────────────────┐
                   │   Qwen3Model       │
                   │   ┌─ 28× DecoderLayer
                   │   │  ├─ Qwen3Attention ── FlashInferBackend
                   │   │  ├─ Qwen3MLP (SwiGLU)
                   │   │  └─ RMSNorm
                   │   ├─ lm_head
                   │   ├─ embed_tokens
                   │   └─ final norm
                   └────────────────────┘
```

### Time Breakdown (nsys)

**Eager (202.3s)**
```
┌──────────────────────────────────────────────┐
│  GPU Kernel (83.8s)   ▓▓▓▓▓▓▓▓▓▓░░  41%     │
├──────────────────────────────────────────────┤
│  cudaLaunchKernel CPU (32.2s)  ▓▓▓▓▓░░  16%  │
├──────────────────────────────────────────────┤
│  cudaDeviceSync (12.0s)  ▓▓░░   6%            │
├──────────────────────────────────────────────┤
│  cudaMemcpyAsync (3.2s)  ░░   2%              │
├──────────────────────────────────────────────┤
│  Python scheduling + other (71.1s) ▓▓▓▓▓▓▓▓░░35%│
└──────────────────────────────────────────────┘
```

**CG (96.5s)**
```
┌──────────────────────────────────────────────┐
│  Graph Replay (71.2s)  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓  74%│
├──────────────────────────────────────────────┤
│  Prefill + Sampling GPU (19.3s)  ▓▓▓▓  20%   │
├──────────────────────────────────────────────┤
│  cudaLaunchKernel (3.1s)  ░   3%              │
├──────────────────────────────────────────────┤
│  Python scheduling (3.0s)  ░   3%             │
└──────────────────────────────────────────────┘
```

---

## Features

- **Continuous batching** — vLLM-style two-phase scheduler (prefill priority, decode preemption), budget preemption
- **Paged KV cache** — block-level memory management via FlashInfer paged attention kernels
- **CUDA Graph decode** — full-forward graph capture, one graph per batch size, eliminates per-layer CPU launch overhead
- **FlashInfer attention** — fused paged attention + RoPE + KV cache append
- **AWQ int4 quantization** — custom fused CUDA kernel, dequant on-the-fly during BF16 inference
- **SwiGLU MLP** — Qwen3-native gated feed-forward network
- **GQA support** — grouped-query attention with QK normalization (Qwen3-specific)
- **Auto memory profiling** — optimal KV cache sizing via dummy forward pass
- **Profiling tooling** — TTFT/ITL/throughput stats, nsys + torch.profiler integration

---

## Quick Start

```python
from core import LLMService, SamplingConfig

service = LLMService(model_path="./models/Qwen3-0.6B", max_num_seqs=8)

output = service.generate(
    "Hello, how are you?",
    SamplingConfig(temperature=0.6, top_p=0.9, max_new_tokens=128)
)
print(output)
```

**Download models:**
```bash
# HuggingFace (default)
python scripts/download_model.py 0.6b

# Direct download
hf download Qwen/Qwen3-0.6B --local-dir ./models/Qwen3-0.6B
```

**Quick start:**
```bash
python scripts/example_simple.py
```

**Run benchmarks:**
```bash
python scripts/bench.py                  # throughput benchmark (supports --backend vllm)
python scripts/bench.py --eager          # disable CUDA Graph for comparison
```

---

## Install

```bash
pip install torch safetensors transformers huggingface_hub flashinfer-python
```

Or use in-place (no pip install required):

```bash
git clone https://github.com/your-org/nanoserve
cd nanoserve
python scripts/download_model.py 0.6b
python scripts/example_simple.py
```

Python ≥ 3.10 required.

---

## Continuous Batching Scaling

**Burst mode (all requests arrive simultaneously):** RTX 3060 · Qwen3-0.6B · block_size=16 · 4533 blocks

| Requests | Wall (s) | Throughput (tok/s) | Avg TTFT (ms) | Avg ITL (ms) |
|----------|---------|-------------------|---------------|--------------|
| 16 | 10.76 | 385 | 55.3 | 38.6 |
| 32 | 11.62 | 714 | 51.3 | 39.9 |
| 64 | 13.03 | 1,273 | 54.1 | 41.7 |
| 128 | 13.52 | 2,454 | 62.5 | 39.0 |

> Wall time grows only 26% (16→128 req). ITL stays constant at 38-40ms — decode is the bottleneck and does not scale with concurrency.

**Staggered mode (wave arrival):**

| Requests | Burst (tok/s) | Staggered (tok/s) | Ratio |
|----------|--------------|-------------------|-------|
| 16 | 385 | 236 | 61% |
| 32 | 714 | 483 | 68% |
| 64 | 1,273 | 805 | 63% |
| 128 | 2,454 | 1,562 | 64% |

---

## CUDA Graph Deep Dive

### Why CG Wins

Each decode step runs 28 transformer layers. In eager mode, every layer launches multiple CUDA kernels (attention + QKV GEMM + MLP GEMM × 3 + elementwise + norm) — roughly **72 kernels/layer × 28 layers ≈ 2000 launches/step**. Each launch incurs CUDA driver serialization on the CPU.

CUDA Graph records the entire decode forward path into a single graph. At runtime, one `cuGraphLaunch` replays every kernel. This eliminates:

- **90%+ of kernel launch overhead** (3.5M → 256K launches)
- **Python per-layer scheduling** (71s → 3s)
- **GPU utilization jumps from 41% to 94%**

### Operator Breakdown

**Prefill single layer (GPU ~98% utilized):**
| Operator | GPU time | Share |
|----------|---------|-------|
| matmul (QKV + attn_out + MLP ×3) | 42.4ms | 57% |
| elementwise (mul/add) | 9.8ms | 13% |
| copy + fill | 7.0ms | 10% |
| flashinfer(attention) | 4.8ms | 7% |
| rms_norm (pow + mean/rsqrt) | 6.3ms | 9% |
| flashinfer(rotary + kv_write) | 2.2ms | 3% |
| silu | 1.3ms | 2% |

**Decode 15 steps aggregated (GPU utilization 27% → 10%):**
| Operator | GPU time | Share | Per-layer avg |
|----------|---------|-------|--------------|
| flashinfer(attention) | 12.1ms | 58% | ~430μs |
| matmul (mlp+qkv) | 5.4ms | 26% | ~190μs |
| eltwise (residual/add/mul) | 1.8ms | 9% | ~64μs |
| rms_norm | 0.8ms | 4% | ~28μs |
| flashinfer(rotary + kv_write + silu) | 0.3ms | 1% | ~9μs |

### Raw Comparison Data

| Metric | Eager | CG | Change |
|--------|-------|----|--------|
| Wall time | 202.3s | 96.5s | **-52%** |
| Total GPU kernel time | 83.8s | ~90.5s | +8% |
| cudaLaunchKernel calls | 3,503,398 | 255,772 | **-92.7%** |
| cudaLaunchKernel CPU time | 32.2s | 3.1s | **-90.4%** |
| cudaDeviceSynchronize | 12.0s | 71.2s | +493% |
| cudaMemcpyAsync | 315,557 calls / 3.2s | 173,327 calls / 1.3s | -59% |

---

## AWQ int4 Quantization

Supports inference with AWQ-quantized models. Weights stay in int4 on VRAM; the fused CUDA kernel dequantizes on-the-fly during forward.

**Implementation:** Custom CUDA kernel (`core/quantization/cuda/awq_kernel_naive.cu`), one thread per output element, inner product loop over input features. Each int32 word packs 8 int4 weights, indexed via `REORDER[8] = {0,4,1,5,2,6,3,7}` (matching AutoAWQ's nibble layout).

```
qweight shape: [in_features, out_features / 8]  (int32)
qzeros  shape: [num_groups,   out_features / 8]  (int32)
scales  shape: [num_groups,   out_features]       (float16)
```

**Memory savings:** ~4× vs BF16 weights (int4 vs 16-bit).

```bash
# Download Qwen3-1.7B-AWQ
python scripts/download_model.py 1.7b-awq
```

---

## Project Structure

```
nanoserve/
├── core/
│   ├── backends/           # FlashInferBackend, TorchBackend
│   ├── models/qwen3/       # Qwen3Model, Qwen3Attention, Qwen3MLP
│   ├── quantization/cuda/  # AWQ fused CUDA kernel + benchmarks
│   ├── llm_service.py      # entry point LLMService
│   ├── model_executor.py   # model executor + CUDA Graph capture
│   ├── scheduler.py        # continuous batching scheduler
│   ├── block_manager.py    # paged KV cache pool
│   ├── config.py           # configuration dataclasses
│   ├── layers_utils.py     # shared layers (RMSNorm, Embedding, Linear)
│   └── utils.py            # profiling utilities
├── scripts/                # download, benchmark, profiling scripts
├── docs/                   # technical documentation
│   ├── benchmark.md        # vLLM comparison benchmark
│   ├── cuda_graph.md       # CUDA Graph implementation notes
│   ├── eager_vs_cg_analysis.md  # nsys deep analysis
│   ├── flashinfer_validation.md  # FlashInfer validation
│   └── nsys_analysis.md    # Nsight Systems analysis commands
└── test/                   # end-to-end tests
```

---

## Model Support

| Model | HF Repo | Precision | Size |
|-------|---------|-----------|------|
| Qwen3-0.6B | `Qwen/Qwen3-0.6B` | BF16 | ~1.2 GB |
| Qwen3-1.7B | `Qwen/Qwen3-1.7B` | BF16 | ~3.4 GB |
| Qwen3-1.7B-AWQ | `Orion-zhen/Qwen3-1.7B-AWQ` | INT4 | ~0.9 GB |

### Adding a New Model

1. Create `core/models/<model>/attention.py`, `mlp.py`, `model.py`
2. Add a branch in `ModelExecutor.__init__()`
3. Implement a new `AttentionBackend` if needed

---

## Documentation

| Doc | Description |
|-----|-------------|
| `docs/benchmark.md` | nanoserve vs vLLM benchmark comparison |
| `docs/cuda_graph.md` | CUDA Graph implementation details & bugs |
| `docs/eager_vs_cg_analysis.md` | nsys deep analysis report |
| `docs/flashinfer_validation.md` | FlashInfer numerical validation |
| `docs/nsys_analysis.md` | Nsight Systems command reference |
| `docs/test_report.md` | Continuous batching test report |

---

## Dependencies

- Python ≥ 3.10
- PyTorch ≥ 2.4.0 (CUDA 12.4 recommended)
- FlashInfer ≥ 0.6.3
- safetensors / transformers / huggingface_hub
- modelscope (optional, for China mirror)

---

## License

MIT

[中文版](README.zh.md)
