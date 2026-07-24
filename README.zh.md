<p align="right"><a href="README.md">English</a> | <strong>中文</strong></p>

<h1 align="center">NanoServe 🚀</h1>

<div align="center">

高性能 LLM 推理引擎 · 连续批处理 · CUDA Graph · FlashInfer · AWQ int4

![Python](https://img.shields.io/badge/python-≥3.10-blue)
![CUDA](https://img.shields.io/badge/CUDA-12.4-green)
![Torch](https://img.shields.io/badge/torch-≥2.4.0-orange)
![License](https://img.shields.io/badge/license-MIT-blue)
![GPU](https://img.shields.io/badge/RTX%203060%20%7C%204060%20Ti-passing-green)

**nanoserve CG (1491 tok/s) ≈ vLLM (1471 tok/s) ·  CG 相比 Eager 加速 2.1× · GPU 利用率 41% → 94%**

</div>

---

## 性能速览

| 后端 | 吞吐量 | 相对 Eager | 相对 vLLM |
|------|--------|-----------|----------|
| nanoserve **CG**（全图捕获） | **1,491 tok/s** | **1.30×** | **1.01×** |
| vLLM（FlashAttn + CUDA Graph） | 1,471 tok/s | 1.28× | 1.0× |
| nanoserve eager（无编译） | 1,147 tok/s | 1.0× | 0.78× |

> **测试条件**：RTX 3060 (12 GB) · Qwen3-0.6B · 256 请求 · 100–1024 input tokens · 100–1024 output tokens · 共计 133,966 tokens · temperature=0.6

| 指标 | nanoserve CG | nanoserve Eager | 提升 |
|------|-------------|----------------|------|
| GPU 利用率 | **~94%** | 41% | **2.3×** |
| Kernel launch 次数 | 255,772 | 3,503,398 | **-92.7%** |
| Kernel launch CPU 耗时 | 3.1s | 32.2s | **-90.4%** |
| Python 调度开销 | ~3s | ~71s | **-95.8%** |
| 端到端耗时 | **89.9s** | 202.3s | **2.1×** |

---

## 架构概览

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
   │ (HF Auto)    │   │  连续批处理    │   │  KV 缓存池分配     │
   └──────────────┘   └──────┬───────┘   └──────────────────┘
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

### CUDA Graph 时间分解

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
│  Python 调度 + 其他 (71.1s)  ▓▓▓▓▓▓▓▓░░  35%  │
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
│  Python 调度 (3.0s)  ░   3%                   │
└──────────────────────────────────────────────┘
```

---

## 特性

- **连续批调度** — vLLM 风格的两阶段调度（prefill 优先，decode 可被抢占），budget 抢占策略
- **Paged KV 缓存** — 通过 FlashInfer 的分页注意力内核实现类 vLLM 的块级内存管理
- **CUDA Graph 解码** — 全前向 CUDA Graph 捕获，每批大小一个图，消除逐层 CPU launch 开销
- **FlashInfer 注意力后端** — 融合的 paged attention + RoPE + KV 缓存追加
- **AWQ int4 量化** — 自定义融合 CUDA 内核，BF16 推理时动态反量化，单线程每输出元素逐点积
- **SwiGLU MLP** — Qwen3 原生门控前馈网络
- **GQA 支持** — 分组查询注意力，附带 QK 归一化（Qwen3 特定）
- **自动内存分析** — 通过虚拟前向传递自动确定最佳 KV 缓存大小
- **性能分析工具** — TTFT/ITL/吞吐量统计，nsys/torch.profiler 集成

---

## 快速开始

```python
from core import LLMService, SamplingConfig

service = LLMService(model_path="./models/Qwen3-0.6B", max_num_seqs=8)

output = service.generate(
    "Hello, how are you?",
    SamplingConfig(temperature=0.6, top_p=0.9, max_new_tokens=128)
)
print(output)
```

**下载模型：**
```bash
# 国内（hf-mirror.com，默认已配好）
python scripts/download_model.py 0.6b

# 官方 HuggingFace Hub
hf download Qwen/Qwen3-0.6B --local-dir ./models/Qwen3-0.6B
```

**运行基准测试：**
```bash
python scripts/testbench.py              # 连续批处理测试（burst vs staggered）
python scripts/bench.py                  # 吞吐量基准（支持 --backend vllm 对比）
python scripts/bench.py --eager           # 禁用 CUDA Graph 对比
python scripts/profile_trace.py           # torch.profiler Chrome trace
```

---

## 安装

```bash
pip install torch safetensors transformers huggingface_hub flashinfer-python modelscope
```

或者本地使用（无需 pip install）：

```bash
git clone https://github.com/your-org/nanoserve
cd nanoserve
python scripts/download_model.py 0.6b
python scripts/example_simple.py
```

支持的 Python 版本：≥ 3.10

---

## 连续批处理扩展性

**Burst 模式（所有请求同时到达）：** RTX 3060 · Qwen3-0.6B · block_size=16 · 4533 blocks

| 请求数 | 耗时 (s) | 吞吐 (tok/s) | TTFT 平均 (ms) | ITL 平均 (ms) |
|--------|---------|-------------|----------------|---------------|
| 16 | 10.76 | 385 | 55.3 | 38.6 |
| 32 | 11.62 | 714 | 51.3 | 39.9 |
| 64 | 13.03 | 1,273 | 54.1 | 41.7 |
| 128 | 13.52 | 2,454 | 62.5 | 39.0 |

> Wall time 增长仅 26%（16→128 请求），ITL 恒定在 38-40ms，说明 decode 是系统瓶颈且瓶颈不随并发数变化。

**Staggered 模式（波浪式到达）：**

| 请求数 | Burst (tok/s) | Staggered (tok/s) | 比例 |
|--------|--------------|-------------------|------|
| 16 | 385 | 236 | 61% |
| 32 | 714 | 483 | 68% |
| 64 | 1,273 | 805 | 63% |
| 128 | 2,454 | 1,562 | 64% |

---

## CUDA Graph 深入分析

### 为什么 CG 快？

解码阶段（decode）每个 token 需要跑 28 层 transformer。在 eager 模式下，每层都要发起多次 CUDA kernel launch（attention + QKV GEMM + MLP GEMM × 3 + elementwise + norm），共约 72 次 / 层 × 28 层 ≈ **2000 次 launch / 步**。每次 launch 都需要 CPU 端 CUDA driver 序列化，产生可观的调度延迟。

CUDA Graph 将整个解码前向路径录制到一个图（graph）中，运行时只需一次 `cuGraphLaunch` 即可回放全部 kernel。这消除了：

- **90%+ 的 kernel launch 开销**（3,503,398 → 255,772 次）
- **Python 逐层调度开销**（71s → 3s）
- **GPU 利用率从 41% → 94%**

### 算子耗时分解

**Prefill 单层（GPU ~98% 利用率）：**
| 算子 | GPU 时间 | 占比 |
|------|---------|------|
| matmul (QKV + attn_out + MLP ×3) | 42.4ms | 57% |
| elementwise (mul/add) | 9.8ms | 13% |
| copy + fill | 7.0ms | 10% |
| flashinfer(attention) | 4.8ms | 7% |
| rms_norm (pow + mean/rsqrt) | 6.3ms | 9% |
| flashinfer(rotary + kv_write) | 2.2ms | 3% |
| silu | 1.3ms | 2% |

**Decode 十五步汇总（GPU 利用率 27% → 10%）：**
| 算子 | GPU 时间 | 占比 | 每层平均 |
|------|---------|------|---------|
| flashinfer(attention) | 12.1ms | 58% | ~430μs |
| matmul (mlp+qkv) | 5.4ms | 26% | ~190μs |
| eltwise (residual/add/mul) | 1.8ms | 9% | ~64μs |
| rms_norm | 0.8ms | 4% | ~28μs |
| flashinfer(rotary + kv_write + silu) | 0.3ms | 1% | ~9μs |

### 原始对比数据

| 指标 | Eager | CG | 变化 |
|------|-------|----|------|
| Wall time | 202.3s | 96.5s | **-52%** |
| 总 GPU kernel 时间 | 83.8s | ~90.5s | +8% |
| cudaLaunchKernel 次数 | 3,503,398 | 255,772 | **-92.7%** |
| cudaLaunchKernel CPU 耗时 | 32.2s | 3.1s | **-90.4%** |
| cudaDeviceSynchronize | 12.0s | 71.2s | +493% |
| cudaMemcpyAsync | 315,557 次 / 3.2s | 173,327 次 / 1.3s | -59% |

---

## AWQ int4 量化

支持 AWQ 量化模型的推理，权重以 int4 格式存储在显存中，前向时通过融合 CUDA kernel 动态反量化。

**核心实现：** 自定义 CUDA kernel（`core/quantization/cuda/awq_kernel_naive.cu`），单线程每输出元素，逐点积循环遍历所有输入特征。每个 int32 字打包 8 个 int4 权重，通过 `REORDER[8] = {0,4,1,5,2,6,3,7}` 索引（与 AutoAWQ 的 nibble 布局一致）。

```
qweight shape: [in_features, out_features / 8]  (int32)
qzeros  shape: [num_groups,   out_features / 8]  (int32)
scales  shape: [num_groups,   out_features]       (float16)
```

**内存节省：** 相比 BF16 权重节省约 4×（int4 vs 16-bit），适合显存受限的推理场景。

**AWQ 模型支持：**
```bash
# 下载 Qwen3-1.7B-AWQ
python scripts/download_model.py 1.7b-awq
```

---

## 项目结构

```
nanoserve/
├── core/
│   ├── backends/           # FlashInferBackend, TorchBackend
│   ├── models/qwen3/       # Qwen3Model, Qwen3Attention, Qwen3MLP
│   ├── quantization/cuda/  # AWQ 融合 CUDA kernel + 基准测试
│   ├── llm_service.py      # 主入口 LLMService
│   ├── model_executor.py   # 模型执行器 + CUDA Graph 捕获
│   ├── scheduler.py        # 连续批调度器
│   ├── block_manager.py    # Paged KV 缓存池
│   ├── config.py           # 配置数据类
│   ├── layers_utils.py     # 通用层（RMSNorm, Embedding, Linear）
│   └── utils.py            # 性能分析工具
├── scripts/                # 下载、基准测试、性能分析脚本
├── docs/                   # 技术文档
│   ├── benchmark.md        # vLLM 对比基准
│   ├── cuda_graph.md       # CUDA Graph 实现笔记
│   ├── eager_vs_cg_analysis.md  # nsys 深度分析
│   ├── flashinfer_validation.md  # FlashInfer 验证
│   └── nsys_analysis.md    # Nsight Systems 分析命令
└── test/                   # 端到端测试
```

---

## 模型支持

| 模型 | HF 仓库 | 精度 | 大小 |
|------|---------|------|------|
| Qwen3-0.6B | `Qwen/Qwen3-0.6B` | BF16 | ~1.2 GB |
| Qwen3-1.7B | `Qwen/Qwen3-1.7B` | BF16 | ~3.4 GB |
| Qwen3-1.7B-AWQ | `Orion-zhen/Qwen3-1.7B-AWQ` | INT4 | ~0.9 GB |

### 添加新模型

1. 在 `core/models/<model>/` 中创建 `attention.py`、`mlp.py`、`model.py`
2. 在 `ModelExecutor.__init__()` 中添加分支
3. 实现新的 `AttentionBackend`（如果需要）

---

## 技术文档

| 文档 | 内容 |
|------|------|
| `docs/benchmark.md` | nanoserve vs vLLM 性能对比 |
| `docs/cuda_graph.md` | CUDA Graph 实现细节与踩坑记录 |
| `docs/eager_vs_cg_analysis.md` | nsys 深度分析报告（完整数据） |
| `docs/flashinfer_validation.md` | FlashInfer 数值验证 |
| `docs/nsys_analysis.md` | Nsight Systems 分析命令速查 |
| `docs/test_report.md` | 连续批处理测试报告 |

---

## 依赖

- Python ≥ 3.10
- PyTorch ≥ 2.4.0（CUDA 12.4 推荐）
- FlashInfer ≥ 0.6.3
- safetensors / transformers / huggingface_hub
- modelscope（可选，国内镜像下载）

---

## License

MIT
