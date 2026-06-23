# Eager vs CUDA Graph: nsys 分析报告

## 测试条件

| 项目 | 值 |
|---|---|
| GPU | RTX 3060 (12 GB) |
| 模型 | Qwen3-0.6B (28 layers, 8 KV heads, bf16) |
| Batch | 256 请求，prompt 100-1024 tok，output 100-1024 tok |
| 总生成 tokens | 133,966 |
| nsys 范围 | 仅 generate 阶段（`cudaProfilerStart/Stop`），不含 init/CG capture |

> 对比参考：相同条件下 vLLM V1（torch.compile + CUDA Graph + Flash Attention）吞吐为 **955.65 tok/s**，
> nanoserve CG（1380.59 tok/s）比 vLLM 快 **44%**，详见 [cuda_graph.md](cuda_graph.md#性能对比总表)。

---

## 1. 宏观对比

| 指标 | Eager | CG | 变化 |
|---|---|---|---|
| Wall time | 202.3s | 96.5s | **-52%** |
| 总 GPU kernel 时间（nsys 可见） | **83.8s** | **19.3s** | -77% |
| CUDA API CPU 时间 | 50.2s | 78.0s | +55% |
| 其中 `cudaDeviceSynchronize` | 12.0s | **71.2s** | +493% |
| GPU 利用率 (kernel/wall) | 41% | ~93%¹ | |

¹ CG 中 `cudaDeviceSynchronize` 的 71.2s 实际是 **graph replay 的 GPU 执行时间**（nsys 将其归为 sync API 时间）。加上可见 kernel 19.3s = ~90.5s GPU 时间，占 wall 的 ~94%。

**核心结论：CG 将 GPU 利用率从 41% 提升到 ~94%，吞吐翻倍。**

---

## 2. Kernel Launch 开销对比

这是 CG 加速的核心来源：

| 指标 | Eager | CG | 降幅 |
|---|---|---|---|
| `cudaLaunchKernel` 调用次数 | **3,503,398** | **255,772** | **-92.7%** |
| `cudaLaunchKernel` CPU 耗时 | **32.2s** | **3.1s** | **-90.4%** |
| 唯一 kernel 种类 | 17 种 | 12 种 | -29% |

Eager 每次 decode 步骤 × 28 层都需要 launch 每个 kernel（attention + MLP GEMM + elementwise + norm）。CG 将解码路径的全部 kernel 合并到一次 `cuGraphLaunch`，绕过了 CUDA driver 的 launch 序列化开销。

### 2.1 最耗时 kernel (Eager)

| Rank | Kernel | 总时间 | 次数 | 说明 |
|---|---|---|---|---|
| 1 | `flashinfer::BatchDecodeWithPagedKVCacheKernel` | **41.9s (50.0%)** | 72,408 | 每层 × 每步的 decode attention |
| 2 | `cutlass relu 256x128` (gate_proj + up_proj) | 4.5s (5.4%) | 8,851 | MLP 第一个 GEMM |
| 3 | `ampere_bf16_s16816gemm 128x64` (多种变体) | 各 2.7-4.4s | 共 ~262K | MLP GEMM 的各种 tile 配置 |
| 4 | `BatchPrefillWithPagedKVCacheKernel` | 1.3s (1.5%) | 4,648 | prefill attention |

**Eager 中 flashinfer decode attention kernel 占了 GPU 时间的一半**（41.9s），在 CG 中这个 kernel 完全消失——被捕获到 CUDA Graph 内部，不再单独可见。

### 2.2 最耗时 kernel (CG)

| Rank | Kernel | 总时间 | 次数 |
|---|---|---|---|
| 1 | `cutlass relu 256x128` (MLP) | 4.2s (21.7%) | 8,625 |
| 2 | `ampere_bf16_s16816gemm 256x128` (MLP) | 2.5s (12.9%) | 4,984 |
| 3 | `BatchPrefillWithPagedKVCacheKernel` | 1.3s (6.8%) | 4,648 |
| 4 | `cutlass relu 128x256` (MLP large) | 1.1s (5.6%) | 1,260 |

CG 顶部的 kernel 全部是 **prefill + 采样相关**（以及 CG 内部不可见的 decode 占大部分时间）。较慢的原因：prefill 的 batch size 随时间变化（5→数百 tokens），导致 GEMM 效率不如固定 size 的 decode。

---

## 3. 时间分解对比

### Eager (202.3s)

```
┌──────────────────────────────────────────────┐
│  GPU Kernel (83.8s)   41%                     │
├──────────────────────────────────────────────┤
│  cudaLaunchKernel CPU (32.2s)  16%            │  ← 每层每步 launch
├──────────────────────────────────────────────┤
│  cudaDeviceSync (12.0s)  6%                   │
├──────────────────────────────────────────────┤
│  cudaMemcpyAsync (3.2s)  2%                   │
├──────────────────────────────────────────────┤
│  Python 调度 + 其他 (71.1s)  35%              │  ← 28 层 for 循环
└──────────────────────────────────────────────┘
```

### CG (96.5s)

```
┌──────────────────────────────────────────────┐
│  Graph Replay (cudaDeviceSync 71.2s)  74%    │  ← 一次 cuGraphLaunch
├──────────────────────────────────────────────┤
│  Prefill + Sampling GPU (19.3s)  20%          │
├──────────────────────────────────────────────┤
│  cudaLaunchKernel (3.1s)  3%                  │
├──────────────────────────────────────────────┤
│  Python 调度 (3.0s)  3%                       │
└──────────────────────────────────────────────┘
```

---

## 4. 关键结论

### 4.1 CG 为什么快
1. **消除 3.25M 次 kernel launch** — 每 decode 步 28 层 × 每个 tensor 一个 launch，CG 合并为 1 次 cuGraphLaunch
2. **消除 Python 调度开销** — 28 层 for 循环 + attention backend plan + 张量创建，CG 全部在 GPU 上执行
3. **提升 GPU 利用率** — 从 41% 到 ~94%，GPU 几乎一直在干活

### 4.2 瓶颈转移到何处
CG 之后瓶颈转移到两个地方：
1. **Prefill 阶段** — 占总 GPU 时间 20%，不如 decode 高效，因为 batch size 和 sequence length 动态变化
2. **`cudaDeviceSynchronize`** — CG 重放需要通过 sync 来保证所有操作完成，这是 cuGraph 的固有特性

### 4.3 进一步优化方向

| 方向 | 预期收益 | 说明 |
|---|---|---|
| Prefill CUDA Graph（chunked prefill） | +10-15% | 当前 prefill 仍是 eager，可用同样方式 capture |
| 减少 `copy_()` 次数 | +3-5% | 当前每步 3 次 copy（input_ids, positions, page_table），可合并为 1 次 |
| 采样也用 CUDA Graph | +1-2% | `softmax` + `multinomial` 可以 capture |
| 异步 graph replay + scheduler | +5-10% | 当前 graph replay 是同步的（cudaDeviceSynchronize），可与其他步骤重叠 |

---

## 5. 原始数据

### Eager

| 指标 | 值 |
|---|---|
| Wall time | 202.3s |
| 总 GPU kernel 时间 | 83,780 ms |
| cudaLaunchKernel 调用 | 3,503,398 次 / 32,185 ms |
| cudaDeviceSynchronize | 11,009 次 / 11,967 ms |
| cudaMemcpyAsync | 315,557 次 / 3,199 ms |
| cudaStreamSynchronize | 155,405 次 / 1,220 ms |

### CG

| 指标 | 值 |
|---|---|
| Wall time | 96.5s |
| 总 GPU kernel 时间（可见） | 19,328 ms |
| cudaLaunchKernel 调用 | 255,772 次 / 3,067 ms |
| cudaDeviceSynchronize | 11,009 次 / 71,207 ms |
| cudaMemcpyAsync | 173,327 次 / 1,309 ms |
| cudaStreamSynchronize | 150,230 次 / 1,134 ms |

### 加速汇总

| 阶段 | Eager | CG | 加速比 |
|---|---|---|---|
| 端到端 | 202.3s | 96.5s | 2.1× |
| GPU kernel 执行 | 83.8s | ~90.5s¹ | 0.93× |
| Kernel launch (CPU) | 32.2s | 3.1s | 10.4× |
| Python 调度 | ~71.1s | ~3s | 23.7× |

¹ CG 的 GPU 执行时间 = 可见 kernel 19.3s + graph replay 内隐式执行 ≈ 71.2s（来自 `cudaDeviceSynchronize` 耗时）。
