# Trace 算子拆解分析 (64 seqs × 4-16 output tokens, Qwen3-0.6B)

Trace 来源：`torch.profiler.profile` 录制了完整的 64 seqs 生成过程（patched forward 使用 `no_grad` 替代 `inference_mode` 以暴露 CUDA kernel 名称）。共 20,691 个 GPU kernel，总 trace 时长 3,213ms。

## 总览

| 阶段 | 步数 | 每步 wall | 每步 GPU | GPU 利用率 |
|------|------|-----------|----------|-----------|
| Prefill | 28 层 (≈2100ms) | 75.1ms | 73.3ms | **~98%** |
| Decode | 15 步 | ~73ms (恒定) | 21ms → 7ms (递减) | **~27% → ~10%** |

## 1. Prefill — 单层分析 (46 kernels, 73.7ms GPU)

一次完整的 28 层 decoder forward，**每层 GPU 利用率 ~98%**（GPU 一直满载，没有 CPU launch 瓶颈）。

```
Operator                    GPU time   占比    说明
────────────────────────────────────────────────────────
matmul (QKV + attn_out + MLP ×3)  42.4ms   57%   CUTLASS GEMM (5 个: 4×256x128 + 1×128x256)
elementwise (mul/add)               9.8ms   13%   residual、SwiGLU 缩放
copy + fill                         7.0ms   10%   bfloat16 类型转换、fill
flashinfer(prefill_attn)            4.8ms    7%   BatchPrefillWithPagedKVCache
pow (x²)                            4.2ms    6%   RMS norm 的 float 精度平方
rms_norm (mean/rsqrt)               2.1ms    3%   float 精度的 mean + rsqrt
flashinfer(rotary)                  1.3ms    2%   RoPE 位置编码
silu                                1.3ms    2%   SwiGLU 激活函数
flashinfer(kv_write)                0.9ms    1%   AppendPagedKVCache
────────────────────────────────────────────────────────
总计:                             73.7ms

GPU 利用率: ~98%  ← 计算核心几乎完全满载
```

**结论：prefill 已经是 GPU 瓶颈，没有优化空间。**

> **修正说明**：原版报告将 matmul 记为 34.1ms (46%)，实际为 **42.4ms (57%)**。原因是每层有 5 个 matmul（QKV + attention 输出投影 + gate_proj + up_proj + down_proj），原版漏掉了 down_proj（128×256 CUTLASS kernel，8.3ms）。相应地，`vectorized` 分类 8.3ms 实为 4.2ms (pow) + 6.6ms (copy)，合 10.8ms，原版将其余 8.3ms 归类有误；`elementwise` 的 21.1ms 也调整为 16.8ms。

## 2. Decode — 全步分析 (15 步, 28 层/步, GPU 时间递减)

```
Operator                    GPU time   占比    每层平均
─────────────────────────────────────────────────────────
flashinfer(attention)         12.1ms   58%    ~430us
matmul(mlp+qkv)                5.4ms   26%    ~190us
eltwise (residual/add/mul)     1.8ms    9%     ~64us
rms_norm + reduce              0.8ms    4%     ~28us
flashinfer(rotary)             0.1ms    0%     ~3us
silu                           0.1ms    0%     ~3us
flashinfer(kv_write)           0.1ms    0%     ~3us
─────────────────────────────────────────────────────────
GPU 总计:                    20.8ms
Wall 总计:                   76.1ms
GPU 利用率:                    27%     ← CPU launch 瓶颈
```

Wall time 基本恒定在 ~73ms，**递减的是 GPU 时间**（随 batch 减小从 21ms 降到 7ms），GPU 利用率从 ~27% 递减到 ~10%。

> Trace 中共 420 个 decode attention kernel ÷ 28 层 = **15 步**（原版记为 ~21 步）。Wall 实际恒定 ~73ms（原版描述的递减趋势实际是 GPU 时间）。

## 3. 核心结论

| | Prefill | Decode |
|---|---|---|
| GPU 利用率 | **~98%** | **~27% → ~10%** |
| 瓶颈 | 计算 (compute-bound) | CPU launch (launch-bound) |
| 优化 | 无（已满） | CUDA graphs → 预期提升 ~3x |

Decode 阶段每个 kernel 的 input 很小（batch_size 个单 token），cuBLAS / FlashInfer 的 kernel launch 开销 > 实际计算时间。CUDA graphs 可以一次性录制 28 层 forward 然后 replay，完全消除 CPU launch 开销，预期 decode GPU 利用率从 27% → 80%+，整体吞吐提升 ~15-20%。随着 batch size 递减（请求逐步完成），GPU 时间从 21ms 降至 7ms，但 wall time 恒定 ~73ms，说明 CPU launch 是恒定的瓶颈。
