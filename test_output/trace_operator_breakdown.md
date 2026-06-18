# Trace 算子拆解分析 (64 seqs × 4-16 output tokens, Qwen3-0.6B)

## 一个 Prefill Step

取自 burst 42（1314 kernels, 68.8ms wall time, 19.5ms GPU time）。28 层 decoder layer 全跑。

```
Operator           GPU时间   占比    每层耗时        备注
─────────────────────────────────────────────────────────────
attention          11.0ms  56.7%    ~400us    FlashInfer BatchDecodeWithPagedKVCache
                     (注意：profile 时用的 patched forward 没有 inference_mode,
                      所以实际 prefill 里的 attention 算子名显示为 decode_attn,
                      但对应的是 prefill 操作——因为 profile 使用了 no_grad 替代
                      inference_mode, 导致 FlashInfer 内部 kernel 选择逻辑变化)

matmul(mlp)         5.1ms  26.3%    ~180us    cuBLAS gate/up/down + splitKreduce
                     (28层 × 3个 linear + 每层一个 splitKreduce)

eltwise残差         1.0ms   5.0%     ~36us     add, mul, silu 等

rms_norm            0.8ms   4.3%     ~29us     mean + rsqrt + mul (float精度)

类型转换 + copy     0.5ms   2.6%     ~18us     bf16 ←→ float 转换

其他                0.7ms   3.6%              rotary, silu, kv_append, assert

─────────────────────────────────────────────────────────────
GPU 总计:          19.5ms
Wall 总计:         68.8ms
GPU 利用率:         28.3%              ← 71.7% 的时间花在 CPU kernel launch 开销
```

## 一个 Decode Step (4层)

取自 burst 35（164 kernels, 9.6ms wall, 2.6ms GPU）。

```
Operator           GPU时间   占比    每层耗时
─────────────────────────────────────────────────
attention           1.7ms  66.2%    ~433us
matmul(mlp)         0.5ms  18.8%    ~33us
eltwise             0.1m   ~5%
其他                0.3ms  ~10%
─────────────────────────────────────────────────
GPU 总计:           2.6ms
Wall 总计:          9.6ms
GPU 利用率:         27.1%
```

## 关键发现

### 1. GPU 利用率极低 (~28%)

每 1ms GPU 计算对应 ~3ms CPU launch overhead。原因是**每个 kernel 都是独立 CPU dispatch**——PyTorch eager mode 下每层每算子都调一次 cuLaunchKernel。

CUDA graphs 可以**一次性录制整个 forward** 然后 replay，完全消除 CPU launch 开销。

### 2. Attention 是最大单项 (57%)

FlashInfer paged attention 每层 ~400us。对比纯 PyTorch SDPA 或 FA2 原生实现，FlashInfer 多了 `append_paged_kv_cache` + `plan()` 的间接开销。

### 3. per-layer prefill ≈ per-layer decode

| | Prefill 单层 | Decode 单层 |
|---|---|---|
| GPU time | ~700us | ~650us |
| attention | ~400us | ~433us |
| MLP | ~180us | ~33us (QKV dim × 1 token) |

Prefill 和 decode 的 attention 耗时相近，但 MLP 在 prefill 时更大（batch=多个 token vs decode 每请求 1 token）。

### 4. 其他开销

- `pow` 0.2ms (0.9%)：RMS norm 里的 x²，float 精度
- `copy` 0.5ms (2.6%)：bf16 ←→ float 转换
- `sanity_assert` 接近 0：只在部分层触发

## 优化方向

1. **CUDA graphs for decode** → 消除 CPU launch 开销，预期 decode GPU 利用率从 27% → 80%+，decode 加速 ~3x
2. **图内算子融合**：pow + mean + rsqrt + mul → 单个 fused RMS norm kernel，减少 kernel count
3. **纯 BF16 推理**：当前大量 float32 中间计算（RMS norm 的 pow/mean/rsqrt），换成全 bf16 可以减少 copy 和精度转换
