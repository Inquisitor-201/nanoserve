# CUDA Graph decode 优化路线图

## 目标

把 decode 阶段 GPU 利用率从 **~30% 提升到 80%+**，通过 CUDA graphs 消除 CPU kernel launch 开销。

## 现状

Trace 分析（64 seqs × 4-16 output tokens, Qwen3-0.6B）显示每步 decode：

```
GPU 总计:     20.8ms
Wall 总计:    69.6ms
GPU 利用率:   30%
```

70% 的时间花在 CPU kernel launch（`cuLaunchKernel`）上。每步 ~2000 次 kernel launch（28 层 × ~72 kernels/层）。

## 关键依赖

FlashInfer 自带的 `CUDAGraphBatchDecodeWithPagedKVCacheWrapper`（decode.py:1461）提供了 CUDA graph 兼容的 decode attention wrapper。

### 规则
1. **固定 batch size** — 每个 wrapper 实例绑定一个 batch size，运行时不能变
2. **固定 buffer 地址** — `indptr_buffer`, `indices_buffer`, `last_page_len_buffer` 地址不变，值可更新
3. **`plan()` 在 graph 外** — 内部分配 workspace，不能在 graph capture 内调用
4. **`run()` 可 graph** — 纯 kernel launch，可 capture

## 实施阶段

### Phase 0：KV block 预分配

**问题**：当前 prefill 只分配 prompt 长度的 blocks，decode 每步 allocate 1 个新 block，page table 长度变化，需要每步 re-plan。

**修复**：prefill 阶段一次性分配给整条序列（prompt + max_new_tokens）的 blocks，固定 page table 长度。这样 `plan()` 只需要在 prefill 结束时调一次。

**代价**：提前占用 blocks，但总量不变（每条序列最终都需要那么多块）。

**收益**：无吞吐提升，但为 CUDA graph 铺路。

### Phase 1：Attention-only CUDA graph

替换每层 decode 的 FlashInfer attention 为 `CUDAGraphBatchDecodeWithPagedKVCacheWrapper`。

每层 decode forward（改后）：
```
  input_layernorm          → eager
  QKV projection           → eager
  RoPE                     → eager
  attention (FlashInfer)   → CUDAGraphBatchDecodeWithPagedKVCacheWrapper.run → REPLAY
  KV append                → eager
  residual + norm + MLP    → eager
```

改动量：
- 在 `FlashInferBackend` 中创建 `CUDAGraphBatchDecodeWithPagedKVCacheWrapper`
- warmup 时对每个 batch size capture graph
- `plan()` 在 graph 外调用

**收益**：省掉 28 个 attention + 28 个 KV append 的 kernel launch，decode wall time -15%

### Phase 2：完整 decode forward CUDA graph

把整个 28 层 `input → logits` 全部 graph 化。

每步 decode（改后）：
```
  1. memcpy input_ids, positions, page table 到固定 buffer
  2. graph.replay()  ← 一次调用，涵盖所有 28 层的所有算子
  3. 从 output buffer 读取 logits
```

需要：
- 预分配所有中间 tensor 的固定 buffer（hidden_states 各层之间复用）
- 把 `FlashInferBackend` 的 `plan()` 提取到 graph 外
- warmup 对每个 batch size capture

**收益**：省掉 ~2000 kernel launch/步，decode GPU 利用率从 30% → 80%，整体吞吐 +15-20%

### Phase 3：多 batch size 管理

decode batch size 逐步递减（请求完成释放 blocks），需要一组 graph：

| batch size | graph |
|---|---|
| max 并发数 | Gₘₐₓ |
| max-1 | Gₘₐₓ₋₁ |
| ... | ... |
| 1 | G₁ |

warmup 时 capture 全部 graphs，每步选当前 batch size 对应的 graph replay。

**代价**：~4.5s warmup + ~0.5 GiB 显存（buffer 预分配）

## 预期收益汇总

| 阶段 | decode GPU 利用率 | 吞吐提升 | 工作量 | 风险 |
|---|---|---|---|---|
| 当前 | ~30% | — | — | — |
| Phase 0 (预分配) | ~30% | 0% | 小 | 低 |
| Phase 1 (attention graph) | ~40% | ~5% | 中 | 低 |
| Phase 2 (全 forward) | ~80% | ~15-20% | 大 | 中 |
| Phase 3 (多 batch) | ~80% | ~20% | 中 | 低 |

## 推荐路线

**先 Phase 0 + Phase 1**：预分配 blocks + attention-only graph。改动量小，验证 FlashInfer CUDA graph wrapper 能正常工作。如果收益可观再推进 Phase 2。
