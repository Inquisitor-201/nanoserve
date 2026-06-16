# vLLM 显存分配明细 (V1 engine, Qwen3-0.6B, 12 GiB GPU)

```
GPU: 11.76 GiB total
gpu_memory_utilization: 0.90 (default)

Profile 过程：
1. 模型权重加载后实测:        1.12 GiB
2. 跑一次 dummy forward:
   - PyTorch 峰值分配:        peak_memory = 2.71 GiB
   - 其中模型权重:             1.12 GiB
   - 净中间激活峰值:           1.59 GiB  (= 2.71 - 1.12)
3. Profile 后空闲:            10.43 GiB

计算 KV cache 预算：
  budget = 11.76 × 0.90 = 10.58 GiB
  available_kv = 10.58 - 2.71 = 7.87 GiB

最终分配：
  KV cache:      7.87 GiB  (73,696 tokens, ~18 seqs × 4096 tokens)
  模型权重:      1.12 GiB
  中间激活:      1.59 GiB  (profile 峰值, 实际运行复用)
  其他开销:      1.18 GiB  (= 11.76 - 7.87 - 1.12 - 1.59)
```

## nanoserve 对齐过程

### 第一次实现 (commit 042bbaf)

照着 vLLM 的逻辑实现 profiling：

1. 创建模型 + 544 block profile pool（0.93 GiB）
2. 跑 dummy forward（4 seqs × 2048 tok → peak=2.85 GiB）
3. 释放 profile pool → `surviving = memory_allocated()` → `intermediate_peak = peak - surviving`
4. `available_kv = total × 0.9 - surviving - intermediate_peak - headroom`

**问题 A：Profile pool 未释放双引用**
- 只 `del profile_bm` 不够，`model.attention_backend.kv_cache_pool` 和 `model_executor.kv_cache_pool` 都持有引用
- 导致 0.93 GiB 无法回收，free 仅 1.52 GiB（应有 2.45+ GiB）
- 修复后 free=3.18 GiB，KV pool=3935 blocks

**问题 B：Profile batch 参数不符**
- Profile 时 4 seqs × 2048 tok（宽裕），实际调度 15 seqs × 不同长度
- FlashInfer workspace 大小跟 seq 数有关，不是总 token 数
- Profile peak=2.85 GiB，实际 forward 增量=3.14 GiB
- 修正在 profile batch 里用 16 seqs × 512 tok 对齐真实分布

### 定位 OOM 根因

经过逐层 memory trace（monkey-patch Attention 和 MLP 的 forward），发现：

```
Layer  0 MLP: +0.166 GiB  (never freed)
Layer  1 MLP: +0.166 GiB  (accumulated)
...
Layer 13 MLP: +0.166 GiB  → OOM at layer 14
```

**根因：`model.forward()` 没有 `torch.no_grad()`**
- autograd 保留了完整计算图，每层的中间变量（QKV、MLP hidden）层间不释放
- 14 层 × 0.166 GiB = 2.33 GiB，加上 attention 的 0.5 GiB 正好吃掉所有空闲
- 之后换用 `torch.inference_mode()`（还禁用了 view tracking 和 version counter，性能更好）

### 调度器问题：阻塞死锁

删 `max_model_len` 并发上限后，256 个请求全部进 running_list，每人分 32 tok。block 耗尽后走 preempt，每次只推进 1-7 个请求。

**修复方向**：one-at-a-time prefill——budget 先喂饱一个人再换下一个，减少 concurrent prefills。

### 当前状态

- ✅ `inference_mode` OOM fix
- ✅ profiling KV 缓存大小正确
- ✅ 256 seqs × 100-1024 tok 跑通（251.58 tok/s）
- ❌ **preempt 逻辑有 CUDA illegal memory access bug**——preempt 时释放的 block 可能还在 attention kernel 中使用
- ❌ 性能还有 ~5.9x 提升空间（vLLM: 1,476 tok/s）

### 结论

- vLLM **先 profile 再分 KV cache**，精确知道峰值
- 中间激活只占 **1.59 GiB**（12 GiB 的 13.5%）
- nanoserve 的 `auto_calculate_num_blocks` 用 safety_factor=0.65 硬算，没有实测，多估了激活、少分了 KV cache
