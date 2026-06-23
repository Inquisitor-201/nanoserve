# Full-Forward CUDA Graph 实现

## 概述

nanoserve 采用 **full-forward** CUDA Graph 捕获策略：将整个 `model.forward()`（embedding → 28 层 decoder → norm → lm_head）**一次性捕获为单个 CUDAGraph**，replay 时一次 cuGraphLaunch 完成全部计算。

与 vLLM 的 **piecewise** 策略（每层单独捕获，`CUDAGraphRunner` 逐层 replay）相比：

| 维度 | nanoserve (full-forward) | vLLM (piecewise) |
|---|---|---|
| Graph 数量 | 每 batch size 1 个 | 每 batch size × layers 个 |
| Replay 开销 | 1 次 cuGraphLaunch | ~num_layers 次 cuGraphLaunch |
| Python 介入 | 0（纯 GPU） | 每层 Python for 循环 |
| 捕获难度 | 需固定所有中间 tensor 地址 | 只需固定每层的输入/输出 |

关键设计：为每个 batch size 预分配**固定地址**的输入/输出/页表缓冲区，捕获时 CUDA kernels 记录的是这些固定地址，replay 时仅需 `copy_()` 更新缓冲区数据然后 `graph.replay()`。

---

## 架构

### 1. Batch Schedule

`_build_batch_schedule()` 生成预捕获的 batch size 列表：

```python
[1, 2, 4, 8, 16, 32, 64, 96, 128, 160, ...]  # capped at max_batch
```

- 小 batch（≤32）按 2 倍递增
- 大 batch（≥64）每 32 递增
- Replay 时选择 **≥ actual batch size 的最小预捕获 batch**，剩余 slot 填充无效数据

### 2. `_CGBatchResources`

每个 batch size 对应一个 `_CGBatchResources` 实例，持有：

| 缓冲区 | 大小 | 用途 |
|---|---|---|
| `input_ids` | `[batch_size]` | 当前 token ID |
| `positions` | `[batch_size]` | 位置编码（seq_len - 1） |
| `indptr` | `[batch_size + 1]` | 页表 indptr |
| `indices` | `[batch_size × 256]` | 页表 block IDs |
| `last_page_len` | `[batch_size]` | 最后一页有效 token 数 |
| `logits` | `[batch_size, vocab_size]` | 输出 logits |
| `metadata` | `AttentionMetadata` | 视图（view）指向上述缓冲区 |

所有缓冲区在 `__init__` 时 `torch.zeros()` 分配，地址固定。`metadata` 是持有这些缓冲区视图的单例对象，捕获时所有 kernel 引用的是底层固定地址。

### 3. `CUDAGraphBatchDecodeWithPagedKVCacheWrapper`

FlashInfer 提供专用的 CG wrapper，其 `run()` 是纯 kernel launch（可被 `torch.cuda.graph()` 捕获）。`init_cg_wrapper()` 创建此 wrapper 并绑定到 `_CGBatchResources` 的 indptr/indices/last_page_len 缓冲区。

```python
# flashinfer_backend.py
def init_cg_wrapper(self, batch_size, indptr_buffer, indices_buffer,
                    last_page_len_buffer):
    cg_workspace = torch.empty(16 * 1024 * 1024, dtype=torch.uint8, device=device)
    wrapper = CUDAGraphBatchDecodeWithPagedKVCacheWrapper(
        cg_workspace, indptr_buffer, indices_buffer, last_page_len_buffer)
    wrapper.plan(...)  # 固定参数
    # Warmup 触发 lazy 分配
    return wrapper
```

### 4. Capture 流程

```python
# model_executor.py
for bs in reversed(batch_sizes):  # 从大到小
    res = _CGBatchResources(bs, ...)
    res.cg_wrapper = backend.init_cg_wrapper(bs, ...)  # ← 存入引用

    # 填充 dummy 页表（1 page/seq → block 0）
    res.indptr = arange(batch_size + 1)
    res.indices[:bs] = 0

    # Warmup
    _ = model(input_ids[:bs], metadata)

    # 捕获全部 forward → 单个 CUDAGraph
    with torch.cuda.graph(g, pool=graph_pool):
        res.logits[:bs] = model(res.input_ids[:bs], res.metadata)
```

### 5. Replay 流程

```python
# model_executor.py
def _execute_cg_decode(self, input_ids, block_tables, seq_lengths):
    res = self._get_cg_resources(batch_size)

    # 切换 backend 的 CG wrapper 到当前 batch size 的 wrapper
    backend._cg_wrapper = res.cg_wrapper

    # 更新输入缓冲区
    res.input_ids[:batch_size].copy_(input_ids)
    res.positions[:batch_size].copy_(positions)
    res.upload_block_tables(block_tables, seq_lengths, ...)

    # 一次 cuGraphLaunch = 全部 28 层 + lm_head
    res.graph.replay()
    return res.logits[:batch_size]
```

---

## 已修复的问题

### Bug 1: CG Wrapper workspace 被 GC 回收 → illegal memory access

**现象**：`CUDA_LAUNCH_BLOCKING=1` 时 bench.py 崩溃，报 `an illegal memory access was encountered`，发生在 CG decode 路径。`--eager` 正常。

**影响版本**：6e6666a ~ ccbfac5（修复前）。

**根因**：

```python
# 修复前：capture_decode_graphs()
for bs in reversed(batch_sizes):
    res = _CGBatchResources(bs, ...)

    # 每次循环创建新 wrapper，覆盖 backend._cg_wrapper
    backend.init_cg_wrapper(bs, ...)      # ← wrapper_W256 被创建
    # ... capture G256 ...
    # 下一轮：wrapper_W224 覆盖 wrapper_W256
    # wrapper_W256 失去引用 → Python GC → cg_workspace 释放
    # 但 G256 的 CUDA ops 还引用着被释放的 workspace 地址
    # → replay G256 → illegal memory access ✅
```

**修复**（3 处改动）：

1. `_CGBatchResources.__slots__` 新增 `"cg_wrapper"`，`__init__` 中 `self.cg_wrapper = None`
2. `capture_decode_graphs()` 中：`res.cg_wrapper = backend.init_cg_wrapper(...)` — 每个 batch size 的 wrapper 引用存在 `_CGBatchResources` 中，不会被 GC
3. `_execute_cg_decode()` 中 replay 前：`backend._cg_wrapper = res.cg_wrapper` — 切换到正确 batch size 的 wrapper，确保 CUDA ops 引用的是存活的 workspace

**验证**：修复后 CG 路径正常，吞吐 1387.88 tok/s。

---

### Bug 2: `_plan_decode` 跳过 CG wrapper plan → page-table drift → 输出退化

**现象**：CG 模式输出质量严重退化（"thinkazazazaazaz" 类型重复），而 eager 模式正常。在长生成（>100 tokens）中尤为明显，表现为"奔跑到某个 token 后突然分叉退化为重复"。

**影响版本**：ccbfac5 ~ 修复前。

**根因**：

```python
# 修复前：flashinfer_backend.py
def _plan_decode(self, metadata):
    if self._cg_wrapper is not None:
        return  # ← ✗ 直接跳过！workspace 不更新
```

`CUDAGraphBatchDecodeWithPagedKVCacheWrapper.plan()` 会将 page table 数据预计算为 attention kernel 的元数据（如每序列需要读哪些 KV page、chunk 偏移等）写入固定地址的 workspace buffer。graph capture 时，kernel 读取的是这个 workspace 的固定地址。

但 `_plan_decode` 在 CG 路径下直接 return，意味着 **plan() 只在 `init_cg_wrapper()`（capture 前）被调用过一次**，用的是 dummy page table（每序列 1 页 → block 0）。

真实推理过程中，序列不断增长、acquire 新 block，page table 从 `[0]` 演进到 `[0, 17, 42, ...]`。但 workspace 里的预计算元数据还是基于 `[0]` 的旧数据 → attention 读到的 KV 位置越来越偏移正确值 → logits 逐渐偏离 → 某一步 argmax 翻转 → cascade 退化。

**为什么不是第一步就错，而是跑到某个 token 才分叉**：
- 前几步序列还在 1 页内，page table = `[0]`，和 dummy 一致 → workspace 正确 → token 一致
- 首次需要第 2 页时：workspace 里没有这个新页的信息 → attention 读到错误 KV 位置 → logits 微小偏移但 argmax 可能还没变
- 累积偏移 → argmax 翻转 → 第一个错误 token → cascade

**修复**：

```python
# 修复后：flashinfer_backend.py
def _plan_decode(self, metadata):
    if self._cg_wrapper is not None:
        # 用当前（已更新）的 page table 重算 workspace 元数据
        self._cg_wrapper.plan(
            metadata.paged_kv_indptr,
            metadata.paged_kv_indices,
            metadata.paged_kv_last_page_len,
            self.num_heads, self.num_key_value_heads, self.head_dim,
            self.page_size,
            q_data_type=self.dtype, kv_data_type=self.dtype,
            pos_encoding_mode="NONE",
        )
        return
```

**关键理解**：
- `plan()` 不在 CUDA Graph 内部（capture 只 capture `model.forward()`），它在 replay 前作为 CPU 操作（或 `cudaMemcpyAsync`）更新 workspace buffer 数据。capture 时 kernel 读的是 workspace 的**地址**，不是数据——所以每次 replay 前更新数据是安全的，也是必要的。
- **顺序要求**：`plan()` 必须在 `upload_block_tables()` **之后**调用。因为 `plan()` 会从 indptr/indices 缓冲区**读取数据**来预计算 workspace 元数据。如果先 plan 再更新 page table，workspace 将基于老旧数据计算，replay 时 kernel 读到的 buffer 数据和 workspace 元数据不一致，可能导致 attention 读到错误偏移甚至 GPU hang。

**验证**：
- 0.6B ABC 测试：CG PASS, Eager PASS, 两者一致
- 1.7B 多序列泛化：CG 和 eager 输出一致，质量正常
- 跨实例重复运行：CG 与 eager 输出 token-by-token 一致

---

## 性能

RTX 3060 (12GB) + Qwen3-0.6B, 256 req, output ~524 tokens/req:

| Mode | Time | Throughput |
|------|------|------------|
| **CG** (full-forward) | **96.53s** | **1387.88 tok/s** |
| Eager | 202.27s | 662.32 tok/s |

CG 加速比：**2.1×**。提速来自：
- 消除 28 层 × Python for 循环的调度开销
- 单次 cuGraphLaunch 替代 28 次 kernel launch
- 减少 CPU-GPU 同步次数

---

## 性能对比总表

RTX 3060 (12GB) + Qwen3-0.6B, 256 req, output ~524 tokens/req:

| 版本 | Time | Throughput | vs nanoserve eager | vs nanoserve CG |
|------|------|------------|-------------------|-----------------|
| vLLM V1 (torch.compile + CUDA Graph + FlashAttn) | 140.18s | 955.65 tok/s | 1.44× | 0.69× |
| **nanoserve CG** | **97.04s** | **1380.59 tok/s** | **2.09×** | **1.0×** |
| nanoserve eager | 202.27s | 662.32 tok/s | 1.0× | 0.48× |
| `f31df09` 基线 (eager) | 208.60s | 642.22 tok/s | 0.97× | 0.47× |

结论：

- **nanoserve CG 比 vLLM V1（已开 torch.compile + CUDA Graph）快 44%**
- nanoserve CG 比 nanoserve eager 快 2.09×
- `f31df09` commit message 中的 "1177 tok/s" 系不同条件测得，同一 bench.py 实际跑得 642 tok/s。无 eager 回归。

vLLM 使用了 Flash Attention 后端 + torch.compile(PIECEWISE) + piecewise CUDA Graph，
nanoserve 使用 FlashInfer 后端 + full-forward CUDA Graph（无 torch.compile）。
差距主要来自 nanoserve 的 full-forward CG 策略（单次 cuGraphLaunch vs vLLM 的逐层 replay），
以及 FlashInfer 的 decode kernel 在 RTX 3060 上比 Flash Attention 更高效。
