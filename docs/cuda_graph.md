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
    res.update_page_table(block_tables, seq_lengths, ...)

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

## TODO

### [Perf] Eager 模式吞吐回归

**现状**：当前 HEAD 的 `--eager` 吞吐仅 662 tok/s，但 commit `f31df09`（引入 CG 前的基线）有 1177 tok/s。回归约 **44%**。

**待排查方向**：

| 可能原因 | 说明 |
|---|---|
| FlashInfer `_plan_decode` 每次都被调用 | `run()` 中 `metadata is not self._current_metadata` 总是 True（新 metadata 是每次创建的临时对象），导致每一步都重新 plan |
| `model.eval()` 副作用 | ccbfac5 在 `Qwen3Model.__init__` 末尾加了 `self.eval()`，虽不应影响推理性能，但需确认 |
| `core/__init__.py` 的 `PYTORCH_CUDA_ALLOC_CONF` | 新增 `expandable_segments:True,max_split_size_mb:512`，从 bench.py 移到 __init__.py 后执行时机不同 |
| ProfileTimer 的 `synchronize` 被移除 | ProfileTimer 现在完全注释掉（`__enter__/__exit__` 都是 pass），但回归前是有 synchronize 的——不应变慢 |
| 需要 bisect | 用 `git bisect` 在 `f31df09..HEAD` 范围内定位精确回归 commit |

**目标**：恢复 eager 到 1177 tok/s 水平，同时保持 CG 性能。
