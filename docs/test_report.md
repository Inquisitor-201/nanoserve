# nanoserve 离线推理测试报告

> **测试日期**: 2026-04-30
> **测试目标**: 评估 nanoserve 推理框架的批量推理性能，对比 Burst（同时到达）和 Staggered（波次到达）两种模式，分析连续批处理的实际效果。

---

## 目录

- [1. 术语说明](#1-术语说明)
- [2. 测试环境](#2-测试环境)
- [3. 测试方法](#3-测试方法)
- [4. 基准测试结果](#4-基准测试结果)
- [5. 性能分析](#5-性能分析)
- [6. 瓶颈定位](#6-瓶颈定位)
- [7. 优化建议](#7-优化建议)
- [附录](#附录)

---

## 1. 术语说明

| 概念 | 含义 |
|------|------|
| **Burst 模式** | 所有请求一次性通过 `add_requests()` 提交，`main_loop` 统一处理。scheduler 全部 prefill 后再全部 decode |
| **Staggered 模式** | 请求分多个波次（wave）在 decode 过程中到达，scheduler 需在 decode 间隙插入 prefill 新请求。触发真正的连续批处理 |
| **Wave** | Staggered 模式中的一批请求。同一 wave 的请求同时到达 |
| **TTFT** | Time To First Token。请求加入 scheduler 到产出第一个 token 的延迟 |
| **ITL** | Inter-Token Latency。后续每生成一个 token 的平均耗时 |

---

## 2. 测试环境

### 2.1 硬件

| 组件 | 规格 |
|------|------|
| GPU | NVIDIA GeForce RTX 3060 12 GB |
| 显存 | 12,288 MiB |
| 驱动版本 | 535.129.03 (CUDA 12.2) |

### 2.2 软件栈

| 组件 | 版本 |
|------|------|
| PyTorch | 2.4.0+cu121 |
| FlashInfer | 0.6.3 |
| Python | 3.11 |

### 2.3 模型

| 属性 | 值 |
|------|-----|
| 模型 | Qwen3-0.6B |
| 参数量 | ~600M |
| 数据类型 | bfloat16 |
| 层数 | 28 |
| Hidden size | 1024 |
| KV heads | 8 (GQA) |
| Head dim | 128 |
| Vocab size | 151,936 |

### 2.4 推理引擎配置

| 参数 | 值 | 说明 |
|------|-----|------|
| num_blocks | **4473**（自动计算） | 65% 显存用于 KV cache |
| block_size | 16 | 每块 token 数 |
| KV cache 总容量 | 4473 × 16 = **71,568 tokens** |
| max_num_seqs | 256 | 最大并发序列数上限 |
| Attention backend | FlashInfer | 分页注意力 |

---

## 3. 测试方法

### 3.1 两种测试模式

**Burst 模式（baseline）：**

```
add_requests(all) → main_loop → collect
                    ├─ schedule() → 一次 prefill 所有请求
                    └─ schedule() → decode 所有请求（256步）
```

所有请求同时到达。scheduler 一次性 prefill，然后全部 decode。这是吞吐上限。

**Staggered 模式（连续批处理）：**

```
wave 0: add_requests  →  prefill → decode...
wave 1:                  at step K → add_requests → prefill → decode...
wave 2:                                   at step 2K → add_requests → prefill → decode...
```

请求分多个 wave 在 decode 过程中注入。scheduler 在 decode 步骤间插入 prefill。这模拟了真实 serving 场景。

### 3.2 测试变量

| 变量 | 取值 |
|------|------|
| 总请求数 | {16, 32, 64, 128} |
| Prompt 类型 | short（2-3 tokens） |
| 每请求 max_new_tokens | 256（固定） |
| Staggered 波次 | 16/32 req → 4 waves, 64/128 req → 8 waves |
| 波次注入间隔 | ~256/(waves+1) decode steps |
| 采样参数 | temperature=0.7, top_p=0.9 |

### 3.3 采集指标

| 指标 | 含义 |
|------|------|
| Wall time | 端到端耗时（最后一条请求完成） |
| Throughput (tok/s) | 所有 token / wall time |
| Throughput (req/s) | 请求数 / wall time |
| TTFT | 每条请求的首 token 延迟均值 |
| ITL | 所有 decode 步骤的延迟均值 |

---

## 4. 基准测试结果

### 4.1 Burst 模式（同时到达）

```
 #Req |   Wall   |  tok/s   |  req/s  |  TTFT avg |  ITL avg
      |    (s)   |          |         |   (ms)    |   (ms)
------+----------+----------+---------+-----------+----------
  16  |   10.76  |    385   |   1.5   |    55.3   |   38.6
  32  |   11.62  |    714   |   2.8   |    51.3   |   39.9
  64  |   13.03  |   1273   |   4.9   |    54.1   |   41.7
 128  |   13.52  |   2454   |   9.5   |    62.5   |   39.0
```

### 4.2 Staggered 模式（波次到达）

**16 requests（4 waves × 4 req）：**

```
 Wave  |  tok/s  |  #Req   |  TTFT avg |  ITL avg
       |(per wave)|         |   (ms)    |   (ms)
-------+----------+---------+-----------+----------
   0   |   58.9  |    4    |    46.0   |   39.6
   1   |   58.9  |    4    |    42.8   |   39.9
   2   |   58.9  |    4    |    44.0   |   40.2
   3   |   58.9  |    4    |    43.0   |   40.5
 Total |  235.5  |   16    |    43.9   |   40.1
 Wall: 17.58s （Burst: 10.76s）
```

**32 requests（4 waves × 8 req）：**

```
 Wave  |  tok/s  |  #Req   |  TTFT avg |  ITL avg
-------+----------+---------+-----------+----------
   0   |  120.8  |    8    |    41.7   |   37.5
   1   |  120.8  |    8    |    41.6   |   37.4
   2   |  120.8  |    8    |    40.9   |   37.7
   3   |  120.8  |    8    |    41.9   |   38.4
 Total |  483.3  |   32    |    41.5   |   37.7
 Wall: 17.15s （Burst: 11.62s）
```

**64 requests（8 waves × 8 req）：**

```
 Wave  |  tok/s  |  #Req   |  TTFT avg |  ITL avg
-------+----------+---------+-----------+----------
   0   |  100.6  |    8    |    44.3   |   40.1
   1   |  100.6  |    8    |    46.5   |   40.0
   2   |  100.6  |    8    |    43.8   |   39.9
   3   |  100.6  |    8    |    47.5   |   39.5
   4   |  100.6  |    8    |    45.5   |   39.3
   5   |  100.6  |    8    |    42.4   |   39.2
   6   |  100.6  |    8    |    41.8   |   39.2
   7   |  100.6  |    8    |    40.9   |   39.1
 Total |  805.2  |   64    |    44.1   |   39.5
 Wall: 20.59s （Burst: 13.03s）
```

**128 requests（8 waves × 16 req）：**

```
 Wave  |  tok/s  |  #Req   |  TTFT avg |  ITL avg
-------+----------+---------+-----------+----------
   0   |  195.3  |   16    |    42.9   |   37.7
   1   |  195.3  |   16    |    41.9   |   37.8
   2   |  195.3  |   16    |    41.3   |   37.8
   3   |  195.3  |   16    |    48.0   |   37.9
   4   |  195.3  |   16    |    41.1   |   37.9
   5   |  195.3  |   16    |    40.2   |   37.7
   6   |  195.3  |   16    |    41.6   |   37.7
   7   |  195.3  |   16    |    41.9   |   37.7
 Total | 1562.3  |  128    |    42.4   |   37.8
 Wall: 21.23s （Burst: 13.52s）
```

### 4.3 核心对比

```
 请求数  |  Burst tok/s  |  Staggered tok/s  |  比例   |  Burst TTFT  |  Staggered TTFT
---------+---------------+-------------------+---------+--------------+-----------------
   16    |     385       |       236         |   61%   |   55.3 ms    |    43.9 ms
   32    |     714       |       483         |   68%   |   51.3 ms    |    41.5 ms
   64    |    1273       |       805         |   63%   |   54.1 ms    |    44.1 ms
  128    |    2454       |      1562         |   64%   |   62.5 ms    |    42.4 ms
```

---

## 5. 性能分析

### 5.1 Burst 模式——吞吐上限

Burst 模式下，墙钟时间几乎不随请求数增加：

```
 16 req: 10.76s
128 req: 13.52s   (+26%)
```

墙钟 = prefill(1步) + decode(256步)。增加请求数只增加每步 batch size，不增加步数。因此吞吐几乎线性扩展：16→128 请求，吞吐提升 6.4×。

这就是 LLM 推理的独特性质：**Burst 模式下，增加请求数的边际成本极低**，因为 decode 步数是固定的。

### 5.2 Staggered 模式——连续批处理实际效果

Staggered 模式的墙钟更长：

| 请求数 | Burst 墙钟 | Staggered 墙钟 | 增幅 |
|--------|-----------|---------------|------|
| 16 | 10.76s | 17.58s | +63% |
| 128 | 13.52s | 21.23s | +57% |

原因是 staggered 模式下，**最后一批请求的 decode 结束时间被推迟了**。每个 wave 的注入间隔（spacing）让总的 decode timeline 被拉长。示意图：

```
Burst:
  16 req: ██prefill██→████decode 256步████→ done at t=10.8s

Staggered (4 waves of 4):
  w0: ██p█→████d████→ ... →████done at t=256
  w1:           at step 51: ██p█→████d████→ ... →done at t=307
  w2:                      at step 102: ██p█→████d████→ ... →done at t=358
  w3:                                at step 153: ██p█→████d████→ ... →done at t=409
                                                                       ↑ done at t=17.6s
```

### 5.3 关键发现：TTFT 一致性

**Staggered 模式下，所有 wave 的 TTFT 均保持在 40-48ms，不随 wave 号递增。**

这意味着：
- 晚到达的请求不需要等待前面的请求 decode 完成
- Scheduler 在 decode 间隙迅速 prefill 新请求（~2-5ms for 16 short prompts）
- **连续的 prefill 不会显著增加 decode 延迟**（ITL 始终 ~38ms）

对比 Burst 模式：TTFT 55-63ms，略高于 staggered 的 42-44ms。Burst 模式下，一次 prefill 所有请求（128 × 3 = 384 tokens）比 staggered 模式一次 prefill 16 个请求（48 tokens）耗时更长。

### 5.4 ITL 完全不受影响

Burst 和 Staggered 的 ITL 完全相同（38-40ms），不受以下因素影响：

- 总请求数（16 vs 128）
- 调度模式（burst vs staggered）
- 是否在 prefill 间隙穿插 decode

这进一步验证了 **decode 是 memory-bandwidth bound**，prefill 的插入不会影响单步 decode 延迟。

### 5.5 时间分解

以 Burst 128 为例：

| 阶段 | 耗时 | 占比 |
|------|------|------|
| Prefill（128 req × 3 tok） | ~63 ms | 0.5% |
| Decode（256 步 × ~39 ms） | ~9,984 ms | 73.9% |
| 其他（调度、采样等开销） | ~3,473 ms | 25.7% |
| **合计** | **~13,520 ms** | **100%** |

> 注：Burst 128 比 Burst 16 多出的 ~3s 开销来自调度器内部循环（每步处理 128 条请求的 block table、token 管理等），以及 attention backend 对更大 batch 的额外处理。

---

## 6. 瓶颈定位

### 6.1 解码显存带宽（主要瓶颈）

- 单步 decode 访存量：28 层 × (KV cache + 权重) ≈ 数百 MB
- RTX 3060 360 GB/s 带宽，实测每 token ~25 tok/s/seq
- 增加 batch 不显著增加每步延迟——说明注意力访问的带宽效率高

### 6.2 调度 overhead（大 batch 显现）

Burst 128 比 Burst 16 多了 ~3s 的非 decode 开销。来源：

- Scheduler 每步更新 128 条请求的 block table
- `update_running_requests` 循环 128 次 × 256 步 = 32,768 次迭代
- `_run_inference_step` 中的采样（128 × 151936 vocab softmax）

### 6.3 KV Cache 未达瓶颈

4473 blocks × 16 = 71,568 tokens 容量。128 请求 × (3 + 256) = 33,152 tokens，利用率仅 46%。继续增加请求到 256 应仍可容纳。

---

## 7. 优化建议

### 7.1 优化调度器循环

大 batch 下，scheduler 的 Python 级循环（逐请求更新 block table、逐请求检查 EOS）成为 overhead。可批量处理这些操作：

- `update_running_requests` 使用向量化操作而非 for 循环
- 使用 PyTorch tensor 跟踪 block table 而非 Python list

### 7.2 动态波次调度

当前 Staggered 测试使用固定 spacing。实际 serving 可根据 KV cache 余量动态决定何时注入新请求，平衡吞吐和延迟。

### 7.3 请求级 max_new_tokens 差异

当前所有请求使用相同 max_new_tokens。混合长短生成的任务下，scheduler 可优先 decode 短请求以释放 block，提高 block 周转率。

---

## 附录

### A. 测试脚本

`testbench.py` 核心流程：

**Burst 模式：**
```python
request_ids = llm_service.add_requests(prompts, sampling_config)
llm_service.main_loop(request_ids, sampling_config)
```

**Staggered 模式：**
```python
# 手动控制 scheduler 循环，在 decode 过程中注入新请求
while scheduler.has_unfinished_requests():
    sched_output = scheduler.schedule()
    sampled_tokens = _run_inference_step(sched_output, ...)
    scheduler.update_running_requests(sampled_tokens, ...)
    step += 1
    if step == inject_at:
        ids = llm_service.add_requests(new_prompts, sampling_config)
```

### B. 使用方式

```bash
python3 testbench.py
```

### C. Prompt 数据集

| 类型 | 长度 | 示例 |
|------|------|------|
| short | 2-3 tokens | "中国是一个", "人工智能是" |

### D. 自动 num_blocks 机制

```python
def auto_calculate_num_blocks(device, dtype, block_size,
                              num_layers, num_kv_heads, head_dim,
                              safety_factor=0.65):
    total = torch.cuda.get_device_properties(0).total_memory
    available = int(total * safety_factor)
    bytes_per_block = num_layers * 2 * block_size * num_kv_heads * head_dim * dtype_size
    return available // bytes_per_block
```
