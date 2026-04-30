# nanoserve 离线推理测试报告

> **测试日期**: 2026-04-30
> **测试目标**: 评估 nanoserve 推理框架在离线批量场景下的吞吐与延迟表现，分析连续批处理的可扩展性。

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

本文档中几个关键概念的区分：

| 概念 | 含义 |
|------|------|
| **num_requests** | 调用 `generate()` 时一次性提交的请求总数 |
| **GPU batch** | Scheduler 在单次 `schedule()` 中实际组批的请求数。由剩余 KV cache 动态决定 |
| **Prefill / Decode** | 推理的两个阶段。Prefill 计算整个 prompt 的注意力；Decode 逐个 token 自回归生成 |
| **Continuous Batching** | Scheduler 在 decode 间隙插入 prefill 的能力（当前实现为"先全部 prefill 再全部 decode"） |

> **注意**: `num_requests` ≠ GPU batch size。Scheduler 根据可用 KV block 决定一次能 prefill 多少请求，解码时所有 running 请求组成一个 batch。本文各测试中，受限于 `max_new_tokens=256` 的 KV 需求，每次 `generate()` 内的所有请求都能一次性进入 GPU batch（无分段 prefill）。

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
| num_blocks | **4473**（自动计算） | 根据可用显存自动算出 |
| block_size | 16 | 每块 token 数 |
| KV cache 总容量 | 4473 × 16 = **71,568 tokens** | 所有序列共享 |
| KV cache 显存占用 | ~7.64 GiB | 28×4473×2×16×8×128×2B |
| max_num_seqs | 256 | 最大并发序列数上限 |
| Attention backend | FlashInfer | 分页注意力实现 |

#### 关于自动计算 num_blocks

当 `EngineArgs(num_blocks=None)` 或省略此参数时，框架自动：

1. 从 HuggingFace 加载模型配置（获取 `num_layers`, `num_kv_heads`, `head_dim`）
2. 查询 `torch.cuda.get_device_properties(0).total_memory`
3. 预留 **35%** 给模型权重 + 激活值 + CUDA 上下文
4. 剩余 **65%** 全部用于 KV cache 的 block 分配

公式：

```
num_blocks = (total_gpu_memory × 0.65)
             / (num_layers × 2 × block_size × num_kv_heads × head_dim × dtype_bytes)
```

本环境计算结果：**4473 blocks**（KV cache 占 ~7.64 GiB，加上模型权重 ~1.2 GiB + 激活值，合计 ~9.31 GiB）

用户可随时手动指定 `num_blocks` 来覆盖自动值。

---

## 3. 测试方法

### 3.1 测试脚本

使用 `testbench.py`，见附录 A。

### 3.2 测试变量

| 变量 | 取值 |
|------|------|
| num_requests（提交请求数） | {1, 2, 4, 8, 16, 32, 64} |
| Prompt 类型 | short (2–3 tok), medium (14–15 tok), long (80–103 tok) |
| 每请求生成长度 | 256 tokens（固定） |
| 采样参数 | temperature=0.7, top_p=0.9 |

### 3.3 采集指标

| 指标 | 含义 |
|------|------|
| Throughput (tok/s) | 每秒处理 token 总数（含 prompt + generation） |
| Throughput (req/s) | 每秒完成请求数 |
| TTFT | Time To First Token, 首 token 延迟 |
| ITL | Inter-Token Latency, 后续每 token 平均延迟 |
| Prefill / Decode steps | 调度器实际执行的 prefill 轮次和 decode 轮次 |

### 3.4 流程

每个 (num_requests, prompt_type) 组合：

1. **Warmup**: 1 次空跑（预热 CUDA kernels）
2. **Timed run**: 记录一次带 `torch.cuda.synchronize()` 时序的完整 `generate()` 调用
3. **采集**: 从 `scheduler.completed_requests` 逐请求提取 TTFT、ITL

---

## 4. 基准测试结果

### 4.1 Short Prompts（2–3 tok input → 256 tok output）

**样例**: `"中国是一个"`, `"人工智能是"`, `"机器学习是"`

```
 #Req | Throughput  | Throughput |  TTFT  |   ITL  |   Steps  |  Wall
      |  tokens/s   |   req/s    | avg ms | avg ms |   P/D    |  (s)
------+-------------+------------+--------+--------+----------+--------
   1  |       26.5  |      0.1   |   40.3 |   36.7 |  1/255   |  9.740
   2  |       51.8  |      0.2   |   56.6 |   37.2 |  2/510   |  9.970
   4  |      104.5  |      0.4   |   40.1 |   36.8 |  4/1020  |  9.908
   8  |      183.5  |      0.7   |   56.9 |   41.4 |  8/2040  | 11.291
  16  |      410.4  |      1.6   |   42.5 |   36.2 | 16/4080  | 10.100
  32  |      790.4  |      3.1   |   42.4 |   35.8 | 32/8160  | 10.488
  64  |     1437.5  |      5.5   |   47.6 |   36.4 | 64/16320 | 11.536
```

### 4.2 Medium Prompts（14–15 tok input → 256 tok output）

**样例**: `"解释一下C++和Python的主要区别，以及各自的应用场景。"`

```
 #Req | Throughput  | Throughput |  TTFT  |   ITL  |   Steps  |  Wall
      |  tokens/s   |   req/s    | avg ms | avg ms |   P/D    |  (s)
------+-------------+------------+--------+--------+----------+--------
   1  |       27.4  |      0.1   |   39.8 |   37.0 |  1/255   |  9.843
   2  |       53.1  |      0.2   |   41.8 |   38.0 |  2/510   | 10.173
   4  |      100.3  |      0.4   |   43.8 |   40.0 |  4/1020  | 10.768
   8  |      213.8  |      0.8   |   44.2 |   37.0 |  8/2040  | 10.104
  16  |      425.6  |      1.6   |   44.2 |   36.4 | 16/4080  | 10.150
  32  |      821.7  |      3.0   |   51.9 |   35.8 | 32/8160  | 10.515
  64  |     1488.6  |      5.5   |   73.8 |   36.6 | 64/16320 | 11.608
```

### 4.3 Long Prompts（80–103 tok input → 256 tok output）

**样例**: `"你是一个精通逻辑推理的数学助手..."`（约 100 字长 prompt）

```
 #Req | Throughput  | Throughput |  TTFT  |   ITL  |   Steps  |  Wall
      |  tokens/s   |   req/s    | avg ms | avg ms |   P/D    |  (s)
------+-------------+------------+--------+--------+----------+--------
   1  |       35.4  |      0.1   |   39.7 |   38.2 |  1/255   | 10.145
   2  |       66.5  |      0.2   |   41.4 |   38.9 |  2/510   | 10.431
   4  |      136.8  |      0.4   |   61.5 |   37.5 |  4/1020  | 10.143
   8  |      279.2  |      0.8   |   63.5 |   36.3 |  8/2040  |  9.944
  16  |      546.7  |      1.6   |  114.5 |   36.1 | 16/4080  | 10.155
  32  |    ⚠ OOM   |            |        |        |          |
```

> **OOM 分析**: long prompts + 32 requests 总 KV 需求 ≈ 32 × (91 + 256) = 11,104 tokens，远低于 71,568 容量。但 prefill 阶段需要同时计算 32 × 91 = 2,912 tokens 的完整注意力，中间激活值（attention scores, softmax 输出等）瞬时撑爆显存。这是 **compute-time memory**（激活内存）瓶颈，不是 KV cache 容量问题。

### 4.4 汇总 - num_requests 扩展性

在所有 prompt 类型下，**每请求吞吐稳定在 25-26 tok/s/seq**（单序列解码速度），因此总吞吐几乎随 num_requests 线性增长：

```
t/s
1400 ┼                                          ● short (1437)
     │                                          ● medium (1488)
1200 ┼
     │
1000 ┼
     │
 800 ┼                                ● (790)
     │                                ● (822)
 600 ┼
     │                     ● (410)
 400 ┼                     ● (426)
     │                     ● (547)
 200 ┼          ● (104)   ● (213)
     │          ● (137)   ● (279)
   0 ┼──● (26)──● (52)──────────────────────────
     1      2       4       8      16      32      64
                                    num_requests →
```

---

## 5. 性能分析

### 5.1 解码延迟（ITL）稳定在 ~36-41ms

无论 prompt 类型和 num_requests 如何变化，**逐 token 解码延迟始终稳定**：

| 配置 | ITL 范围 | 偏离 |
|------|---------|------|
| Short, 1→64 req | 35.7–41.4 ms | ±8% |
| Medium, 1→64 req | 35.8–40.0 ms | ±6% |
| Long, 1→16 req | 36.1–38.9 ms | ±4% |

这表明：

- **解码阶段是 memory-bandwidth bound**。RTX 3060 的 360 GB/s 显存带宽是瓶颈
- 增加 batch 不会显著增加单 token 延迟，因为注意力计算的增量 KVCache 读取与 batch 大小呈正比，但带宽利用率已接近饱和
- batch=64 时 ITL 仍然只有 ~36ms，说明 GPU 可以很好地利用并行性隐藏增加的访存量

### 5.2 Prefill 延迟（TTFT）受 prompt 长度支配

| num_requests | Short TTFT | Long TTFT |
|-------------|-----------|----------|
| 1 | 40 ms | 40 ms |
| 4 | 40 ms | 62 ms |
| 16 | 43 ms | 115 ms |

- Short prompt 的 TTFT 基本不随请求数变化（prefill 计算量小，GPU 可并行）
- Long prompt 的 TTFT 随 batch 增长明显：每新增一条 91 tok 的请求，prefill 需多算 91 个位置的注意力分数

### 5.3 时间分解

以 short prompts、64 请求为例：

| 阶段 | 耗时 | 占比 |
|------|------|------|
| Prefill（1 步） | ~48 ms | ~0.4% |
| Decode（255 步 × ~36 ms） | ~11,488 ms | ~99.6% |
| **合计** | **~11,536 ms** | **100%** |

**Decode 占据 >99% 的时间。** 这是因为 `max_new_tokens=256` 下，每个请求要自回归解码 255 步，而 prefill 只需 1 步。

### 5.4 KV Cache 容量

| 参数 | 400 blocks（旧） | 4473 blocks（自动） |
|------|-----------------|-------------------|
| 总容量 | 6,400 tokens | **71,568 tokens** |
| 显存占用 | ~730 MB | ~7.64 GiB |
| 最大 short 请求 | ~24 | ~276 |
| 最大 medium 请求 | ~23 | ~265 |
| 最大 long 请求 | ~17 | ~206 |

实际限制更多来自 prefill 阶段的激活内存而非 KV cache 容量。

### 5.5 全表汇总

```
             Short (2-3 tok)         Medium (14-15 tok)        Long (80-103 tok)
 #Req   |  tok/s   TTFT   ITL   |   tok/s   TTFT   ITL   |   tok/s   TTFT   ITL
--------+------------------------+------------------------+------------------------
   1    |    26.5  40.3  36.7   |    27.4  39.8  37.0   |    35.4  39.7  38.2
   2    |    51.8  56.6  37.2   |    53.1  41.8  38.0   |    66.5  41.4  38.9
   4    |   104.5  40.1  36.8   |   100.3  43.8  40.0   |   136.8  61.5  37.5
   8    |   183.5  56.9  41.4   |   213.8  44.2  37.0   |   279.2  63.5  36.3
  16    |   410.4  42.5  36.2   |   425.6  44.2  36.4   |   546.7 114.5  36.1
  32    |   790.4  42.4  35.8   |   821.7  51.9  35.8   |    OOM   -     -
  64    |  1437.5  47.6  36.4   |  1488.6  73.8  36.6   |    -     -     -
```

---

## 6. 瓶颈定位

### 6.1 显存带宽瓶颈（Decode 主瓶颈）

- 单步 decode 需读取 28 层 KV cache: 每层 2×16×8×128×2B = 65,536 B
- 加上权重（QKV 投影 + FFN），单步总访存量远超计算量
- RTX 3060 **360 GB/s**  显存带宽，实测单序列 decode 约 25 tok/s
- 算术强度极低（~1 FLOP/byte），典型的 memory-bound 场景

### 6.2 激活内存瓶颈（Prefill OOM）

Long prompt × 32 requests 时 prefill OOM 的原因：

- Prefill 需要同时计算 2,912 tokens 的注意力
- Attention score 矩阵: batch × num_heads × seq_len² = 32 × 16 × (91)² ≈ 4.2M 元素
- 中间激活值（QKV 投影输出、softmax 输出、残差流等）每个 ~2912 × 1024 × 2B ≈ 6 MB/层
- 28 层合计中间激活 ≈ 200-300 MB
- 加上 KV cache 写入（~2.1 GB for 32 requests）和模型权重（1.2 GB）

### 6.3 调度器单步限制

当前 scheduler 的 `schedule()` 逻辑是 **全 prefill 后全 decode**：

```
if waiting_list: → _schedule_prefill()   # 一次 prefill 所有 waiting 请求
else:             → _schedule_decode()   # decode 所有 running 请求
```

这不是真正的 continuous batching（interleaved prefill + decode）。改进为连续批处理后，长 prompt 和大 batch 场景下 decode 步骤可穿插 prefill 新请求，平滑显存峰值。

---

## 7. 优化建议

### 7.1 连续批处理（Continuous Batching）

将调度策略从"全 prefill 再全 decode"改为"每步最多 prefill 一个请求 + decode 已有请求"：

```
当前:  prefill all → decode all → done
改进:  decode + prefill(1) → decode + prefill(1) → decode → ...
```

收益：
- 平滑 prefill 显存峰值，缓解长 prompt OOM
- 解码过程中可插入新请求，降低首个请求的排队延迟

### 7.2 减小 block_size 降低内部碎片

当前每个请求始终需要 `ceil((prompt_len + max_new_tokens) / 16)` 块。若 `block_size=8`，碎片更少，相同容量可服务更多序列。

### 7.3 长 prompt 分段 prefill（Chunked Prefill）

对于长 prompt，可以将 prefill 拆成多个 chunk，与 decode 交替执行，避免单步显存尖峰。

### 7.4 KV cache 量化

当前 KV cache 使用 bfloat16（2B/element）。如果实现 FP8 KV cache，容量翻倍且带宽需求减半。

---

## 附录

### A. 测试脚本

位于 `testbench.py`，核心流程：

```
LLMService.from_engine_args(engine_args)
  └─ generate(prompts, sampling_config)
       ├─ add_requests(prompts, sampling_config)  # tokenize + 入 scheduler 队列
       └─ main_loop(request_ids, sampling_config)  # schedule → execute 循环
            ├─ scheduler.schedule()                  # 选择 prefill 或 decode
            ├─ model_executor.execute_batch()        # GPU forward pass
            └─ scheduler.update_running_requests()   # 更新 token / 回收 block
```

### B. 使用方式

```bash
# 完整测试（自动 num_blocks）
python3 testbench.py

# 如需手动指定 KV cache 容量
# 修改 testbench.py 内的 engine_args:
#   EngineArgs(model_path=..., num_blocks=2000)
```

### C. 合成 Prompt 数据集

| 类型 | 长度 | 数量 | 示例 |
|------|------|------|------|
| short | 2–3 tokens | 15 | `"中国是一个"`, `"人工智能是"` |
| medium | 14–15 tokens | 4 | `"解释一下C++和Python的主要区别"` |
| long | 80–103 tokens | 2 | 带逻辑推理步骤的长 prompt |

### D. 自动 num_blocks 机制

当 `EngineArgs` 不传 `num_blocks` 或设为 `None` 时，由 `auto_calculate_num_blocks()` 在 `from_engine_args()` 中自动计算：

```python
def auto_calculate_num_blocks(device, dtype, block_size,
                              num_layers, num_kv_heads, head_dim,
                              safety_factor=0.65):
    total = torch.cuda.get_device_properties(0).total_memory
    available = int(total * safety_factor)
    bytes_per_block = num_layers * 2 * block_size * num_kv_heads * head_dim * dtype_size
    return available // bytes_per_block
```

`safety_factor=0.65` 预留 35% 给模型权重 + 激活值 + CUDA 上下文。

### E. 术语表

| 术语 | 英文 | 含义 |
|------|------|------|
| TTFT | Time To First Token | 请求到首个生成 token 的延迟 |
| ITL | Inter-Token Latency | 后续每生成一个 token 的平均耗时 |
| num_requests | Number of Requests | 一次 `generate()` 调用提交的请求总数 |
| GPU batch | GPU Batch | Scheduler 单步实际处理的请求数 |
| Continuous Batching | Continuous Batching | 混合 prefill 和 decode 请求的批处理策略 |
| KV Cache | Key-Value Cache | 缓存的注意力 Key/Value 张量，避免重复计算 |
| Block / Page | Physical Block | PagedAttention 的物理块，存储固定数量 token 的 KV cache |
