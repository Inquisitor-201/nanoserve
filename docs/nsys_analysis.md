# Nsight Systems Trace 分析命令

## 操作方式

在项目根目录 `./` 下执行（相对路径），trace 文件在 `test_output/` 中。

```bash
NSYS=/opt/nvidia/nsight-compute/2024.1.1/host/target-linux-x64/nsys
```

---

## 1. GPU Kernel 耗时分布（最核心）

三份报告一起出，看哪些 kernel 最耗时：

```bash
# Eager
$NSYS stats --report cuda_gpu_kern_sum \
  --format column --output - \
  --timeunit msec test_output/nsys_eager.nsys-rep

# CG
$NSYS stats --report cuda_gpu_kern_sum \
  --format column --output - \
  --timeunit msec test_output/nsys_cg.nsys-rep
```

重点关注：

| 列 | 含义 |
|---|---|
| `Total Time` | 该 kernel 总耗时 |
| `Pct.` | 占总 GPU 时间百分比（最大的几项就是瓶颈） |
| `Mean` | 单次平均耗时（Eager vs CG 同 kernel 的差异） |
| `Count` | 调用次数（Eager ≈ layers × steps × batch_sizes = 28× steps） |

对比预期：Eager 的 `flashinfer::decode` kernel 应该出现很多次（每层 × 每步），CG 里数量和种类都极少。

---

## 2. 算子级 Melting Point（cuda_gpu_kern_gb_sum 含 grid/block 维度）

```bash
$NSYS stats --report cuda_gpu_kern_gb_sum \
  --format column --output - \
  --timeunit msec test_output/nsys_eager.nsys-rep

$NSYS stats --report cuda_gpu_kern_gb_sum \
  --format column --output - \
  --timeunit msec test_output/nsys_cg.nsys-rep
```

多了 `Grid X`, `Block X`, `Registers`, `Static SMem` 列——可以看 CG 是否改变了 kernel launch 参数。

---

## 3. CUDA Kernel Launch + Exec 分离（看 launch overhead）

```bash
$NSYS stats --report cuda_kern_exec_sum \
  --format column --output - \
  --timeunit usec test_output/nsys_eager.nsys-rep

$NSYS stats --report cuda_kern_exec_sum \
  --format column --output - \
  --timeunit usec test_output/nsys_cg.nsys-rep
```

| 列 | 含义 |
|---|---|
| `Launch` | CPU 端 launch API 耗时（driver 调度延迟） |
| `Exec` | GPU 实际执行时间 |
| `Ratio` | Launch/Exec 比例 |

CG 的目的是消除每层的 Launch 开销——Launch 应该几乎降为 0。

---

## 4. CUDA API 调用统计（验证 CG 消除了哪些 API）

```bash
$NSYS stats --report cuda_api_sum \
  --format column --output - \
  --timeunit msec test_output/nsys_eager.nsys-rep

$NSYS stats --report cuda_api_sum \
  --format column --output - \
  --timeunit msec test_output/nsys_cg.nsys-rep
```

Eager 中每层每次 decode 都会有 `cuLaunchKernel`（× layers × steps）。
CG 中应该只有 graph replay 前的一次 `cudaMemcpyAsync`（更新 input_ids/positions/page table） + 一次 `cuGraphLaunch`。

---

## 5. GPU MemOps 分类（Memcpy vs Kernel 占比）

```bash
$NSYS stats --report cuda_gpu_mem_time_sum \
  --format column --output - \
  --timeunit msec test_output/nsys_eager.nsys-rep

$NSYS stats --report cuda_gpu_mem_time_sum \
  --format column --output - \
  --timeunit msec test_output/nsys_cg.nsys-rep
```

看 CG 是否减少了 Memcpy 的占比。CG 额外有 `copy_()` 更新 input buffers，但那点量远小于模型权重读取。

---

## 6. GPU 整体汇总（Kernels + MemOps 一次性）

```bash
$NSYS stats --report cuda_gpu_sum \
  --format column --output - \
  --timeunit msec test_output/nsys_eager.nsys-rep

$NSYS stats --report cuda_gpu_sum \
  --format column --output - \
  --timeunit msec test_output/nsys_cg.nsys-rep
```

3 行汇总：`[CUDA kernel]`, `[CUDA memcpy]`, `[CUDA memset]`——快速看 CG 加速来自 kernel 还是 memcpy。

---

## 7. OS Runtime 开销（每步 Python 级调度开销）

```bash
$NSYS stats --report osrt_sum \
  --format column --output - \
  --timeunit msec test_output/nsys_eager.nsys-rep

$NSYS stats --report osrt_sum \
  --format column --output - \
  --timeunit msec test_output/nsys_cg.nsys-rep
```

看 `Python` 和 `write` 等 syscall 耗时。CG 省掉了 28 层 Python for 循环的调度开销。

---

## 8. 统一输出到文件（一次性全跑）

全部报告导出到 `test_output/stats_*.csv`：

```bash
# Eager — all reports
$NSYS stats \
  --report cuda_gpu_kern_sum,cuda_gpu_kern_gb_sum,cuda_kern_exec_sum,cuda_api_sum,cuda_gpu_mem_time_sum,cuda_gpu_sum,osrt_sum \
  --format csv \
  --output test_output/stats_eager \
  test_output/nsys_eager.nsys-rep

# CG — all reports
$NSYS stats \
  --report cuda_gpu_kern_sum,cuda_gpu_kern_gb_sum,cuda_kern_exec_sum,cuda_api_sum,cuda_gpu_mem_time_sum,cuda_gpu_sum,osrt_sum \
  --format csv \
  --output test_output/stats_cg \
  test_output/nsys_cg.nsys-rep
```

---

## 关键对比指标速查

| 指标 | 想看什么 | 命令 |
|---|---|---|
| Top-5 最耗时 kernel | CG 是否去掉了 per-layer kernel | `cuda_gpu_kern_sum` |
| 总 GPU kernel 时间 | 纯计算加速比 | `cuda_gpu_sum` |
| Launch overhead | CG 是否消除了 launch 延迟 | `cuda_kern_exec_sum` |
| Kernel 数量 | Eager 比 CG 多多少 | `cuda_gpu_kern_sum` 的 Count |
| OS/Python 开销 | Python 调度循环占多少 | `osrt_sum` |

---

## GPU Trace 逐步导出

如需逐步看（某个 time range 内的所有 kernel launch）：

```bash
# 只输出前 500 行看格式
$NSYS stats --report cuda_gpu_trace \
  --format csv \
  --timeunit usec \
  --output - \
  test_output/nsys_eager.nsys-rep | head -500
```
