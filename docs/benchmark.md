# 性能基准测试：nanoserve vs vLLM

## 测试环境

- **GPU**: RTX 3060 (12GB)
- **模型**: Qwen3-0.6B
- **输入长度**: 100–1024 tokens（随机）
- **输出长度**: 100–1024 tokens（随机）
- **请求数**: 256
- **总输出 tokens**: 133,966
- **采样参数**: temperature=0.6, ignore EOS

## 结果

| 后端 | 耗时 (s) | 吞吐 (tok/s) | vs Eager | vs vLLM | vs CG |
|---------|----------|-------------------|----------|---------|-------|
| nanoserve eager | 116.82 | **1146.81** | 1.0× | 0.78× | 0.77× |
| vLLM (FlashAttn + CUDA Graph) | 91.05 | **1471.29** | 1.28× | 1.0× | 0.99× |
| nanoserve CG (全图捕获) | 89.85 | **1491.03** | 1.30× | 1.01× | 1.0× |

## 分析

- **nanoserve CG 比 eager 快 30%** — full-forward CUDA Graph 策略消除了 Python 逐层调度开销，将每步 decode 的 kernel launch 次数从 28 次减少到 1 次。
- **nanoserve CG 与 vLLM 基本持平**（1491 vs 1471 tok/s，在噪声范围内）。修复 page-table drift 后两者性能趋同。
- **nanoserve eager 比 vLLM 慢 22%** — 符合预期，因为 vLLM 使用了 CUDA Graph 加速。

## 关键差异

| 维度 | nanoserve CG | vLLM |
|--------|-------------|------|
| Graph 策略 | 全图捕获（每 batch size 1 个图） | 逐层捕获（每 batch size × 层数个图） |
| Attention 后端 | FlashInfer（自定义 CUDA kernel） | Flash Attention（Triton） |
| 每步 Python 开销 | 0（纯 GPU） | 很小（逐层 for 循环） |
| 捕获复杂度 | 高（需固定所有中间 tensor 地址） | 中等（只需固定每层 I/O） |

## 结论

nanoserve CG 在 RTX 3060 + Qwen3-0.6B 上达到 **1491 tok/s**，与 vLLM 优化后的管线持平。Full-forward CUDA Graph 策略能有效消除 Python 调度开销，且不牺牲输出质量。
