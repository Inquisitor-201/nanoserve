# throughput 对比 (Qwen3-0.6B, 256 seqs, 100-1024 tokens, 12 GiB GPU)

| Backend | tok/s | 条件 |
|---|---|---|
| vLLM 0.8.5 | 1476 | `enforce_eager=False` (default): CUDA graphs + full torch.compile |
| vLLM 0.8.5 | 1200 | `enforce_eager=True`: 无 compile |
| nanoserve | 1178 | `enforce_eager=True` (default): 无 compile |
| nanoserve | 1070 | MLP-only torch.compile: **反而更慢** |

## 结论

- nanoserve 与 vLLM 在同等条件下（均无 compile）性能**对齐到 2% 以内**
- vLLM 的 CUDA graphs + full torch.compile 带来约 **20%** 加速，代价是首次加载耗时 ~40s，显存 +0.46 GiB
- nanoserve 启动耗时 **~2.5s**
- **MLP-only compile 无收益**：小模型 MLP 只有 3 个 matmul，torch.compile 调用开销 > cuBLAS 本身优化
- **完整 model compile 不可行**：FlashInfer 自定义 CUDA kernel 对 torch.compile 不透明，编译崩溃
- **后续方向**：手动 CUDA graphs for decode stage（思路 2）

## 影响吞吐的核心因素

1. **调度器并发**：origin/main 的 scheduler 没有并发上限，通过 KV block 自然 backpressure 限制并发。decode 每步处理 60-100 seqs。
2. **`torch.inference_mode()`**：禁用 autograd view tracking 和 version counter，减少每层前向开销。
3. **`max_num_batched_tokens=8192`**：与 vLLM 对齐，确保 batch 大小一致。
4. **prefill fallthrough**：prefill 拉不到 block 时 fallthrough 到 decode，避免死锁。
