# FlashInfer 验证结论

## 环境

- **GPU**: RTX 4060 Ti 16GB (sm89)
- **vLLM**: 0.8.5
- **FlashInfer**: 已安装
- **Model**: Qwen3-1.7B-AWQ (AWQ Marlin 量化)

---

## 结论

### 1. V1 引擎默认同时开启 torch.compile + CUDA Graph

```json
compilation_config={
  "level": 3,
  "use_inductor": true,     // torch.compile ✅
  "use_cudagraph": true,    // CUDA Graph ✅
  "cudagraph_capture_sizes": [512, 504, ..., 2, 1]
}
```

不需要任何额外参数。只要不传 `--enforce-eager`，两者自动全开。

### 2. V1 + FlashInfer 正常工作

```
Using FlashInfer backend on V1 engine.
```

启动命令：

```bash
VLLM_ATTENTION_BACKEND=FLASHINFER \
python -m vllm.entrypoints.openai.api_server \
    --model models/Qwen3-1.7B-AWQ \
    --port 8000 \
    --max-model-len 4096 \
    --max-num-seqs 256
```

### 3. `VLLM_TORCH_COMPILE` 环境变量不存在

vLLM 0.8.5 源码中没有 `VLLM_TORCH_COMPILE` 这个环境变量。
正确的控制方式：

| 方式 | 命令 |
|---|---|
| 快捷 (-O 级别) | `-O 3` (PIECEWISE, 推荐) |
| JSON 完整控制 | `--compilation-config '{"level": 3}'` |

`VLLM_USE_V1=0` 降级到 V0 后，`-O 3` 同样生效。

### 4. 首次编译耗时

```
Dynamo bytecode transform time: 15.26 s
Compiling a graph for general shape takes 53.61 s
```

编译产物缓存到 `~/.cache/vllm/torch_compile_cache/`，后续启动复用。

### 5. 已知小问题

```
FlashInfer>=v0.2.3 is not backward compatible.
Falling back to PyTorch-native implementation of top-p & top-k sampling.
```

FlashInfer 的 top-p/top-k sampling 因版本不兼容被禁用，回退到 PyTorch 原生实现。
不影响功能，采样正确性不变。

### 6. Model name 规则

```bash
# vLLM 用路径全名作为 model id
curl http://localhost:8000/v1/models
# → "id": "models/Qwen3-1.7B-AWQ"

# API 请求时必须用同样的全名
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "models/Qwen3-1.7B-AWQ", ...}'
# 不能用短名 "Qwen3-1.7B-AWQ"，会 404
```

### 7. 推理验证结果

```
Q: "Hello, who are you?"
A: <think>Okay, the user asked... I need to respond appropriately...
    First, I should acknowledge their greeting...</think>
```

Qwen3 的 `<think>` 块正常输出，推理功能完整。

---

## 快速启动命令（一键）

```bash
VLLM_ATTENTION_BACKEND=FLASHINFER \
python -m vllm.entrypoints.openai.api_server \
    --model models/Qwen3-1.7B-AWQ \
    --port 8000 \
    --max-model-len 4096 \
    --max-num-seqs 256 \
    --generation-config '{"temperature": 0.6, "top_p": 0.95}'
```

测试：

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "models/Qwen3-1.7B-AWQ", "messages": [{"role": "user", "content": "Hello"}], "max_tokens": 50}' | python -m json.tool
```
