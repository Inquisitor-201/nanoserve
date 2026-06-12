# Download Models

## tl;dr

```bash
# 国内（hf-mirror.com，默认已配好）
python scripts/download_model.py 1.7b

# 或者一行命令自己跑：
HF_ENDPOINT=https://hf-mirror.com hf download Qwen/Qwen3-1.7B --local-dir ./models/Qwen3-1.7B

# 官方 Hub（需要梯子）
hf download Qwen/Qwen3-0.6B --local-dir ./models/Qwen3-0.6B
```

## Available models

| shortcut      | HuggingFace repo                        | Size   | Quant |
|---------------|-----------------------------------------|--------|-------|
| `0.6b`        | `Qwen/Qwen3-0.6B`                       | 0.6B   | BF16  |
| `1.7b`        | `Qwen/Qwen3-1.7B`                       | 1.7B   | BF16  |
| `1.7b-awq`    | `Orion-zhen/Qwen3-1.7B-AWQ`             | 1.7B   | AWQ   |

## How the script works

`download_model.py` 是个极薄包装，把参数映射成 repo 和路径后直接调 `hf download`。等于替你敲了上面那行命令，没别的魔法。

```python
# 实际就做了这件事：
cmd = ["hf", "download", repo, "--local-dir", local_dir]
subprocess.check_call(cmd)
```

每次运行会打印真实命令，看得见在做什么，`hf download` 自带进度条。

## 接码云 ModelScope

默认配了 `HF_ENDPOINT=https://hf-mirror.com`。如果你想从 ModelScope 下，把镜像地址改了就行：

```bash
HF_ENDPOINT="" python scripts/download_model.py 0.6b
```
