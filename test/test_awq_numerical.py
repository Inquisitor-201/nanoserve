#!/usr/bin/env python3
"""Compare nanoserve AWQ naive kernel output vs float32 reference for first layer q_proj."""
import os, sys
os.environ['FLASHINFER_DISABLE_VERSION_CHECK'] = '1'
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from safetensors import safe_open
from safetensors.torch import load_file

MODEL = "./models/Qwen3-1.7B-AWQ"
DEVICE = "cuda"

# ── Load raw AWQ weights for first layer q_proj ──────────────────────
print("Loading AWQ weights for layer 0 q_proj ...")
sd = load_file(f"{MODEL}/model.safetensors", device=DEVICE)

prefix = "model.layers.0.self_attn.q_proj"
qweight = sd[f"{prefix}.qweight"]   # [in_features, out_features // 8], int32
qzeros  = sd[f"{prefix}.qzeros"]    # [num_groups, out_features // 8], int32
scales  = sd[f"{prefix}.scales"]    # [num_groups, out_features], fp16

in_features  = qweight.shape[0]
out_features = scales.shape[1]
group_size   = in_features // qzeros.shape[0]
pack_factor  = 8  # 32 bits / 4 bits

print(f"  qweight: {qweight.shape} {qweight.dtype}")
print(f"  qzeros:  {qzeros.shape} {qzeros.dtype}")
print(f"  scales:  {scales.shape} {scales.dtype}")
print(f"  in_features={in_features}, out_features={out_features}, group_size={group_size}")

# ── Create random input ──────────────────────────────────────────────
torch.manual_seed(42)
x = torch.randn(1, in_features, dtype=torch.bfloat16, device=DEVICE)

# ── Reference: manual dequant + matmul in float32 ────────────────────
def ref_awq_linear(x_bf16, qweight, qzeros, scales, group_size):
    """Pure PyTorch float32 reference for AWQ dequant + linear."""
    x = x_bf16.float()  # [B, in]
    B = x.shape[0]
    in_f = qweight.shape[0]
    out_f = scales.shape[1]
    pf = 8

    # Vectorized dequant: build weight matrix in float32
    # qweight[i, j//8] → ((qweight >> (j%8)*4) & 0xF) for each i,j
    # We'll do it column-by-column for the output features
    cols = []
    for j in range(out_f):
        shift = (j % pf) * 4
        col_idx = j // pf
        w_col = ((qweight[:, col_idx] >> shift) & 0xF).float()  # [in_f]

        # Zero-point per group
        group = torch.arange(in_f, device=qweight.device) // group_size
        z_col = ((qzeros[group, col_idx] >> shift) & 0xF).float()  # [in_f]
        s_col = scales[group, j].float()  # [in_f]

        w_deq = (w_col - z_col) * s_col  # [in_f]
        cols.append(w_deq)

    w = torch.stack(cols, dim=1)  # [in_f, out_f]
    out = x @ w  # [B, out_f]
    return out

print("Computing float32 reference...")
out_ref = ref_awq_linear(x, qweight, qzeros, scales, group_size)
print(f"  ref output[0,:8]: {out_ref[0,:8]}")
print(f"  ref output sum: {out_ref.sum().item():.6f}")

# ── Nanoserve naive kernel ───────────────────────────────────────────
print("Computing nanoserve naive kernel...")
from core.quantization.cuda.awq_kernel import awq_linear_forward
out_nano = awq_linear_forward(x, qweight, qzeros, scales, group_size)  # returns bf16
out_nano_f32 = out_nano.float()
print(f"  nano output[0,:8]: {out_nano_f32[0,:8]}")
print(f"  nano output sum: {out_nano_f32.sum().item():.6f}")

# ── Compare ──────────────────────────────────────────────────────────
diff = (out_ref - out_nano_f32).abs()
print(f"\n  Max diff: {diff.max().item():.6f}")
print(f"  Mean diff: {diff.mean().item():.6f}")
print(f"  Relative max diff: {(diff / (out_ref.abs().max() + 1e-10)).max().item():.6e}")

if diff.max().item() < 0.1:
    print("  ✅ AWQ naive kernel matches float32 reference!")
else:
    print("  ❌ AWQ naive kernel MISMATCHES float32 reference!")
    idx = diff.view(-1).argmax()
    b = idx // out_features
    j = idx % out_features
    print(f"  Worst at batch={b}, out_feat={j}: ref={out_ref[b,j].item():.6f}, nano={out_nano_f32[b,j].item():.6f}")
