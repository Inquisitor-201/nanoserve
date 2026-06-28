"""Benchmark: naive vs shared-memory AWQ kernel. Must pass test_awq_kernel first."""
import torch, sys, time, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Step 1: run unit test
import test_awq_kernel
print(f"  {test_awq_kernel.__doc__}")
x_t, qw_t, qz_t, sc_t = test_awq_kernel.mock()
o = test_awq_kernel.kernel(x_t, qw_t, qz_t, sc_t, 16)
assert (test_awq_kernel.ref(x_t, qw_t, qz_t, sc_t, 16) - o).abs().max() < 0.1
print("  ✅ test_awq_kernel passed\n")

# Step 2: benchmark with realistic dims
from awq_kernel import awq_linear_forward as naive
from awq_kernel_sm import forward as sm_kernel
R = test_awq_kernel.R
dev = "cuda"

def make_data(b, i, o, gs):
    pf, ng = 8, i // gs
    qwr = torch.randint(0, 16, (i, o), device=dev)
    qzr = torch.randint(0, 16, (ng, o), device=dev)
    qwp = torch.zeros(i, o//pf, dtype=torch.int32, device=dev)
    qzp = torch.zeros(ng, o//pf, dtype=torch.int32, device=dev)
    for j in range(o):
        nb, ci = R[j%pf], j//pf
        qwp[:,ci] |= qwr[:,j].int() << (nb*4)
        qzp[:,ci] |= qzr[:,j].int() << (nb*4)
    sc = torch.randn(ng, o, device=dev, dtype=torch.bfloat16).abs() * 0.01
    x = torch.randn(b, i, device=dev, dtype=torch.bfloat16)
    return x, qwp, qzp, sc

def bench(name, fn, x, qw, qz, sc, gs, iters=200):
    # Warmup
    for _ in range(20): fn(x, qw, qz, sc, gs)
    torch.cuda.synchronize()
    # Time
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters): fn(x, qw, qz, sc, gs)
    end.record()
    torch.cuda.synchronize()
    ms = start.elapsed_time(end) / iters
    print(f"  {name:30s}  {ms:8.3f} ms")
    return ms

# Qwen3-1.7B-AWQ real linear dimensions  (hidden=2048, intermediate=6144)
cfgs = [
    ("gate/up decode  (b=1,   2048×6144)", 1,  2048, 6144, 128),
    ("gate/up prefill (b=16,  2048×6144)", 16, 2048, 6144, 128),
    ("qkv  decode     (b=1,   2048×4096)", 1,  2048, 4096, 128),
    ("qkv  prefill    (b=16,  2048×4096)", 16, 2048, 4096, 128),
    ("down decode     (b=1,  6144×2048)",  1,  6144, 2048, 128),
    ("down prefill    (b=16, 6144×2048)",  16, 6144, 2048, 128),
    ("o    decode     (b=1,   2048×2048)", 1,  2048, 2048, 128),
]

for name, b, i, o, gs in cfgs:
    print(f"\n── {name} ──")
    x, qw, qz, sc = make_data(b, i, o, gs)
    t_n = bench("naive (global load x)", naive, x, qw, qz, sc, gs)
    t_s = bench("sm    (shared mem x)",  sm_kernel, x, qw, qz, sc, gs)
    ratio = t_n / t_s
    print(f"  {'speedup':30s}  {ratio:7.2f}×  {'🚀' if ratio > 1.05 else '❌' if ratio < 0.95 else '≈'}")

# ── Summary: why SM tiling doesn't help ──────────────────────────────────
# gate/up decode:  qweight=6.3MB (97% of traffic), x=4KB (0.06%)
# x fits in L1 cache anyway → SM tiling adds sync overhead for no gain.
# The bottleneck is qweight random access: 16× more traffic than x.
print(f"\n{'='*60}")
print(f"  x is only ~0.06% of total global load traffic (gate/up 2048×6144)")
print(f"  Bottleneck: qweight (97%) — each thread reads random columns")
print(f"  Next optimization: weight tiling / pre-dequant / warp-cooperative load")
