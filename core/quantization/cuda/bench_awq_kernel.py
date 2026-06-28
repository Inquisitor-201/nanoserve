"""Benchmark AWQ naive kernel. Must pass test_awq_kernel first."""
import torch, sys
sys.path.insert(0, __file__.rsplit("/", 1)[0])

import test_awq_kernel
x, qw, qz, sc = test_awq_kernel.mock()
assert (test_awq_kernel.ref(x, qw, qz, sc, 16) - test_awq_kernel.kernel(x, qw, qz, sc, 16)).abs().max() < 0.1
print("  ✅ unit test passed\n")

from awq_kernel import awq_linear_forward as fn
dev = "cuda"

def data(b, i, o, gs=128):
    pf, ng = 8, i // gs
    qwr = torch.randint(0, 16, (i, o), device=dev)
    qzr = torch.randint(0, 16, (ng, o), device=dev)
    qwp = torch.zeros(i, o//pf, dtype=torch.int32, device=dev)
    qzp = torch.zeros(ng, o//pf, dtype=torch.int32, device=dev)
    for j in range(o):
        nb, ci = test_awq_kernel.R[j%pf], j//pf
        qwp[:,ci] |= qwr[:,j].int() << (nb*4)
        qzp[:,ci] |= qzr[:,j].int() << (nb*4)
    sc = torch.randn(ng, o, device=dev, dtype=torch.bfloat16).abs() * 0.01
    x = torch.randn(b, i, device=dev, dtype=torch.bfloat16)
    return x, qwp, qzp, sc

def bench(name, x, qw, qz, sc, gs=128, iters=200):
    for _ in range(20): fn(x, qw, qz, sc, gs)
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters): fn(x, qw, qz, sc, gs)
    e.record(); torch.cuda.synchronize()
    ms = s.elapsed_time(e) / iters
    print(f"  {ms:8.3f} ms  {name}")
    return ms

# Real Qwen3-1.7B-AWQ linear dims (hidden=2048, intermediate=6144)
cfgs = [
    ("gate/up decode   b=1  2048×6144", 1, 2048, 6144),
    ("gate/up prefill  b=16 2048×6144", 16, 2048, 6144),
    ("qkv decode       b=1  2048×4096", 1, 2048, 4096),
    ("qkv prefill      b=16 2048×4096", 16, 2048, 4096),
    ("down decode      b=1  6144×2048", 1, 6144, 2048),
    ("down prefill     b=16 6144×2048", 16, 6144, 2048),
    ("o decode         b=1  2048×2048", 1, 2048, 2048),
]
for n, b, i, o in cfgs:
    bench(n, *data(b, i, o))
