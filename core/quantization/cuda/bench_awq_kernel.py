"""Benchmark naive vs SM AWQ kernel. Runs test first."""
import torch, sys
sys.path.insert(0, __file__.rsplit("/", 1)[0])
import test_awq_kernel
x_t, qw_t, qz_t, sc_t = test_awq_kernel.mock()
r = test_awq_kernel.ref(x_t, qw_t, qz_t, sc_t, 16)
for fn in [test_awq_kernel.naive, test_awq_kernel.sm]:
    assert (r - fn(x_t, qw_t, qz_t, sc_t, 16)).abs().max() < 0.1
print("  ✅ tests passed\n")

from awq_kernel_naive import awq_linear_forward as naive
from awq_kernel import forward as sm

device, gs = "cuda", 128
def data(b, i, o):
    pf, ng = 8, i//gs
    qwr = torch.randint(0,16,(i,o),device=device)
    qzr = torch.randint(0,16,(ng,o),device=device)
    qwp = torch.zeros(i,o//pf,dtype=torch.int32,device=device)
    qzp = torch.zeros(ng,o//pf,dtype=torch.int32,device=device)
    for j in range(o):
        nb,ci = test_awq_kernel.R[j%pf],j//pf
        qwp[:,ci] |= qwr[:,j].int()<<(nb*4)
        qzp[:,ci] |= qzr[:,j].int()<<(nb*4)
    sc = torch.randn(ng,o,device=device,dtype=torch.bfloat16).abs()*0.01
    x = torch.randn(b,i,device=device,dtype=torch.bfloat16)
    return x,qwp,qzp,sc

def bench(fn, x, qw, qz, sc, iters=200):
    if x.size(0) >= 256: iters = 50
    for _ in range(20): fn(x,qw,qz,sc,gs)
    torch.cuda.synchronize()
    s,e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters): fn(x,qw,qz,sc,gs)
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e)/iters

kernels = [("naive", naive), ("sm", sm)]
cfgs = [("gate/up",1,2048,6144),("gate/up",16,2048,6144),("gate/up",256,2048,6144),("qkv",1,2048,4096),("qkv",16,2048,4096),("qkv",256,2048,4096),("down",1,6144,2048),("down",16,6144,2048),("down",256,6144,2048),("o",1,2048,2048),("o",256,2048,2048)]

hdr = f"  {'layer':<22} {'bs':>3} {'dims':<13}"
for n,_ in kernels: hdr += f" {n+'(ms)':>10}"
hdr += f" {'speedup':>8}"
print(hdr)
print(f"  {'─'*22} {'─'*3} {'─'*13} {'─'*10} {'─'*10} {'─'*8}")

for name,b,i,o in cfgs:
    d = data(b,i,o); times = []
    line = f"  {name:<22} {b:>3} {i}x{o:<9}"
    for _,fn in kernels: times.append(bench(fn,*d)); line += f" {times[-1]:>9.3f}"
    r = times[0]/times[1]
    tag = " 🚀" if r>1.05 else " ❌" if r<0.95 else " ≈"
    print(f"{line} {r:>6.2f}×{tag}")
