"""Unit test: AWQ CUDA kernel vs correct reference."""
import torch, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from awq_kernel import awq_linear_forward as kernel
R = [0, 4, 1, 5, 2, 6, 3, 7]

def ref(x, qw, qz, sc, gs):
    xf = x.float(); i, o = qw.shape[0], sc.shape[1]
    cols = []
    for j in range(o):
        nb, ci = R[j % 8], j // 8
        g = torch.arange(i, device=qw.device) // gs
        w = ((qw[:, ci] >> (nb*4)) & 0xF).float()
        z = ((qz[g, ci] >> (nb*4)) & 0xF).float()
        cols.append((w - z) * sc[g, j].float())
    return (xf @ torch.stack(cols, 1)).to(x.dtype)

def mock(b=1, i=32, o=16, gs=16):
    pf, ng = 8, i//gs; d = "cuda"
    qwr = torch.randint(0,16,(i,o),device=d)
    qzr = torch.randint(0,16,(ng,o),device=d)
    qwp = torch.zeros(i,o//pf,dtype=torch.int32,device=d)
    qzp = torch.zeros(ng,o//pf,dtype=torch.int32,device=d)
    for j in range(o):
        nb,ci = R[j%pf],j//pf
        qwp[:,ci] |= qwr[:,j].int() << (nb*4)
        qzp[:,ci] |= qzr[:,j].int() << (nb*4)
    sc = torch.randn(ng,o,device=d,dtype=torch.bfloat16).abs()*0.01
    x = torch.randn(b,i,device=d,dtype=torch.bfloat16)
    return x,qwp,qzp,sc

x,qw,qz,sc = mock()
r = ref(x,qw,qz,sc,16)
o = kernel(x,qw,qz,sc,16)
d = (r-o).abs().max().item()
print(f"  maxdiff={d:.6f}  {'✅' if d<0.1 else '❌'}KERNEL")
assert d < 0.1, f"FAIL: maxdiff={d}"
print("  PASS")
