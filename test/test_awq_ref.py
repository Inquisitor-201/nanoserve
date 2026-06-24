"""End-to-end AWQ test: dequant all weights to bf16 once, then run."""
import torch, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.quantization.awq_linear import AWQLinear
from core import LLMService, SamplingConfig
sh = torch.tensor([0, 4, 1, 5, 2, 6, 3, 7], device="cuda") * 4

def dequant(awq):
    qw, qz, sc, gs = awq.qweight, awq.qzeros, awq.scales, awq.group_size
    i, o = qw.shape[0], sc.shape[1]
    w = ((qw[:, :, None] >> sh) & 0xF).float().reshape(i, o)
    g = torch.arange(i, device=qw.device) // gs
    z = ((qz[:, :, None] >> sh) & 0xF).float().reshape(qz.shape[0], o)
    wd = ((w - z[g]) * sc[g]).bfloat16().t().contiguous()
    lin = torch.nn.Linear(wd.shape[1], wd.shape[0], bias=False, device=wd.device, dtype=torch.bfloat16)
    lin.weight.data = torch.nn.Parameter(wd)
    return lin

llm = LLMService(model_path="./models/Qwen3-1.7B-AWQ", max_num_seqs=1, enforce_eager=True)
for l in llm.model_executor.model.layers:
    for n in ['qkv_proj', 'o_proj']:
        m = getattr(l.self_attn, n)
        if isinstance(m, AWQLinear): setattr(l.self_attn, n, dequant(m))
    for n in ['gate_proj', 'up_proj', 'down_proj']:
        m = getattr(l.mlp, n)
        if isinstance(m, AWQLinear): setattr(l.mlp, n, dequant(m))

o = llm.generate("The capital of France is", SamplingConfig(temperature=0.0, top_p=1.0, max_new_tokens=1))
print(f"Output: {o[0] if isinstance(o, list) else o}")
