"""
Profile a few decode steps with torch.profiler → Chrome trace.
Usage:
    python scripts/profile_trace.py              # CG (default)
    python scripts/profile_trace.py --eager      # Eager
    python scripts/profile_trace.py --eager --steps 10
"""
import sys, os, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF",
                      "expandable_segments:True,max_split_size_mb:512")

import torch
from random import randint, seed
from core import LLMService, SamplingConfig

seed(0)

eager = "--eager" in sys.argv
profile_steps = 3  # default: just 3 decode steps
for i, a in enumerate(sys.argv):
    if a == "--steps" and i + 1 < len(sys.argv):
        profile_steps = int(sys.argv[i + 1])

num_seqs = 8  # small batch so trace is readable
min_input_len = 10
max_input_len = 20
min_output_len = profile_steps + 5  # ensure enough decode steps
max_output_len = profile_steps + 5

model_path = os.path.abspath("./models/Qwen3-0.6B")

prompt_token_ids = [
    [randint(0, 10000) for _ in range(randint(min_input_len, max_input_len))]
    for _ in range(num_seqs)
]
output_lens = [randint(min_output_len, max_output_len)
               for _ in range(num_seqs)]

llm = LLMService(model_path=model_path, max_num_seqs=num_seqs,
                 enforce_eager=eager)

sampling_params = [
    SamplingConfig(temperature=0.6, top_p=1.0, ignore_eos=True,
                   max_new_tokens=out_len)
    for out_len in output_lens
]

import gc; gc.collect(); torch.cuda.synchronize()

tag = "eager" if eager else "cg"
# ── Profile the whole generate with torch profiler ──
with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    record_shapes=False,
    with_stack=False,
    with_flops=False,
) as prof:
    llm.generate(prompt_token_ids, sampling_params)

torch.cuda.synchronize()
trace_path = f"test_output/trace_{tag}_8seqs.json"
prof.export_chrome_trace(trace_path)
print(f"Trace exported to {trace_path}")

# Also print a quick summary
kernels = prof.key_averages(group_by_input_shape=False)
print(kernels.table(sort_by="cuda_time_total", row_limit=15))
