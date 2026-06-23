"""Wrapper: init → profiler start → generate → profiler stop → exit."""
import sys, os, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:512")

import torch
from random import randint, seed
from core import LLMService, SamplingConfig

seed(0)
num_seqs = 256
model_path = os.path.abspath("./models/Qwen3-0.6B")

prompt_token_ids = [
    [randint(0, 10000) for _ in range(randint(100, 1024))]
    for _ in range(num_seqs)
]
output_lens = [randint(100, 1024) for _ in range(num_seqs)]

eager = "--eager" in sys.argv
llm = LLMService(model_path=model_path, max_num_seqs=num_seqs,
                 enforce_eager=eager)

sampling_params = [
    SamplingConfig(temperature=0.6, top_p=1.0, ignore_eos=True,
                   max_new_tokens=out_len)
    for out_len in output_lens
]

import gc; gc.collect(); torch.cuda.synchronize()

# ── Only profile the generate phase ──
torch.cuda.cudart().cudaProfilerStart()
llm.generate(prompt_token_ids, sampling_params)
torch.cuda.synchronize()
torch.cuda.cudart().cudaProfilerStop()

total = sum(output_lens)
print(f"[{'EAGER' if eager else 'CG'}] {total} tokens")
