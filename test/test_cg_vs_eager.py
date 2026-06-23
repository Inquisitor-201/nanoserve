#!/usr/bin/env python3
"""CG vs eager: find first divergence point with identical prompts."""
import os, sys, gc, torch, re
os.environ['FLASHINFER_DISABLE_VERSION_CHECK'] = '1'
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core import LLMService, SamplingConfig

PROMPT = '请输出a-z全部小写字母。不准输出任何多余字符，只准有26个。'
CFG = SamplingConfig(temperature=0.0, top_p=1.0, max_new_tokens=100)

def run(mode):
    torch.cuda.empty_cache(); gc.collect(); torch.cuda.empty_cache()
    llm = LLMService(model_path='./models/Qwen3-0.6B', device='cuda', block_size=16, enforce_eager=(mode == 'eager'))
    msg = llm.tokenizer.apply_chat_template([{'role':'user','content':PROMPT}], tokenize=False, add_generation_prompt=True)
    out = llm.generate([msg], CFG)[0]
    out = re.sub(r'<think>.*?</think>', '', out, flags=re.DOTALL).strip()
    return out

eager_out = run('eager')
cg_out = run('cg')
print("eager_out=", eager_out)
print("cg_out=", cg_out)

# print(f"eager len={len(eager_out)}  cg len={len(cg_out)}")
# print(f"eager letters ok={re.sub(r'[^a-zA-Z]','',eager_out).lower()=='abcdefghijklmnopqrstuvwxyz'}")
# print(f"cg   letters ok={re.sub(r'[^a-zA-Z]','',cg_out).lower()=='abcdefghijklmnopqrstuvwxyz'}")
# for i, (a, b) in enumerate(zip(eager_out, cg_out)):
#     if a != b:
#         ctx = max(0, i-10)
#         print(f"First diff at char {i}: eager={eager_out[ctx:i+10]!r}  cg={cg_out[ctx:i+10]!r}")
#         break
