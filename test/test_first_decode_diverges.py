#!/usr/bin/env python3
"""Reproduce: prefill logits match, first decode logits differ."""
import os, sys, gc, torch
os.environ['FLASHINFER_DISABLE_VERSION_CHECK'] = '1'
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core import LLMService, SamplingConfig

torch.cuda.empty_cache(); gc.collect(); torch.cuda.empty_cache()
llm = LLMService(model_path='./models/Qwen3-0.6B', device='cuda', block_size=16, enforce_eager=False)
msg = llm.tokenizer.apply_chat_template([{'role':'user','content':'白日依山尽，'}], tokenize=False, add_generation_prompt=True)

def run():
    rid = llm.add_requests([msg], SamplingConfig(temperature=0.0, top_p=1.0, max_new_tokens=2))
    sched = llm.scheduler.schedule()
    last_tok = torch.cumsum(torch.tensor(sched.seq_lengths, device='cuda'), dim=0) - 1
    pre_logits = llm.model_executor.execute_batch(sched.input_ids, sched.block_tables, sched.seq_lengths, True, last_tok)
    tok = llm.model_executor.sample(pre_logits, 0.0, 1.0)
    llm.scheduler.update_running_requests([t.view(1) for t in tok])
    sched2 = llm.scheduler.schedule()
    dec_logits = llm.model_executor.execute_batch(sched2.input_ids, sched2.block_tables, sched2.seq_lengths, False)
    tok2 = llm.model_executor.sample(dec_logits, 0.0, 1.0)
    llm.scheduler.update_running_requests([t.view(1) for t in tok2])
    return pre_logits, dec_logits

p1, d1 = run()
p2, d2 = run()
print(f"prefill same={torch.equal(p1,p2)}  decode1 same={torch.equal(d1,d2)}")
if not torch.equal(d1, d2):
    print(f"Run 0 decode1 top5={d1[0].topk(8).values.tolist()}")
    print(f"Run 1 decode1 top5={d2[0].topk(8).values.tolist()}")
    print("BUG: residual KV cache data makes decode non-deterministic")
    sys.exit(1)
else:
    print("PASS")
    sys.exit(0)
