#!/usr/bin/env python3
"""Eager with torch.zeros() KV cache: output should be deterministic."""
import os, sys, re
os.environ['FLASHINFER_DISABLE_VERSION_CHECK'] = '1'
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core import LLMService, SamplingConfig

llm = LLMService(model_path='./models/Qwen3-0.6B', device='cuda', block_size=16, enforce_eager=True)
msg = llm.tokenizer.apply_chat_template([{'role':'user','content':'请输出a-z全部小写字母。不准输出任何多余字符，只准有26个。'}], tokenize=False, add_generation_prompt=True)
out = llm.generate([msg], SamplingConfig(temperature=0.0, top_p=1.0, max_new_tokens=512))[0]
out = re.sub(r'<think>.*?</think>', '', out, flags=re.DOTALL).strip()
letters = re.sub(r'[^a-zA-Z]', '', out).lower()
print(f'letters={letters!r}')
if letters == 'abcdefghijklmnopqrstuvwxyz':
    print('PASS')
else:
    print('FAIL')
