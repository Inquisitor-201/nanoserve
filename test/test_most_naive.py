#!/usr/bin/env python3
"""Test: greedy decode must output exactly a-z 26 letters."""
import os, sys, re
os.environ['FLASHINFER_DISABLE_VERSION_CHECK'] = '1'
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core import LLMService, SamplingConfig

EXPECTED = 'abcdefghijklmnopqrstuvwxyz'
PROMPT = '请输出a-z全部小写字母。不准输出任何多余字符，只准有26个。'

llm = LLMService(model_path='./models/Qwen3-0.6B', device='cuda', block_size=16)
msg = llm.tokenizer.apply_chat_template([{'role':'user','content':PROMPT}], tokenize=False, add_generation_prompt=True)
out = llm.generate([msg], SamplingConfig(temperature=0.0, top_p=1.0, max_new_tokens=1024))[0]
out = re.sub(r'<think>.*?</think>', '', out, flags=re.DOTALL).strip()
out = re.sub(r'<\|im_start\|>.*?<\|im_end\|>', '', out, flags=re.DOTALL).strip()
letters = re.sub(r'[^a-zA-Z]', '', out).lower()
if letters == EXPECTED:
    print('PASS')
    sys.exit(0)
else:
    print(f'FAIL: got {letters!r}')
    print(f'  raw={out[:200]!r}')
    sys.exit(1)
