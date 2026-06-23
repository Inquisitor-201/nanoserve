#!/usr/bin/env python3
import os, sys
os.environ['FLASHINFER_DISABLE_VERSION_CHECK'] = '1'
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core import LLMService, SamplingConfig

import logging
logging.basicConfig(level=logging.DEBUG, format="%(levelname)s %(name)s: %(message)s")
# Keep core at INFO unless debugging
logging.getLogger("core").setLevel(logging.INFO)
logging.getLogger("core.model_executor").setLevel(logging.DEBUG)
logging.getLogger("core.backends.flashinfer_backend").setLevel(logging.DEBUG)

PROMPT = '请输出a-z全部小写字母。不准输出任何多余字符，只准有26个。'
# PROMPT = '白日依山尽，'

llm = LLMService(model_path='./models/Qwen3-0.6B', device='cuda', block_size=16, enforce_eager=True)
out = llm.generate([PROMPT], SamplingConfig(temperature=0, top_p=1.0, max_new_tokens=32))[0]

print('out=', out)