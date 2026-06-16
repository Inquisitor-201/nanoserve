#!/usr/bin/env python3
"""
Offline throughput benchmark — side-by-side nanoserve vs vLLM.

Usage:
    python scripts/bench.py                        # nanoserve
    python scripts/bench.py --backend vllm         # vLLM
"""

import os
import sys

# Enable PyTorch memory allocator to expand segments on demand
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF",
                      "expandable_segments:True,max_split_size_mb:512")

# Ensure project root is on sys.path so `from core import ...` works
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import time
from random import randint, seed
import torch


def main():
    backend = "nanoserve"
    if "--backend" in sys.argv:
        idx = sys.argv.index("--backend")
        if idx + 1 < len(sys.argv):
            backend = sys.argv[idx + 1]

    seed(0)

    # ── Benchmark parameters ───────────────────────────────────────────
    num_seqs = 256
    min_input_len = 100
    max_input_len = 1024
    min_output_len = 100
    max_output_len = 1024
    model_path = os.path.abspath("./models/Qwen3-0.6B")

    # ── Generate random token IDs (no real text) ───────────────────────
    prompt_token_ids = [
        [randint(0, 10000) for _ in range(randint(min_input_len, max_input_len))]
        for _ in range(num_seqs)
    ]
    output_lens = [randint(min_output_len, max_output_len) for _ in range(num_seqs)]

    # ── Init model ─────────────────────────────────────────────────────
    if backend == "nanoserve":
        import logging, warnings
        warnings.filterwarnings("ignore", message=".*torch_dtype.*")
        logging.getLogger("core").setLevel(logging.WARNING)
    if backend == "vllm":
        from vllm import LLM, SamplingParams
        llm = LLM(model_path, max_num_seqs=num_seqs, max_model_len=4096)
        # vLLM expects dict format for pre-tokenized inputs
        prompts = [dict(prompt_token_ids=p) for p in prompt_token_ids]
        sampling_params = [
            SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=out_len)
            for out_len in output_lens
        ]
        gen_kwargs = dict(use_tqdm=False)
    else:
        from core import LLMService, SamplingConfig
        llm = LLMService(model_path=model_path, max_num_seqs=num_seqs)
        prompts = prompt_token_ids
        sampling_params = [
            SamplingConfig(temperature=0.6, top_p=1.0, ignore_eos=True,
                           max_new_tokens=out_len)
            for out_len in output_lens
        ]
        gen_kwargs = {}

    # ── Warmup (vLLM only — nanoserve warmup causes allocator frag) ────
    if backend == "vllm":
        print("Warming up ...")
        llm.generate(
            [dict(prompt_token_ids=[randint(0, 10000) for _ in range(64)])],
            SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=32),
            use_tqdm=False,
        )

    # ── Benchmark ──────────────────────────────────────────────────────
    import gc
    gc.collect()
    torch.cuda.synchronize()
    t = time.time()
    llm.generate(prompts, sampling_params, **gen_kwargs)
    torch.cuda.synchronize()
    t = time.time() - t

    total_tokens = sum(output_lens)
    throughput = total_tokens / t
    print(f"[{backend}] num_seqs={num_seqs}, total_output_tokens={total_tokens}, "
          f"time={t:.2f}s, throughput={throughput:.2f} tok/s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
