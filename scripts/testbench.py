#!/usr/bin/env python3
"""
NanoServe — Continuous Batching Testbench

Two modes:
  burst     — all requests submitted at once (baseline)
  staggered — requests arrive in waves during decode (triggers continuous batching)

Usage:
  python testbench.py                              # default: Qwen3-0.6B
  python testbench.py --model 0.6b
  python testbench.py --model 1.7b
  python testbench.py --model ./models/Qwen3-1.7B
"""

import os
os.environ["FLASHINFER_DISABLE_VERSION_CHECK"] = "1"

import logging
import time
import sys
import argparse
from dataclasses import dataclass, field
from typing import List, Dict
from pathlib import Path
import torch

# Ensure project root is on sys.path so `from core import ...` works
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from core import LLMService, SamplingConfig

logging.basicConfig(level=logging.WARNING)


# ── Model resolution ─────────────────────────────────────────────────────

DEFAULT_MODEL_MAP = {
    "0.6b": "./models/Qwen3-0.6B",
    "1.7b": "./models/Qwen3-1.7B",
}


def resolve_model_path(model_arg: str) -> str:
    """Resolve short name to path, or return as-is if it's a directory."""
    if Path(model_arg).exists():
        return model_arg
    if model_arg.lower() in DEFAULT_MODEL_MAP:
        return DEFAULT_MODEL_MAP[model_arg.lower()]
    # Maybe it's a partial name match
    for key, path in DEFAULT_MODEL_MAP.items():
        if key in model_arg.lower().replace("_", "."):
            return path
    raise FileNotFoundError(
        f"Cannot resolve model '{model_arg}'. "
        f"Available shortcuts: {list(DEFAULT_MODEL_MAP.keys())}, "
        f"or provide a direct path."
    )

# ── Synthetic test dataset ────────────────────────────────────────────

SHORT_PROMPTS = [
    "中国是一个", "人工智能是", "机器学习是",
    "深度学习的核心是", "自然语言处理是", "计算机视觉是",
    "强化学习是", "数据挖掘是", "Python是一种",
    "Transformer架构是", "神经网络由", "反向传播算法是",
    "卷积神经网络", "循环神经网络", "生成对抗网络",
]

PROMPT_POOLS = {"short": SHORT_PROMPTS}

# ── Per-wave metrics ──────────────────────────────────────────────────

@dataclass
class WaveMetrics:
    wave_idx: int
    num_requests: int
    inject_step: int
    ttft_list: List[float] = field(default_factory=list)
    itl_list: List[float] = field(default_factory=list)
    gen_tokens_list: List[int] = field(default_factory=list)
    prompt_lens_list: List[int] = field(default_factory=list)

    @property
    def avg_ttft(self) -> float:
        return (sum(self.ttft_list) / len(self.ttft_list)) if self.ttft_list else 0.0
    @property
    def avg_itl(self) -> float:
        return (sum(self.itl_list) / len(self.itl_list)) if self.itl_list else 0.0
    @property
    def avg_gen_tokens(self) -> float:
        return (sum(self.gen_tokens_list) / len(self.gen_tokens_list)) if self.gen_tokens_list else 0.0
    @property
    def avg_prompt_len(self) -> float:
        return (sum(self.prompt_lens_list) / len(self.prompt_lens_list)) if self.prompt_lens_list else 0.0


# ── Helpers ───────────────────────────────────────────────────────────

def build_prompts(n: int) -> List[str]:
    pool = PROMPT_POOLS["short"]
    return [pool[i % len(pool)] for i in range(n)]


def collect_metrics(llm_service, request_ids: List[str],
                    wave_of_request: Dict[str, int],
                    n_waves: int) -> List[WaveMetrics]:
    """Collect per-wave metrics from completed requests."""
    waves = [WaveMetrics(wave_idx=i, num_requests=0, inject_step=0)
             for i in range(n_waves)]

    for rid in request_ids:
        req = llm_service.scheduler.completed_requests.get(rid)
        if req is None:
            continue
        w = wave_of_request.get(rid, 0)
        m = req.metrics
        gen_cnt = len(req.token_ids) - req.prompt_length
        waves[w].num_requests += 1
        waves[w].ttft_list.append(m.ttft)
        waves[w].itl_list.extend(m.decode_latencies)
        waves[w].gen_tokens_list.append(gen_cnt)
        waves[w].prompt_lens_list.append(req.prompt_length)

    return waves


# ── Burst mode (baseline) ─────────────────────────────────────────────

def run_burst(llm_service, n_reqs: int, sampling_config) -> tuple:
    """All requests at once. Returns (wall_time, request_ids)."""
    prompts = build_prompts(n_reqs)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    request_ids = llm_service.add_requests(prompts, sampling_config)
    llm_service.main_loop(request_ids, sampling_config)
    torch.cuda.synchronize()
    wall_time = time.perf_counter() - t0
    return wall_time, request_ids


# ── Staggered mode (continuous batching) ──────────────────────────────

def run_staggered(llm_service, n_total: int, n_waves: int,
                  sampling_config) -> tuple:
    """
    Requests arrive in equal waves, injected every `spacing` decode steps.

    Returns (wall_time, request_ids, wave_of_request).
    """
    n_per_wave = n_total // n_waves
    wave_of_request: Dict[str, int] = {}
    all_request_ids: List[str] = []
    step = 0

    # Wave 0: submit immediately
    prompts = build_prompts(n_per_wave)
    ids = llm_service.add_requests(prompts, sampling_config)
    all_request_ids.extend(ids)
    for rid in ids:
        wave_of_request[rid] = 0

    # Estimate spacing: prefill happens in 1 step, decode takes max_new_tokens steps.
    # Inject next wave at ~25% of decode to give current wave time to start decoding.
    spacing = max(1, sampling_config.max_new_tokens // (n_waves + 1))

    torch.cuda.synchronize()
    t0 = time.perf_counter()

    while llm_service.scheduler.has_unfinished_requests():
        sched_output = llm_service.scheduler.schedule()
        sampled_tokens = llm_service._run_inference_step(
            sched_output, sampling_config)
        new_tokens = [t.view(1) for t in sampled_tokens]
        active_ids = [r.request_id
                      for r in sched_output.scheduled_requests]
        llm_service.scheduler.update_running_requests(new_tokens, active_ids)
        step += 1

        # Inject next wave?
        wave_idx = step // spacing
        if 0 < wave_idx < n_waves and len([rid for rid in all_request_ids
                                           if wave_of_request.get(rid, -1) == wave_idx - 1]):
            # Check if this wave hasn't been injected yet
            already_injected = any(
                w >= wave_idx for w in wave_of_request.values())
            if not already_injected:
                prompts = build_prompts(n_per_wave)
                ids = llm_service.add_requests(prompts, sampling_config)
                all_request_ids.extend(ids)
                for rid in ids:
                    wave_of_request[rid] = wave_idx

    torch.cuda.synchronize()
    wall_time = time.perf_counter() - t0
    return wall_time, all_request_ids, wave_of_request


# ── Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="NanoServe Continuous Batching Testbench")
    parser.add_argument("--model", type=str, default="0.6b",
                       help="Model path or shortcut (0.6b, 1.7b, or direct path)")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    args = parser.parse_args()

    model_path = resolve_model_path(args.model)
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[args.dtype]

    print("\n" + "█" * 70)
    print("  nanoserve  Continuous Batching Testbench")
    print(f"  Model: {model_path}")
    print("█" * 70)

    llm_service = LLMService(
        model_path=model_path,
        device=args.device, block_size=16,
    )

    num_blocks = llm_service.cache_config.num_blocks
    kv_cap = num_blocks * llm_service.cache_config.block_size
    print(f"  KV capacity: {num_blocks} blk × {llm_service.cache_config.block_size}"
          f" = {kv_cap} tokens\n")

    sampling_config = SamplingConfig(
        temperature=0.7, top_p=0.9, max_new_tokens=256)

    # Warmup: one small burst to compile CUDA kernels
    print("  Warming up (1 round of 4 requests)...")
    warmup_prompts = build_prompts(4)
    warmup_cfg = SamplingConfig(temperature=0.7, top_p=0.9, max_new_tokens=32)
    llm_service.generate(warmup_prompts, warmup_cfg)

    # ── Test matrix ──
    configs = [
        # (burst_n, staggered_n, n_waves, label)
        (16,  16,  4,  "16 requests"),
        (32,  32,  4,  "32 requests"),
        (64,  64,  8,  "64 requests"),
        (128, 128, 8,  "128 requests"),
    ]

    for burst_n, stag_n, n_waves, label in configs:
        total_tok_needed = burst_n * (5 + 256)
        if total_tok_needed > kv_cap:
            print(f"  [skip] {label}: ~{total_tok_needed} tok > {kv_cap} capacity\n")
            continue

        # ── Burst ──
        try:
            wall_b, ids_b = run_burst(llm_service, burst_n, sampling_config)
            waves_b = collect_metrics(llm_service, ids_b,
                                      {r: 0 for r in ids_b}, 1)
            w = waves_b[0]
        except Exception as e:
            print(f"  [burst {label}] ✗ {e}")
            continue

        # ── Staggered ──
        try:
            wall_s, ids_s, wor_s = run_staggered(
                llm_service, stag_n, n_waves, sampling_config)
            waves_s = collect_metrics(llm_service, ids_s, wor_s, n_waves)
        except Exception as e:
            print(f"  [stag  {label}] ✗ {e}")
            continue

        # ── Print ──
        bar = "─" * 70

        print(f"\n{bar}")
        print(f"  {label}  |  prompt ~tokens, max_new=256")
        print(f"{bar}")
        print(f"  {'Mode':>12} | {'Wall':>7} | {'tok/s':>8} | {'req/s':>6} |"
              f" {'TTFT':>7} | {'ITL':>7}")
        print(f"  {'':>12} | {'(s)':>7} | {'':>8} | {'':>6} |"
              f" {'avg ms':>7} | {'avg ms':>7}")
        print(f"  {'-'*12} + {'-'*7} + {'-'*8} + {'-'*6} + {'-'*7} + {'-'*7}")

        # Burst line
        tok_b = sum(w.avg_prompt_len * w.num_requests
                    + sum(w.gen_tokens_list) for w in waves_b)
        print(f"  {'Burst':>12} | {wall_b:>7.3f} | {tok_b/wall_b:>8.1f} |"
              f" {burst_n/wall_b:>6.1f} | {w.avg_ttft*1000:>7.1f} |"
              f" {w.avg_itl*1000:>7.1f}")

        # Staggered lines (one per wave + total)
        tok_s = 0
        for wi, w in enumerate(waves_s):
            tok_w = w.avg_prompt_len * w.num_requests + sum(w.gen_tokens_list)
            tok_s += tok_w
            tag = f"  Wave {wi}" if wi < 3 else f"  Wave {wi}"
            t_str = f"  {tag:>12} | {wall_s:>7.3f} | {tok_w/wall_s:>8.1f} |"
            if w.num_requests > 0:
                t_str += f" {w.num_requests:>4}  | {w.avg_ttft*1000:>7.1f} | {w.avg_itl*1000:>7.1f}"
            else:
                t_str += f" {'   -':>6} | {'   -':>7} | {'   -':>7}"
            print(t_str)

        # Staggered total
        print(f"  {'──Total──':>12} | {wall_s:>7.3f} | {tok_s/wall_s:>8.1f} |"
              f" {stag_n/wall_s:>6.1f} |"
              f" {sum(w.avg_ttft*w.num_requests for w in waves_s)/stag_n*1000:>7.1f} |"
              f" {sum(w.avg_itl*len(w.itl_list) for w in waves_s)/max(1,sum(len(w.itl_list) for w in waves_s))*1000:>7.1f}")

    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
