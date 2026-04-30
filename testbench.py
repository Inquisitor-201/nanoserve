#!/usr/bin/env python3
import os
os.environ["FLASHINFER_DISABLE_VERSION_CHECK"] = "1"
"""
Offline inference testbench for nanoserve.

Measures throughput and latency under varying request counts.
The scheduler internally decides the actual GPU batch size via continuous batching.
"""

import logging
import time
import sys
from dataclasses import dataclass, field
from typing import List
import torch

from core import LLMService, SamplingConfig, EngineArgs

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# ── Synthetic test dataset ────────────────────────────────────────────

SHORT_PROMPTS = [
    "中国是一个", "人工智能是", "机器学习是",
    "深度学习的核心是", "自然语言处理是", "计算机视觉是",
    "强化学习是", "数据挖掘是", "Python是一种",
    "Transformer架构是", "神经网络由", "反向传播算法是",
    "卷积神经网络", "循环神经网络", "生成对抗网络",
]

MEDIUM_PROMPTS = [
    "解释一下C++和Python的主要区别，以及各自的应用场景。",
    "请详细说明云计算和边缘计算的区别，以及各自的优缺点。",
    "什么是区块链技术？请说明其基本原理和主要应用场景。",
    "解释一下TCP/IP协议的层次结构，以及每层的主要功能。",
]

LONG_PROMPTS = [
    "你是一个精通逻辑推理的数学助手。在回答任何数学问题之前，你必须遵循以下步骤："
    "1. 提取题目中的关键数字和条件；2. 分步骤列出计算过程，每一步只做一个简单的运算；"
    "3. 最后给出最终结果。请务必保持逻辑严密，不要跳步。"
    "请你计算：小红买了3个苹果，单价12元；又买了2个梨，单价16元。"
    "她给了老板68元，应该找回多少钱？",
    "你是一个资深的编程导师。请根据以下要求编写一个Python函数："
    "1. 函数名为fibonacci；2. 接收一个整数n作为参数；"
    "3. 返回斐波那契数列的第n项；4. 要求使用动态规划方法实现；"
    "5. 添加适当的类型注解和文档字符串。请给出完整的代码实现。",
]

PROMPT_POOLS = {
    "short":  SHORT_PROMPTS,
    "medium": MEDIUM_PROMPTS,
    "long":   LONG_PROMPTS,
}


# ── Benchmark types ───────────────────────────────────────────────────

@dataclass
class BenchmarkConfig:
    """Configuration for a single benchmark run."""
    name: str
    num_requests: int             # total requests submitted via add_requests()
    prompt_type: str = "medium"   # short / medium / long
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    warmup_rounds: int = 1        # warmup rounds before timed run


@dataclass
class RunMetrics:
    """Profiling results from one benchmark run."""
    num_requests: int
    total_time: float               # wall-clock of main_loop()
    total_prompt_tokens: int = 0
    total_generated_tokens: int = 0
    throughput_tok_s: float = 0.0
    throughput_req_s: float = 0.0

    ttft_list: List[float] = field(default_factory=list)
    itl_list: List[float] = field(default_factory=list)
    total_latency_list: List[float] = field(default_factory=list)
    gen_tokens_list: List[int] = field(default_factory=list)
    prompt_lens_list: List[int] = field(default_factory=list)
    n_prefill: int = 0
    n_decode: int = 0

    # ── computed properties ──
    @property
    def avg_ttft(self) -> float:
        return (sum(self.ttft_list) / len(self.ttft_list)) if self.ttft_list else 0.0

    @property
    def avg_itl(self) -> float:
        return (sum(self.itl_list) / len(self.itl_list)) if self.itl_list else 0.0

    @property
    def avg_total_latency(self) -> float:
        return (sum(self.total_latency_list) / len(self.total_latency_list)
                if self.total_latency_list else 0.0)

    @property
    def total_requests(self) -> int:
        return len(self.ttft_list)

    def print_detail(self):
        bar = "─" * 72
        print(f"\n{bar}")
        print(f"  Requests submitted: {self.num_requests}  |  "
              f"Requests completed: {self.total_requests}")
        print(f"  Prompt: avg {self._avg_or_0(self.prompt_lens_list):.0f} tok  →  "
              f"Generate: avg {self._avg_or_0(self.gen_tokens_list):.0f} tok")
        print(f"{bar}")
        print(f"  Wall time:            {self.total_time:>8.3f} s")
        print(f"  Throughput:           {self.throughput_tok_s:>8.1f} tokens/s  "
              f"({self.throughput_req_s:>5.1f} req/s)")
        print(f"  ── Latency ──")
        print(f"  TTFT  (avg):          {self.avg_ttft * 1000:>8.1f} ms")
        print(f"  ITL   (avg):          {self.avg_itl * 1000:>8.1f} ms")
        if self.ttft_list:
            print(f"  TTFT (min/max):       {min(self.ttft_list)*1000:.1f} / "
                  f"{max(self.ttft_list)*1000:.1f} ms")
        print(f"  ── Steps ──")
        print(f"  Prefill steps:        {self.n_prefill:>8}")
        print(f"  Decode  steps:        {self.n_decode:>8}")
        print(f"{bar}")

    @staticmethod
    def _avg_or_0(lst: List[float]) -> float:
        return sum(lst) / len(lst) if lst else 0.0


# ── Helpers ───────────────────────────────────────────────────────────

def build_prompts(n: int, prompt_type: str) -> List[str]:
    pool = PROMPT_POOLS.get(prompt_type, PROMPT_POOLS["medium"])
    return [pool[i % len(pool)] for i in range(n)]


def count_prefill_decode_steps(llm_service: LLMService,
                                request_ids: List[str]) -> tuple:
    """Count prefill and decode steps by inspecting scheduler output."""
    n_pre = 0
    n_dec = 0
    # Hack: re-run a lightweight trace by peeking at completed_requests'
    # decode_latencies count = decode steps, ttft = 1 prefill step.
    for rid in request_ids:
        req = llm_service.scheduler.completed_requests.get(rid)
        if req is None:
            continue
        m = req.metrics
        if m.ttft > 0:
            n_pre += 1
        n_dec += len(m.decode_latencies)
    return n_pre, n_dec


# ── Single benchmark runner ───────────────────────────────────────────

def run_benchmark(
    llm_service: LLMService,
    cfg: BenchmarkConfig,
) -> RunMetrics:
    """Submit `num_requests` prompts, run the scheduler loop, collect metrics."""
    prompts = build_prompts(cfg.num_requests, cfg.prompt_type)
    sampling_config = SamplingConfig(
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        max_new_tokens=cfg.max_new_tokens,
    )

    # ── Warmup ──
    for _ in range(cfg.warmup_rounds):
        llm_service.generate(prompts, sampling_config)

    # ── Timed run ──
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    request_ids = llm_service.add_requests(prompts, sampling_config)
    llm_service.main_loop(request_ids, sampling_config)

    torch.cuda.synchronize()
    wall_time = time.perf_counter() - t0

    # ── Collect per-request metrics ──
    m = RunMetrics(num_requests=cfg.num_requests, total_time=wall_time)

    total_prompt = 0
    total_gen = 0

    for rid in request_ids:
        req = llm_service.scheduler.completed_requests.get(rid)
        if req is None:
            continue
        r_metrics = req.metrics
        gen_cnt = len(req.token_ids) - req.prompt_length

        m.ttft_list.append(r_metrics.ttft)
        m.itl_list.extend(r_metrics.decode_latencies)
        m.total_latency_list.append(r_metrics.total_latency)
        m.gen_tokens_list.append(gen_cnt)
        m.prompt_lens_list.append(req.prompt_length)

        total_prompt += req.prompt_length
        total_gen += gen_cnt

    m.total_prompt_tokens = total_prompt
    m.total_generated_tokens = total_gen
    m.throughput_tok_s = (total_prompt + total_gen) / wall_time
    m.throughput_req_s = cfg.num_requests / wall_time
    m.n_prefill, m.n_decode = count_prefill_decode_steps(llm_service, request_ids)

    return m


# ── Main ──────────────────────────────────────────────────────────────

def main():
    print("\n" + "█" * 72)
    print("  nanoserve  Offline Inference Testbench")
    print("  Measures throughput & latency under varying load")
    print("█" * 72)

    # ── Engine (num_blocks=None → auto-calculate from GPU memory) ──
    engine_args = EngineArgs(
        model_path="./models/Qwen3-0.6B",
        device="cuda",
        block_size=16,
        dtype=torch.bfloat16,
    )

    print(f"\n  Model:        {engine_args.model_path}")
    print(f"  Device:       {engine_args.device}")

    # ── Load model ──
    print("\n  Loading model...", end=" ", flush=True)
    t0 = time.perf_counter()
    llm_service = LLMService.from_engine_args(engine_args)
    torch.cuda.synchronize()
    print(f"done ({time.perf_counter() - t0:.2f}s)")

    num_blocks = llm_service.cache_config.num_blocks
    block_size = llm_service.cache_config.block_size
    kv_capacity = num_blocks * block_size
    print(f"  KV cache:     {num_blocks} blocks × {block_size} tok/block"
          f"  = {kv_capacity} token capacity")

    if torch.cuda.is_available():
        total_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        used_mem = torch.cuda.memory_allocated(0) / 1024**3
        peak_mem = torch.cuda.max_memory_allocated(0) / 1024**3
        print(f"  GPU memory:   {used_mem:.2f} / {total_mem:.2f} GiB  "
              f"(peak so far: {peak_mem:.2f} GiB)")

    # ── Test configurations ──
    request_counts = [1, 2, 4, 8, 16, 32, 64]
    prompt_types = ["short", "medium", "long"]
    max_new_tokens = 256

    for ptype in prompt_types:
        print(f"\n{'=' * 72}")
        sample = PROMPT_POOLS[ptype][0]
        print(f"  Prompt type: {ptype}  ({sample[:60]}{'…' if len(sample)>60 else ''})")
        print(f"  Max new tokens per request: {max_new_tokens}")
        print(f"  KV capacity: {kv_capacity} tokens  "
              f"(→ max theoretical requests ≈ {kv_capacity // (20 + max_new_tokens)})")
        print(f"{'=' * 72}")

        results: List[RunMetrics] = []

        for n_req in request_counts:
            # Skip if tokens needed exceed KV capacity
            # rough estimate: prompt ~20tok avg + max_new_tokens per request
            tok_needed = n_req * (20 + max_new_tokens)
            if tok_needed > kv_capacity:
                print(f"\n  [skip] {n_req} requests: ~{tok_needed} tokens needed"
                      f" > {kv_capacity} capacity")
                continue

            cfg = BenchmarkConfig(
                name=f"{ptype}_n{n_req}",
                num_requests=n_req,
                prompt_type=ptype,
                max_new_tokens=max_new_tokens,
                warmup_rounds=1 if n_req <= 32 else 0,
            )

            try:
                print(f"\n  ● {n_req} requests ...", end=" ", flush=True)
                m = run_benchmark(llm_service, cfg)
                results.append(m)
                m.print_detail()
            except Exception as e:
                print(f"  ✗ Error: {e}")
                import traceback
                traceback.print_exc()
                break

            # Check OOM
            if torch.cuda.is_available():
                used = torch.cuda.memory_allocated(0) / 1024**3
                print(f"  GPU memory now: {used:.2f} GiB")
                if used > 10.5:
                    print("  ⚠  GPU nearly full, stopping this prompt type")
                    break

        # ── Summary table ──
        if results:
            print(f"\n  ── Summary: {ptype} ──")
            print(f"  {'#Req':>5} | {'Throughput':>10} | {'Throughput':>7} |"
                  f" {'TTFT':>7} | {'ITL':>7} | {'Steps':>8} | {'Wall':>7}")
            print(f"  {'':>5} | {'tokens/s':>10} | {'req/s':>7} |"
                  f" {'avg ms':>7} | {'avg ms':>7} | {'P/D':>8} | {'(s)':>7}")
            print(f"  {'-'*5} + {'-'*10} + {'-'*7} + {'-'*7} + {'-'*7} + {'-'*8} + {'-'*7}")
            for m in results:
                print(f"  {m.num_requests:>5} | {m.throughput_tok_s:>10.1f} |"
                      f" {m.throughput_req_s:>7.1f} | {m.avg_ttft*1000:>7.1f} |"
                      f" {m.avg_itl*1000:>7.1f} | {m.n_prefill}/{m.n_decode:<4} |"
                      f" {m.total_time:>7.3f}")

    print_footer()
    return 0


def print_footer():
    print("\n" + "█" * 72)
    print("  Benchmark complete.")
    print("█" * 72 + "\n")


if __name__ == "__main__":
    sys.exit(main())
