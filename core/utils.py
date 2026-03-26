import time
import torch
from typing import List

class StatsCollector:
    def __init__(self):
        self.ttft = 0.0
        self.itl_list = []  # 存每一次 decode 的耗时

    def add_record(self, latency, is_prefill):
        if is_prefill:
            self.ttft = latency
        else:
            self.itl_list.append(latency)

    def get_stats(self):
        avg_itl = sum(self.itl_list) / len(self.itl_list) if self.itl_list else 0
        return {
            "ttft": self.ttft,
            "avg_itl": avg_itl,
            "total_tokens": len(self.itl_list) + 1  # +1 for prefill
        }

class ProfileTimer:
    def __init__(self, collector, is_prefill):
        self.collector = collector
        self.is_prefill = is_prefill

    def __enter__(self):
        # 翻转沙漏：开始计时
        torch.cuda.synchronize() # 保证 GPU 之前的活干完了
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # 沙漏流完：结束计时
        torch.cuda.synchronize() # 保证 GPU 刚干的活干完了
        latency = time.perf_counter() - self.start
        
        # 【核心步骤】：把结果传给账本
        self.collector.add_record(latency, self.is_prefill)

class ContinuousBatchTimer:
    def __init__(self, active_requests: List):
        self.active_requests = active_requests

    def __enter__(self):
        torch.cuda.synchronize()
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        torch.cuda.synchronize()
        step_latency = time.perf_counter() - self.start
        
        # 核心逻辑：给这轮所有的 Request 记账
        for req in self.active_requests:
            m = req.metrics  # 获取账本
            if req.is_prefill:
                # 如果是 prefill 状态，这一轮的耗时就是它的 TTFT
                m.ttft = time.perf_counter() - m.arrival_time
                m.start_inference_time = time.perf_counter()
            else:
                # 如果是 decode 状态，记入 ITL 列表
                m.decode_latencies.append(step_latency)
