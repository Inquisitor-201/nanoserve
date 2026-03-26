import time
import torch
from typing import List

class StatsCollector:
    def __init__(self):
        self.ttft = 0.0
        self.itl_list = []  # Store each decode latency

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
        # Flip the hourglass: start timing
        torch.cuda.synchronize() # Ensure GPU has finished previous work
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Hourglass finished: end timing
        torch.cuda.synchronize() # Ensure GPU has finished current work
        latency = time.perf_counter() - self.start
        
        # [Core step]: Pass the result to the ledger
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
        
        # Core logic: Record time for all requests in this batch
        for req in self.active_requests:
            m = req.metrics  # Get ledger
            if req.is_prefill:
                # If in prefill state, this batch's latency is its TTFT
                m.ttft = time.perf_counter() - m.arrival_time
                m.start_inference_time = time.perf_counter()
            else:
                # If in decode state, add to ITL list
                m.decode_latencies.append(step_latency)
