"""
Scheduler module implementing vLLM-like scheduling for continuous batching.

This module implements a scheduler that manages requests in waiting and running lists,
and makes decisions about which requests to include in the next batch to maximize
GPU utilization while avoiding OOM conditions.
"""

"""
Scheduler module with Preemption support for preventing OOM in continuous batching.
"""

import logging
from typing import List, Dict, Optional, Deque, NamedTuple
from collections import deque
import torch
from dataclasses import dataclass
from enum import Enum

import time
from .block_manager import BlockManager
from .config import SamplingConfig, ModelConfig, SchedulerConfig


logger = logging.getLogger(__name__)

class RequestState(Enum):
    WAITING = "waiting"
    RUNNING = "running"
    PREEMPTED = "preempted" 

@dataclass
class RequestMetrics:
    """Dedicated to recording request performance profiles"""
    arrival_time: float = 0.0
    start_inference_time: float = None
    ttft: float = 0.0
    decode_latencies: List[float] = None
    
    def __post_init__(self):
        if self.decode_latencies is None:
            self.decode_latencies = []
        if self.arrival_time == 0.0:
            self.arrival_time = time.perf_counter()
    
    @property
    def total_latency(self) -> float:
        if self.start_inference_time is None:
            return 0.0
        return time.perf_counter() - self.arrival_time

@dataclass
class Request:
    request_id: str
    input_ids: torch.Tensor
    block_tables: List[int]
    computed_num_tokens: int
    sampling_config: SamplingConfig
    created_at: float = 0.0
    state: RequestState = RequestState.WAITING
    eos_token_id: int = 2
    # Profile data wrapped in a single object
    metrics: RequestMetrics = None
    # Unified storage for all token_ids
    token_ids: List[int] = None
    # Original prompt length, used to determine if it's still in prefill phase
    prompt_length: int = 0
    
    def __post_init__(self):
        if self.metrics is None:
            self.metrics = RequestMetrics()
        if self.token_ids is None:
            # Convert input_ids to list for storage during initialization
            self.token_ids = self.input_ids.tolist()
            self.prompt_length = len(self.token_ids)
    
    @property
    def is_prefill(self):
        # If token_ids length equals original prompt length, it means prefill hasn't finished yet
        return len(self.token_ids) == self.prompt_length
    
    def get_last_token(self) -> int:
        return self.token_ids[-1]
    
    def add_token(self, token: int):
        self.token_ids.append(token)
    
    def is_finished(self) -> bool:
        if not self.sampling_config.ignore_eos and self.token_ids and self.token_ids[-1] == self.eos_token_id:
            return True
        # Number of generated tokens equals total tokens minus original prompt length
        generated_tokens_count = len(self.token_ids) - self.prompt_length
        return generated_tokens_count >= self.sampling_config.max_new_tokens
    
    def get_last_token(self) -> int:
        return self.token_ids[-1]

class SchedulerOutput(NamedTuple):
    is_prefill: bool
    scheduled_requests: List[Request]
    running_requests: List[Request]
    blocks_needed: List[int]
    input_ids: torch.Tensor
    block_tables: List[List[int]]
    seq_lengths: List[int]

class Scheduler:
    def __init__(
        self,
        scheduler_config: SchedulerConfig,
        block_manager: BlockManager,
    ):
        """
        Initialize Scheduler with config-based parameters.
        
        Args:
            scheduler_config: SchedulerConfig containing scheduling parameters
            block_manager: BlockManager instance for KV cache allocation
        """
        self.scheduler_config = scheduler_config
        self.block_manager = block_manager
        self.waiting_list: Deque[Request] = deque()
        self.running_list: Deque[Request] = deque()
        self.completed_requests: Dict[str, Request] = {}
        self.request_id_counter = 0

    def add_request(
        self,
        input_ids: torch.Tensor,
        sampling_config: SamplingConfig,
        eos_token_id: int = 2,
    ) -> str:
        request_id = f"req_{self.request_id_counter}"
        self.request_id_counter += 1
        
        request = Request(
            request_id=request_id,
            input_ids=input_ids,
            block_tables=[],
            computed_num_tokens=0,
            sampling_config=sampling_config,
            eos_token_id=eos_token_id
        )
        self.waiting_list.append(request)
        return request_id

    def _preempt_request(self):
        """Preempt the youngest request (from the end of running_list)."""
        assert self.running_list, "No running requests to preempt"
        victim = self.running_list.pop()
        self.block_manager.free_blocks(victim.block_tables)
        victim.block_tables = []
        victim.computed_num_tokens = 0
        # Preserve token_ids so that prefill can start from "prompt + generated tokens" when rescheduling
        # victim.generated_tokens = []
        victim.state = RequestState.PREEMPTED
        self.waiting_list.appendleft(victim) # Priority resume

    def _get_decode_resource_requirement(self) -> int:
        """Calculate how many NEW blocks are needed for the current running_list."""
        new_blocks_needed = 0
        for req in self.running_list:
            # We assume BlockManager has 'needs_new_block' or we check manually
            # Example logic: if (current_tokens + 1) exceeds current_blocks * block_size
            if self.block_manager.needs_new_block(req.block_tables, req.computed_num_tokens + 1):
                new_blocks_needed += 1
        return new_blocks_needed

    def schedule(self) -> SchedulerOutput:
        # --- Phase 1: PREFILL (Priority) ---
        # If prefill can't schedule anything (blocks exhausted), fall
        # through to decode rather than returning empty.
        if self.waiting_list:
            out = self._schedule_prefill()
            if out.scheduled_requests:
                return out

        # --- Phase 2: DECODE ---
        if self.running_list:
            return self._schedule_decode()

        return self._build_output([], [], [], [])

    def _schedule_prefill(self) -> SchedulerOutput:
        """Handle prefill requests with budget-based scheduling."""
        scheduled = []
        # In prefill, we don't preempt running requests usually, 
        # but we only start if enough blocks exist.
        while self.waiting_list:
            req = self.waiting_list[0]
            # Use unified token_ids storage
            blocks = self.block_manager.allocate_blocks([], len(req.token_ids))
            
            if blocks:
                req = self.waiting_list.popleft()
                req.block_tables = blocks
                req.state = RequestState.RUNNING
                scheduled.append(req)
                self.running_list.append(req)
            else:
                break
        
        if not scheduled:
            # If no prefill can start, try to decode existing ones instead
            return self._schedule_decode() if self.running_list else self._build_output([], [], [], [])
        
        # Build output using unified token_ids
        input_ids_list = []
        seq_lengths = []
        for r in scheduled:
            input_ids_list.extend(r.token_ids)
            seq_lengths.append(len(r.token_ids))

        return self._build_output(
            scheduled, 
            input_ids_list,
            [r.block_tables for r in scheduled],
            seq_lengths,
            is_prefill=True
        )

    def _schedule_decode(self) -> SchedulerOutput:
        """Decode phase with global estimation to prevent thrashing."""
        # 1. Estimation: How many blocks do we need for EVERYONE to take one step?
        needed = self._get_decode_resource_requirement()
        
        # 2. Batch Preemption: Evict until the requirement is met
        while self.block_manager.num_free_blocks < needed and len(self.running_list) > 0:
            if self.block_manager.needs_new_block(self.running_list[-1].block_tables, self.running_list[-1].computed_num_tokens + 1):
                needed -= 1
            self._preempt_request()

        # 3. Execution: Now it's safe to allocate
        scheduled = []
        ids, tables, lens = [], [], []
        
        for req in list(self.running_list):
            new_tables = self.block_manager.allocate_blocks(req.block_tables, req.computed_num_tokens + 1)
            if new_tables:
                req.block_tables = new_tables
                ids.append(req.get_last_token())
                tables.append(req.block_tables)
                lens.append(req.computed_num_tokens + 1)
                scheduled.append(req)
            else:
                # This should theoretically not happen due to the while-loop above
                raise RuntimeError(f"Critical error: OOM for {req.request_id} despite preemption.")

        return self._build_output(scheduled, ids, tables, lens, is_prefill=False)

    def _build_output(self, reqs, ids, tables, lens, is_prefill=False) -> SchedulerOutput:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return SchedulerOutput(
            is_prefill=is_prefill,
            scheduled_requests=reqs,
            running_requests=list(self.running_list),
            blocks_needed=[b for r in self.running_list for b in r.block_tables],
            input_ids=torch.tensor(ids, dtype=torch.long, device=device) if ids else torch.empty(0, dtype=torch.long, device=device),
            block_tables=tables,
            seq_lengths=lens
        )

    def update_running_requests(self, new_token_ids: List[torch.Tensor], request_ids: List[str] = None):
        # Survival filtering logic (as previously optimized)
        if request_ids is not None:
            token_update_map = {rid: token for rid, token in zip(request_ids, new_token_ids)}
        else:
            token_update_map = {req.request_id: token for req, token in zip(self.running_list, new_token_ids)}

        new_running_list = deque()
        updated_requests = []

        while self.running_list:
            request = self.running_list.popleft()
            req_id = request.request_id

            if req_id in token_update_map:
                new_token_raw = token_update_map[req_id]
                new_token = new_token_raw.item() if isinstance(new_token_raw, torch.Tensor) else int(new_token_raw)
                
                # Use unified token_ids storage
                request.add_token(new_token)
                request.computed_num_tokens = len(request.token_ids)
                updated_requests.append(request)

                if request.is_finished():
                    self.completed_requests[req_id] = request
                    self.block_manager.free_blocks(request.block_tables)
                    request.block_tables = []
                    continue
            
            new_running_list.append(request)

        self.running_list = new_running_list

        return updated_requests

    def has_unfinished_requests(self) -> bool:
        return len(self.waiting_list) > 0 or len(self.running_list) > 0