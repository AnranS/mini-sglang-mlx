"""
第 6 章实验：观察调度器每一步在做什么 —— continuous batching、chunked prefill、准入控制。

    python tutorials/ch06_scheduler_trace.py

我们故意把参数调得很小，让各种调度行为都能在几十步内出现：
  * --max-extend-tokens=64   一个长 prompt 会被切成多个 chunk 分几轮 prefill
  * num_tokens=400           KV pool 很小，后到的请求要排队，直到前面的请求结束、释放了 KV
  * 请求的 max_tokens 各不相同 -> 有的先结束、新的请求随即补进来（continuous batching）
"""

import numpy as np

from minisgl_mlx import LLM, SamplingParams
from minisgl_mlx.scheduler.prefill import ChunkedReq

llm = LLM("mlx-community/Qwen3-0.6B-bf16", num_tokens_override=400, max_extend_tokens=64, max_running_req=8)

step = 0
orig_schedule = llm._schedule_next_batch


def traced_schedule():
    global step
    fwd = orig_schedule()
    if fwd is None:
        return None
    step += 1
    b = fwd.batch
    pending = len(llm.prefill_manager.pending_list)
    free = llm.cache_manager.available_size
    if b.is_prefill:
        desc = ", ".join(
            f"uid{r.uid}[{r.cached_len}:{r.device_len}]{'(chunk)' if isinstance(r, ChunkedReq) else ''}"
            for r in b.reqs
        )
    else:
        desc = "uids " + ",".join(str(r.uid) for r in b.reqs)
    print(f"step {step:3d} {b.phase:7s} | {desc:60s} | pending={pending} kv_free={free}")
    return fwd


llm._schedule_next_batch = traced_schedule

rng = np.random.default_rng(0)
prompts = [rng.integers(100, 5000, n).tolist() for n in (150, 40, 30, 300, 20, 25)]
max_tokens = [8, 30, 5, 10, 12, 6]
outs = llm.generate(prompts, [SamplingParams(max_tokens=m, ignore_eos=True) for m in max_tokens])
print("\n每个请求的输出长度:", [len(o["token_ids"]) for o in outs], "（应等于 max_tokens", max_tokens, "）")
llm.cache_manager.check_integrity()
print("所有请求结束后 KV 完整性检查通过：空闲页 + 前缀缓存页 == 总页数")
