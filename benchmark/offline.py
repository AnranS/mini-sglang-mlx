"""
离线吞吐 benchmark（参数与原版 benchmark/offline/bench.py 对齐，规模默认缩小一些）。

    python benchmark/offline.py                                   # 默认配置
    python benchmark/offline.py --attn naive                      # 对比 attention 后端
    python benchmark/offline.py --disable-overlap-scheduling      # 对比 overlap 调度
    python benchmark/offline.py --baseline mlx-lm                 # 对比 mlx-lm 的 BatchGenerator

随机 token 作为 prompt、ignore_eos，保证每个请求恰好生成 max_tokens 个 token。
"""

from __future__ import annotations

import argparse
import time
from random import randint, seed


def make_workload(num_seqs: int, max_input: int, max_output: int):
    seed(0)
    prompts = [[randint(0, 10000) for _ in range(randint(100, max_input))] for _ in range(num_seqs)]
    out_lens = [randint(100, max_output) for _ in range(num_seqs)]
    return prompts, out_lens


def run_minisgl(args, prompts, out_lens) -> float:
    from minisgl_mlx import LLM, SamplingParams

    llm = LLM(
        args.model,
        attention_backend=args.attn,
        cache_type=args.cache,
        overlap_scheduling=args.overlap_scheduling,
        max_running_req=args.max_running_requests,
        max_extend_tokens=args.max_prefill_length,
        num_tokens_override=args.num_tokens,
    )
    llm.generate(["Benchmark warmup: "] * 4, SamplingParams(max_tokens=16))  # 编译 Metal kernel 等
    timer = None
    if args.profile:  # 第 10 章：给调度器挂上分阶段计时器
        from minisgl_mlx.profiling import PhaseTimer

        timer = PhaseTimer().attach(llm)
    sps = [SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=n) for n in out_lens]
    import mlx.core as mx

    mx.reset_peak_memory()
    t = time.perf_counter()
    outs = llm.generate(prompts, sps)
    elapsed = time.perf_counter() - t
    assert [len(o["token_ids"]) for o in outs] == out_lens
    print(f"peak memory {mx.get_peak_memory() / 2**30:.2f} GiB (KV pool {llm.engine.kv_cache.nbytes / 2**30:.2f} GiB)")
    if timer is not None:
        print(timer.summary())
    return elapsed


def run_mlx_lm(args, prompts, out_lens) -> float:
    import mlx.core as mx
    from mlx_lm import load
    from mlx_lm.generate import BatchGenerator
    from mlx_lm.sample_utils import make_sampler

    model, _ = load(args.model)

    def run(ps, ns):
        gen = BatchGenerator(
            model,
            stop_tokens=[],  # 等价于 ignore_eos
            sampler=make_sampler(temp=0.6),
            completion_batch_size=args.max_running_requests,
            prefill_batch_size=8,
        )
        gen.insert(ps, ns)
        n = 0
        while responses := gen.next_generated():
            n += len(responses)
        gen.close()
        return n

    run([[1, 2, 3]] * 4, [16] * 4)  # warmup
    mx.clear_cache()
    t = time.perf_counter()
    n = run(prompts, out_lens)
    elapsed = time.perf_counter() - t
    assert n == sum(out_lens), (n, sum(out_lens))
    return elapsed


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="mlx-community/Qwen3-0.6B-bf16")
    p.add_argument("--num-seqs", type=int, default=256)
    p.add_argument("--max-input", type=int, default=1024)
    p.add_argument("--max-output", type=int, default=1024)
    p.add_argument("--attn", default="metal", choices=["metal", "naive"])
    p.add_argument("--cache", default="radix", choices=["radix", "naive"])
    p.add_argument("--disable-overlap-scheduling", dest="overlap_scheduling", action="store_false")
    p.add_argument("--max-running-requests", type=int, default=128)
    p.add_argument("--max-prefill-length", type=int, default=4096)
    p.add_argument("--num-tokens", type=int, default=None)
    p.add_argument("--baseline", choices=["mlx-lm"], default=None)
    p.add_argument("--profile", action="store_true", help="打印调度器各阶段耗时（见 docs/10-profiling.md）")
    args = p.parse_args()

    prompts, out_lens = make_workload(args.num_seqs, args.max_input, args.max_output)
    if args.baseline == "mlx-lm":
        name, elapsed = "mlx-lm BatchGenerator", run_mlx_lm(args, prompts, out_lens)
    else:
        name = f"mini-sglang-mlx (attn={args.attn}, overlap={args.overlap_scheduling})"
        elapsed = run_minisgl(args, prompts, out_lens)
    total = sum(out_lens)
    print(
        f"[{name}] {args.num_seqs} seqs, input {sum(map(len, prompts))} tok, output {total} tok, "
        f"time {elapsed:.2f}s, output throughput {total / elapsed:.1f} tok/s"
    )


if __name__ == "__main__":
    main()
