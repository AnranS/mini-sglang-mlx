"""
第 7 章实验：overlap 调度。

    python tutorials/ch07_overlap.py

Part 1：复现"buffer 捐赠失败"——同样是重叠 CPU 与 GPU，提交时机不同，性能差 10 倍以上。
Part 2：在真实模型上对比 overlap 开 / 关，观察 batch 越小收益越大。
"""

import time

import mlx.core as mx
import numpy as np


def part1():
    print("=" * 15, "Part 1: overlap 的两种写法与 buffer 捐赠", "=" * 15)
    N, H, D = 1_000_000, 8, 128  # 2 GB 的 "KV pool"
    val = mx.ones((64, H, D), dtype=mx.bfloat16)

    def step(i, pool):
        pool[mx.arange(i * 64, (i + 1) * 64)] = val  # 写 KV（scatter）
        read = pool[mx.arange(0, 4096)]  # 读 KV（attention）
        return pool, read.astype(mx.float32).sum()

    def fresh():
        p = mx.zeros((N, H, D), dtype=mx.bfloat16)
        mx.eval(p)
        return p

    # 写法 A：构图后立刻 async_eval，再等上一步（mlx-lm generate_step 的写法，也是原版 CUDA 的思路）
    pool = fresh()
    pool, s = step(0, pool)
    mx.async_eval(s)
    t = time.perf_counter()
    for i in range(1, 51):
        pool, s_next = step(i, pool)
        mx.async_eval(s_next)  # 此时上一步对 pool 的"读"可能还在 GPU 上进行 -> pool 不能被捐赠
        s.item()
        s = s_next
    s.item()
    print(f"A. 构图 -> 立即提交 -> 等上一步:   {(time.perf_counter() - t) / 50 * 1e3:6.3f} ms/step")

    # 写法 B：构图 -> 等上一步 -> 提交（本项目 scheduler.overlap_loop 的写法）
    pool = fresh()
    pool, s = step(0, pool)
    mx.async_eval(s)
    t = time.perf_counter()
    for i in range(1, 51):
        pool, s_next = step(i, pool)  # 构图仍然与上一步的 GPU 计算重叠
        s.item()  # 上一步完成，pool 不再被任何 GPU 任务引用
        mx.async_eval(s_next)  # 现在提交，scatter 可以直接复用 pool 的 buffer
        s = s_next
    s.item()
    print(f"B. 构图 -> 等上一步 -> 再提交:     {(time.perf_counter() - t) / 50 * 1e3:6.3f} ms/step")
    print("=> A 每一步都把 2GB 的 pool 拷贝了一遍；B 是真正的原地写入。")


def part2():
    print("\n" + "=" * 15, "Part 2: 真实模型上的 overlap 收益", "=" * 15)
    from minisgl_mlx import LLM, SamplingParams

    rng = np.random.default_rng(0)
    for overlap in [False, True]:
        llm = LLM("mlx-community/Qwen3-0.6B-bf16", overlap_scheduling=overlap, num_tokens_override=65536)
        llm.generate(["warmup"] * 4, SamplingParams(max_tokens=8))
        for B in [1, 4, 16, 64]:
            prompts = [rng.integers(0, 10000, 200).tolist() for _ in range(B)]
            t = time.perf_counter()
            llm.generate(prompts, SamplingParams(max_tokens=256, ignore_eos=True))
            elapsed = time.perf_counter() - t
            print(f"overlap={overlap!s:5} batch={B:3d}: {B * 256 / elapsed:8.1f} tok/s")
        llm.shutdown()


if __name__ == "__main__":
    part1()
    part2()
