"""
在线 benchmark：通过 OpenAI 兼容接口并发发送请求，统计 TTFT / TPOT / 吞吐。先启动服务：

    python -m minisgl_mlx --model mlx-community/Qwen3-0.6B-bf16
    python benchmark/online.py --num-requests 64 --concurrency 32
"""

from __future__ import annotations

import argparse
import asyncio
import random
import time

import numpy as np
from openai import AsyncOpenAI


async def one_request(client, model, prompt, max_tokens, sem):
    async with sem:
        t0 = time.perf_counter()
        tics = []
        stream = await client.completions.create(
            model=model, prompt=prompt, max_tokens=max_tokens, temperature=0.0,
            stream=True, extra_body={"ignore_eos": True},
        )
        async for chunk in stream:
            if chunk.choices[0].text:
                tics.append(time.perf_counter())
        return t0, tics


async def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=1919)
    p.add_argument("--num-requests", type=int, default=64)
    p.add_argument("--concurrency", type=int, default=32)
    p.add_argument("--max-input-words", type=int, default=800)
    p.add_argument("--max-output", type=int, default=256)
    args = p.parse_args()

    random.seed(0)
    client = AsyncOpenAI(base_url=f"http://127.0.0.1:{args.port}/v1", api_key="none")
    model = (await client.models.list()).data[0].id
    words = "the quick brown fox jumps over a lazy dog while it is raining in the city".split()
    prompts = [" ".join(random.choices(words, k=random.randint(16, args.max_input_words)))
               for _ in range(args.num_requests)]
    out_lens = [random.randint(16, args.max_output) for _ in range(args.num_requests)]

    sem = asyncio.Semaphore(args.concurrency)
    t = time.perf_counter()
    results = await asyncio.gather(*[one_request(client, model, pr, n, sem) for pr, n in zip(prompts, out_lens)])
    elapsed = time.perf_counter() - t

    ttft = np.array([tics[0] - t0 for t0, tics in results if tics]) * 1e3
    tpot = np.array([(tics[-1] - tics[0]) / (len(tics) - 1) for _, tics in results if len(tics) > 1]) * 1e3
    n_out = sum(len(tics) for _, tics in results)
    print(f"requests={args.num_requests} concurrency={args.concurrency} time={elapsed:.2f}s")
    print(f"TTFT  ms: mean {ttft.mean():7.1f}  p50 {np.percentile(ttft, 50):7.1f}  p99 {np.percentile(ttft, 99):7.1f}")
    print(f"TPOT  ms: mean {tpot.mean():7.1f}  p50 {np.percentile(tpot, 50):7.1f}  p99 {np.percentile(tpot, 99):7.1f}")
    print(f"output throughput: {n_out / elapsed:.1f} chunks/s (≈ tok/s)")


if __name__ == "__main__":
    asyncio.run(main())
