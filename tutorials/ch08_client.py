"""
第 8 章实验：作为 OpenAI 兼容服务使用。先在另一个终端启动服务：

    python -m minisgl_mlx --model mlx-community/Qwen3-0.6B-bf16

然后运行：

    python tutorials/ch08_client.py

观察点：
  1. 流式输出：中文/emoji 不会出现乱码（感谢增量 detokenize）
  2. 并发：8 个请求同时发出，服务端 continuous batching，总耗时远小于 8 倍单请求耗时
  3. 中途断开：客户端 break 后，服务端日志会出现 "Abort request"，KV 立即被释放
"""

import asyncio
import time

from openai import AsyncOpenAI

client = AsyncOpenAI(base_url="http://127.0.0.1:1919/v1", api_key="none")


async def chat(content: str, max_tokens: int = 128, show: bool = False):
    t0 = time.perf_counter()
    ttft, text = None, ""
    stream = await client.chat.completions.create(
        model="default",
        messages=[{"role": "user", "content": content}],
        max_tokens=max_tokens,
        temperature=0.7,
        stream=True,
    )
    async for chunk in stream:
        delta = chunk.choices[0].delta.content or ""
        if delta and ttft is None:
            ttft = time.perf_counter() - t0
        text += delta
        if show:
            print(delta, end="", flush=True)
    return text, ttft, time.perf_counter() - t0


async def main():
    print("1) 单个流式请求：")
    _, ttft, total = await chat("用三句话介绍一下 Apple Silicon 的统一内存。/no_think", show=True)
    print(f"\n   TTFT {ttft * 1e3:.0f} ms, 总耗时 {total:.2f}s")

    print("\n2) 8 个并发请求：")
    t = time.perf_counter()
    results = await asyncio.gather(*[chat(f"写一句关于数字 {i} 的诗。/no_think", 64) for i in range(8)])
    print(f"   全部完成用时 {time.perf_counter() - t:.2f}s；各请求 TTFT(ms):",
          [round(r[1] * 1e3) for r in results])

    print("\n3) 中途断开：")
    stream = await client.chat.completions.create(
        model="default", messages=[{"role": "user", "content": "从 1 数到 1000"}], max_tokens=4000, stream=True
    )
    n = 0
    async for _ in stream:
        n += 1
        if n >= 10:
            break
    await stream.close()
    print("   收到 10 个 chunk 后断开 —— 去服务端日志里找 'Abort request'")


asyncio.run(main())
