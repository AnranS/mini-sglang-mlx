"""交互式命令行聊天（python -m minisgl_mlx --model ... --shell），直接复用 FrontendManager，不经过 HTTP。"""

from __future__ import annotations

import asyncio
import os
from typing import Dict, List

from minisgl_mlx.core import SamplingParams

from .api_server import FrontendManager

HELP = "输入消息开始对话；/reset 清空历史，/exit 退出。Qwen3 可在消息末尾加 /no_think 关闭思考模式。"


async def run_shell(state: FrontendManager) -> None:
    sp = SamplingParams(
        temperature=float(os.getenv("MINISGL_SHELL_TEMPERATURE", "0.6")),
        top_p=float(os.getenv("MINISGL_SHELL_TOP_P", "0.95")),
        top_k=int(os.getenv("MINISGL_SHELL_TOP_K", "20")),
        max_tokens=int(os.getenv("MINISGL_SHELL_MAX_TOKENS", "2048")),
    )
    history: List[Dict[str, str]] = []
    print(HELP)
    while True:
        try:
            cmd = (await asyncio.to_thread(input, "\n$ ")).strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not cmd:
            continue
        if cmd == "/exit":
            break
        if cmd == "/reset":
            history = []
            print("(history cleared)")
            continue
        history.append({"role": "user", "content": cmd})
        uid = await state.submit(list(history), SamplingParams(**vars(sp)))
        answer = ""
        try:
            async for reply in state.wait_for_replies(uid):
                answer += reply.incremental_output
                print(reply.incremental_output, end="", flush=True)
        except (KeyboardInterrupt, asyncio.CancelledError):
            await state.abort(uid)
        print()
        history.append({"role": "assistant", "content": answer})
    print("Bye.")
