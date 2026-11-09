"""
API server（FastAPI），对应原版 minisgl/server/api_server.py。

它是一个纯 asyncio 的"前台"：不做任何计算，只负责
    1. 给每个 HTTP 请求分配 uid，把 TokenizeMsg 发给 tokenizer 进程；
    2. 后台协程 listen() 不停地从 detokenizer 收 UserReply，按 uid 放进对应的队列并唤醒等待者；
    3. 把增量文本以 SSE (text/event-stream) 的形式流式返回给客户端。
客户端断开时发送 AbortMsg，调度器会立即释放该请求占用的 KV。
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import AsyncIterator, Dict, List, Literal

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from minisgl_mlx.core import SamplingParams
from minisgl_mlx.message import AbortMsg, TokenizeMsg, UserReply, decode, encode, unwrap
from minisgl_mlx.utils import ZmqAsyncPullQueue, ZmqAsyncPushQueue, init_logger

from .args import ServerArgs

logger = init_logger(__name__, "api")

DEFAULT_MAX_TOKENS = 1024


class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class OpenAIRequest(BaseModel):
    """/v1/chat/completions 与 /v1/completions 共用的请求体（只实现常用字段）。"""

    model: str = ""
    messages: List[Message] | None = None
    prompt: str | None = None
    max_tokens: int | None = None
    max_completion_tokens: int | None = None
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    stream: bool = False
    ignore_eos: bool = False

    def sampling_params(self) -> SamplingParams:
        return SamplingParams(
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
            ignore_eos=self.ignore_eos,
            max_tokens=self.max_completion_tokens or self.max_tokens or DEFAULT_MAX_TOKENS,
        )


class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = DEFAULT_MAX_TOKENS
    temperature: float = 0.0
    ignore_eos: bool = False


@dataclass
class RequestState:
    replies: List[UserReply] = field(default_factory=list)
    event: asyncio.Event = field(default_factory=asyncio.Event)
    num_tokens: int = 0
    num_prompt_tokens: int = 0


@dataclass
class FrontendManager:
    config: ServerArgs
    send_tokenizer: ZmqAsyncPushQueue
    recv_tokenizer: ZmqAsyncPullQueue
    uid_counter: int = 0
    states: Dict[int, RequestState] = field(default_factory=dict)
    listener: asyncio.Task | None = None

    async def submit(self, text, sampling_params: SamplingParams) -> int:
        if self.listener is None:
            self.listener = asyncio.create_task(self.listen())
        uid = self.uid_counter
        self.uid_counter += 1
        self.states[uid] = RequestState()
        await self.send_tokenizer.put(TokenizeMsg(uid=uid, text=text, sampling_params=sampling_params))
        return uid

    async def listen(self) -> None:
        while True:
            for reply in unwrap(await self.recv_tokenizer.get()):
                if (state := self.states.get(reply.uid)) is None:
                    continue  # 已经 abort 的请求
                state.replies.append(reply)
                state.event.set()

    async def wait_for_replies(self, uid: int) -> AsyncIterator[UserReply]:
        state = self.states[uid]
        try:
            while True:
                await state.event.wait()
                state.event.clear()
                pending, state.replies = state.replies, []
                for reply in pending:
                    state.num_tokens += 1
                    state.num_prompt_tokens = max(state.num_prompt_tokens, reply.num_prompt_tokens)
                    yield reply
                    if reply.finished:
                        return
        finally:
            self.states.pop(uid, None)

    async def abort(self, uid: int) -> None:
        self.states.pop(uid, None)
        logger.info(f"Abort request {uid}")
        await self.send_tokenizer.put(AbortMsg(uid=uid))

    async def stream_with_cancellation(self, gen: AsyncIterator[bytes], request: Request, uid: int):
        finished = False
        try:
            async for chunk in gen:
                if await request.is_disconnected():
                    break
                yield chunk
            else:
                finished = True
        finally:
            if not finished:  # 客户端断开：告诉调度器尽快释放资源（用 task，避免在取消过程中 await）
                asyncio.create_task(self.abort(uid))

    def shutdown(self) -> None:
        self.send_tokenizer.stop()
        self.recv_tokenizer.stop()


_STATE: FrontendManager | None = None


def get_state() -> FrontendManager:
    assert _STATE is not None
    return _STATE


app = FastAPI(title="mini-sglang-mlx", version="0.1.0")


def _finish_reason(state_tokens: int, max_tokens: int) -> str:
    return "length" if state_tokens >= max_tokens else "stop"


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/v1/models")
async def models():
    path = get_state().config.model_path
    return {"object": "list", "data": [{"id": path, "object": "model", "owned_by": "mini-sglang-mlx"}]}


@app.post("/generate")
async def generate(req: GenerateRequest, request: Request):
    """最简单的流式接口：每行 `data: <增量文本>`。"""
    state = get_state()
    sp = SamplingParams(temperature=req.temperature, ignore_eos=req.ignore_eos, max_tokens=req.max_tokens)
    uid = await state.submit(req.prompt, sp)

    async def gen():
        async for reply in state.wait_for_replies(uid):
            yield f"data: {json.dumps(reply.incremental_output, ensure_ascii=False)}\n\n".encode()
        yield b"data: [DONE]\n\n"

    return StreamingResponse(state.stream_with_cancellation(gen(), request, uid), media_type="text/event-stream")


@app.post("/v1/chat/completions")
async def chat_completions(req: OpenAIRequest, request: Request):
    return await _openai_endpoint(req, request, chat=True)


@app.post("/v1/completions")
async def completions(req: OpenAIRequest, request: Request):
    return await _openai_endpoint(req, request, chat=False)


async def _openai_endpoint(req: OpenAIRequest, request: Request, chat: bool):
    state = get_state()
    if chat:
        assert req.messages, "'messages' is required for chat completions"
        text = [m.model_dump() for m in req.messages]
    else:
        assert req.prompt is not None, "'prompt' is required for completions"
        text = req.prompt
    sp = req.sampling_params()
    uid = await state.submit(text, sp)
    rid, created = f"cmpl-{uid}", int(time.time())
    obj = "chat.completion" if chat else "text_completion"

    def choice(content: str, finish_reason: str | None, delta: bool) -> Dict:
        if not chat:
            return {"index": 0, "text": content, "finish_reason": finish_reason}
        key = "delta" if delta else "message"
        body = {"role": "assistant", "content": content} if (not delta or content) else {}
        return {"index": 0, key: body, "finish_reason": finish_reason}

    if req.stream:

        async def gen():
            rs = state.states[uid]
            async for reply in state.wait_for_replies(uid):
                reason = _finish_reason(rs.num_tokens, sp.max_tokens) if reply.finished else None
                chunk = {"id": rid, "object": obj + (".chunk" if chat else ""), "created": created,
                         "model": req.model, "choices": [choice(reply.incremental_output, reason, True)]}
                yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n".encode()
            yield b"data: [DONE]\n\n"

        return StreamingResponse(state.stream_with_cancellation(gen(), request, uid), media_type="text/event-stream")

    rs = state.states[uid]
    content = "".join([r.incremental_output async for r in state.wait_for_replies(uid)])
    return {
        "id": rid,
        "object": obj,
        "created": created,
        "model": req.model,
        "choices": [choice(content, _finish_reason(rs.num_tokens, sp.max_tokens), False)],
        "usage": {
            "prompt_tokens": rs.num_prompt_tokens,
            "completion_tokens": rs.num_tokens,
            "total_tokens": rs.num_prompt_tokens + rs.num_tokens,
        },
    }


def create_frontend(config: ServerArgs) -> FrontendManager:
    global _STATE
    _STATE = FrontendManager(
        config=config,
        recv_tokenizer=ZmqAsyncPullQueue(config.zmq_frontend_addr, create=True, decoder=decode),
        send_tokenizer=ZmqAsyncPushQueue(config.zmq_tokenizer_addr, create=False, encoder=encode),
    )
    return _STATE
