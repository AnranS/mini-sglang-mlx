"""
tokenizer 进程：同时负责分词（API server -> scheduler）和反分词（scheduler -> API server），
对应原版 minisgl/tokenizer/server.py。

为什么要单独一个进程？分词/反分词是纯 CPU 的 Python 工作，放在调度器进程里会挤占调度器的 CPU 时间
（调度器的 CPU 开销直接决定了 GPU 能不能被喂饱）；放在 API server 里又会阻塞 asyncio 事件循环。
"""

from __future__ import annotations

import multiprocessing as mp
from typing import List

from minisgl_mlx.message import (
    AbortBackendMsg,
    AbortMsg,
    BatchBackendMsg,
    BatchFrontendMsg,
    DetokenizeMsg,
    TokenizeMsg,
    UserMsg,
    UserReply,
    decode,
    encode,
    unwrap,
)
from minisgl_mlx.utils import ZmqPullQueue, ZmqPushQueue, init_logger, load_hf_config, load_tokenizer


def _put(queue: ZmqPushQueue, batch_cls, items: List) -> None:
    if len(items) == 1:
        queue.put(items[0])
    elif len(items) > 1:
        queue.put(batch_cls(data=items))


def tokenize_worker(
    *,
    model_path: str,
    listen_addr: str,
    backend_addr: str,
    frontend_addr: str,
    ack_queue: mp.Queue | None = None,
) -> None:
    from .detokenize import DetokenizeManager
    from .tokenize import TokenizeManager

    logger = init_logger(__name__, "tokenizer")
    send_backend = ZmqPushQueue(backend_addr, create=False, encoder=encode)
    send_frontend = ZmqPushQueue(frontend_addr, create=False, encoder=encode)
    recv = ZmqPullQueue(listen_addr, create=True, decoder=decode)

    tokenizer = load_tokenizer(model_path)
    eos = load_hf_config(model_path).get("eos_token_id", [])
    eos_ids = set([eos] if isinstance(eos, int) else eos or []) | {tokenizer.eos_token_id}
    tokenize_manager = TokenizeManager(tokenizer)
    detokenize_manager = DetokenizeManager(tokenizer, eos_ids)
    num_prompt_tokens = {}  # uid -> prompt 长度，第一条回复时带给前端用于 usage 统计

    if ack_queue is not None:
        ack_queue.put("Tokenizer is ready")

    try:
        while True:
            msgs = unwrap(recv.get())
            while not recv.empty():  # 尽量攒一批，batch_decode 更高效
                msgs.extend(unwrap(recv.get()))

            detok = [m for m in msgs if isinstance(m, DetokenizeMsg)]
            tok = [m for m in msgs if isinstance(m, TokenizeMsg)]
            abort = [m for m in msgs if isinstance(m, AbortMsg)]

            if detok:
                texts = detokenize_manager.detokenize(detok)
                replies = []
                for m, text in zip(detok, texts, strict=True):
                    n = num_prompt_tokens.pop(m.uid, 0) if m.finished else num_prompt_tokens.get(m.uid, 0)
                    replies.append(UserReply(uid=m.uid, incremental_output=text, finished=m.finished, num_prompt_tokens=n))
                _put(send_frontend, BatchFrontendMsg, replies)
            if tok:
                ids_list = tokenize_manager.tokenize(tok)
                for m, ids in zip(tok, ids_list, strict=True):
                    num_prompt_tokens[m.uid] = len(ids)
                _put(
                    send_backend,
                    BatchBackendMsg,
                    [UserMsg(uid=m.uid, input_ids=ids, sampling_params=m.sampling_params) for m, ids in zip(tok, ids_list)],
                )
            if abort:
                for m in abort:
                    num_prompt_tokens.pop(m.uid, None)
                    detokenize_manager.decode_map.pop(m.uid, None)
                _put(send_backend, BatchBackendMsg, [AbortBackendMsg(uid=m.uid) for m in abort])
            logger.debug(f"tokenize={len(tok)} detokenize={len(detok)} abort={len(abort)}")
    except KeyboardInterrupt:
        pass
