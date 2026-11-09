"""杂项工具：日志、数学小函数、模型路径解析、tokenizer 加载、ZMQ 队列。"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Callable, Dict, Generic, TypeVar

# ----------------------------------------------------------------------------
# 日志
# ----------------------------------------------------------------------------

_LOG_LEVEL = os.getenv("MINISGL_LOG_LEVEL", "INFO").upper()


def init_logger(name: str, tag: str | None = None) -> logging.Logger:
    logger = logging.getLogger(name if tag is None else f"{name}.{tag}")
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        prefix = f"[{tag}] " if tag else ""
        handler.setFormatter(
            logging.Formatter(f"%(asctime)s %(levelname).1s {prefix}%(message)s", "%H:%M:%S")
        )
        logger.addHandler(handler)
        logger.setLevel(_LOG_LEVEL)
        logger.propagate = False
    return logger


# ----------------------------------------------------------------------------
# 数学小工具
# ----------------------------------------------------------------------------


def div_ceil(a: int, b: int) -> int:
    return (a + b - 1) // b


def align_down(a: int, b: int) -> int:
    return a // b * b


def align_up(a: int, b: int) -> int:
    return div_ceil(a, b) * b


def mem_GB(size: int) -> str:
    return f"{size / (1024**3):.2f} GiB"


# ----------------------------------------------------------------------------
# 模型路径 & tokenizer
# ----------------------------------------------------------------------------


def resolve_model_path(model_path: str) -> str:
    """本地目录直接返回；否则当作 HuggingFace repo id，下载（或命中本地缓存）后返回快照目录。"""
    if os.path.isdir(model_path):
        return model_path
    from huggingface_hub import snapshot_download

    return snapshot_download(
        model_path,
        allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.jinja", "*.tiktoken"],
    )


def load_hf_config(model_path: str) -> dict:
    import json

    with open(Path(resolve_model_path(model_path)) / "config.json") as f:
        return json.load(f)


def load_tokenizer(model_path: str):
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")  # 我们只用 tokenizer，不需要 PyTorch
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(resolve_model_path(model_path))


# ----------------------------------------------------------------------------
# ZMQ + msgpack 队列（进程间通信用，与原版 minisgl/utils/mp.py 相同的设计）
# ----------------------------------------------------------------------------

T = TypeVar("T")


class ZmqPushQueue(Generic[T]):
    def __init__(self, addr: str, create: bool, encoder: Callable[[T], Dict]):
        import zmq

        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUSH)
        self.socket.bind(addr) if create else self.socket.connect(addr)
        self.encoder = encoder

    def put(self, obj: T) -> None:
        import msgpack

        self.socket.send(msgpack.packb(self.encoder(obj), use_bin_type=True), copy=False)

    def stop(self) -> None:
        self.socket.close(linger=0)
        self.context.term()


class ZmqPullQueue(Generic[T]):
    def __init__(self, addr: str, create: bool, decoder: Callable[[Dict], T]):
        import zmq

        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PULL)
        self.socket.bind(addr) if create else self.socket.connect(addr)
        self.decoder = decoder

    def get(self) -> T:
        import msgpack

        return self.decoder(msgpack.unpackb(self.socket.recv(), raw=False))

    def empty(self) -> bool:
        import zmq

        return self.socket.poll(timeout=0, flags=zmq.POLLIN) == 0

    def stop(self) -> None:
        self.socket.close(linger=0)
        self.context.term()


class ZmqAsyncPushQueue(Generic[T]):
    def __init__(self, addr: str, create: bool, encoder: Callable[[T], Dict]):
        import zmq
        import zmq.asyncio

        self.context = zmq.asyncio.Context()
        self.socket = self.context.socket(zmq.PUSH)
        self.socket.bind(addr) if create else self.socket.connect(addr)
        self.encoder = encoder

    async def put(self, obj: T) -> None:
        import msgpack

        await self.socket.send(msgpack.packb(self.encoder(obj), use_bin_type=True), copy=False)

    def stop(self) -> None:
        self.socket.close(linger=0)
        self.context.term()


class ZmqAsyncPullQueue(Generic[T]):
    def __init__(self, addr: str, create: bool, decoder: Callable[[Dict], T]):
        import zmq
        import zmq.asyncio

        self.context = zmq.asyncio.Context()
        self.socket = self.context.socket(zmq.PULL)
        self.socket.bind(addr) if create else self.socket.connect(addr)
        self.decoder = decoder

    async def get(self) -> T:
        import msgpack

        return self.decoder(msgpack.unpackb(await self.socket.recv(), raw=False))

    def stop(self) -> None:
        self.socket.close(linger=0)
        self.context.term()
