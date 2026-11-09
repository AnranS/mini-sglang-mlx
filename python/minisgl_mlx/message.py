"""
进程间消息，对应原版 minisgl/message/。所有消息都是 dataclass，通过 msgpack 在 ZMQ 上传输。

消息流向（与原版相同）：

    API Server ──TokenizeMsg/AbortMsg──> Tokenizer ──UserMsg/AbortBackendMsg──> Scheduler
    API Server <──────UserReply─────── Detokenizer <────────DetokenizeMsg──────── Scheduler

(本实现与原版默认配置一样，tokenizer 和 detokenizer 是同一个进程。)
"""

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
from typing import Any, Dict, List

import numpy as np

from minisgl_mlx.core import SamplingParams

# ----------------------------------------------------------------------------
# 通用序列化：dataclass <-> dict，numpy 数组 <-> bytes
# ----------------------------------------------------------------------------


def _serialize(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return {"__ndarray__": value.tobytes(), "dtype": str(value.dtype)}
    if is_dataclass(value):
        out = {"__type__": type(value).__name__}
        for f in fields(value):
            out[f.name] = _serialize(getattr(value, f.name))
        return out
    if isinstance(value, dict):
        return {k: _serialize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialize(v) for v in value]
    return value


def _deserialize(value: Any) -> Any:
    if isinstance(value, dict):
        if "__ndarray__" in value:
            return np.frombuffer(value["__ndarray__"], dtype=value["dtype"]).copy()
        if "__type__" in value:
            cls = _REGISTRY[value["__type__"]]
            return cls(**{k: _deserialize(v) for k, v in value.items() if k != "__type__"})
        return {k: _deserialize(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_deserialize(v) for v in value]
    return value


def encode(msg: Any) -> Dict:
    return _serialize(msg)


def decode(data: Dict) -> Any:
    return _deserialize(data)


# ----------------------------------------------------------------------------
# Tokenizer -> Scheduler
# ----------------------------------------------------------------------------


@dataclass
class UserMsg:
    uid: int
    input_ids: np.ndarray  # int32
    sampling_params: SamplingParams


@dataclass
class AbortBackendMsg:
    uid: int


@dataclass
class ExitMsg:
    pass


@dataclass
class BatchBackendMsg:
    data: List[Any]


# ----------------------------------------------------------------------------
# API Server -> Tokenizer, Scheduler -> Detokenizer
# ----------------------------------------------------------------------------


@dataclass
class TokenizeMsg:
    uid: int
    text: str | List[Dict[str, str]]  # 纯文本 prompt，或 chat messages
    sampling_params: SamplingParams


@dataclass
class DetokenizeMsg:
    uid: int
    next_token: int
    finished: bool


@dataclass
class AbortMsg:
    uid: int


@dataclass
class BatchTokenizerMsg:
    data: List[Any]


# ----------------------------------------------------------------------------
# Detokenizer -> API Server
# ----------------------------------------------------------------------------


@dataclass
class UserReply:
    uid: int
    incremental_output: str
    finished: bool
    num_prompt_tokens: int = 0  # 只在第一条回复里填写，用于 usage 统计


@dataclass
class BatchFrontendMsg:
    data: List[Any]


def unwrap(msg: Any) -> List[Any]:
    """把 Batch*Msg 拆成消息列表。"""
    if isinstance(msg, (BatchBackendMsg, BatchTokenizerMsg, BatchFrontendMsg)):
        return list(msg.data)
    return [msg]


_REGISTRY = {
    cls.__name__: cls
    for cls in [
        SamplingParams,
        UserMsg,
        AbortBackendMsg,
        ExitMsg,
        BatchBackendMsg,
        TokenizeMsg,
        DetokenizeMsg,
        AbortMsg,
        BatchTokenizerMsg,
        UserReply,
        BatchFrontendMsg,
    ]
}
