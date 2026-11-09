from __future__ import annotations

import os
from dataclasses import dataclass, field

from minisgl_mlx.engine import EngineConfig


def _unique_suffix() -> str:
    return f"_{os.getpid()}"


@dataclass(frozen=True)
class SchedulerConfig(EngineConfig):
    max_extend_tokens: int = 4096  # 一个 prefill batch 最多处理多少 token（chunked prefill 的 chunk 大小）
    cache_type: str = "radix"  # radix / naive
    overlap_scheduling: bool = True
    offline_mode: bool = False

    _unique_suffix: str = field(default_factory=_unique_suffix)

    # ZMQ 地址（ipc 本质是 unix domain socket）
    @property
    def zmq_backend_addr(self) -> str:  # tokenizer -> scheduler
        return f"ipc:///tmp/minisgl_mlx_backend{self._unique_suffix}"

    @property
    def zmq_detokenizer_addr(self) -> str:  # scheduler -> detokenizer, api server -> tokenizer
        return f"ipc:///tmp/minisgl_mlx_tokenizer{self._unique_suffix}"
