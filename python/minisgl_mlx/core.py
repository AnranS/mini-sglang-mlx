"""
核心数据结构，对应原版 minisgl/core.py。

    SamplingParams  用户给的采样参数
    Req             一个"正在 GPU 上跑"的请求（已分配了 page table 的一行）
    Batch           一次 forward 要处理的一组请求（要么全是 prefill，要么全是 decode）
    Context         全局上下文：page table、KV pool、attention 后端、当前 batch

和原版最大的区别：原版的 input_ids、page_table 等放在 GPU 显存里 (torch.cuda)，
而 Apple Silicon 是统一内存，CPU 和 GPU 共用一块物理内存。所以这里所有"调度用的元数据"
都放在 CPU 侧的 numpy 数组里，每一步只把这一个 batch 需要的索引转成 mx.array 交给 GPU。
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, List, Literal

import numpy as np

if TYPE_CHECKING:
    import mlx.core as mx

    from minisgl_mlx.attention import BaseAttnBackend
    from minisgl_mlx.kvcache import BaseCacheHandle, MHAKVCache


@dataclass
class SamplingParams:
    temperature: float = 0.0
    top_k: int = -1
    top_p: float = 1.0
    ignore_eos: bool = False
    max_tokens: int = 1024

    @property
    def is_greedy(self) -> bool:
        return (self.temperature <= 0.0 or self.top_k == 1) and self.top_p == 1.0


@dataclass(eq=False)
class Req:
    """
    一个请求在调度器里的状态。理解三个长度是理解整个调度器的关键：

        input_ids   [ prompt ........................ | 已生成的 token ... ]
                    0            cached_len        device_len      max_device_len
                    |<-- KV 已在 cache 中 -->|<- 本轮要算 ->|

        cached_len      : 前 cached_len 个 token 的 KV 已经写进 KV pool（可能来自前缀缓存命中）
        device_len      : 前 device_len 个 token 已经"交给设备"了；[cached_len, device_len) 是本轮 extend 的部分
        max_device_len  : prompt 长度 + 最多生成的 token 数

    每次 forward 之后调用 complete_one()：
        cached_len = device_len   （本轮算完，这些 token 的 KV 都写好了）
        device_len += 1           （新采样出的 token 下一轮要喂给模型）
    """

    input_ids: np.ndarray  # host 上的 int32 数组
    table_idx: int  # 在 page table 中占用的行号
    cached_len: int
    output_len: int
    uid: int
    sampling_params: SamplingParams
    cache_handle: BaseCacheHandle

    def __post_init__(self) -> None:
        self.device_len = len(self.input_ids)
        self.max_device_len = len(self.input_ids) + self.output_len
        assert 0 <= self.cached_len < self.device_len <= self.max_device_len

    @property
    def remain_len(self) -> int:
        return self.max_device_len - self.device_len

    @property
    def extend_len(self) -> int:
        return self.device_len - self.cached_len

    def complete_one(self) -> None:
        self.cached_len = self.device_len
        self.device_len += 1

    def append_host(self, next_token: int) -> None:
        self.input_ids = np.append(self.input_ids, np.int32(next_token))

    @property
    def can_decode(self) -> bool:
        return self.remain_len > 0

    def __repr__(self) -> str:
        return (
            f"Req(uid={self.uid}, table_idx={self.table_idx}, cached_len={self.cached_len}, "
            f"device_len={self.device_len}, max_device_len={self.max_device_len})"
        )


@dataclass
class Batch:
    reqs: List[Req]
    phase: Literal["prefill", "decode"]
    # 下面这些字段由 scheduler 在 _prepare_batch 中填写
    input_ids: mx.array = field(init=False)  # [T]  所有请求本轮要算的 token 拼在一起（packed）
    positions: mx.array = field(init=False)  # [T]  每个 token 的位置（给 RoPE 用）
    out_loc: mx.array = field(init=False)  # [T]  每个 token 的 KV 要写到 KV pool 的哪个槽位
    # 由 attention 后端在 prepare_metadata 中填写
    attn_metadata: Any = field(init=False)

    @property
    def is_prefill(self) -> bool:
        return self.phase == "prefill"

    @property
    def is_decode(self) -> bool:
        return self.phase == "decode"

    @property
    def size(self) -> int:
        return len(self.reqs)


@dataclass
class Context:
    page_size: int
    # page_table[table_idx, pos] = 第 table_idx 个请求的第 pos 个 token 的 KV 槽位
    # 注意：和原版一样，这张表总是按 token 粒度存储（相当于 page_size = 1 的视角）
    page_table: np.ndarray = field(init=False)
    kv_cache: MHAKVCache = field(init=False)
    attn_backend: BaseAttnBackend = field(init=False)
    _batch: Batch | None = field(default=None, init=False)

    @property
    def batch(self) -> Batch:
        assert self._batch is not None, "No active batch in context"
        return self._batch

    @contextmanager
    def forward_batch(self, batch: Batch):
        assert self._batch is None, "Nested forward_batch is not allowed"
        try:
            self._batch = batch
            yield
        finally:
            self._batch = None


_GLOBAL_CTX: Context | None = None


def set_global_ctx(ctx: Context | None) -> None:
    global _GLOBAL_CTX
    _GLOBAL_CTX = ctx


def get_global_ctx() -> Context:
    assert _GLOBAL_CTX is not None, "Global context is not set"
    return _GLOBAL_CTX
