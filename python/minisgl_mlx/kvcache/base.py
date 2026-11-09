"""前缀缓存（prefix cache）的抽象接口，对应原版 minisgl/kvcache/base.py。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np


@dataclass(frozen=True)
class BaseCacheHandle(ABC):
    """一次前缀匹配的结果：命中了多长（cached_len），以及能找回这些 KV 槽位的"句柄"。"""

    cached_len: int

    @abstractmethod
    def get_matched_indices(self) -> np.ndarray:
        """返回命中前缀对应的 KV 槽位（长度为 cached_len 的 int32 数组）。"""


class SizeInfo(NamedTuple):
    evictable_size: int  # 没有任何请求在用，可以被淘汰的 token 数
    protected_size: int  # 正被某些请求引用（加了锁），不能淘汰的 token 数

    @property
    def total_size(self) -> int:
        return self.evictable_size + self.protected_size


class InsertResult(NamedTuple):
    cached_len: int  # 插入前树里已经存在的前缀长度（这部分槽位是重复的，调用方应释放）
    handle: BaseCacheHandle


class MatchResult(NamedTuple):
    handle: BaseCacheHandle


class BasePrefixCache(ABC):
    @abstractmethod
    def lock_handle(self, handle: BaseCacheHandle, unlock: bool = False) -> None:
        """
        给 handle 对应的整条路径加锁（引用计数 +1）或解锁。
        加锁的前缀不会被 evict。在使用 match_prefix 返回的槽位之前必须先加锁。
        """

    @abstractmethod
    def match_prefix(self, input_ids: np.ndarray) -> MatchResult:
        """查找 input_ids 在缓存中的最长前缀。不修改缓存内容（但可能分裂节点）。"""

    @abstractmethod
    def insert_prefix(self, input_ids: np.ndarray, indices: np.ndarray) -> InsertResult:
        """把 (token 序列 -> KV 槽位) 插入缓存。"""

    @abstractmethod
    def evict(self, size: int) -> np.ndarray:
        """淘汰至少 size 个 token，返回被淘汰的槽位。"""

    @property
    @abstractmethod
    def size_info(self) -> SizeInfo: ...

    def check_integrity(self) -> None:
        pass
