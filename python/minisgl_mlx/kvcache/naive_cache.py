"""
"不缓存"的前缀缓存：永远匹配长度 0，插入什么都不记。用来做对照实验（--cache naive），
感受 radix cache 在多轮对话 / 共享 system prompt 场景下带来的收益。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .base import BaseCacheHandle, BasePrefixCache, InsertResult, MatchResult, SizeInfo

_EMPTY = np.empty(0, dtype=np.int32)


@dataclass(frozen=True)
class NaiveCacheHandle(BaseCacheHandle):
    def get_matched_indices(self) -> np.ndarray:
        return _EMPTY


class NaivePrefixCache(BasePrefixCache):
    def __init__(self, page_size: int = 1) -> None:
        self.page_size = page_size

    def lock_handle(self, handle: BaseCacheHandle, unlock: bool = False) -> None:
        pass

    def match_prefix(self, input_ids: np.ndarray) -> MatchResult:
        return MatchResult(NaiveCacheHandle(0))

    def insert_prefix(self, input_ids: np.ndarray, indices: np.ndarray) -> InsertResult:
        return InsertResult(0, NaiveCacheHandle(0))

    def evict(self, size: int) -> np.ndarray:
        if size == 0:
            return _EMPTY
        raise RuntimeError("NaivePrefixCache does not hold anything to evict.")

    @property
    def size_info(self) -> SizeInfo:
        return SizeInfo(evictable_size=0, protected_size=0)
