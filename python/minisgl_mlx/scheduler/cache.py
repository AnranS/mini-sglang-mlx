"""
CacheManager：管理 KV pool 的槽位分配 & 前缀缓存，对应原版 minisgl/scheduler/cache.py。

一个 KV 槽位在任意时刻处于以下三种状态之一：
    1. 空闲            在 free_slots 里
    2. 被请求私有      写在某个请求的 page table 行里，还没进入前缀缓存
    3. 在前缀缓存里    可能被多个请求共享（ref_count > 0，受保护）或无人使用（可淘汰）

check_integrity() 检查的就是：空闲页数 + 前缀缓存中的页数 == 总页数（在没有运行中请求时）。
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING, List, Tuple

import numpy as np

from minisgl_mlx.kvcache import BaseCacheHandle, MatchResult, create_prefix_cache
from minisgl_mlx.utils import div_ceil

if TYPE_CHECKING:
    from minisgl_mlx.core import Req

    from .utils import PendingReq


class CacheManager:
    def __init__(self, num_pages: int, page_size: int, page_table: np.ndarray, type: str) -> None:
        # free_slots 按页对齐：page_size = 2 时形如 [0, 2, 4, ...]，每个元素代表一页的首个槽位
        self.free_slots = np.arange(num_pages, dtype=np.int32) * page_size
        self.prefix_cache = create_prefix_cache(type, page_size)
        self.num_pages = num_pages
        self.page_table = page_table
        self.page_size = page_size

    def match_req(self, req: PendingReq) -> MatchResult:
        # 注意只匹配 input_ids[:-1]：最后一个 token 必须真正算一遍，才能拿到它的 logits 来采样
        assert req.input_len > 0
        return self.prefix_cache.match_prefix(req.input_ids[: req.input_len - 1])

    @property
    def available_size(self) -> int:
        """还能分配多少 token：空闲的 + 前缀缓存中可淘汰的。"""
        return self.prefix_cache.size_info.evictable_size + len(self.free_slots) * self.page_size

    def lock(self, handle: BaseCacheHandle) -> None:
        self.prefix_cache.lock_handle(handle, unlock=False)

    def unlock(self, handle: BaseCacheHandle) -> None:
        self.prefix_cache.lock_handle(handle, unlock=True)

    def allocate_paged(self, reqs: List[Req]) -> None:
        """为每个请求本轮要算的 token [cached_len, device_len) 分配槽位，写进 page table。"""
        needed_pages = 0
        allocation_info: List[Tuple[int, int, int]] = []
        for req in reqs:
            first_page = div_ceil(req.cached_len, self.page_size)
            last_page = div_ceil(req.device_len, self.page_size)
            if last_page > first_page:  # page_size > 1 时，上一页可能还没写满，不需要新页
                needed_pages += last_page - first_page
                allocation_info.append((req.table_idx, first_page, last_page))
        if needed_pages == 0:
            return
        allocated = self._page_to_token(self._allocate(needed_pages))
        offset = 0
        for table_idx, first_page, last_page in allocation_info:
            first_pos, last_pos = first_page * self.page_size, last_page * self.page_size
            length = last_pos - first_pos
            self.page_table[table_idx, first_pos:last_pos] = allocated[offset : offset + length]
            offset += length
        assert offset == len(allocated)

    def cache_req(self, req: Req, *, finished: bool) -> None:
        """
        把请求已经算好的 KV 插入前缀缓存（prefill 结束时 & 请求结束时调用）。

        设插入前 handle 覆盖 [0, old)，插入后树中覆盖 [0, new)，请求自己算到了 [0, cached_len)：
          [0, old)               本来就是从缓存里借来的
          [old, prefix_len)      插入时发现别的请求已经把这段放进树里了 —— 我们的这份是重复的，释放
          [prefix_len, new)      新插入树中的部分，所有权交给前缀缓存
          [new, cached_len)      page_size > 1 时不足一页的尾巴：请求结束则释放，否则继续留给请求自己
        """
        insert_ids = req.input_ids[: req.cached_len]
        page_indices = self.page_table[req.table_idx, : req.cached_len]
        old_handle = req.cache_handle
        prefix_len, new_handle = self.prefix_cache.insert_prefix(insert_ids, page_indices)
        self.unlock(old_handle)  # 所有对 handle 的操作结束后再解锁
        self._free(page_indices[old_handle.cached_len : prefix_len])
        if finished:
            self._free(page_indices[new_handle.cached_len :])
        else:
            req.cache_handle = new_handle
            self.lock(new_handle)

    def check_integrity(self) -> None:
        self.prefix_cache.check_integrity()
        cache_pages = self.prefix_cache.size_info.total_size // self.page_size
        if len(self.free_slots) + cache_pages != self.num_pages:
            raise RuntimeError(
                "CacheManager integrity check failed: "
                f"free_pages({len(self.free_slots)}) + cache_pages({cache_pages}) != num_pages({self.num_pages})"
            )
        if self.page_size > 1:
            assert np.all(self.free_slots % self.page_size == 0)

    @contextmanager
    def lazy_free_region(self):
        """在区域内的 free 先攒起来，退出时一次性归还（避免反复 concatenate）。"""
        lazy_list: List[np.ndarray] = []

        def lazy_free(indices: np.ndarray) -> None:
            if len(indices) > 0:
                lazy_list.append(indices[:: self.page_size])

        try:
            self._free = lazy_free  # type: ignore[method-assign]
            yield
        finally:
            del self._free
            if lazy_list:
                self.free_slots = np.concatenate([self.free_slots] + lazy_list)

    def _allocate(self, needed_pages: int) -> np.ndarray:
        if needed_pages > (free_pages := len(self.free_slots)):
            # 空闲页不够：从前缀缓存里淘汰一些（LRU）
            evicted = self.prefix_cache.evict((needed_pages - free_pages) * self.page_size)
            self.free_slots = np.concatenate([self.free_slots, evicted[:: self.page_size]])
            assert len(self.free_slots) >= needed_pages, "Eviction did not free enough space."
        allocated = self.free_slots[:needed_pages]
        self.free_slots = self.free_slots[needed_pages:]
        return allocated

    def _free(self, indices: np.ndarray) -> None:
        if len(indices) > 0:
            self.free_slots = np.concatenate([self.free_slots, indices[:: self.page_size]])

    def _page_to_token(self, pages: np.ndarray) -> np.ndarray:
        if self.page_size == 1:
            return pages
        # [X * page_size] -> [X * page_size, ..., X * page_size + page_size - 1]
        offsets = np.arange(self.page_size, dtype=np.int32)
        return (pages[:, None] + offsets[None, :]).reshape(-1)
