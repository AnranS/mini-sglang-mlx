"""
PrefillManager：决定"下一个 prefill batch 放哪些请求"，对应原版 minisgl/scheduler/prefill.py。

核心是 PrefillAdder 的两个预算：
  * token_budget  : 本 batch 最多算多少个 token（--max-extend-tokens）。一个长 prompt 放不下时，
                    只放一部分进来，剩下的下次再算 —— 这就是 chunked prefill。
  * reserved_size : 为已经在跑的请求"预留"的 KV 空间。我们按"最坏情况"估计：每个请求都会生成到
                    max_tokens。如果新请求的 prompt + max_tokens 放不下，就不接纳它。
                    这个策略很保守，但保证了运行中的请求永远不会因为 KV 不够而被迫中断。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Tuple

from minisgl_mlx.core import Batch, Req

from .utils import PendingReq

if TYPE_CHECKING:
    from minisgl_mlx.kvcache import BaseCacheHandle
    from minisgl_mlx.message import UserMsg

    from .cache import CacheManager
    from .decode import DecodeManager
    from .table import TableManager


class ChunkedReq(Req):
    """只算了一部分 prompt 的请求：这一轮不采样，也不进入 decode。"""

    def append_host(self, next_token: int) -> None:
        raise NotImplementedError("ChunkedReq should not be sampled")

    @property
    def can_decode(self) -> bool:
        return False


@dataclass
class PrefillAdder:
    token_budget: int
    reserved_size: int
    cache_manager: CacheManager
    table_manager: TableManager

    def _try_allocate_one(self, req: PendingReq) -> Tuple[BaseCacheHandle, int] | None:
        if self.table_manager.available_size == 0:
            return None

        handle = self.cache_manager.match_req(req).handle
        cached_len = handle.cached_len
        estimated_len = (req.input_len - cached_len) + req.output_len
        if estimated_len + self.reserved_size > self.cache_manager.available_size:
            return None
        # 先加锁（命中的前缀不再可淘汰，available_size 会变小），再检查一次
        self.cache_manager.lock(handle)
        if estimated_len + self.reserved_size > self.cache_manager.available_size:
            return self.cache_manager.unlock(handle)

        table_idx = self.table_manager.allocate()
        if cached_len > 0:  # 命中的前缀：直接把缓存里的槽位抄进 page table，KV 不用重算
            self.table_manager.page_table[table_idx, :cached_len] = handle.get_matched_indices()
        return handle, table_idx

    def _add_one_req(
        self, pending_req: PendingReq, cache_handle: BaseCacheHandle, table_idx: int, cached_len: int
    ) -> Req:
        remain_len = pending_req.input_len - cached_len
        chunk_size = min(self.token_budget, remain_len)
        cls = ChunkedReq if chunk_size < remain_len else Req
        self.token_budget -= chunk_size
        self.reserved_size += remain_len + pending_req.output_len
        return cls(
            input_ids=pending_req.input_ids[: cached_len + chunk_size],
            table_idx=table_idx,
            cached_len=cached_len,
            output_len=pending_req.output_len,
            uid=pending_req.uid,
            cache_handle=cache_handle,
            sampling_params=pending_req.sampling_params,
        )

    def try_add_one(self, pending_req: PendingReq) -> Req | None:
        if self.token_budget <= 0:
            return None
        if chunked := pending_req.chunked_req:  # 上次没算完的请求：资源已分配，接着算
            return self._add_one_req(pending_req, chunked.cache_handle, chunked.table_idx, chunked.cached_len)
        if resource := self._try_allocate_one(pending_req):
            handle, table_idx = resource
            return self._add_one_req(pending_req, handle, table_idx, handle.cached_len)
        return None


@dataclass
class PrefillManager:
    cache_manager: CacheManager
    table_manager: TableManager
    decode_manager: DecodeManager
    pending_list: List[PendingReq] = field(default_factory=list)

    def add_one_req(self, msg: UserMsg) -> None:
        self.pending_list.append(PendingReq(msg.uid, msg.input_ids, msg.sampling_params))

    def schedule_next_batch(self, prefill_budget: int) -> Batch | None:
        if not self.pending_list:
            return None
        adder = PrefillAdder(
            token_budget=prefill_budget,
            reserved_size=self.decode_manager.inflight_tokens,
            cache_manager=self.cache_manager,
            table_manager=self.table_manager,
        )
        reqs: List[Req] = []
        chunked_list: List[PendingReq] = []
        for pending_req in self.pending_list:  # FCFS：按到达顺序，放不下就停
            req = adder.try_add_one(pending_req)
            if req is None:
                break
            pending_req.chunked_req = None
            if isinstance(req, ChunkedReq):
                pending_req.chunked_req = req
                chunked_list.append(pending_req)
            reqs.append(req)
        if not reqs:
            return None
        # 没做完的 chunked 请求放回队首，下次优先继续
        self.pending_list = chunked_list + self.pending_list[len(reqs) :]
        return Batch(reqs=reqs, phase="prefill")

    def abort_req(self, uid: int) -> Req | None:
        for i, req in enumerate(self.pending_list):
            if req.uid == uid:
                self.pending_list.pop(i)
                return req.chunked_req  # 如果做了一半，需要调用方释放资源
        return None

    @property
    def runnable(self) -> bool:
        return len(self.pending_list) > 0
