from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Set

from minisgl_mlx.core import Batch, Req


@dataclass
class DecodeManager:
    """维护所有处于 decode 阶段的请求，对应原版 minisgl/scheduler/decode.py。"""

    page_size: int
    running_reqs: Set[Req] = field(default_factory=set)

    def filter_reqs(self, reqs: Iterable[Req]) -> None:
        # 刚做完 prefill 的请求加入进来；已经生成到上限的请求移出去
        self.running_reqs = {req for req in self.running_reqs.union(reqs) if req.can_decode}

    def remove_req(self, req: Req) -> None:
        self.running_reqs.discard(req)

    def abort_req(self, uid: int) -> Req | None:
        for req in self.running_reqs:
            if req.uid == uid:
                self.running_reqs.remove(req)
                return req
        return None

    @property
    def inflight_tokens(self) -> int:
        """运行中的请求未来最多还需要多少 KV（每个请求再额外预留一页）。"""
        reserved = (self.page_size - 1) * len(self.running_reqs)
        return sum(req.remain_len for req in self.running_reqs) + reserved

    def schedule_next_batch(self) -> Batch | None:
        if not self.runnable:
            return None
        # 排序是为了让 batch 中请求的顺序是确定的，方便调试与复现
        return Batch(reqs=sorted(self.running_reqs, key=lambda r: r.uid), phase="decode")

    @property
    def runnable(self) -> bool:
        return len(self.running_reqs) > 0
