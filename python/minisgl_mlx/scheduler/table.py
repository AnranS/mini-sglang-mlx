from __future__ import annotations

import mlx.core as mx
import numpy as np


class TableManager:
    """
    管理 page table 的"行"：每个正在运行的请求占一行（table_idx），对应原版 minisgl/scheduler/table.py。

    原版还维护了一个 GPU 上的 token_pool[max_req, max_seq_len]，存放每个请求的全部 token。
    这里只需要一个小得多的 token_pool[max_req]：每个请求"最后一个 token"。原因是：
      * prefill 的输入（prompt）在 CPU 上本来就有，直接转成 mx.array 即可；
      * decode 的输入是上一步刚采样出来的 token —— 在 overlap 调度下，CPU 构建下一步时
        上一步还没算完，CPU 根本不知道这个 token 是什么！所以必须让 GPU 自己把它写进
        token_pool，下一步再从 token_pool 里读。这样两步之间只有"计算图上的依赖"，
        不需要 CPU 等待，见 scheduler.py 的 _forward。
    """

    def __init__(self, max_running_reqs: int, page_table: np.ndarray) -> None:
        self._max_running_reqs = max_running_reqs
        self._free_slots = list(range(max_running_reqs))
        self.page_table = page_table
        self.token_pool = mx.zeros((max_running_reqs,), dtype=mx.int32)

    @property
    def available_size(self) -> int:
        return len(self._free_slots)

    def allocate(self) -> int:
        return self._free_slots.pop()

    def free(self, slot: int) -> None:
        self._free_slots.append(slot)
