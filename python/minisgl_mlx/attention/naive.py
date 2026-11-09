"""
naive decode 后端：gather + mask + SDPA，只用 MLX 自带的算子。

    kv_indices [B, L_max] ──gather──> K [B, L_max, Hkv, D] ──transpose──> [B, Hkv, L_max, D]
    mask       [B, 1, 1, L_max]       第 j 个位置是否 < 该请求的长度

优点：简单、好懂、一定正确。
缺点：gather 会把所有历史 KV 复制一遍（读一次 + 写一次），然后 SDPA 再读一次，
      访存量是理想情况的约 3 倍；再加上 padding 到 L_max 的浪费。
      在 B=32, L=2048 的场景下 gather 本身就比 attention 计算还慢（见 tutorials/ch04_attention.py）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import mlx.core as mx
import numpy as np

from minisgl_mlx.utils import align_up

from .base import BaseAttnBackend

if TYPE_CHECKING:
    from minisgl_mlx.core import Batch


@dataclass
class NaiveDecodeMetadata:
    kv_indices: mx.array  # [B, L_max]
    mask: mx.array  # [B, 1, 1, L_max] bool


class NaiveAttnBackend(BaseAttnBackend):
    def prepare_decode(self, batch: Batch) -> NaiveDecodeMetadata:
        lens = np.array([r.device_len for r in batch.reqs])
        rows = np.array([r.table_idx for r in batch.reqs])
        # 向上对齐到 128：否则每一步 L_max 都不同，gather 出的临时 buffer 尺寸各异，
        # MLX 的 buffer 缓存无法复用它们（实测不对齐时缓存能涨到几十 GB）
        max_len = min(align_up(int(lens.max()), 128), self.page_table.shape[1])
        # page table 中超出请求长度的部分是脏数据，但一定是合法槽位，会被 mask 掉
        kv_indices = self.page_table[rows, :max_len]
        mask = np.arange(max_len)[None, :] < lens[:, None]
        return NaiveDecodeMetadata(
            kv_indices=mx.array(kv_indices),
            mask=mx.array(mask)[:, None, None, :],
        )

    def forward_decode(self, q: mx.array, layer_id: int, meta: NaiveDecodeMetadata) -> mx.array:
        k = self.kv_cache.k_cache(layer_id)[meta.kv_indices].transpose(0, 2, 1, 3)
        v = self.kv_cache.v_cache(layer_id)[meta.kv_indices].transpose(0, 2, 1, 3)
        o = mx.fast.scaled_dot_product_attention(
            q[:, :, None, :],  # [B, Hq, 1, D]
            k,
            v,
            scale=self.scale,
            mask=meta.mask,
        )
        return o[:, :, 0, :]  # [B, Hq, D]
