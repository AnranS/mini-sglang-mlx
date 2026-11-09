"""
attention 后端的接口 + 两个后端共用的 prefill 实现，对应原版 minisgl/attention/base.py。

attention 后端要解决的问题：q/k/v 是 packed 的 [T, H, D]，但每个 token 只能 attend 到
"同一个请求"里、位置不超过自己的那些 token，而这些历史 token 的 KV 散落在 KV pool 的各个槽位上。

我们把 prefill 和 decode 分开处理（原版也允许 prefill/decode 用不同后端，见 HybridBackend）：

  * prefill：计算密集（一个请求有很多 query token）。逐个请求调用 mx.fast.scaled_dot_product_attention，
    mask="causal"（MLX 的 causal 是右下角对齐，正好处理"前缀已缓存、只算后半段"的情况）。
  * decode ：访存密集（每个请求只有 1 个 query token，却要读全部历史 KV）。必须把整个 batch
    一起算，两个后端的差别就在这里：
        naive.py  —— 先 gather 出 [B, L_max, H, D] 的连续 KV，再配合 mask 调用 SDPA
        metal.py  —— 自己写的 Metal kernel，直接按索引从 KV pool 读取（PagedAttention）
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, List

import mlx.core as mx
import numpy as np

if TYPE_CHECKING:
    from minisgl_mlx.core import Batch
    from minisgl_mlx.kvcache import MHAKVCache
    from minisgl_mlx.models import ModelConfig


@dataclass
class PrefillMetadata:
    last_indices: mx.array  # [B]   每个请求最后一个 token 在 packed 序列中的下标
    q_offsets: List[int]  # [B+1] 请求 i 的 query 是 q[q_offsets[i] : q_offsets[i+1]]
    # 请求 i 完整 KV（长度 device_len）的槽位；None 表示没有命中前缀，直接用本轮算出的 k/v 即可
    kv_indices: List[mx.array | None]


class BaseAttnBackend(ABC):
    def __init__(self, config: ModelConfig, kv_cache: MHAKVCache, page_table: np.ndarray) -> None:
        self.config = config
        self.kv_cache = kv_cache
        self.page_table = page_table
        self.scale = config.head_dim**-0.5

    # ------------------------------------------------------------------ 接口

    def prepare_metadata(self, batch: Batch) -> None:
        """在 forward 之前（CPU 侧）准备好 attention 需要的索引。"""
        if batch.is_prefill:
            batch.attn_metadata = self._prepare_prefill(batch)
        else:
            batch.attn_metadata = self.prepare_decode(batch)

    def forward(self, q: mx.array, k: mx.array, v: mx.array, layer_id: int, batch: Batch) -> mx.array:
        """q: [T, Hq, D], k/v: [T, Hkv, D]（本轮新 token，已写入 KV pool） -> o: [T, Hq, D]"""
        if batch.is_prefill:
            return self._forward_prefill(q, k, v, layer_id, batch.attn_metadata)
        return self.forward_decode(q, layer_id, batch.attn_metadata)

    @abstractmethod
    def prepare_decode(self, batch: Batch): ...

    @abstractmethod
    def forward_decode(self, q: mx.array, layer_id: int, metadata) -> mx.array: ...

    # ------------------------------------------------------------------ prefill（共用）

    def _prepare_prefill(self, batch: Batch) -> PrefillMetadata:
        extend_lens = [r.extend_len for r in batch.reqs]
        q_offsets = np.concatenate([[0], np.cumsum(extend_lens)]).tolist()
        kv_indices: List[mx.array | None] = []
        for req in batch.reqs:
            if req.cached_len == 0:
                kv_indices.append(None)
            else:
                kv_indices.append(mx.array(self.page_table[req.table_idx, : req.device_len]))
        return PrefillMetadata(
            last_indices=mx.array(np.array(q_offsets[1:], dtype=np.int32) - 1),
            q_offsets=q_offsets,
            kv_indices=kv_indices,
        )

    def _forward_prefill(
        self, q: mx.array, k: mx.array, v: mx.array, layer_id: int, meta: PrefillMetadata
    ) -> mx.array:
        k_pool, v_pool = self.kv_cache.k_cache(layer_id), self.kv_cache.v_cache(layer_id)
        outputs: List[mx.array] = []
        for i, kv_idx in enumerate(meta.kv_indices):
            s, e = meta.q_offsets[i], meta.q_offsets[i + 1]
            qi = q[s:e]  # [Lq, Hq, D]
            if kv_idx is None:
                ki, vi = k[s:e], v[s:e]  # 没有前缀：KV 就是本轮算出来的这些
            else:
                ki, vi = k_pool[kv_idx], v_pool[kv_idx]  # 有前缀：从 pool 里读出完整 KV
            oi = mx.fast.scaled_dot_product_attention(
                qi.transpose(1, 0, 2)[None],  # [1, Hq, Lq, D]
                ki.transpose(1, 0, 2)[None],  # [1, Hkv, Lkv, D]   GQA 无需手动复制 head
                vi.transpose(1, 0, 2)[None],
                scale=self.scale,
                mask="causal" if e - s > 1 else None,
            )
            outputs.append(oi[0].transpose(1, 0, 2))  # [Lq, Hq, D]
        return outputs[0] if len(outputs) == 1 else mx.concatenate(outputs, axis=0)
