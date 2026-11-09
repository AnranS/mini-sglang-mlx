"""
KV pool：所有请求共享的一大块 KV 存储，对应原版 minisgl/kvcache/mha_pool.py。

    k[layer] : [num_tokens, num_kv_heads, head_dim]
    v[layer] : [num_tokens, num_kv_heads, head_dim]

"槽位 (slot)" 就是第 0 维的下标。一个请求的 KV 散落在 pool 的任意槽位上，
由 page table 记录 "第 i 个请求的第 pos 个 token -> 槽位"。这就是 PagedAttention 的核心思想：
像操作系统的虚拟内存一样，逻辑上连续的序列，物理上可以不连续，于是不会产生碎片。

MLX 小知识：mx.array 在语义上是"不可变"的，`a[idx] = b` 会生成一个 scatter 操作，
并让 Python 对象 a 指向新结果。只要旧的 buffer 没有被其它地方引用，MLX 会直接复用（donate）
这块内存，从而实现真正的原地写入。什么时候会 donate 失败、退化成整块拷贝，
是 overlap 调度里必须小心的问题，见 docs/07-overlap.md。
"""

from __future__ import annotations

from typing import List

import mlx.core as mx


class MHAKVCache:
    def __init__(
        self,
        num_layers: int,
        num_tokens: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: mx.Dtype,
    ) -> None:
        shape = (num_tokens, num_kv_heads, head_dim)
        self.k: List[mx.array] = [mx.zeros(shape, dtype=dtype) for _ in range(num_layers)]
        self.v: List[mx.array] = [mx.zeros(shape, dtype=dtype) for _ in range(num_layers)]
        mx.eval(self.k, self.v)  # MLX 是惰性的：不 eval 的话内存不会真正分配
        self.num_layers = num_layers
        self.num_tokens = num_tokens
        self.dtype = dtype

    def k_cache(self, layer_id: int) -> mx.array:
        return self.k[layer_id]

    def v_cache(self, layer_id: int) -> mx.array:
        return self.v[layer_id]

    def store_kv(self, k: mx.array, v: mx.array, out_loc: mx.array, layer_id: int) -> None:
        """把本轮新算出的 k/v（[T, Hkv, D]）写进槽位 out_loc（[T]）。"""
        self.k[layer_id][out_loc] = k
        self.v[layer_id][out_loc] = v

    @property
    def nbytes(self) -> int:
        return 2 * self.num_layers * self.k[0].nbytes
