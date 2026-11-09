"""
Metal 后端：手写一个 paged decode attention kernel（相当于 FlashInfer 的 BatchDecodeWithPagedKVCache）。

KV 索引使用和 FlashInfer 一样的 CSR 格式（page_size = 1 的视角）：

    kv_indptr  = [0, L0, L0+L1, ...]              长度 B+1
    kv_indices = [请求0 的 L0 个槽位, 请求1 的 L1 个槽位, ...]

请求 b 的历史 KV 就在 k_pool[kv_indices[kv_indptr[b] : kv_indptr[b+1]]]。
不需要 padding，也不需要先把 KV gather 成连续内存 —— kernel 直接按索引去 pool 里读。

kernel 的并行方式（与 MLX 内置的 sdpa_vector 思路一致）：
  * 一个 threadgroup 负责一个 (请求 b, kv head h)；GQA 下共享这个 kv head 的 G 个 q head
    也在同一个 threadgroup 里算，于是每个 K/V 只从内存里读一次。
  * threadgroup 内有 NSG 个 simdgroup（每个 32 线程），第 s 个 simdgroup 处理第 s, s+NSG, ... 个 key。
  * simdgroup 内 32 个线程各持有 head_dim/32 个元素：点积先在线程内累加，再用 simd_sum 归约。
  * 每个 simdgroup 用 online softmax（FlashAttention 的核心技巧）维护 (max, sum, acc)，
    最后通过 threadgroup memory 把 NSG 份部分结果合并起来。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import mlx.core as mx
import numpy as np

from .base import BaseAttnBackend

if TYPE_CHECKING:
    from minisgl_mlx.core import Batch


_PAGED_DECODE_SOURCE = """
    // grid = (NSG * 32, Hkv, B)，每个 threadgroup 负责 (请求 b, kv head kvh)
    const uint b   = threadgroup_position_in_grid.z;
    const uint kvh = threadgroup_position_in_grid.y;
    const uint sg  = simdgroup_index_in_threadgroup;   // 第几个 simdgroup
    const uint ln  = thread_index_in_simdgroup;        // simdgroup 内第几个线程 (0..31)
    const uint tid = thread_index_in_threadgroup;
    constexpr int EPT = D / 32;                        // 每个线程负责 head_dim 中的几个元素
    const int Hq  = q_shape[1];
    const int Hkv = k_pool_shape[1];
    const float sc = scale[0];

    // 1. 读入 G 个 q head 的切片（预先乘好 scale）
    float qv[G][EPT], acc[G][EPT], m[G], l[G];
    for (int g = 0; g < G; ++g) {
        const device T* qp = q + (b * Hq + kvh * G + g) * D + ln * EPT;
        for (int e = 0; e < EPT; ++e) { qv[g][e] = float(qp[e]) * sc; acc[g][e] = 0.0f; }
        m[g] = -INFINITY;   // 当前见过的最大 score
        l[g] = 0.0f;        // sum(exp(score - m))
    }

    // 2. 遍历该请求的所有 key（按 simdgroup 交错分配），online softmax
    const int start = kv_indptr[b], end = kv_indptr[b + 1];
    for (int j = start + sg; j < end; j += NSG) {
        const int64_t base = (int64_t(kv_indices[j]) * Hkv + kvh) * D + ln * EPT;
        float kk[EPT], vv[EPT];
        for (int e = 0; e < EPT; ++e) { kk[e] = float(k_pool[base + e]); vv[e] = float(v_pool[base + e]); }
        for (int g = 0; g < G; ++g) {
            float s = 0.0f;
            for (int e = 0; e < EPT; ++e) s += qv[g][e] * kk[e];
            s = simd_sum(s);                               // 32 个线程的部分点积求和
            const float m_new = max(m[g], s);
            const float alpha = fast::exp(m[g] - m_new);   // 旧结果需要缩放的系数
            const float p = fast::exp(s - m_new);
            l[g] = l[g] * alpha + p;
            for (int e = 0; e < EPT; ++e) acc[g][e] = acc[g][e] * alpha + p * vv[e];
            m[g] = m_new;
        }
    }

    // 3. 合并 NSG 个 simdgroup 的部分结果
    threadgroup float sm_m[NSG], sm_l[NSG];
    threadgroup float sm_o[NSG * D];
    for (int g = 0; g < G; ++g) {
        if (ln == 0) { sm_m[sg] = m[g]; sm_l[sg] = l[g]; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float M = -INFINITY;
        for (int i = 0; i < NSG; ++i) M = max(M, sm_m[i]);
        float L = 0.0f;
        for (int i = 0; i < NSG; ++i) L += sm_l[i] * fast::exp(sm_m[i] - M);
        const float f = fast::exp(m[g] - M);
        for (int e = 0; e < EPT; ++e) sm_o[sg * D + ln * EPT + e] = acc[g][e] * f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int d = tid; d < D; d += NSG * 32) {
            float o = 0.0f;
            for (int i = 0; i < NSG; ++i) o += sm_o[i * D + d];
            out[(b * Hq + kvh * G + g) * D + d] = T(o / L);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
"""

_kernel = None


def _get_kernel():
    global _kernel
    if _kernel is None:  # 第一次调用时编译（MLX 会按模板参数缓存编译结果）
        _kernel = mx.fast.metal_kernel(
            name="minisgl_paged_decode",
            input_names=["q", "k_pool", "v_pool", "kv_indptr", "kv_indices", "scale"],
            output_names=["out"],
            source=_PAGED_DECODE_SOURCE,
        )
    return _kernel


def paged_decode_attention(
    q: mx.array,  # [B, Hq, D]
    k_pool: mx.array,  # [N, Hkv, D]
    v_pool: mx.array,  # [N, Hkv, D]
    kv_indptr: mx.array,  # [B+1] int32
    kv_indices: mx.array,  # [sum(L)] int32
    scale: mx.array,  # [1] float32
) -> mx.array:
    B, Hq, D = q.shape
    Hkv = k_pool.shape[1]
    assert D % 32 == 0 and Hq % Hkv == 0
    nsg = 32 if D <= 128 else 16  # threadgroup memory 上限 32KB：NSG * D * 4 bytes
    return _get_kernel()(
        inputs=[q, k_pool, v_pool, kv_indptr, kv_indices, scale],
        template=[("T", q.dtype), ("G", Hq // Hkv), ("D", D), ("NSG", nsg)],
        grid=(nsg * 32, Hkv, B),
        threadgroup=(nsg * 32, 1, 1),
        output_shapes=[q.shape],
        output_dtypes=[q.dtype],
    )[0]


@dataclass
class PagedDecodeMetadata:
    kv_indptr: mx.array  # [B+1]
    kv_indices: mx.array  # [sum(L)]


class MetalAttnBackend(BaseAttnBackend):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.scale_array = mx.array([self.scale], dtype=mx.float32)

    def prepare_decode(self, batch: Batch) -> PagedDecodeMetadata:
        lens = np.array([r.device_len for r in batch.reqs])
        rows = np.array([r.table_idx for r in batch.reqs])
        max_len = int(lens.max())
        valid = np.arange(max_len)[None, :] < lens[:, None]
        kv_indices = self.page_table[rows, :max_len][valid]  # 按行展平，正好是 CSR 的顺序
        kv_indptr = np.concatenate([[0], np.cumsum(lens)]).astype(np.int32)
        return PagedDecodeMetadata(kv_indptr=mx.array(kv_indptr), kv_indices=mx.array(kv_indices))

    def forward_decode(self, q: mx.array, layer_id: int, meta: PagedDecodeMetadata) -> mx.array:
        return paged_decode_attention(
            q,
            self.kv_cache.k_cache(layer_id),
            self.kv_cache.v_cache(layer_id),
            meta.kv_indptr,
            meta.kv_indices,
            self.scale_array,
        )
