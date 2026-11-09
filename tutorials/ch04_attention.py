"""
第 4 章实验：decode attention 的两种实现 —— gather + SDPA vs 手写 paged Metal kernel。

    python tutorials/ch04_attention.py

观察点：
  1. 两者数值一致（bf16 误差 ~1e-3）
  2. gather 本身的开销：把散落的 KV 拷成连续内存，比真正的 attention 计算还贵
  3. 随着 batch 变大，paged kernel 的优势越来越明显
"""

import time

import mlx.core as mx
import numpy as np

from minisgl_mlx.attention import paged_decode_attention

N, Hq, Hkv, D = 300_000, 16, 8, 128  # Qwen3-0.6B 的 head 配置；pool 约 1.2 GB
k_pool = mx.random.normal((N, Hkv, D)).astype(mx.bfloat16)
v_pool = mx.random.normal((N, Hkv, D)).astype(mx.bfloat16)
scale = mx.array([D**-0.5], dtype=mx.float32)
rng = np.random.default_rng(0)


def bench(fn, iters=20):
    for _ in range(3):
        mx.eval(fn())
    t = time.perf_counter()
    for _ in range(iters):
        mx.eval(fn())
    return (time.perf_counter() - t) / iters * 1e3


print(f"{'B':>4} {'L':>6} | {'gather':>8} {'gather+sdpa':>12} {'paged kernel':>13} | {'max err':>8}")
for B, L in [(1, 4096), (8, 2048), (32, 2048), (64, 1024), (128, 1024)]:
    lens = rng.integers(L // 2, L + 1, B)
    lens[0] = L
    indptr = np.concatenate([[0], np.cumsum(lens)]).astype(np.int32)
    flat = rng.choice(N, int(indptr[-1]), replace=False).astype(np.int32)  # 随机、不连续的槽位
    padded = np.zeros((B, L), dtype=np.int32)
    for i in range(B):
        padded[i, : lens[i]] = flat[indptr[i] : indptr[i + 1]]
    q = mx.random.normal((B, Hq, D)).astype(mx.bfloat16)
    padded_mx, flat_mx, indptr_mx = mx.array(padded), mx.array(flat), mx.array(indptr)
    mask = mx.array(np.arange(L)[None, :] < lens[:, None])[:, None, None, :]

    def gather_only():
        return k_pool[padded_mx], v_pool[padded_mx]

    def naive():  # attention/naive.py 的做法
        k = k_pool[padded_mx].transpose(0, 2, 1, 3)
        v = v_pool[padded_mx].transpose(0, 2, 1, 3)
        return mx.fast.scaled_dot_product_attention(q[:, :, None], k, v, scale=D**-0.5, mask=mask)[:, :, 0]

    def paged():  # attention/metal.py 的做法
        return paged_decode_attention(q, k_pool, v_pool, indptr_mx, flat_mx, scale)

    err = mx.abs(naive().astype(mx.float32) - paged().astype(mx.float32)).max().item()
    print(f"{B:4d} {L:6d} | {bench(gather_only):6.3f}ms {bench(naive):10.3f}ms {bench(paged):11.3f}ms | {err:8.4f}")

print("\n每一行是 *一层* attention 的耗时；Qwen3-0.6B 有 28 层，乘以 28 就是每个 decode step 的 attention 开销。")
