"""
采样，对应原版 minisgl/engine/sample.py（原版调用 FlashInfer 的 sampling kernel）。

一个 batch 里每个请求的采样参数可以不同，所以 temperature / top_k / top_p 都是 [B, 1] 的数组，
整个 batch 一次性向量化完成，不需要 Python 循环。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import mlx.core as mx

if TYPE_CHECKING:
    from minisgl_mlx.core import Batch


@dataclass
class BatchSamplingArgs:
    temperatures: mx.array | None  # [B, 1]；None 表示整个 batch 都是 greedy
    greedy_mask: mx.array | None = None  # [B] 哪些请求用 greedy
    top_k: mx.array | None = None  # [B, 1] int32
    top_p: mx.array | None = None  # [B, 1] float32


class Sampler:
    def __init__(self, vocab_size: int) -> None:
        self.vocab_size = vocab_size

    def prepare(self, batch: Batch) -> BatchSamplingArgs:
        params = [r.sampling_params for r in batch.reqs]
        if all(p.is_greedy for p in params):
            return BatchSamplingArgs(temperatures=None)

        ts = [1.0 if p.is_greedy else max(p.temperature, 1e-5) for p in params]
        ks = [p.top_k if p.top_k >= 1 else self.vocab_size for p in params]
        ps = [min(max(p.top_p, 1e-6), 1.0) for p in params]
        args = BatchSamplingArgs(
            temperatures=mx.array(ts, dtype=mx.float32)[:, None],
            greedy_mask=mx.array([p.is_greedy for p in params]),
        )
        if any(k < self.vocab_size for k in ks):
            args.top_k = mx.array(ks, dtype=mx.int32)[:, None]
        if any(p < 1.0 for p in ps):
            args.top_p = mx.array(ps, dtype=mx.float32)[:, None]
        return args

    def sample(self, logits: mx.array, args: BatchSamplingArgs) -> mx.array:
        """logits: [B, V] -> next_tokens: [B] int32（惰性，还没真正计算）"""
        greedy = mx.argmax(logits, axis=-1).astype(mx.int32)
        if args.temperatures is None:
            return greedy

        logits = logits.astype(mx.float32) / args.temperatures
        if args.top_k is not None or args.top_p is not None:
            sorted_desc = -mx.sort(-logits, axis=-1)  # 每行从大到小
            if args.top_k is not None:
                # 第 k 大的值作为阈值，小于它的全部屏蔽
                kth = mx.take_along_axis(sorted_desc, args.top_k - 1, axis=-1)
                logits = mx.where(logits < kth, -mx.inf, logits)
                rank = mx.arange(sorted_desc.shape[-1])[None, :]
                sorted_desc = mx.where(rank < args.top_k, sorted_desc, -mx.inf)
            if args.top_p is not None:
                # nucleus sampling：保留累计概率刚好超过 top_p 的最小集合
                probs = mx.softmax(sorted_desc, axis=-1)
                keep = (mx.cumsum(probs, axis=-1) - probs) < args.top_p
                threshold = mx.min(mx.where(keep, sorted_desc, mx.inf), axis=-1, keepdims=True)
                logits = mx.where(logits < threshold, -mx.inf, logits)

        sampled = mx.random.categorical(logits, axis=-1).astype(mx.int32)
        return mx.where(args.greedy_mask, greedy, sampled)
