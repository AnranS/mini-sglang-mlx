"""
"无状态"的算子层：RoPE 和 AttentionLayer，对应原版 minisgl/layers/{rotary,attention}.py。

原版里 RMSNorm / SiLU / RoPE 都要调用 FlashInfer 的 CUDA kernel；
在 MLX 里它们都有现成的融合实现：mx.fast.rms_norm（nn.RMSNorm 内部使用）、mx.fast.rope。
"""

from __future__ import annotations

import math
from typing import Any, Dict

import mlx.core as mx

from minisgl_mlx.core import get_global_ctx
from minisgl_mlx.models.config import RotaryConfig


class RotaryEmbedding:
    """
    RoPE。难点在于：我们的 batch 是 "packed" 的 —— 所有请求的 token 拼成一条长度为 T 的序列，
    每个 token 的位置各不相同（请求 A 的第 0..99 个 token，紧接着请求 B 的第 37..40 个 token ...）。

    mx.fast.rope 要求输入形如 (B, *, L, D)，并允许 offset 是长度为 B 的向量（每个 batch 一个起始位置）。
    技巧：把 [T, H, D] reshape 成 [T, H, 1, D]，即"每个 token 自己是一个长度为 1 的序列"，
    再把 positions[T] 作为 offset 传进去，就得到了逐 token 位置的 RoPE，而且仍然是一个融合 kernel。
    """

    def __init__(self, config: RotaryConfig) -> None:
        self.dims = config.rotary_dim
        self.base: float | None = config.base
        self.scale = 1.0
        self.freqs: mx.array | None = None
        scaling: Dict[str, Any] | None = config.scaling
        if scaling:
            rope_type = scaling.get("rope_type", scaling.get("type", "default"))
            if rope_type == "linear":
                self.scale = 1.0 / scaling["factor"]
            elif rope_type == "llama3":
                self.freqs = _llama3_freqs(self.dims, config.base, scaling)
                self.base = None
            elif rope_type != "default":
                raise NotImplementedError(f"rope_scaling type {rope_type!r} is not supported yet")

    def __call__(self, x: mx.array, positions: mx.array) -> mx.array:
        T, H, D = x.shape
        y = mx.fast.rope(
            x.reshape(T, H, 1, D),
            self.dims,
            traditional=False,  # Llama/Qwen 系列都是 "rotate half" 风格
            base=self.base,
            scale=self.scale,
            offset=positions,
            freqs=self.freqs,
        )
        return y.reshape(T, H, D)


def _llama3_freqs(dims: int, base: float, scaling: Dict[str, Any]) -> mx.array:
    # 与 mlx-lm 中 Llama3RoPE 的实现一致（mx.fast.rope 的 freqs 参数表示"周期"而非角频率）
    factor = scaling["factor"]
    low_freq_factor = scaling.get("low_freq_factor", 1.0)
    high_freq_factor = scaling.get("high_freq_factor", 4.0)
    old_context_len = scaling.get("original_max_position_embeddings", 8192)
    low_freq_wavelen = old_context_len / low_freq_factor
    high_freq_wavelen = old_context_len / high_freq_factor

    freqs = base ** (mx.arange(0, dims, 2) / dims)
    wavelens = 2 * math.pi * freqs
    freqs = mx.where(wavelens > low_freq_wavelen, freqs * factor, freqs)
    is_medium = (wavelens > high_freq_wavelen) & (wavelens < low_freq_wavelen)
    smooth = (old_context_len / wavelens - low_freq_factor) / (high_freq_factor - low_freq_factor)
    smooth_freqs = freqs / ((1 - smooth) / factor + smooth)
    return mx.where(is_medium, smooth_freqs, freqs)


class AttentionLayer:
    """
    一层 attention 中与"KV cache / 调度"相关的部分：
        1. 对 q/k 施加 RoPE
        2. 把本轮新算出的 k/v 写进 KV pool 的 out_loc 槽位
        3. 调用 attention 后端，从 KV pool 中读出完整的历史 KV 做 attention

    它不持有任何参数（q/k/v/o 投影在 models/transformer.py 的 Attention 模块里），
    所需的一切元数据都从全局 Context 中获取 —— 这和原版的设计完全相同。
    """

    def __init__(
        self,
        layer_id: int,
        num_qo_heads: int,
        num_kv_heads: int,
        head_dim: int,
        rotary_config: RotaryConfig,
    ) -> None:
        assert num_qo_heads % num_kv_heads == 0
        self.layer_id = layer_id
        self.num_qo_heads = num_qo_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.rotary = RotaryEmbedding(rotary_config)

    def __call__(self, q: mx.array, k: mx.array, v: mx.array) -> mx.array:
        """q: [T, Hq, D], k/v: [T, Hkv, D]  ->  o: [T, Hq * D]"""
        ctx = get_global_ctx()
        batch = ctx.batch
        q = self.rotary(q, batch.positions)
        k = self.rotary(k, batch.positions)
        ctx.kv_cache.store_kv(k, v, batch.out_loc, self.layer_id)
        o = ctx.attn_backend.forward(q, k, v, self.layer_id, batch)
        return o.reshape(o.shape[0], self.num_qo_heads * self.head_dim)
