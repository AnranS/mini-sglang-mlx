"""
Llama 家族 decoder-only 模型（覆盖 Llama / Mistral / Qwen2 / Qwen3），对应原版 minisgl/models/*.py。

几个模型之间的差别其实很小，用 ModelConfig 中的开关表达：
    Qwen2 : q/k/v 投影带 bias            (attention_bias=True)
    Qwen3 : 对 q/k 做 per-head RMSNorm   (qk_norm=True)
    Llama : 都没有；Llama3 有特殊的 RoPE scaling（见 layers.py）

与"普通"实现（如 mlx-lm）最大的不同：输入不是 [B, L] 的二维 batch，而是 [T] 的 packed 序列。
所有请求本轮要算的 token 被拼接在一起，线性层/MLP/Norm 对 token 是逐行独立的，天然支持 packed；
只有 attention 需要知道"哪些 token 属于同一个请求"，这部分信息由 attention 后端从 Context 中获取。
"""

from __future__ import annotations

from functools import partial

import mlx.core as mx
import mlx.nn as nn

from minisgl_mlx.core import get_global_ctx
from minisgl_mlx.layers import AttentionLayer

from .config import ModelConfig


@partial(mx.compile, shapeless=True)
def swiglu(gate: mx.array, up: mx.array) -> mx.array:
    # mx.compile 把 silu 和乘法融合成一个 kernel（不编译的话是两个）
    return nn.silu(gate) * up


class Attention(nn.Module):
    def __init__(self, config: ModelConfig, layer_id: int) -> None:
        super().__init__()
        self.num_qo_heads = Hq = config.num_qo_heads
        self.num_kv_heads = Hkv = config.num_kv_heads
        self.head_dim = D = config.head_dim
        # q/k/v 三个投影融合成一个矩阵乘（权重在加载时拼接，见 weight.py）。
        # decode 时每个矩阵乘都是一次"矩阵 × 向量"，kernel 数量比计算量更要紧：每层少 2 个 kernel。
        self.qkv_proj = nn.Linear(config.hidden_size, (Hq + 2 * Hkv) * D, bias=config.attention_bias)
        self.o_proj = nn.Linear(Hq * D, config.hidden_size, bias=False)
        self.use_qk_norm = config.qk_norm
        if config.qk_norm:
            self.q_norm = nn.RMSNorm(D, eps=config.rms_norm_eps)
            self.k_norm = nn.RMSNorm(D, eps=config.rms_norm_eps)
        # 普通 Python 对象（不是 nn.Module），不会被当成参数
        self.attn = AttentionLayer(layer_id, Hq, Hkv, D, config.rotary_config)

    def __call__(self, x: mx.array) -> mx.array:
        T, Hq, Hkv, D = x.shape[0], self.num_qo_heads, self.num_kv_heads, self.head_dim
        q, k, v = mx.split(self.qkv_proj(x), [Hq * D, (Hq + Hkv) * D], axis=-1)
        q, k, v = q.reshape(T, Hq, D), k.reshape(T, Hkv, D), v.reshape(T, Hkv, D)
        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)
        return self.o_proj(self.attn(q, k, v))


class MLP(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        # gate / up 融合成一个矩阵乘（同上）
        self.gate_up_proj = nn.Linear(config.hidden_size, 2 * config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        gate, up = mx.split(self.gate_up_proj(x), 2, axis=-1)
        return self.down_proj(swiglu(gate, up))


class DecoderLayer(nn.Module):
    def __init__(self, config: ModelConfig, layer_id: int) -> None:
        super().__init__()
        self.self_attn = Attention(config, layer_id)
        self.mlp = MLP(config)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(self, x: mx.array) -> mx.array:
        h = x + self.self_attn(self.input_layernorm(x))
        return h + self.mlp(self.post_attention_layernorm(h))


class Backbone(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [DecoderLayer(config, i) for i in range(config.num_layers)]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)


class CausalLM(nn.Module):
    """参数命名沿用 HuggingFace（model.layers.0.self_attn.o_proj.weight ...）；q/k/v 与 gate/up 在加载时融合（见 weight.py）。"""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.model = Backbone(config)
        if not config.tie_word_embeddings:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def __call__(self, input_ids: mx.array) -> mx.array:
        """input_ids: [T] (packed)  ->  logits: [B, vocab]（每个请求只返回最后一个 token 的 logits）"""
        batch = get_global_ctx().batch
        h = self.model.embed_tokens(input_ids)
        for layer in self.model.layers:
            h = layer(h)
        if batch.is_prefill:
            # prefill 时一个请求有很多 token，但只有最后一个 token 的 logits 用于采样。
            # 先挑出来再过 lm_head，可以省掉一个 [T, vocab] 的巨大矩阵乘。
            h = h[batch.attn_metadata.last_indices]
        h = self.model.norm(h)
        if self.config.tie_word_embeddings:
            return self.model.embed_tokens.as_linear(h)
        return self.lm_head(h)
