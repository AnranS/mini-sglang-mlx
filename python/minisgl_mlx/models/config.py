"""从 HuggingFace 的 config.json 解析出我们需要的模型结构参数。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass(frozen=True)
class RotaryConfig:
    head_dim: int
    rotary_dim: int
    base: float
    max_position: int
    scaling: Dict[str, Any] | None


@dataclass(frozen=True)
class ModelConfig:
    model_type: str
    architecture: str
    num_layers: int
    hidden_size: int
    intermediate_size: int
    num_qo_heads: int
    num_kv_heads: int
    head_dim: int
    vocab_size: int
    rms_norm_eps: float
    tie_word_embeddings: bool
    attention_bias: bool  # q/k/v 投影是否带 bias（Qwen2 有）
    qk_norm: bool  # 是否对 q/k 做 per-head RMSNorm（Qwen3 有）
    rotary_config: RotaryConfig
    quantization: Dict[str, Any] | None  # mlx-community 量化模型的配置
    eos_token_ids: List[int]

    @classmethod
    def from_hf(cls, cfg: Dict[str, Any]) -> ModelConfig:
        model_type = cfg["model_type"]
        num_qo_heads = cfg["num_attention_heads"]
        hidden_size = cfg["hidden_size"]
        head_dim = cfg.get("head_dim") or hidden_size // num_qo_heads

        # transformers >= 5 会把 rope_theta / rope_scaling 收进 rope_parameters 里
        rope_params = cfg.get("rope_parameters") or {}
        rope_theta = cfg.get("rope_theta", rope_params.get("rope_theta", 10000.0))
        rope_scaling = cfg.get("rope_scaling") or None
        if rope_scaling is None and rope_params.get("rope_type", "default") != "default":
            rope_scaling = rope_params

        eos = cfg.get("eos_token_id", [])
        eos = [eos] if isinstance(eos, int) else list(eos or [])

        return cls(
            model_type=model_type,
            architecture=(cfg.get("architectures") or ["LlamaForCausalLM"])[0],
            num_layers=cfg["num_hidden_layers"],
            hidden_size=hidden_size,
            intermediate_size=cfg["intermediate_size"],
            num_qo_heads=num_qo_heads,
            num_kv_heads=cfg.get("num_key_value_heads", num_qo_heads),
            head_dim=head_dim,
            vocab_size=cfg["vocab_size"],
            rms_norm_eps=cfg.get("rms_norm_eps", 1e-6),
            tie_word_embeddings=cfg.get("tie_word_embeddings", False),
            # Qwen2 的 config 里没有 attention_bias 字段，但它的 qkv 是带 bias 的
            attention_bias=cfg.get("attention_bias", model_type == "qwen2"),
            qk_norm=model_type in ("qwen3",),
            rotary_config=RotaryConfig(
                head_dim=head_dim,
                rotary_dim=int(head_dim * cfg.get("partial_rotary_factor", 1.0)),
                base=float(rope_theta),
                max_position=cfg.get("max_position_embeddings", 4096),
                scaling=rope_scaling,
            ),
            quantization=cfg.get("quantization") or cfg.get("quantization_config"),
            eos_token_ids=eos,
        )

    @property
    def kv_bytes_per_token(self) -> int:
        """每个 token 在所有层上的 K + V 占用多少字节（按 bf16 计算）。"""
        return 2 * self.num_layers * self.num_kv_heads * self.head_dim * 2
