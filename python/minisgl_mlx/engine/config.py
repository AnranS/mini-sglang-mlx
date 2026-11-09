from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EngineConfig:
    model_path: str
    dtype: str = "auto"  # auto / bfloat16 / float16 / float32
    max_running_req: int = 128  # 同时在跑的请求数上限（= page table 的行数）
    attention_backend: str = "metal"  # metal / naive
    page_size: int = 1
    # KV cache 能用多少内存：memory_ratio * GPU 推荐工作集大小 - 模型权重，并且不超过 max_kv_tokens
    memory_ratio: float = 0.6
    max_kv_tokens: int = 131072
    num_tokens_override: int | None = None  # 直接指定 KV pool 的 token 数（做实验时很有用）
    max_seq_len_override: int | None = None
    # MLX 会把释放掉的 buffer 缓存起来复用。不设上限时，形状不断变化的临时 buffer 会让缓存无限增长，
    # 在统一内存的 Mac 上直接挤占系统内存、引发 swap（见 docs/04-attention.md 的"踩坑记录"）
    mlx_cache_limit_gb: float = 4.0
    seed: int = 42
