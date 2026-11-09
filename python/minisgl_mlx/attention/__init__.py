from __future__ import annotations

from typing import TYPE_CHECKING

from .base import BaseAttnBackend, PrefillMetadata
from .metal import MetalAttnBackend, paged_decode_attention
from .naive import NaiveAttnBackend

if TYPE_CHECKING:
    import numpy as np

    from minisgl_mlx.kvcache import MHAKVCache
    from minisgl_mlx.models import ModelConfig

SUPPORTED_ATTENTION_BACKENDS = {
    "naive": NaiveAttnBackend,  # gather + mask + mx.fast.scaled_dot_product_attention
    "metal": MetalAttnBackend,  # 自定义 paged decode Metal kernel
}


def create_attention_backend(
    name: str, config: ModelConfig, kv_cache: MHAKVCache, page_table: np.ndarray
) -> BaseAttnBackend:
    if name not in SUPPORTED_ATTENTION_BACKENDS:
        raise ValueError(f"Unknown attention backend {name!r}: {list(SUPPORTED_ATTENTION_BACKENDS)}")
    return SUPPORTED_ATTENTION_BACKENDS[name](config, kv_cache, page_table)


__all__ = [
    "BaseAttnBackend",
    "PrefillMetadata",
    "NaiveAttnBackend",
    "MetalAttnBackend",
    "paged_decode_attention",
    "SUPPORTED_ATTENTION_BACKENDS",
    "create_attention_backend",
]
