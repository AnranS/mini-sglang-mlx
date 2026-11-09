from __future__ import annotations

from .base import BaseCacheHandle, BasePrefixCache, InsertResult, MatchResult, SizeInfo
from .naive_cache import NaivePrefixCache
from .pool import MHAKVCache
from .radix_cache import RadixPrefixCache

SUPPORTED_PREFIX_CACHES = {
    "radix": RadixPrefixCache,
    "naive": NaivePrefixCache,
}


def create_prefix_cache(type: str, page_size: int) -> BasePrefixCache:
    if type not in SUPPORTED_PREFIX_CACHES:
        raise ValueError(f"Unknown cache type {type!r}, supported: {list(SUPPORTED_PREFIX_CACHES)}")
    return SUPPORTED_PREFIX_CACHES[type](page_size=page_size)


__all__ = [
    "BaseCacheHandle",
    "BasePrefixCache",
    "InsertResult",
    "MatchResult",
    "SizeInfo",
    "MHAKVCache",
    "RadixPrefixCache",
    "NaivePrefixCache",
    "SUPPORTED_PREFIX_CACHES",
    "create_prefix_cache",
]
