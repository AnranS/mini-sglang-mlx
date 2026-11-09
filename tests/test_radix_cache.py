import numpy as np
import pytest

from minisgl_mlx.kvcache import RadixPrefixCache


def ids(*xs):
    return np.array(xs, dtype=np.int32)


def test_insert_and_match():
    cache = RadixPrefixCache()
    prefix_len, handle = cache.insert_prefix(ids(1, 2, 3, 4), ids(10, 11, 12, 13))
    assert prefix_len == 0 and handle.cached_len == 4
    m = cache.match_prefix(ids(1, 2, 3, 9, 9)).handle
    assert m.cached_len == 3
    assert m.get_matched_indices().tolist() == [10, 11, 12]
    assert cache.size_info.evictable_size == 4
    cache.check_integrity()


def test_insert_existing_prefix_returns_duplicate_len():
    cache = RadixPrefixCache()
    cache.insert_prefix(ids(1, 2, 3), ids(10, 11, 12))
    # 另一个请求算了同样的前缀 + 新的后缀：前 3 个是重复的，调用方应释放自己那份
    prefix_len, handle = cache.insert_prefix(ids(1, 2, 3, 4, 5), ids(20, 21, 22, 23, 24))
    assert prefix_len == 3 and handle.cached_len == 5
    assert handle.get_matched_indices().tolist() == [10, 11, 12, 23, 24]
    assert cache.size_info.evictable_size == 5


def test_lock_protects_from_eviction():
    cache = RadixPrefixCache()
    _, h1 = cache.insert_prefix(ids(1, 2, 3), ids(10, 11, 12))
    _, h2 = cache.insert_prefix(ids(7, 8), ids(20, 21))
    cache.lock_handle(h1)
    assert cache.size_info == (2, 3)
    with pytest.raises(AssertionError):
        cache.evict(3)
    assert sorted(cache.evict(2).tolist()) == [20, 21]
    cache.lock_handle(h1, unlock=True)
    assert cache.size_info == (3, 0)
    cache.check_integrity()


def test_split_keeps_ref_count():
    cache = RadixPrefixCache()
    _, h = cache.insert_prefix(ids(1, 2, 3, 4), ids(10, 11, 12, 13))
    cache.lock_handle(h)
    m = cache.match_prefix(ids(1, 2, 9)).handle  # 触发节点分裂 [1,2] + [3,4]
    assert m.cached_len == 2
    cache.check_integrity()
    cache.lock_handle(h, unlock=True)
    assert cache.size_info == (4, 0)
    cache.check_integrity()


def test_lru_eviction_order():
    cache = RadixPrefixCache()
    cache.insert_prefix(ids(1, 1), ids(10, 11))
    cache.insert_prefix(ids(2, 2), ids(20, 21))
    cache.match_prefix(ids(1, 1))  # 访问 [1,1]，它变成"最近使用"
    assert cache.evict(1).tolist() == [20, 21]  # 先淘汰最久没用的 [2,2]（至少淘汰一整个节点）


def test_page_size_alignment():
    cache = RadixPrefixCache(page_size=2)
    prefix_len, handle = cache.insert_prefix(ids(1, 2, 3, 4, 5), ids(10, 11, 12, 13, 14))
    assert handle.cached_len == 4  # 最后不足一页的 token 不会被插入
    assert cache.match_prefix(ids(1, 2, 3, 9)).handle.cached_len == 2  # 只能按整页匹配
