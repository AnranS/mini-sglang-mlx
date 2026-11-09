import numpy as np

from minisgl_mlx.core import Req, SamplingParams
from minisgl_mlx.scheduler.cache import CacheManager


def make_req(cm: CacheManager, tokens, table_idx=0):
    handle = cm.match_req(_Pending(np.array(tokens, dtype=np.int32))).handle
    cm.lock(handle)
    cm.page_table[table_idx, : handle.cached_len] = handle.get_matched_indices()
    return Req(
        input_ids=np.array(tokens, dtype=np.int32),
        table_idx=table_idx,
        cached_len=handle.cached_len,
        output_len=4,
        uid=0,
        sampling_params=SamplingParams(),
        cache_handle=handle,
    )


class _Pending:
    def __init__(self, input_ids):
        self.input_ids = input_ids
        self.input_len = len(input_ids)


def run_prefill_and_finish(cm: CacheManager, tokens, table_idx=0):
    req = make_req(cm, tokens, table_idx)
    cm.allocate_paged([req])
    req.complete_one()  # 模拟 forward
    cm.cache_req(req, finished=True)
    return req


def test_allocate_free_integrity():
    page_table = np.zeros((4, 64), dtype=np.int32)
    cm = CacheManager(num_pages=32, page_size=1, page_table=page_table, type="radix")
    run_prefill_and_finish(cm, [1, 2, 3, 4, 5])
    cm.check_integrity()
    assert cm.prefix_cache.size_info.evictable_size == 5

    # 完全相同的请求：只命中 4 个 token —— 最后一个 token 必须重新算，才能拿到 logits 去采样
    same = make_req(cm, [1, 2, 3, 4, 5], table_idx=2)
    assert same.cached_len == 4
    cm.unlock(same.cache_handle)

    # 共享前缀的请求：整个 [1..5] 都能命中
    req = make_req(cm, [1, 2, 3, 4, 5, 6, 7], table_idx=1)
    assert req.cached_len == 5
    cm.allocate_paged([req])
    req.complete_one()
    cm.cache_req(req, finished=True)
    cm.check_integrity()


def test_eviction_when_full():
    page_table = np.zeros((4, 64), dtype=np.int32)
    cm = CacheManager(num_pages=8, page_size=1, page_table=page_table, type="radix")
    run_prefill_and_finish(cm, [1, 2, 3, 4, 5, 6])
    assert len(cm.free_slots) == 2 and cm.available_size == 8
    run_prefill_and_finish(cm, [9, 9, 9, 9, 9], table_idx=1)  # 需要淘汰旧前缀
    cm.check_integrity()


def test_page_size_4():
    page_table = np.zeros((4, 64), dtype=np.int32)
    cm = CacheManager(num_pages=8, page_size=4, page_table=page_table, type="radix")
    req = run_prefill_and_finish(cm, list(range(10)))  # 10 个 token -> 分配 3 页
    pages = page_table[req.table_idx, :12].reshape(3, 4)
    assert np.all(pages[:, 0] % 4 == 0) and np.all(pages - pages[:, :1] == np.arange(4))
    # 完整的 2 页进入前缀缓存，不足一页的尾巴（第 3 页）被释放
    assert cm.prefix_cache.size_info.evictable_size == 8 and len(cm.free_slots) == 6
    assert np.all(cm.free_slots % 4 == 0)
    cm.check_integrity()


def test_naive_cache():
    page_table = np.zeros((4, 64), dtype=np.int32)
    cm = CacheManager(num_pages=16, page_size=1, page_table=page_table, type="naive")
    run_prefill_and_finish(cm, [1, 2, 3])
    assert len(cm.free_slots) == 16  # 什么都不缓存，全部归还
    cm.check_integrity()
