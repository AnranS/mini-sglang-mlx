"""
第 5 章实验：Radix cache。

    python tutorials/ch05_radix_cache.py

第一部分：直接操作 RadixPrefixCache，打印树的结构，观察插入、匹配、分裂、加锁、淘汰。
第二部分：用真实模型跑"共享 system prompt"的请求，对比 radix / naive 两种 cache 的 prefill 量与耗时。
"""

import time

import numpy as np

from minisgl_mlx.kvcache import RadixPrefixCache

MODEL = "mlx-community/Qwen3-0.6B-bf16"


def part1():
    print("=" * 20, "Part 1: 树的结构", "=" * 20)
    from minisgl_mlx.utils import load_tokenizer

    tok = load_tokenizer(MODEL)
    cache = RadixPrefixCache()
    next_slot = 0

    def insert(text):
        nonlocal next_slot
        ids = np.array(tok.encode(text), dtype=np.int32)
        slots = np.arange(next_slot, next_slot + len(ids), dtype=np.int32)  # 假装分配了这些 KV 槽位
        next_slot += len(ids)
        dup, handle = cache.insert_prefix(ids, slots)
        print(f"\ninsert {text!r}: {len(ids)} tokens, 其中 {dup} 个已在树中（这部分槽位应释放）")
        print(cache.pretty_print(tok))
        return handle

    insert("今天天气很好，我们去公园散步吧。")
    h = insert("今天天气很好，我们去爬山吧。")  # 共享前缀 -> 节点分裂
    insert("Hello, how are you?")

    query = np.array(tok.encode("今天天气很好，我们去公园玩"), dtype=np.int32)
    m = cache.match_prefix(query).handle
    print(f"\nmatch {tok.decode(query)!r}: 命中 {m.cached_len}/{len(query)} tokens -> 槽位 {m.get_matched_indices().tolist()}")

    cache.lock_handle(h)  # 模拟"有请求正在使用 爬山 这条路径"
    print("\nlock '...爬山吧。' 之后:", cache.size_info)
    print(cache.pretty_print(tok))
    evicted = cache.evict(1)
    print(f"\nevict(1): 淘汰了 {len(evicted)} 个 token（只能淘汰没被锁住的叶子，按 LRU 顺序）")
    print(cache.pretty_print(tok))


def part2():
    print("\n" + "=" * 20, "Part 2: 共享 system prompt 的真实收益", "=" * 20)
    from minisgl_mlx import LLM, SamplingParams

    system = "你是一个乐于助人的助手。请遵守以下规则：" + "回答要准确、简洁、有礼貌。" * 60
    questions = [f"问题 {i}: 请解释一下数字 {i} 的含义。" for i in range(32)]
    msgs = [[{"role": "system", "content": system}, {"role": "user", "content": q}] for q in questions]
    for cache_type in ["naive", "radix"]:
        llm = LLM(MODEL, cache_type=cache_type, num_tokens_override=65536)
        llm.chat(msgs[:1], SamplingParams(max_tokens=1))  # 预热（radix 模式下也会把 system prompt 放进树里）
        llm.stats.num_prefill_tokens = llm.stats.num_cached_tokens = 0
        t = time.perf_counter()
        llm.chat(msgs, SamplingParams(max_tokens=1))  # 只看 prefill：每个请求只生成 1 个 token
        elapsed = time.perf_counter() - t
        print(
            f"cache={cache_type:5s}: 实际 prefill {llm.stats.num_prefill_tokens:6d} tokens, "
            f"命中前缀 {llm.stats.num_cached_tokens:6d} tokens, 耗时 {elapsed:.2f}s"
        )
        llm.shutdown()


if __name__ == "__main__":
    part1()
    part2()
