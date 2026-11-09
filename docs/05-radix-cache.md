# 第 5 章：Radix 前缀缓存

> 代码：`kvcache/radix_cache.py`、`scheduler/cache.py`（`CacheManager.cache_req`）
> 实验：`python tutorials/ch05_radix_cache.py`，测试：`tests/test_radix_cache.py`、`tests/test_cache_manager.py`

## 5.1 动机

多轮对话、共享 system prompt、few-shot 示例……大量请求的开头是相同的。相同前缀的 KV 完全相同，
没有理由每个请求都重新 prefill 一遍。

## 5.2 数据结构

一棵基数树（压缩前缀树），每个节点存一段 token 序列（key）和对应的 KV 槽位（value）：

```
root
├── '今天天气很好，我们去' (len=6, ref=0)
│   ├── '公园散步吧。' (len=4, ref=0)
│   └── '爬山吧。' (len=4, ref=0)
└── 'Hello, how are you?' (len=6, ref=0)
```

（这是 `ch05_radix_cache.py` 的真实输出：插入第二句时，与第一句的公共前缀让节点分裂成了两段。）

- 子节点用"第一个 token"（page_size > 1 时是第一页的 token 元组）做字典 key，查找是 O(1)；
- 比较一段 key 与输入的公共前缀长度用 numpy 向量化（`fast_compare_key`，原版用 C++ 写）。

## 5.3 四个操作

**match_prefix(input_ids)**：从根向下走，返回命中长度和末端节点。如果只命中了某个节点的前半段，就把它分裂。

> 注意 `CacheManager.match_req` 只匹配 `input_ids[:-1]`：即使整个 prompt 都在缓存里，
> 最后一个 token 也必须真正算一遍，否则拿不到用于采样的 logits。

**insert_prefix(input_ids, indices)**：插入一段（token → 槽位）。返回值里的 `cached_len` 表示
"插入前树里已经有了多长" —— 这一段调用方自己的槽位是重复的，应该释放。

**lock_handle / 引用计数**：请求正在使用一条路径时，路径上每个节点 `ref_count += 1`；
`ref_count > 0` 的节点受保护，不能被淘汰。`size_info` 维护可淘汰 / 受保护的 token 数。

**evict(size)**：收集所有 `ref_count == 0` 的叶子，按 LRU（timestamp）用堆依次淘汰；
父节点因此变成叶子时也加入候选。被淘汰节点的槽位归还给 `free_slots`。

## 5.4 一个请求与 radix cache 的完整交互

```
到达    match_prefix -> 命中 40 个 token -> lock -> page_table[row, :40] = 命中的槽位
prefill 为 [40, P) 分配新槽位，计算
prefill 结束  cache_req(finished=False)：把 [0, P) 插入树，换成新的 handle 并加锁
decode  ...
结束    cache_req(finished=True)：把 [0, 全长) 插入树，解锁；树里已有的重复部分释放
```

`cache_req` 是整个缓存管理里最精巧的函数，它把一个请求的 KV 槽位分成四段处理（见代码注释）：

```
[0, old)               本来就是从缓存里借来的
[old, prefix_len)      插入时发现别的请求已经放进树里了 —— 我们这份是重复的，释放
[prefix_len, new)      新插入的部分，所有权移交给树
[new, cached_len)      page_size > 1 时不足一页的尾巴：请求结束则释放，否则留给请求自己
```

为什么会有第二段？两个相同前缀的请求同时 prefill（都没命中），先结束的那个把前缀插入了树；
后结束的那个插入时就会发现前缀已经存在，自己算的那份就是重复的。

## 5.5 正确性的守护者：check_integrity

调度器每次空闲时都会调用 `CacheManager.check_integrity()`：

```
空闲页数 + 前缀缓存中的页数 == 总页数
```

只要有一个槽位泄漏或被重复释放，这个等式就会被打破。`tests/` 和每个 tutorial 结束时也都会检查它。
写调度器时，这类"守恒检查"非常值得。

## 5.6 实测收益

`ch05_radix_cache.py` 第二部分：32 个请求共享一个约 580 token 的 system prompt，每个请求只生成 1 个 token（只看 prefill）：

```
cache=naive: 实际 prefill  18604 tokens, 命中前缀      0 tokens, 耗时 0.27s
cache=radix: 实际 prefill    556 tokens, 命中前缀  18048 tokens, 耗时 0.10s
```

prefill 计算量减少了 97%。耗时只降了 63%，因为剩下的主要是每个请求的固定开销（调度、Python 循环、采样）。
prompt 越长、模型越大，radix cache 的收益越明显。
