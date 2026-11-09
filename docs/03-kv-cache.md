# 第 3 章：KV cache —— KV pool、page table 与请求的三个长度

> 代码：`kvcache/pool.py`、`core.py`（`Req`）、`scheduler/cache.py`（分配）、`engine/engine.py`（预算）
> 实验：`python tutorials/ch03_manual_engine.py`

## 3.1 为什么不给每个请求一块连续的 KV cache

mlx-lm 给每个请求一块连续 buffer，按需增长。服务端这样做有两个问题：

- **不知道请求会生成多长**：预留 `max_tokens` 会浪费大量内存；按需增长又会产生碎片、需要反复拷贝。
- **无法共享**：两个请求有相同的 system prompt，它们的 KV 完全相同，却要各存一份。

PagedAttention 的思路和操作系统的虚拟内存一样：**所有请求共享一个大的 KV pool，按槽位分配；
每个请求用一张"页表"记录自己的每个 token 存在哪个槽位**。逻辑上连续，物理上可以随便放。

## 3.2 KV pool

```python
# kvcache/pool.py
k[layer] : [num_tokens, num_kv_heads, head_dim]
v[layer] : [num_tokens, num_kv_heads, head_dim]
```

"槽位"就是第 0 维的下标。写入新 token 的 KV：

```python
self.k[layer_id][out_loc] = k      # out_loc: [T]，scatter
```

第 1 章讲过，这是一次真正的原地写入（前提是 buffer 可以被捐赠）。

**每个 token 占多少内存？** `2 (K,V) × 层数 × kv_heads × head_dim × 2 bytes (bf16)`

| 模型 | 层数 | kv_heads | head_dim | 每 token | 13 万 token |
|---|---|---|---|---|---|
| Qwen3-0.6B | 28 | 8 | 128 | 112 KiB | 14 GiB |
| Qwen3-8B | 36 | 8 | 128 | 144 KiB | 18 GiB |

`Engine._determine_num_tokens` 按 `memory_ratio × 推荐工作集 − 模型权重` 计算预算，再用 `--max-kv-tokens`（默认 131072）封顶；
做实验时可以用 `--num-tokens` 直接指定。

## 3.3 page table

```python
# engine/engine.py —— 注意它是 numpy 数组，在 CPU 上
page_table = np.zeros((max_running_req, max_seq_len), dtype=np.int32)
# page_table[table_idx, pos] = 第 table_idx 个运行中请求的第 pos 个 token 的 KV 槽位
```

- 每个正在运行的请求占一行，行号是 `req.table_idx`，由 `TableManager` 分配和回收。
- 原版把这张表放在 GPU 上；统一内存下没有必要，CPU 上的 numpy 读写更方便。
  每一步只把本 batch 需要的部分（`out_loc`、attention 的索引）转成 `mx.array`。

## 3.4 请求的三个长度（最重要的一节）

```
input_ids  [ prompt ........................ | 已生成的 token ... ]
           0            cached_len        device_len      max_device_len
           |<-- KV 已在 pool 中 -->|<- 本轮要算 ->|
```

- `cached_len`：前 `cached_len` 个 token 的 KV 已经在 pool 里了（可能是前缀缓存命中，也可能是之前的轮次算的）。
- `device_len`：前 `device_len` 个 token 已经"交给设备"。`[cached_len, device_len)` 就是本轮的 extend 部分。
- `max_device_len = prompt 长度 + max_tokens`。

每一轮：

1. 调度器为 `[cached_len, device_len)` 分配 KV 槽位，写进 page table（`CacheManager.allocate_paged`）。
2. 构建这批 token 的 `positions = arange(cached_len, device_len)` 和 `out_loc = page_table[row, positions]`。
3. forward 之后调用 `req.complete_one()`：`cached_len = device_len; device_len += 1`。
   新采样出的 token 在位置 `device_len - 1`，下一轮喂给模型。

于是 prefill 和 decode 统一了：

| | cached_len | device_len | 本轮计算 |
|---|---|---|---|
| prefill（无缓存） | 0 | P | P 个 token |
| prefill（命中 40 个） | 40 | P | P − 40 个 token |
| chunked prefill 第 2 块 | 512 | 1024 | 512 个 token |
| decode | n | n + 1 | 1 个 token |

`tutorials/ch03_manual_engine.py` 让你手动做一遍：分配槽位、拼 batch、prefill、decode 5 步，
最后打印出 page table —— 你会看到逻辑上连续的序列，KV 在物理上散落在各处。

## 3.5 page_size

`page_size = 1` 时按 token 分配（SGLang 的经典做法）。`page_size > 1` 时按页分配：

- `free_slots` 里存的是每页的首槽位，如 `[0, 4, 8, ...]`；
- 一个请求的最后一页可能没写满，下一轮继续写同一页，不需要新分配（`allocate_paged` 里 `div_ceil` 的作用）；
- 前缀缓存只能按整页匹配/插入（第 5 章）。

page table 始终按 token 记录（与原版一致），所以 attention 后端不需要关心 page_size。

> 思考：page_size 变大有什么好处和坏处？（提示：radix 命中粒度、尾页浪费、kernel 的访存连续性。）
