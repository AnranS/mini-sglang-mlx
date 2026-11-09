# 第 4 章：Attention —— 从 gather + SDPA 到手写 paged Metal kernel

> 代码：`attention/base.py`（prefill）、`attention/naive.py`、`attention/metal.py`
> 实验：`python tutorials/ch04_attention.py`，测试：`tests/test_attention.py`

## 4.1 问题是什么

经过第 2、3 章，attention 后端拿到的是：

- `q: [T, Hq, D]`、`k, v: [T, Hkv, D]` —— packed 的、本轮新 token 的 q/k/v（k/v 已经写进了 KV pool）；
- 每个请求的历史 KV 散落在 KV pool 的哪些槽位上 —— 由 page table 给出。

要算的是：每个新 token 对"同一个请求中、位置不超过它的所有 token"做 attention。

prefill 和 decode 的性质完全不同，所以分开处理（原版也是如此，甚至允许二者用不同的后端，见原版的 `HybridBackend`）：

| | prefill | decode |
|---|---|---|
| 每个请求的 query 数 | 很多（几十到几千） | 1 |
| 瓶颈 | 计算（矩阵乘） | 访存（读全部历史 KV） |
| batch 内请求数 | 通常不多 | 可能上百 |

## 4.2 prefill：逐请求调用 SDPA

```python
# attention/base.py: _forward_prefill
for 每个请求 i:
    qi = q[s:e]                                     # [Lq, Hq, D]
    ki = k[s:e] if 没有前缀命中 else k_pool[kv_idx]   # [Lkv, Hkv, D]
    oi = mx.fast.scaled_dot_product_attention(qi, ki, vi, mask="causal")
```

两个细节：

1. **`mask="causal"` 正好能处理前缀命中的情况。** 前缀命中时 `Lq < Lkv`（只算后半段 query，但要看完整的 KV）。
   MLX 的 causal mask 是**右下角对齐**的（最后一个 query 对齐最后一个 key），这恰好就是我们要的语义。
2. **没有前缀时不必从 pool 里读。** 本轮算出的 k/v 就是完整的 KV，直接用，省掉一次 gather。

GQA 不需要手动复制 kv head，`mx.fast.scaled_dot_product_attention` 原生支持 `Hq ≠ Hkv`。

逐请求循环有 Python 开销，但 prefill 是计算密集的，这点开销相对很小。
（更好的做法是一个支持变长的 kernel，见练习。）

## 4.3 decode 方案一：gather + mask + SDPA（`naive`）

```python
kv_indices: [B, L_max]   # page_table[rows, :L_max]，超出长度的部分是脏数据
mask:       [B, 1, 1, L_max]   # j < len_b
k = k_pool[kv_indices].transpose(0, 2, 1, 3)   # [B, Hkv, L_max, D]  <- gather：把散落的 KV 拷成连续的
o = mx.fast.scaled_dot_product_attention(q[:, :, None], k, v, mask=mask)
```

简单、一定正确，但代价很高：

- gather 读一遍 KV、写一遍 KV，SDPA 再读一遍 —— 访存量是理想情况的 ~3 倍；
- padding 到 `L_max`：batch 里只要有一个长请求，所有请求都按它的长度算。

## 4.4 decode 方案二：手写 paged decode kernel（`metal`，默认）

FlashInfer 的思路：**kernel 直接按索引去 KV pool 里读，不需要先把 KV 拷成连续内存。**
索引用 CSR 格式（和 FlashInfer 一样）：

```
kv_indptr  = [0, L0, L0+L1, ...]                 # [B+1]
kv_indices = [请求0 的 L0 个槽位, 请求1 的 L1 个槽位, ...]   # [ΣL]，没有 padding
```

`mx.fast.metal_kernel` 允许在 Python 里写一段 Metal shader，运行时编译。kernel 的并行方式（`attention/metal.py`）：

```
grid = (NSG*32, Hkv, B)：一个 threadgroup 负责一个 (请求 b, kv head h)
  ├─ 共享这个 kv head 的 G = Hq/Hkv 个 q head 也在这里算 —— 每个 K/V 只读一次（GQA 友好）
  ├─ NSG 个 simdgroup：第 s 个处理第 s, s+NSG, s+2·NSG, ... 个 key
  │    └─ 32 个线程各持 D/32 个元素；点积先线程内累加，再 simd_sum
  │    └─ online softmax：维护 (m, l, acc)，新 score 到来时按 exp(m_old - m_new) 缩放旧结果
  └─ 最后通过 threadgroup memory 合并 NSG 份 (m, l, acc)
```

online softmax 是 FlashAttention 的核心技巧：不需要先算出所有 score 再 softmax，可以边读 key 边更新，
因此只需要扫一遍 KV。

### 实测（M5 Ultra，Qwen3-0.6B 的 head 配置，单层耗时）

```
   B      L |   gather  gather+sdpa  paged kernel |  max err
   1   4096 |  0.284ms      0.304ms       0.458ms |   0.0005
   8   2048 |  0.417ms      0.588ms       0.382ms |   0.0010
  32   2048 |  0.766ms      1.883ms       0.589ms |   0.0010
  64   1024 |  0.848ms      1.753ms       0.568ms |   0.0020
 128   1024 |  1.365ms      3.378ms       0.665ms |   0.0020
```

几个观察：

- **光是 gather 就比 paged kernel 的全部计算还慢**（B ≥ 32）。这就是 FlashInfer 这类 kernel 存在的意义。
- batch 越大优势越明显：B=128 时快 5 倍。
- **B=1 时 paged kernel 反而更慢**：只有 `B × Hkv = 8` 个 threadgroup，GPU 大部分核心闲着。
  解决办法是 split-K（flash-decoding）：把一个请求的 KV 切成几段并行算，再合并 —— 留作练习。

端到端（`benchmark/offline.py`，256 个请求，并发上限 128）：`metal` 4427 tok/s，`naive` 645 tok/s。

## 4.5 踩坑记录：MLX 的 buffer 缓存把内存吃光了

第一次跑 `naive` 后端的 benchmark 时，吞吐只有 268 tok/s，而且每次运行差异很大。排查发现：

```
attn=naive: active 8.8 GiB, cache 65.0 GiB   <- !!!
attn=metal: active 8.8 GiB, cache  2.3 GiB
```

MLX 会把释放的 buffer 留在缓存里供后续复用。naive 后端每一步都要 gather 出 `[B, L_max, H, D]` 的临时数组，
而 `L_max` 每一步都在变 —— 尺寸各不相同的 buffer 很难被复用，缓存就一直涨，最后挤占系统内存、触发 swap。

两个修复（都在代码里）：

1. `Engine.__init__` 调用 `mx.set_cache_limit(4 GiB)`（`--mlx-cache-limit-gb`）；
2. naive 后端把 `L_max` 向上对齐到 128 的倍数，让临时 buffer 的尺寸可以复用。

修复后 naive 稳定在 645 tok/s。教训：**在统一内存的机器上，MLX 的缓存就是系统内存**，
服务端程序一定要给它设上限；张量形状的"分桶"不只是为了编译缓存，也是为了内存复用。
