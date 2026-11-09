# 第 7 章：Overlap 调度 —— 以及 MLX 下的三个坑

> 代码：`scheduler/scheduler.py`（`overlap_loop`、`_forward`、`_process_last_data`）、`scheduler/table.py`
> 实验：`python tutorials/ch07_overlap.py`

## 7.1 动机

每一步 decode，CPU 都要做这些事（B=64 时实测）：

| 工作 | 耗时 |
|---|---|
| 调度 + 分配槽位 + 准备元数据（`_schedule_next_batch`） | ~0.2 ms |
| 构建计算图（`_forward`，28 层） | ~0.3–0.6 ms |
| 处理结果：追加 token、判断结束、radix 插入、发消息（`_process_last_data`） | ~0.2 ms |

如果 CPU 和 GPU 轮流干活，这 ~1 ms 里 GPU 是空闲的。batch 越小、模型越小，GPU 一步越快，这部分占比就越大。

overlap 调度的思路（来自 NanoFlow，SGLang 采用）：**GPU 计算第 t 步的同时，CPU 处理第 t−1 步的结果、并准备第 t+1 步。**

## 7.2 关键问题：还不知道上一步的结果，怎么准备下一步？

准备第 t+1 步时，第 t 步还没算完：

- **不知道第 t 步采样出的 token** —— 但第 t+1 步 decode 正需要它作为输入；
- **不知道哪些请求在第 t 步遇到了 EOS** —— 它们本不该出现在第 t+1 步。

解法：

1. **CPU 假装第 t 步已经完成。** `Engine.forward_batch` 构图后立刻调用 `req.complete_one()`
   推进 `cached_len / device_len`，于是调度器可以按"t 步之后"的状态去调度 t+1。
2. **新 token 不经过 CPU。** 第 t 步把采样结果写进 GPU 上的 `token_pool[table_idx]`，
   第 t+1 步 decode 直接从 `token_pool` 读输入（`scheduler._prepare_batch` / `_forward`）。
   在 MLX 里，这两步之间只是计算图上的一条依赖边 —— 完全不需要 CPU 等待。
3. **遇到 EOS 的请求多算一步。** 它已经被放进第 t+1 步了，那就让它算，结果丢掉即可（见 7.5）。

原版（CUDA）用两个 stream 实现：调度器在自己的 stream 上准备元数据，engine stream 上跑模型。
MLX 的惰性求值让这件事更自然：`forward_batch` 本来就只是"搭图"，什么时候提交由我们决定。

## 7.3 坑一：提交时机与 buffer 捐赠

最直接的写法（mlx-lm 的 `generate_step` 就是这样）：

```python
# 写法 A
next_out = build(t + 1)
mx.async_eval(next_out)     # 立即提交第 t+1 步
wait(out_t)                 # 再等第 t 步
```

第 1 章讲过：第 t+1 步要往 KV pool 里 scatter，如果此刻第 t 步的 attention 还在 GPU 上读 KV pool ——
旧 buffer 仍被引用，**无法捐赠，MLX 只能把整个 KV pool 复制一份再写**。

本项目的写法（`Scheduler.overlap_loop`）：

```python
# 写法 B
ongoing = build(t + 1)          # ① 构图：纯 CPU 工作，与第 t 步的 GPU 计算重叠
mx.eval(last)                   # ② 等第 t 步完成 —— 此后没有任何 GPU 任务引用旧的 KV pool
mx.async_eval(ongoing)          # ③ 提交第 t+1 步，scatter 可以原地写
process(last)                   # ④ 处理第 t 步的结果，与第 t+1 步的 GPU 计算重叠
```

`ch07_overlap.py` 第一部分用一个 2 GB 的 "KV pool" 复现了两种写法：

```
A. 构图 -> 立即提交 -> 等上一步:    5.098 ms/step
B. 构图 -> 等上一步 -> 再提交:      0.430 ms/step
```

> **更正（第 10 章的测量结果）**：上面是一个**合成的最坏情况**：只有一个 pool，读完立刻就写。
> 在真实引擎里，每层有自己独立的 pool，第 t+1 步写第 i 层时，第 t 步对第 i 层的读取早就结束了。
> 用第 10 章的工具实测（Qwen3-0.6B，14 GB KV pool），写法 A **并没有触发拷贝**（峰值内存不变），
> 吞吐则互有胜负、差距在测量波动范围内。所以本项目保留写法 B：它不依赖"读写恰好错开"这种隐含条件，实测也不慢。
> 真正会让捐赠失败的，往往是一个不起眼的引用。第 10 章的 `donation` 实验演示了这种 bug，
> 以及怎样用"每步耗时 / 峰值内存是否随 pool 变大"来发现它。

单请求时本项目比 mlx-lm 慢：

| | 单请求 decode 速度 |
|---|---|
| mlx-lm `stream_generate` | 384 tok/s |
| 本项目 | ~260–280 tok/s |

我最初的猜测是"写法 B 每步留下了空隙"，但上面的测量否定了"换成写法 A 就能好"。
第 10 章用 GPU 时间线把差距拆开了：一部分是我们每步的 GPU 计算本身更多（多 0.4~0.5 ms，B=1 时自定义 kernel、scatter 等效率较低），
另一部分是步与步之间 GPU 的空闲更长（多 0.4~0.8 ms）。batch ≥ 4 时，continuous batching 带来的收益远大于这部分差距。

## 7.4 坑二：off-by-one —— overlap 让请求少生成了一个 token

最初的实现（照搬原版）在处理第 t 步的结果时这样判断长度上限：

```python
req.append_host(token)
finished = not req.can_decode        # can_decode = device_len < max_device_len
```

非 overlap 模式下完全正确。但 overlap 模式下，处理第 t 步结果时，第 t+1 步已经被调度，
`complete_one()` 已经把 `device_len` 又推进了一格 —— **于是提前一个 token 判定结束**。
`max_tokens=2` 的请求只输出 1 个 token。

修复：用 CPU 侧真正处理过的 token 数判断：

```python
finished = len(req.input_ids) >= req.max_device_len
```

回归测试：`tests/test_e2e.py::test_exact_max_tokens`（overlap 开 / 关两种模式）。
原版 mini-sglang 用的也是 `not req.can_decode`。它的离线接口之所以没有少 token，
是因为第 t+1 步的那个 token 仍然会被发出去，见下一节。

> 教训：overlap 调度下，**CPU 侧任何"读取请求状态"的代码都要问自己：这个状态是"已完成的"还是"已调度的"？**

## 7.5 坑三：已结束的请求出现在下一个 batch 里

请求在第 t 步遇到 EOS，但处理第 t 步结果时它已经在第 t+1 步里了。处理第 t+1 步的结果时：

- 不能再释放一次资源（double free）—— `finished_reqs` 记录了上一轮刚结束的请求；
- 也**不应该**把这个多余的 token 发给 detokenizer。原版只跳过了释放，仍然会发送这个 token；
  本项目直接 `continue`，跳过这个请求的所有处理。

## 7.6 收益实测

`ch07_overlap.py` 第二部分（Qwen3-0.6B，prompt 200 token，每个请求生成 512 token）：

| batch | overlap 关 | overlap 开 | 提升 |
|---|---|---|---|
| 1 | 238 tok/s | 258 tok/s | +8% |
| 4 | 941 | 1183 | +26% |
| 16 | 2832 | 3241 | +14% |
| 64 | 7050 | 8027 | +14% |

还有一个观察：实测 `mx.async_eval` 并不总是"立即返回"。B=64 时它要阻塞约 10 ms，接近一步 GPU 计算的时间。
推测是 MLX 对在途的 command buffer 数量有上限，提交到一定量就会等前面的完成。
所以 MLX 上的 overlap 是"部分重叠"，收益比 CUDA 上小。第 10 章的 `phases` 和 `timeline` 两个实验
分别从 CPU 侧和 GPU 侧量化了这一点。
