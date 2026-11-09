# 第 6 章：调度器 —— continuous batching、chunked prefill、准入控制

> 代码：`scheduler/scheduler.py`、`scheduler/prefill.py`、`scheduler/decode.py`
> 实验：`python tutorials/ch06_scheduler_trace.py`

## 6.1 主循环

```python
while True:
    msgs = receive_msg()               # 新请求 -> PrefillManager.pending_list；abort -> 释放资源
    batch = prefill_manager.schedule_next_batch(budget) \
            or decode_manager.schedule_next_batch()      # prefill 优先
    forward_input = _prepare_batch(batch)                # 分配槽位、准备元数据
    next_tokens = _forward(forward_input)                # 构建计算图 & 执行
    _process_last_data(...)                              # 追加 token、判断结束、释放资源
```

**prefill 优先**：只要有能做的 prefill 就先做，所有正在 decode 的请求暂停一轮。
这让新请求尽快拿到第一个 token（降低 TTFT），代价是正在 decode 的请求会偶尔"卡"一下（TPOT 抖动）。
chunked prefill 限制了每个 prefill batch 的大小，从而限制了这种卡顿的长度。

## 6.2 准入控制：PrefillAdder 的两个预算

```python
token_budget  = max_extend_tokens          # 本 batch 最多算多少 token
reserved_size = decode_manager.inflight_tokens   # 运行中的请求"未来最多还要多少 KV"
```

一个新请求能被接纳，当且仅当：

```
(prompt 未命中部分 + max_tokens) + reserved_size <= 空闲 KV + 可淘汰的前缀缓存
```

这是一个**保守**的策略：按"每个请求都会生成到 max_tokens"来预留空间。好处是运行中的请求
永远不会因为 KV 不够而被打断（不需要实现 preemption / 换出）；坏处是 KV 利用率偏低 ——
大多数请求在 max_tokens 之前就遇到 EOS 了。SGLang 正式版用的是"乐观估计 + 不够时撤回请求（retract）"，见练习。

注意 `_try_allocate_one` 里检查了两次：第一次在加锁前，第二次在加锁后 ——
因为给命中的前缀加锁后，这部分不再可淘汰，`available_size` 会变小。

## 6.3 chunked prefill

如果一个 prompt 剩余部分比 `token_budget` 长，就只放进 `token_budget` 个 token，构造一个 `ChunkedReq`：

- 它不采样（这个 batch 算出来的 token 没有意义），也不进入 decode；
- 它留在 `pending_list` 的**队首**，下一轮继续；
- 已分配的资源（page table 行、命中的前缀）保存在 `pending_req.chunked_req` 里，继续使用。

## 6.4 continuous batching

`DecodeManager.running_reqs` 是一个集合：

- `_forward` 之后 `filter_reqs(batch.reqs)`：刚做完 prefill 的请求加入，生成到上限的请求移出；
- 请求遇到 EOS 时在 `_process_last_data` 里移出；
- 每轮 decode 把当前所有运行中的请求组成一个 batch。

请求随时加入、随时离开，batch 的组成每一步都可能不同 —— 这就是 continuous batching（也叫 iteration-level scheduling）。
对比 static batching：一批请求必须全部结束，才能开始下一批，先结束的请求留下的空位只能空转。

## 6.5 看一次真实的调度过程

`ch06_scheduler_trace.py` 故意把 KV pool 设得很小（400 token）、chunk 设成 64，6 个请求的 trace 如下（节选）：

```
step   1 prefill | uid0[0:64](chunk)                            | pending=1 kv_free=336
step   2 prefill | uid0[64:128](chunk)                          | pending=3 kv_free=272
step   3 prefill | uid0[128:150], uid1[0:40], uid2[0:2](chunk)  | pending=2 kv_free=208
step   4 prefill | uid2[2:30]                                   | pending=3 kv_free=180
step   5 decode  | uids 0,1,2                                   | pending=3 kv_free=177
 ...                                                          ↑ uid3 需要 300+10 个 token，放不下，排队
step   9 decode  | uids 0,1                                     | pending=3 kv_free=166   <- uid2 结束
step  12 decode  | uids 1                                       | pending=3 kv_free=195   <- uid0 结束
step  13 prefill | uid3[0:64](chunk)                            | pending=3 kv_free=288   <- 空间够了，uid3 进场
 ...
step  17 prefill | uid3[256:300]                                | pending=2 kv_free=52
step  18 decode  | uids 1,3                                     | ...                      <- uid3 加入 decode
 ...
step  28 prefill | uid4[0:20], uid5[0:25]                       | pending=0 kv_free=297
step  29 decode  | uids 1,4,5                                   | ...
```

可以看到：

- **chunked prefill**：uid0（150 token）分 3 轮，uid3（300 token）分 5 轮；chunk 可以与其他请求拼在同一个 batch 里（step 3）。
- **准入控制**：step 5~12 期间 uid3 一直在排队，直到 uid0、uid2 结束释放了 KV。
- **continuous batching**：uid1 从 step 5 一直 decode 到最后，期间 uid3、uid4、uid5 先后加入。
- step 13 的 `kv_free` 比 step 12 多出约 90：uid0 / uid2 结束后 KV 留在 radix cache 里（可淘汰），
  `available_size` 把这部分也算作可用。

## 6.6 abort

客户端断开时，API server 发送 `AbortMsg`。调度器在 `pending_list` 或 `running_reqs` 里找到这个请求，
调用 `_free_req_resources`：归还 page table 行，KV 插入 radix cache 后解锁（已经算好的 KV 仍然可以被复用）。
