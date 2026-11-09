# 第 1 章：写推理引擎之前，必须知道的 MLX 特性

> 实验：`python tutorials/ch01_mlx_basics.py`

MLX 的 API 长得很像 NumPy/PyTorch，但有四个特性会**从根本上影响推理引擎的设计**。
后面每一章的很多设计决策，都能在这里找到原因。

## 1.1 统一内存（Unified Memory）

Apple Silicon 的 CPU 和 GPU 共享同一块物理内存。`mx.array` 不区分 "在 CPU 上" 还是 "在 GPU 上"，
只区分 "在哪个 stream 上计算"。

对推理引擎的影响：

- **调度元数据可以放在 CPU 上。** 原版把 page table（`[max_req, max_seq_len]`）放在显存里，
  因为 CPU → GPU 的拷贝要过 PCIe，必须尽量少传、异步传。在 Mac 上 `mx.array(np_array)` 只是一次内存拷贝，
  所以本项目的 page table 是一个普通的 numpy 数组，调度器可以随意读写，每一步只把当前 batch 需要的那几行转成 `mx.array`。
- **读回结果几乎零成本。** `next_tokens.tolist()` 不需要 device → host 拷贝。
- **KV cache 预算要和系统共享。** 显存就是内存。GPU 能用多少由
  `mx.device_info()["max_recommended_working_set_size"]` 给出（一般是物理内存的 ~75%），
  分给 KV cache 太多会挤占系统内存、引发 swap（见第 3、4 章）。

## 1.2 惰性求值（Lazy Evaluation）

```python
c = a @ b + 1     # 只是记录了一个计算图，没有任何计算发生
mx.eval(c)        # 这时才真正把计算提交给 GPU，并等待完成
```

对推理引擎的影响：

- `Engine.forward_batch()` 只是"搭建计算图"，返回一个还没算出来的 `next_tokens`。
  **什么时候真正计算，由调度器决定** —— 这是 overlap 调度的基础（第 7 章）。
- 原版用 CUDA Graph 来消除 decode 阶段的 kernel launch 开销。MLX 本身就是"先建图、后执行"，
  构图只是 C++ 层面创建一些节点，实测 28 层的 Qwen3-0.6B 构建一步 decode 的图约 0.3~0.6 ms，所以本项目没有 CUDA Graph 的对应物。
  （想进一步消除这部分开销，可以研究 `mx.compile`，见练习。）
- 一个容易踩的坑：`mx.zeros(...)` 也是惰性的，**不 eval 就不会真正分配内存**。所以 `MHAKVCache` 创建完 pool 后立刻 `mx.eval`。

## 1.3 异步求值（`mx.async_eval`）

```python
out = model(x)
mx.async_eval(out)   # 提交给 GPU，立即返回
do_cpu_work()        # GPU 计算的同时，CPU 做别的事
mx.eval(out)         # 需要结果时再等待（如果已经算完，立即返回）
```

这就是 overlap 调度的核心原语：GPU 算第 t 步时，CPU 去处理第 t-1 步的结果、准备第 t+1 步。
原版 CUDA 实现用两个 CUDA stream 达到同样的目的。

## 1.4 原地更新与 buffer 捐赠（Donation）

MLX 的数组在语义上是不可变的。下面这行代码：

```python
pool[idx] = new_kv
```

实际上生成了一个 scatter 操作，产生一个**新数组**，然后让 Python 对象 `pool` 指向它。
那它会把整个 pool 复制一遍吗？

- 如果旧的 `pool` buffer **没有被任何其他东西引用**，MLX 会把这块 buffer 直接"捐赠"给 scatter 的输出 ——
  只写入 `idx` 处的数据，这才是我们想要的原地更新，开销与写入量成正比。
- 如果旧的 buffer 还被引用着（比如另一个还没算完的操作要读它），MLX 只能**先把整个 buffer 复制一份**再写。
  对于一个几 GB 的 KV pool，这是灾难性的。

实验脚本的第 4 部分演示了这一点（2GB 的 pool，每次写 64 行，M5 Ultra 实测）：

```
可以捐赠时，每次 scatter: ~1 ms     <- 基本都是"提交 + 同步等待"的固定开销
不能捐赠时，每次 scatter: ~4.3 ms   <- 多出来的 3 ms 是把 2GB 整块拷贝了一遍
```

pool 越大差距越大：真实的 KV pool 有十几 GB、每层一个，一旦捐赠失败，每一步 decode 都会复制整个 KV cache。

对推理引擎的影响：

- KV pool 必须每层一个独立的 `mx.array`（`kvcache/pool.py`），并且只通过 `pool[out_loc] = k` 的方式写入。
- **overlap 调度的提交时机必须小心**：如果第 t 步对 KV pool 的读取还在 GPU 上进行，就提交第 t+1 步，
  第 t+1 步的 scatter 就无法捐赠，每一步、每一层都会复制整个 KV pool。第 7 章会详细讲本项目是怎么避开这个坑的。

## 小结

| 特性 | 带来的好处 | 需要注意的地方 |
|---|---|---|
| 统一内存 | 元数据放 CPU，读回结果零成本 | KV 预算与系统内存共享 |
| 惰性求值 | 构图即"录制"，不需要 CUDA Graph | 不 eval 就不分配、不计算 |
| async_eval | 天然支持 CPU/GPU 重叠 | 提交时机影响 buffer 捐赠 |
| buffer 捐赠 | scatter 可以真正原地写 | 旧 buffer 被引用时会整块拷贝 |
