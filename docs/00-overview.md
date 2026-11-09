# 第 0 章：全局视角

> 目标：知道系统由哪几部分组成、一个请求从进来到出去经过了哪些模块、读代码应该按什么顺序。

## 0.1 一个 LLM 推理服务要解决什么问题

如果只服务一个用户，推理很简单：prefill 一遍 prompt，然后一个 token 一个 token 地 decode。
mlx-lm 的 `generate` 就是这么做的。

难点在于**同时服务很多用户**：

| 问题 | 朴素做法的代价 | 本项目的解法 | 所在章节 |
|---|---|---|---|
| 请求长度各不相同 | padding 浪费算力和显存 | packed batch + paged KV cache | 2、3 |
| 每个请求的 KV 大小在增长 | 预留最大长度，大量浪费 | KV pool + page table（PagedAttention） | 3 |
| decode 时 attention 读取散落的 KV | 先拼成连续内存再算，多一倍访存 | 手写 paged decode Metal kernel | 4 |
| 很多请求共享同一个 system prompt | 每个请求都重新 prefill | Radix 前缀缓存 | 5 |
| 请求随时来、随时走 | 等一批都结束再开始下一批（static batching） | continuous batching | 6 |
| 一个超长 prompt 卡住所有人 | prefill 时其他请求全部停顿 | chunked prefill | 6 |
| Python 调度开销让 GPU 空等 | CPU、GPU 轮流工作 | overlap 调度 | 7 |
| 分词、网络 I/O 拖慢调度 | 都挤在一个进程里 | 多进程 + ZMQ | 8 |

## 0.2 进程结构

```
                     HTTP / SSE
   用户  <────────────────────────>  API Server (主进程, FastAPI, asyncio)
                                      │  TokenizeMsg            ▲  UserReply
                                      ▼  (ZMQ)                  │  (ZMQ)
                                  Tokenizer / Detokenizer 进程
                                      │  UserMsg                ▲  DetokenizeMsg
                                      ▼  (ZMQ)                  │  (ZMQ)
                                  Scheduler 进程  ── 持有 Engine（MLX 模型 + KV cache）
```

和原版 mini-sglang 的唯一结构差异：原版每张 GPU 一个 Scheduler 进程（tensor parallel），
进程之间还要用 NCCL 做 all-reduce；Apple Silicon 只有一个 GPU，所以只有一个 Scheduler。

## 0.3 一个请求的一生

以 `POST /v1/chat/completions` 为例：

1. **API Server** 分配 `uid`，发送 `TokenizeMsg(uid, messages, sampling_params)` 给 tokenizer 进程，
   然后在 `wait_for_replies(uid)` 上等待。（`server/api_server.py`）
2. **Tokenizer** 用 chat template 拼出 prompt、分词，发送 `UserMsg(uid, input_ids)` 给调度器。（`tokenizer/worker.py`）
3. **Scheduler** 收到后放进 `PrefillManager.pending_list` 排队。（`scheduler/prefill.py`）
4. 轮到它时，`PrefillAdder` 先去 **radix cache** 里匹配最长前缀，再检查 KV 空间够不够；
   够的话分配 page table 的一行，把命中前缀的 KV 槽位抄进去。
5. `_prepare_batch` 为需要计算的 token 分配新的 KV 槽位，准备 `positions` / `out_loc` / attention 元数据。
6. **Engine** 构建 forward + 采样的计算图（此刻还没计算！），调度器在合适的时机提交给 GPU。（`engine/engine.py`）
7. 算完后，调度器把新 token 追加到请求上：prefill 刚结束的请求把 prompt 的 KV 插入 radix cache；
   然后请求进入 `DecodeManager`，之后每一轮 decode 生成一个 token。
8. 每个新 token 以 `DetokenizeMsg` 发给 **Detokenizer**，增量地转成文本，`UserReply` 发回 API Server，
   再以 SSE 的形式推给用户。
9. 遇到 EOS 或达到 `max_tokens`，请求结束：释放 page table 的行，KV 交给 radix cache 管理（可被后续请求复用或淘汰）。

## 0.4 代码阅读顺序

推荐按"由内向外"的顺序读，每一章配一个可以直接运行的实验脚本：

| 顺序 | 文件 | 行数 | 教程 / 实验 |
|---|---|---|---|
| 1 | `core.py` | ~150 | [第 3 章](03-kv-cache.md) 的 `Req` 三个长度 |
| 2 | `models/transformer.py`, `layers.py` | ~200 | [第 2 章](02-model.md)，`tutorials/ch03_manual_engine.py` |
| 3 | `kvcache/pool.py`, `engine/engine.py` | ~180 | [第 3 章](03-kv-cache.md) |
| 4 | `attention/` | ~300 | [第 4 章](04-attention.md)，`tutorials/ch04_attention.py` |
| 5 | `kvcache/radix_cache.py`, `scheduler/cache.py` | ~400 | [第 5 章](05-radix-cache.md)，`tutorials/ch05_radix_cache.py` |
| 6 | `scheduler/prefill.py`, `decode.py`, `scheduler.py` | ~450 | [第 6、7 章](06-scheduler.md)，`ch06_scheduler_trace.py`, `ch07_overlap.py` |
| 7 | `llm.py` | ~100 | 离线接口 |
| 8 | `message.py`, `tokenizer/`, `server/` | ~600 | [第 8 章](08-server.md)，`tutorials/ch08_client.py` |
| 9 | `profiling.py` | ~300 | [第 10 章](10-profiling.md)，`tutorials/ch10_profiling.py` |

## 0.5 从 CUDA 到 MLX：概念对照

读原版代码时遇到的 CUDA 概念，在这里分别变成了什么：

| 原版 (PyTorch + CUDA) | 本项目 (MLX) | 说明 |
|---|---|---|
| `torch.cuda.Stream`，多 stream 重叠 | MLX 惰性计算图 + `mx.async_eval` | 第 1、7 章 |
| CUDA Graph 捕获 decode | 不需要 | MLX 本来就是"先建图后执行"，构图的 Python 开销很小（实测 28 层模型约 0.3~0.6 ms） |
| `pin_memory()` + `non_blocking=True` 拷贝 | `mx.array(numpy_array)` | 统一内存，没有 PCIe 传输 |
| page table / token pool 放在 GPU 上 | page table 放在 CPU（numpy）上 | 每步只把本 batch 需要的索引转成 `mx.array` |
| FlashInfer / FlashAttention | `mx.fast.scaled_dot_product_attention` + 自定义 Metal kernel | 第 4 章 |
| FlashInfer 的 RMSNorm / RoPE / SiLU | `mx.fast.rms_norm` / `mx.fast.rope` / `nn.silu` | 第 2 章 |
| FlashInfer sampling kernel | `mx.sort` / `mx.random.categorical` | `engine/sample.py` |
| `torch.cuda.mem_get_info` | `mx.device_info()["max_recommended_working_set_size"]` | 第 3 章 |
| 自定义 CUDA kernel（tvm-ffi JIT） | `mx.fast.metal_kernel`（运行时编译 Metal shader） | 第 4 章 |
| NCCL / tensor parallel | 无（可用 `mx.distributed` 扩展，见练习） | |
| `torch.cuda.empty_cache()` | `mx.set_cache_limit` / `mx.clear_cache()` | 第 4 章踩坑记录 |
