# 第 9 章：性能总结、与原版的差异、练习

## 9.1 性能总结（Apple M5 Ultra，Qwen3-0.6B bf16）

离线吞吐（`benchmark/offline.py`：256 个请求，输入 100~1024、输出 100~1024 个随机 token，ignore_eos，并发上限 128）：

| 配置 | 输出吞吐 |
|---|---|
| **mini-sglang-mlx**（metal attention + overlap，默认） | **4427 tok/s** |
| mini-sglang-mlx，关闭 overlap | 4125 tok/s |
| mini-sglang-mlx，naive attention（gather + SDPA） | 645 tok/s |
| mlx-lm `BatchGenerator`（`completion_batch_size=128`） | 963 tok/s |

单请求：mlx-lm `stream_generate` 384 tok/s，本项目 ~260–280 tok/s（第 10.4 节用 GPU 时间线拆解了原因）。

在线（`benchmark/online.py`，64 个请求，并发 32）：TTFT p50 40 ms，TPOT p50 8.1 ms。

> 这些数字只用于理解各个优化的相对效果，不是严格的 benchmark：不同机器、不同负载下差异很大。
> 自己跑一遍，比读数字更有意义。

## 9.2 与原版 mini-sglang 的差异

| 方面 | 原版 | 本项目 |
|---|---|---|
| 硬件 | NVIDIA GPU，Linux | Apple Silicon，macOS |
| 张量库 | PyTorch + FlashInfer / FlashAttention / sgl-kernel | MLX（+ 一个自定义 Metal kernel） |
| Tensor parallel | 支持（NCCL） | 不支持 |
| CUDA Graph | 有 | 无（MLX 惰性图，构图开销本来就小） |
| page table / token pool | GPU tensor | CPU numpy（token pool 只存每个请求最后一个 token，放在 GPU） |
| overlap | 两个 CUDA stream，构图即提交 | 惰性图 + async_eval，**等上一步完成后再提交**（为了 buffer 捐赠） |
| decode attention | FlashInfer / FA3 / TRT-LLM | 手写 paged Metal kernel |
| MoE | 支持（Triton fused MoE） | 未实现（练习 5） |
| 量化模型 | — | 支持 mlx-community 的量化格式 |
| 代码量 | ~5000 行 Python + CUDA | ~3500 行 Python（含大量中文注释） |

本项目修正 / 调整的几个行为（详见对应章节）：

- overlap 下按 `can_decode` 判断长度上限会少生成一个 token → 改为按 host 侧长度判断（7.4）；
- overlap 下已结束请求在下一个 batch 中的多余 token 不再发送（7.5）；
- 流式反分词在结束时输出残余文本（8.3）；
- 输入长度为 0 或超长的请求会立即返回结束消息，而不是静默丢弃（否则客户端会一直等待）；
- 前缀命中统计不把 chunked prefill 的后续 chunk 算作命中。

## 9.3 练习

难度：⭐ 入门 / ⭐⭐ 中等 / ⭐⭐⭐ 挑战。每个练习都可以用 `tests/test_e2e.py`（float32 下与 mlx-lm 逐 token 对比）验证正确性。

1. ⭐ **stop 字符串 / stop token ids。** 在 `SamplingParams` 里加上 `stop`，在 detokenizer 或调度器里实现，并接到 OpenAI API。
   思考：在调度器里判断 stop 字符串需要 detokenize，代价是什么？放在 detokenizer 里又需要什么额外的消息？

2. ⭐ **融合投影（已实现，可以对照阅读）。** 现在的代码已经把 `q_proj/k_proj/v_proj` 合并成 `qkv_proj`、
   `gate_proj/up_proj` 合并成 `gate_up_proj`（`models/weight.py` 的 `_fuse_projections`），并编译了 swiglu。
   练习：把它改回不融合的版本，用 `ch10_profiling.py ops --batch 1` 测一下差别
   （绕开调度器的流水线测试中，单请求从约 294 提到约 311 tok/s）。

3. ⭐⭐ **split-K paged decode（flash-decoding）。** 第 4 章看到 B=1 时 paged kernel 比 SDPA 还慢，因为只有 `Hkv` 个 threadgroup。
   把每个请求的 KV 切成 S 段，grid 变成 `(NSG*32, Hkv, B*S)`，kernel 输出每段的 `(o, m, l)`，再用 MLX 算子合并：
   `M = max(m); o = Σ exp(m_s - M) · l_s · o_s / Σ exp(m_s - M) · l_s`。

4. ⭐⭐ **变长 prefill kernel。** 现在的 prefill 逐请求循环调用 SDPA。实现一个接受 `cu_seqlens` 的 prefill attention
   （可以先用"padding + mask 的一次 SDPA"做基线），测 64 个短 prompt 同时 prefill 的速度。

5. ⭐⭐ **MoE 模型（Qwen3-30B-A3B）。** 参考 mlx-lm 的 `SwitchGLU`（`mx.gather_qmm`），实现 MoE 层。
   `mlx-community/Qwen3-30B-A3B-4bit` 是一个很好的测试对象，它和 dense 模型共用全部调度代码。

6. ⭐⭐ **更激进的准入控制 + retract。** 现在按"每个请求都生成到 max_tokens"预留 KV（第 6.2 节），利用率偏低。
   改为乐观估计（例如按 max_tokens 的一半），并实现 KV 不够时把一部分 decode 请求"撤回"到 pending 队列
   （释放 KV，之后重新 prefill，radix cache 会让重新 prefill 很便宜）。

7. ⭐⭐ **logprobs。** 在采样时顺便返回所选 token 的 log 概率（以及 top-k），一路传到 OpenAI API。

8. ⭐⭐ **调度策略实验。** 现在是"prefill 优先 + FCFS"。试试 decode 优先、或者按 prompt 长度（SJF）排队，
   用 `benchmark/online.py` 观察 TTFT / TPOT 的变化。

9. ⭐⭐⭐ **追上 mlx-lm 的单请求速度。** 第 10.4 节用 GPU 时间线把 B=1 的差距拆成了两半：
   每步 GPU 计算多约 0.5 ms，步与步之间空闲多约 0.4–0.8 ms。分别优化，并用
   `python tutorials/ch10_profiling.py timeline --with-mlx-lm --batch 1` 验证每一步改动：
   计算那一半可以从 split-K 的 paged kernel（练习 3）入手（融合投影已经做了，见练习 2）；
   空闲那一半要先弄清 `async_eval` 为什么会阻塞（第 10.2 节），可以试试用 `mx.compile` 减少每步的算子数量。

10. ⭐⭐⭐ **KV cache 量化。** 把 KV pool 存成 int8 / fp8（每个 token 每个 head 一个 scale），
    相应修改 `store_kv` 和 paged decode kernel。同样的内存可以多放一倍的 token。

11. ⭐⭐⭐ **多机 tensor parallel。** 用 `mx.distributed`（支持 Thunderbolt / 以太网）把一个大模型切到两台 Mac 上。
    这时原版的 `distributed/`、`layers/linear.py` 里的切分逻辑就有用了。
