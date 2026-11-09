# mini-sglang-mlx

**一个面向学习的、跑在 Apple Silicon 上的 LLM 推理服务框架** —— 用 [MLX](https://github.com/ml-explore/mlx)
重新实现 [mini-sglang](https://github.com/sgl-project/mini-sglang)。

原版 mini-sglang 用约 5000 行 Python 展示了现代 LLM serving 系统的核心技术，但它深度依赖 CUDA
（FlashInfer、sgl-kernel、NCCL、CUDA Graph……），只能跑在 NVIDIA GPU 上。
本项目保留原版的架构和模块划分，把所有 CUDA 相关的部分换成 MLX 实现，约 3500 行 Python（含大量中文注释），
并配了一套**从 MLX 基础讲到 overlap 调度**的中文教程，每一章都有可以直接运行的实验脚本。

## 特性

- **Paged KV cache**：所有请求共享一个 KV pool，page table 记录每个 token 的槽位
- **Radix 前缀缓存**：自动复用共享前缀（system prompt、多轮对话）的 KV
- **Continuous batching + chunked prefill**：请求随时加入/离开，长 prompt 分块 prefill
- **Overlap 调度**：GPU 计算的同时，CPU 处理上一步的结果、准备下一步
- **手写 paged decode Metal kernel**：直接按索引读 KV pool，不需要先 gather 成连续内存（`mx.fast.metal_kernel`）
- **OpenAI 兼容 API**：`/v1/chat/completions`、`/v1/completions`，支持流式输出；还有交互式 shell
- **模型**：Llama / Mistral / Qwen2 / Qwen3，支持 mlx-community 的量化模型（如 `Qwen3-8B-4bit`）
- **正确性**：float32 下与 mlx-lm 的 greedy 输出逐 token 一致（覆盖两种 attention 后端、开/关 overlap、chunked prefill、page_size>1、radix 命中）

## 快速开始

需要 Apple Silicon Mac 和 Python ≥ 3.10。

```bash
cd mini-sglang-mlx
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -e ".[dev]"     # dev 依赖：pytest、mlx-lm（正确性对照）、openai（客户端）
```

**交互式聊天**

```bash
python -m minisgl_mlx --model mlx-community/Qwen3-0.6B-bf16 --shell
```

**OpenAI 兼容服务**

```bash
python -m minisgl_mlx --model mlx-community/Qwen3-0.6B-bf16          # 默认端口 1919

curl http://127.0.0.1:1919/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"你好 /no_think"}],"max_tokens":64}'
```

**离线批量推理（单进程，最适合学习和调试调度器）**

```python
from minisgl_mlx import LLM, SamplingParams

llm = LLM("mlx-community/Qwen3-0.6B-bf16")
outs = llm.generate(["The capital of France is", "1, 2, 3,"], SamplingParams(max_tokens=32))
print([o["text"] for o in outs])
```

常用参数：`--attn metal|naive`、`--cache radix|naive`、`--max-prefill-length`（chunk 大小）、
`--num-tokens`（KV pool 大小）、`--page-size`、`--disable-overlap-scheduling`，完整列表见 `python -m minisgl_mlx --help`。

## 教程

建议按顺序阅读。每章都对应一个实验脚本，**先跑脚本、看输出，再读文档和代码**。

| 章节 | 内容 | 实验 |
|---|---|---|
| [0. 全局视角](docs/00-overview.md) | 系统组成、请求的一生、CUDA → MLX 概念对照 | |
| [1. MLX 基础](docs/01-mlx-basics.md) | 统一内存、惰性求值、async_eval、buffer 捐赠 | `tutorials/ch01_mlx_basics.py` |
| [2. 模型](docs/02-model.md) | packed batch、逐 token 位置的 RoPE、权重与量化 | `tutorials/ch03_manual_engine.py` |
| [3. KV cache](docs/03-kv-cache.md) | KV pool、page table、请求的三个长度 | `tutorials/ch03_manual_engine.py` |
| [4. Attention](docs/04-attention.md) | prefill / decode、手写 paged Metal kernel、MLX 缓存踩坑 | `tutorials/ch04_attention.py` |
| [5. Radix cache](docs/05-radix-cache.md) | 前缀树、引用计数、LRU 淘汰、`cache_req` 的四段 | `tutorials/ch05_radix_cache.py` |
| [6. 调度器](docs/06-scheduler.md) | continuous batching、chunked prefill、准入控制 | `tutorials/ch06_scheduler_trace.py` |
| [7. Overlap 调度](docs/07-overlap.md) | CPU/GPU 重叠，以及 MLX 下的三个坑 | `tutorials/ch07_overlap.py` |
| [8. 服务化](docs/08-server.md) | 多进程、ZMQ、流式反分词、OpenAI API | `tutorials/ch08_client.py` |
| [9. 总结与练习](docs/09-exercises.md) | 性能数据、与原版的差异、11 个练习 | |
| [10. 性能分析](docs/10-profiling.md) | 理论下限、分阶段计时、消融法、GPU 时间线、内存与 buffer 捐赠 | `tutorials/ch10_profiling.py` |

## 性能

Apple M5 Ultra，Qwen3-0.6B bf16，`benchmark/offline.py`（256 个请求，输入 / 输出各 100~1024 个随机 token，并发上限 128）：

| | 输出吞吐 |
|---|---|
| mini-sglang-mlx（默认：metal attention + overlap） | 4427 tok/s |
| mini-sglang-mlx（关闭 overlap） | 4125 tok/s |
| mini-sglang-mlx（naive attention） | 645 tok/s |
| mlx-lm `BatchGenerator`（同样并发 128） | 963 tok/s |

单请求生成时，mlx-lm（384 tok/s）比本项目（~270 tok/s）快。[第 10 章](docs/10-profiling.md) 用 GPU 时间线
把这个差距拆成了"GPU 计算多"和"GPU 空等多"两部分，并给出了分析一次推理为什么慢的完整工具箱。

## 代码结构

```
python/minisgl_mlx/                       对应原版 python/minisgl/
├── core.py           Req / Batch / Context      core.py
├── layers.py         RoPE、AttentionLayer       layers/
├── models/           模型结构、权重加载、量化    models/
├── kvcache/          KV pool、radix / naive 前缀缓存   kvcache/
├── attention/        prefill、naive decode、Metal paged decode   attention/
├── engine/           Engine、采样               engine/
├── scheduler/        调度器、prefill/decode 管理、KV 分配   scheduler/
├── tokenizer/        分词、流式反分词 worker     tokenizer/
├── server/           启动、API server、shell    server/
├── message.py        进程间消息                 message/
└── llm.py            离线接口 LLM               llm/
tutorials/            每章的实验脚本
tests/                单元测试 + 与 mlx-lm 的端到端对比
benchmark/            离线 / 在线 benchmark
docs/                 中文教程
```

## 测试

```bash
python -m pytest -q
```

需要本地已缓存 `mlx-community/Qwen3-0.6B-bf16`（不会自动下载；运行一次上面的快速开始即可），
可以用环境变量 `MINISGL_TEST_MODEL` 换成别的模型。

## 已知限制

- 只支持单 GPU（不支持 tensor parallel），不支持 MoE 模型
- 采样参数只支持 temperature / top_k / top_p / max_tokens / ignore_eos（没有 stop、logprobs、penalty）
- 单请求延迟不如 mlx-lm（见上）

## 致谢

- [mini-sglang](https://github.com/sgl-project/mini-sglang) 和 [SGLang](https://github.com/sgl-project/sglang)：架构、调度器、radix cache、流式反分词的设计
- [MLX](https://github.com/ml-explore/mlx) 和 [mlx-lm](https://github.com/ml-explore/mlx-lm)：张量库与参考实现
