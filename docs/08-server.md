# 第 8 章：服务化 —— 多进程、ZMQ、流式反分词、OpenAI API

> 代码：`server/`、`tokenizer/`、`message.py`、`utils.py`（ZMQ 队列）
> 实验：`python tutorials/ch08_client.py`（需先启动服务）

## 8.1 为什么要三个进程

```
主进程        API server（FastAPI + uvicorn，asyncio）
子进程 1      tokenizer / detokenizer
子进程 2      scheduler（MLX 模型、KV cache，所有 GPU 计算）
```

- **调度器的 CPU 时间是最宝贵的**：第 7 章看到，调度器每多花 1 ms，GPU 就可能多空闲 1 ms。
  分词、反分词、HTTP 解析都是纯 Python 的 CPU 工作，必须从调度器进程里拿走。
- **API server 是 asyncio**：同步的分词调用会阻塞事件循环，影响所有连接。
- Python 有 GIL，多线程无法真正并行，所以用多进程。

`server/launch.py` 用 `spawn` 方式启动子进程（macOS 上 Python 默认就是 spawn；fork 出的子进程里再使用 Metal 并不安全），
并通过一个 `multiprocessing.Queue` 等待所有子进程报告"ready"。

## 8.2 消息与 ZMQ

所有消息都是 dataclass（`message.py`），用 msgpack 序列化（numpy 数组转成 bytes），通过 ZMQ 的 PUSH/PULL socket
（`ipc://` 地址，本质是 unix domain socket）传输：

```
API server ──TokenizeMsg/AbortMsg──> tokenizer ──UserMsg/AbortBackendMsg──> scheduler
API server <──────UserReply──────── detokenizer <────────DetokenizeMsg──────── scheduler
```

一些小优化：接收方每次把队列里积压的消息一次取完；发送方把多条消息打包成 `Batch*Msg` 一次发送。

## 8.3 流式反分词（`tokenizer/detokenize.py`）

为什么不能"来一个 token 就 decode 一个 token"？

1. 一个汉字或 emoji 可能被 BPE 拆成多个 byte-level token，单独 decode 会得到 `�`；
2. 有些 tokenizer decode 时会根据上下文决定空格，单独 decode 会丢空格。

做法（来自 SGLang）：每个请求维护 `surr_offset ≤ read_offset ≤ len(ids)`：

```
new_text = decode(ids[surr_offset:]) - decode(ids[surr_offset:read_offset])
```

- 如果 `new_text` 以 `�` 结尾，说明字符还不完整：先不推进偏移量，只输出"一定不会再变"的部分；
- 否则输出 `new_text`，推进 `surr_offset = read_offset; read_offset = len(ids)`。

本项目处理了一个边界情况：如果请求结束时最后一段文本恰好停在一个多字节字符的中间（例如 max_tokens 正好用完），原版会把这段不完整的文本留在缓冲区里、永远不输出；这里在 `finished` 时把剩余部分原样输出，保证流式结果与一次性 decode 完全一致。
测试：`tests/test_detokenize.py` 逐 token 流式反分词一段中英文混合 + emoji 的文本，结果必须与一次性 decode 完全一致。

## 8.4 API server（`server/api_server.py`）

`FrontendManager` 的设计：

- `submit()`：分配 uid，为它建一个 `RequestState`（回复列表 + `asyncio.Event`），发送 `TokenizeMsg`；
- 后台协程 `listen()`：不停地从 detokenizer 收 `UserReply`，放进对应 uid 的列表并 `event.set()`；
- `wait_for_replies(uid)`：`await event.wait()`，取出积压的回复逐个 yield，直到 `finished`。

一个 HTTP 请求对应一个 `wait_for_replies` 协程。成千上万个并发连接也只有一个 `listen` 协程在读 ZMQ。

**客户端断开**：`stream_with_cancellation` 检测到断开（或生成器被取消）后发送 `AbortMsg`，
调度器立即释放该请求的 KV —— 否则一个断开的请求会一直占着资源生成到 max_tokens。

## 8.5 用起来

```bash
# 启动（默认端口 1919）
python -m minisgl_mlx --model mlx-community/Qwen3-0.6B-bf16

# curl
curl http://127.0.0.1:1919/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"你好 /no_think"}],"max_tokens":64,"stream":true}'

# 交互式 shell（不启动 HTTP，直接复用 FrontendManager）
python -m minisgl_mlx --model mlx-community/Qwen3-0.6B-bf16 --shell
```

支持的接口：`/v1/chat/completions`、`/v1/completions`（均支持 `stream`）、`/v1/models`、`/generate`、`/health`。
采样参数支持 `temperature`、`top_p`、`top_k`、`max_tokens`、`ignore_eos`；`stop`、`n`、`logprobs` 等尚未实现（见练习）。

`benchmark/online.py` 实测（Qwen3-0.6B，64 个请求，并发 32，输入最多约 800 词，输出 16~256 token）：

```
TTFT  ms: mean   114.9  p50    40.0  p99   254.3
TPOT  ms: mean     8.0  p50     8.1  p99    14.4
output throughput: 2878.8 tok/s
```
