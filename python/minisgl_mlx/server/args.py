from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import List, Tuple

from minisgl_mlx.attention import SUPPORTED_ATTENTION_BACKENDS
from minisgl_mlx.kvcache import SUPPORTED_PREFIX_CACHES
from minisgl_mlx.scheduler import SchedulerConfig


@dataclass(frozen=True)
class ServerArgs(SchedulerConfig):
    server_host: str = "127.0.0.1"
    server_port: int = 1919

    @property
    def zmq_frontend_addr(self) -> str:  # detokenizer -> api server
        return f"ipc:///tmp/minisgl_mlx_frontend{self._unique_suffix}"

    @property
    def zmq_tokenizer_addr(self) -> str:  # api server -> tokenizer（与 detokenizer 共用一个进程/地址）
        return self.zmq_detokenizer_addr


def parse_args(argv: List[str], run_shell: bool = False) -> Tuple[ServerArgs, bool]:
    p = argparse.ArgumentParser(description="mini-sglang-mlx server")
    p.add_argument("--model-path", "--model", dest="model_path", required=True,
                   help="本地目录或 HuggingFace repo id，如 mlx-community/Qwen3-0.6B-bf16")
    p.add_argument("--host", dest="server_host", default=ServerArgs.server_host)
    p.add_argument("--port", dest="server_port", type=int, default=ServerArgs.server_port)
    p.add_argument("--dtype", default="auto", choices=["auto", "bfloat16", "float16", "float32"])
    p.add_argument("--max-running-requests", dest="max_running_req", type=int, default=ServerArgs.max_running_req)
    p.add_argument("--attn", "--attention-backend", dest="attention_backend",
                   default=ServerArgs.attention_backend, choices=list(SUPPORTED_ATTENTION_BACKENDS))
    p.add_argument("--cache", dest="cache_type", default=ServerArgs.cache_type, choices=list(SUPPORTED_PREFIX_CACHES))
    p.add_argument("--page-size", type=int, default=ServerArgs.page_size)
    p.add_argument("--memory-ratio", type=float, default=ServerArgs.memory_ratio,
                   help="KV cache + 模型最多占用 GPU 推荐工作集的比例")
    p.add_argument("--max-kv-tokens", type=int, default=ServerArgs.max_kv_tokens)
    p.add_argument("--num-tokens", dest="num_tokens_override", type=int, default=None,
                   help="直接指定 KV pool 能容纳的 token 数（覆盖上面两个参数）")
    p.add_argument("--max-seq-len", dest="max_seq_len_override", type=int, default=None)
    p.add_argument("--max-prefill-length", "--max-extend-tokens", dest="max_extend_tokens", type=int,
                   default=ServerArgs.max_extend_tokens, help="chunked prefill 的 chunk 大小")
    p.add_argument("--disable-overlap-scheduling", dest="overlap_scheduling", action="store_false")
    p.add_argument("--mlx-cache-limit-gb", type=float, default=ServerArgs.mlx_cache_limit_gb,
                   help="MLX buffer 缓存上限（GB）")
    p.add_argument("--shell", action="store_true", help="启动交互式命令行聊天，而不是 HTTP 服务")
    ns = vars(p.parse_args(argv))
    run_shell = ns.pop("shell") or run_shell
    return ServerArgs(**ns), run_shell
