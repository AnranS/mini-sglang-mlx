"""
Engine：持有模型、KV pool、page table、attention 后端和采样器，对应原版 minisgl/engine/engine.py。

原版 Engine 初始化时要做的 CUDA 相关工作，在这里都消失或换了形式：

    原版 (CUDA)                               本实现 (MLX)
    ---------------------------------------   ----------------------------------------------
    torch.cuda.set_device / Stream            MLX 默认 GPU stream，无需手动管理
    NCCL / pynccl 初始化（TP 通信）            单机统一内存，不做 TP
    torch.cuda.mem_get_info 计算 KV 预算       mx.device_info()["max_recommended_working_set_size"]
    page_table 是 GPU tensor                  page_table 是 CPU 上的 numpy 数组（统一内存）
    CUDA Graph 捕获 decode                    不需要：MLX 本身就是"先建图再执行"的惰性模型
    forward 后 .to("cpu", non_blocking)       返回惰性的 mx.array，由调度器决定何时 eval

最后一行是理解 overlap 调度的关键：forward_batch 只"搭建计算图"，不触发任何 GPU 计算。
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np

from minisgl_mlx.attention import create_attention_backend
from minisgl_mlx.core import Batch, Context, set_global_ctx
from minisgl_mlx.kvcache import MHAKVCache
from minisgl_mlx.models import ModelConfig, create_model, load_weights
from minisgl_mlx.utils import init_logger, load_hf_config, mem_GB, resolve_model_path

from .config import EngineConfig
from .sample import BatchSamplingArgs, Sampler

logger = init_logger(__name__)

_DTYPE_MAP = {"bfloat16": mx.bfloat16, "float16": mx.float16, "float32": mx.float32}


def _device_info() -> dict:
    if hasattr(mx, "device_info"):
        return mx.device_info()
    return mx.metal.device_info()


class Engine:
    def __init__(self, config: EngineConfig) -> None:
        mx.random.seed(config.seed)
        mx.set_cache_limit(int(config.mlx_cache_limit_gb * (1 << 30)))
        self.config = config
        self.model_path = resolve_model_path(config.model_path)
        hf_config = load_hf_config(self.model_path)
        self.model_config = model_config = ModelConfig.from_hf(hf_config)
        self.dtype = _resolve_dtype(config.dtype, hf_config)

        # ======================= 模型 =======================
        mem_before = mx.get_active_memory()
        self.model = create_model(model_config)
        load_weights(self.model, self.model_path, model_config, self.dtype)
        self.model_bytes = model_bytes = mx.get_active_memory() - mem_before
        logger.info(f"Loaded {model_config.architecture} ({mem_GB(model_bytes)}) from {self.model_path}")

        # ======================= KV pool =======================
        self.num_tokens = self._determine_num_tokens(model_bytes)
        self.kv_cache = MHAKVCache(
            num_layers=model_config.num_layers,
            num_tokens=self.num_tokens,
            num_kv_heads=model_config.num_kv_heads,
            head_dim=model_config.head_dim,
            dtype=self.dtype,
        )

        # ======================= page table（CPU 侧） =======================
        self.max_seq_len = config.max_seq_len_override or min(
            model_config.rotary_config.max_position, self.num_tokens
        )
        self.page_table = np.zeros((config.max_running_req, self.max_seq_len), dtype=np.int32)

        # ======================= 全局上下文 & attention 后端 =======================
        self.ctx = Context(page_size=config.page_size)
        self.ctx.page_table = self.page_table
        self.ctx.kv_cache = self.kv_cache
        self.attn_backend = create_attention_backend(
            config.attention_backend, model_config, self.kv_cache, self.page_table
        )
        self.ctx.attn_backend = self.attn_backend
        set_global_ctx(self.ctx)

        self.sampler = Sampler(model_config.vocab_size)
        logger.info(
            f"Engine ready: attention={config.attention_backend}, kv_tokens={self.num_tokens}, "
            f"max_seq_len={self.max_seq_len}, active_memory={mem_GB(mx.get_active_memory())}"
        )

    def _determine_num_tokens(self, model_bytes: int) -> int:
        cfg = self.model_config
        bytes_per_token = 2 * cfg.num_layers * cfg.num_kv_heads * cfg.head_dim * self.dtype.size
        if self.config.num_tokens_override is not None:
            num_tokens = self.config.num_tokens_override
        else:
            working_set = _device_info()["max_recommended_working_set_size"]
            budget = int(self.config.memory_ratio * working_set) - model_bytes
            num_tokens = min(budget // bytes_per_token, self.config.max_kv_tokens)
            logger.info(
                f"KV budget = {self.config.memory_ratio} * {mem_GB(working_set)} (working set) "
                f"- {mem_GB(model_bytes)} (model) = {mem_GB(budget)}; "
                f"{bytes_per_token / 1024:.1f} KiB/token, capped by max_kv_tokens={self.config.max_kv_tokens}"
            )
        num_tokens = num_tokens // self.config.page_size * self.config.page_size
        assert num_tokens > 0, "Not enough memory for KV cache"
        logger.info(f"Allocating KV cache: {num_tokens} tokens = {mem_GB(num_tokens * bytes_per_token)}")
        return int(num_tokens)

    def forward_batch(self, batch: Batch, args: BatchSamplingArgs) -> mx.array:
        """搭建"一次 forward + 采样"的计算图，返回惰性的 next_tokens [B]。"""
        with self.ctx.forward_batch(batch):
            logits = self.model(batch.input_ids)
        for req in batch.reqs:
            req.complete_one()  # 在 CPU 侧"假装"这一步已经算完 —— overlap 调度的前提
        return self.sampler.sample(logits, args)

    def shutdown(self) -> None:
        set_global_ctx(None)


def _resolve_dtype(dtype: str, hf_config: dict) -> mx.Dtype:
    if dtype != "auto":
        return _DTYPE_MAP[dtype]
    # 与原版一致：bf16 模型用 bf16，其他（fp16 / fp32）用 fp16
    hf_dtype = hf_config.get("torch_dtype") or hf_config.get("dtype") or "bfloat16"
    return mx.bfloat16 if hf_dtype == "bfloat16" else mx.float16
