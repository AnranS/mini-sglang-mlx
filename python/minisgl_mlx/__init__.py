"""
mini-sglang-mlx：一个面向学习的、跑在 Apple Silicon (MLX) 上的 mini-sglang 实现。

模块与原版 mini-sglang (python/minisgl) 一一对应，方便对照阅读：

    core.py        Req / Batch / Context / SamplingParams      <-> minisgl/core.py
    models/        模型结构与权重加载                           <-> minisgl/models, minisgl/layers
    layers.py      RoPE + AttentionLayer                        <-> minisgl/layers
    kvcache/       KV pool + 前缀缓存 (radix / naive)            <-> minisgl/kvcache
    attention/     attention 后端 (naive gather / Metal kernel)  <-> minisgl/attention
    engine/        Engine：模型 + KV + 采样                      <-> minisgl/engine
    scheduler/     调度器：continuous batching / chunked prefill / overlap
    tokenizer/     分词 & 流式反分词 worker
    server/        多进程启动、OpenAI 兼容 API、交互式 shell
    llm.py         离线批量推理接口 LLM
"""

__version__ = "0.1.0"


def __getattr__(name: str):
    # 延迟导入，避免 `import minisgl_mlx` 时就把 MLX / transformers 都拉起来
    if name == "LLM":
        from .llm import LLM

        return LLM
    if name == "SamplingParams":
        from .core import SamplingParams

        return SamplingParams
    raise AttributeError(name)
