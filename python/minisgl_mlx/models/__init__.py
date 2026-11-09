from __future__ import annotations

from .config import ModelConfig, RotaryConfig
from .transformer import CausalLM
from .weight import load_weights

# 架构名 -> 模型类。Llama 家族共享同一份实现，差异由 ModelConfig 中的开关表达。
SUPPORTED_ARCHITECTURES = {
    "LlamaForCausalLM": CausalLM,
    "MistralForCausalLM": CausalLM,
    "Qwen2ForCausalLM": CausalLM,
    "Qwen3ForCausalLM": CausalLM,
}


def create_model(config: ModelConfig) -> CausalLM:
    if config.architecture not in SUPPORTED_ARCHITECTURES:
        raise NotImplementedError(
            f"Architecture {config.architecture} is not supported. "
            f"Supported: {list(SUPPORTED_ARCHITECTURES)}"
        )
    return SUPPORTED_ARCHITECTURES[config.architecture](config)


__all__ = ["ModelConfig", "RotaryConfig", "CausalLM", "create_model", "load_weights"]
