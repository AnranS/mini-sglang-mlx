"""
权重加载，对应原版 minisgl/models/weight.py。

原版需要处理 TP 切分（把一个大矩阵按列/行切给不同的 GPU）；单机 Apple Silicon 不需要 TP，
所以这里只剩下三件事：读 safetensors、处理 tie embedding、处理 mlx-community 的量化格式。
"""

from __future__ import annotations

import glob
import os
from typing import Dict

import mlx.core as mx
import mlx.nn as nn

from minisgl_mlx.utils import init_logger

from .config import ModelConfig

logger = init_logger(__name__)


def load_weights(model: nn.Module, model_path: str, config: ModelConfig, dtype: mx.Dtype) -> None:
    files = sorted(glob.glob(os.path.join(model_path, "*.safetensors")))
    if not files:
        raise FileNotFoundError(f"No *.safetensors found in {model_path}")

    weights: Dict[str, mx.array] = {}
    for f in files:
        # mx.load 直接把 safetensors 映射为 mx.array（统一内存，不需要"拷到显存"这一步）
        weights.update(mx.load(f))

    # 一些 checkpoint 里会带上用不到的张量
    weights = {k: v for k, v in weights.items() if "rotary_emb.inv_freq" not in k}
    if config.tie_word_embeddings:
        weights.pop("lm_head.weight", None)
    _fuse_projections(weights, config)

    # mlx-community 的量化模型：先把对应的 Linear / Embedding 替换成 Quantized 版本再加载
    if (q := config.quantization) is not None:
        logger.info(f"Loading quantized model: bits={q.get('bits')}, group_size={q.get('group_size')}")

        def predicate(path: str, module: nn.Module):
            if f"{path}.scales" not in weights:
                return False
            # 混合精度量化时，config 里会针对某些层单独给出参数
            return q[path] if isinstance(q.get(path), dict) else True

        nn.quantize(
            model,
            group_size=q.get("group_size", 64),
            bits=q.get("bits", 4),
            mode=q.get("mode", "affine"),
            class_predicate=predicate,
        )

    # 统一浮点权重的 dtype（量化后的 uint32 权重保持原样）
    weights = {
        k: (v.astype(dtype) if mx.issubdtype(v.dtype, mx.floating) else v) for k, v in weights.items()
    }
    model.load_weights(list(weights.items()), strict=True)
    mx.eval(model.parameters())


# HuggingFace 的权重是分开的 q/k/v、gate/up 投影，我们的模型用融合后的 qkv_proj / gate_up_proj
_FUSIONS = [("self_attn", ("q_proj", "k_proj", "v_proj"), "qkv_proj"), ("mlp", ("gate_proj", "up_proj"), "gate_up_proj")]


def _fuse_projections(weights: Dict[str, mx.array], config: ModelConfig) -> None:
    """
    把同一输入的几个 Linear 沿输出维拼接成一个。对量化权重同样成立：mlx 的量化是"每个输出行、
    沿输入维分组"，所以 weight / scales / biases 都可以直接沿第 0 维拼接。
    """
    q = config.quantization
    for i in range(config.num_layers):
        for module, parts, fused in _FUSIONS:
            prefix = f"model.layers.{i}.{module}"
            if q is not None:  # 混合精度量化：只有各部分量化参数相同时才能融合
                overrides = [q.get(f"{prefix}.{p}") for p in parts]
                if any(o != overrides[0] for o in overrides):
                    raise NotImplementedError(f"{prefix}: parts are quantized differently, cannot fuse")
                if isinstance(overrides[0], dict):
                    q[f"{prefix}.{fused}"] = overrides[0]
            for suffix in ("weight", "scales", "biases", "bias"):
                keys = [f"{prefix}.{p}.{suffix}" for p in parts]
                if all(k in weights for k in keys):
                    weights[f"{prefix}.{fused}.{suffix}"] = mx.concatenate([weights.pop(k) for k in keys], axis=0)
