"""
端到端正确性：在 float32 下，我们的 greedy 输出必须与 mlx-lm（逐请求、连续 KV cache 的参考实现）逐 token 完全一致。

为什么用 float32？bf16 下两种实现的累加顺序不同，logits 有 1 个 ULP 量级的差异（~0.1），
遇到"并列第一"的 token 时 argmax 会翻转，之后整个序列就分叉了 —— 这不是 bug。
float32 下这种差异小到不会改变 argmax，于是可以做严格对比。
"""

import pytest

pytest.importorskip("mlx_lm")

import mlx.core as mx  # noqa: E402
from mlx_lm import load  # noqa: E402
from mlx_lm.generate import generate_step  # noqa: E402

from minisgl_mlx import LLM, SamplingParams  # noqa: E402

N = 32
PROMPTS = [
    "The capital of France is",
    "请用一句话介绍一下北京。",
    "def quicksort(arr):",
    "The capital of France is Paris. The capital of Germany is",  # 与第一个 prompt 共享前缀
    "Once upon a time, " * 20,  # 较长的 prompt，配合小 chunk 测试 chunked prefill
]


@pytest.fixture(scope="module")
def reference(model_path):
    model, tok = load(model_path)
    model.set_dtype(mx.float32)
    outs = []
    for p in PROMPTS:
        gen = generate_step(mx.array(tok.encode(p)), model, sampler=lambda x: mx.argmax(x, axis=-1))
        outs.append([int(t) for t, _ in zip((t for t, _ in gen), range(N))])
    return outs


CONFIGS = {
    "metal+overlap": dict(attention_backend="metal"),
    "naive+no_overlap": dict(attention_backend="naive", overlap_scheduling=False),
    "chunked_prefill_16": dict(max_extend_tokens=16),
    "page_size_4": dict(page_size=4),
    "naive_cache": dict(cache_type="naive"),
}


@pytest.mark.parametrize("name", list(CONFIGS))
def test_greedy_matches_mlx_lm(model_path, reference, name):
    llm = LLM(model_path, dtype="float32", num_tokens_override=8192, **CONFIGS[name])
    sp = SamplingParams(max_tokens=N, ignore_eos=True)
    for round_ in range(2):  # 第二轮会命中 radix cache
        outs = llm.generate(PROMPTS, sp)
        for i, (o, ref) in enumerate(zip(outs, reference)):
            assert o["token_ids"] == ref, f"{name} round {round_} prompt {i}"
    llm.cache_manager.check_integrity()
    llm.shutdown()


@pytest.mark.parametrize("overlap", [True, False])
def test_exact_max_tokens(model_path, overlap):
    # 回归测试：overlap 调度下曾经因为用 req.can_decode 判断结束而少生成一个 token
    llm = LLM(model_path, num_tokens_override=4096, overlap_scheduling=overlap)
    lens = (1, 2, 3, 7)
    outs = llm.generate(["Hello"] * len(lens), [SamplingParams(max_tokens=n, ignore_eos=True) for n in lens])
    assert [len(o["token_ids"]) for o in outs] == list(lens)
    llm.shutdown()


def test_eos(model_path):
    llm = LLM(model_path, num_tokens_override=4096)
    # 很容易遇到 EOS 的 chat 请求：输出里不应包含 EOS，并且请求结束后资源全部归还
    outs = llm.chat([[{"role": "user", "content": "只回答一个字：好 /no_think"}]] * 4, SamplingParams(max_tokens=64))
    assert all(llm.tokenizer.eos_token_id not in o["token_ids"] for o in outs)
    llm.cache_manager.check_integrity()
    llm.shutdown()
