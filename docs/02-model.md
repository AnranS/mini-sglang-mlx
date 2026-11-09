# 第 2 章：模型 —— packed batch 与逐 token 的 RoPE

> 代码：`models/transformer.py`、`models/weight.py`、`layers.py`
> 实验：`python tutorials/ch03_manual_engine.py`（前半部分）

## 2.1 模型结构

Llama / Mistral / Qwen2 / Qwen3 都是同一个骨架，差别用 `ModelConfig` 里的开关表达：

```
embed_tokens
for each layer:
    h = x + Attention(RMSNorm(x))         # q/k/v 投影 -> [q/k norm] -> RoPE -> attention -> o 投影
    x = h + MLP(RMSNorm(h))               # down(silu(gate(x)) * up(x))
norm -> lm_head（或与 embed_tokens 共享权重）
```

| 模型 | 差异 | 开关 |
|---|---|---|
| Qwen2 / Qwen2.5 | q/k/v 投影带 bias | `attention_bias=True` |
| Qwen3 | 对 q/k 做 per-head RMSNorm | `qk_norm=True` |
| Llama 3 | 特殊的 RoPE scaling | `layers._llama3_freqs` |

模块的命名沿用 HuggingFace（`model.layers.0.self_attn.o_proj.weight`），只有一处例外：
q/k/v 三个投影融合成了 `qkv_proj`，gate/up 融合成了 `gate_up_proj`（加载时沿输出维拼接权重，见 `weight.py`）。
decode 时每个矩阵乘都是"矩阵 × 向量"，kernel 的数量比计算量更要紧，融合后每层少 3 个 kernel。

## 2.2 packed batch：为什么输入是 `[T]` 而不是 `[B, L]`

mlx-lm 这类单用户实现，输入是 `[B, L]`，batch 里每个序列等长（不等长就 padding）。

服务端 batch 的特点是长度极不均匀：一个 prefill batch 里可能有一个 3000 token 的 prompt 和十个 20 token 的 prompt。
padding 到 `[11, 3000]` 会浪费 90% 的算力。

所以我们和原版一样，把所有请求本轮要算的 token **首尾拼接**成一条长度为 T 的序列：

```
请求 A：prompt 5 个 token，前缀缓存命中 2 个 -> 本轮算 3 个（位置 2,3,4）
请求 B：decode，本轮算 1 个（位置 37）
请求 C：prompt 4 个 token，没有命中        -> 本轮算 4 个（位置 0,1,2,3）

input_ids = [a2 a3 a4 | b37 | c0 c1 c2 c3]        T = 8
positions = [ 2  3  4 |  37 |  0  1  2  3]
```

关键观察：Embedding、Linear、RMSNorm、MLP 都是**逐 token 独立**的，`[T, hidden]` 直接算就行，完全没有浪费。
只有两个地方需要知道"哪些 token 属于同一个请求"：

1. **RoPE**：每个 token 的位置不同 —— 用 `batch.positions`。
2. **Attention**：每个 token 只能看到同一个请求里位置不超过自己的 token —— 交给 attention 后端（第 4 章）。

这些信息都放在全局 `Context` 里（`core.py`），模型代码本身非常干净。

## 2.3 逐 token 位置的 RoPE

`mx.fast.rope` 是一个融合 kernel，它要求输入形如 `(B, *, L, D)`，并且 `offset` 可以是长度为 B 的向量
—— 每个 batch 元素一个起始位置，序列内的位置依次为 `offset, offset+1, ...`。

packed batch 里没有"序列"，每个 token 位置都不连续，怎么办？技巧（`layers.py`）：

```python
# x: [T, H, D]  ->  [T, H, 1, D]：把每个 token 当成一个长度为 1 的序列
y = mx.fast.rope(x.reshape(T, H, 1, D), dims, traditional=False, base=base,
                 scale=1.0, offset=positions)          # positions: [T]
```

于是 `B=T`、`L=1`，每个 token 用自己的位置作为 offset。仍然是一个融合 kernel，不需要手写 cos/sin 查表。

> 小练习：写一个朴素版本（预计算 cos/sin 表，按 positions 取出，再做 rotate-half），验证结果一致。

## 2.4 只对最后一个 token 算 logits

prefill 一个 1000 token 的 prompt，只有最后一个 token 的 logits 会被用来采样。
`lm_head` 是 `[hidden, vocab]`（Qwen3 的 vocab 是 151936），对 1000 个 token 都算一遍很浪费。
所以 `CausalLM.__call__` 在 prefill 时先用 `last_indices` 把每个请求的最后一个 hidden state 挑出来，再过 `norm` 和 `lm_head`。

## 2.5 权重加载与量化模型

`models/weight.py` 只做三件事：

1. `mx.load()` 读取所有 `*.safetensors`（统一内存下不存在"拷到显存"这一步）；
2. tie embedding 的模型丢掉 `lm_head.weight`，用 `embed_tokens.as_linear()` 代替；
3. **mlx-community 的量化模型**（如 `mlx-community/Qwen3-8B-4bit`）：config 里有 `quantization` 字段，
   权重里有 `xxx.scales` / `xxx.biases`。先用 `nn.quantize` 把对应的 `Linear` / `Embedding` 换成量化版本，再加载。

量化对推理引擎的其它部分完全透明 —— KV cache、attention、调度都不需要改。

```bash
python -m minisgl_mlx --model mlx-community/Qwen3-8B-4bit --shell
```

## 2.6 验证：packed batch 的结果与逐请求计算一致

`tutorials/ch03_manual_engine.py` 把两个 prompt 打包成一个 batch 前向，与 mlx-lm 逐个计算的 logits 对比。
更严格的端到端验证在 `tests/test_e2e.py`：float32 下，各种配置（两种 attention 后端、开/关 overlap、
chunked prefill、page_size=4、radix 命中）生成的 token 与 mlx-lm 逐 token 完全一致。

> 为什么不用 bf16 做严格对比？两种实现的浮点累加顺序不同，bf16 下 logits 会有约 1 个 ULP（~0.1）的差异。
> 遇到两个 token 的 logits 几乎并列时 argmax 会翻转，之后序列就分叉了。这不是 bug：
> 用 teacher forcing 检查，我们选出的每个 token 在 mlx-lm 下都与最大 logit 相差不超过 0.125。
