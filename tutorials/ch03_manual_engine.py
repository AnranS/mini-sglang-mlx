"""
第 2、3 章实验：不用调度器，"手动驾驶" Engine。

    python tutorials/ch03_manual_engine.py

你会亲手做调度器平时替你做的事：
  1. 给每个请求分配 page table 的一行、为每个 token 分配 KV 槽位
  2. 拼出 packed 的 prefill batch（两个请求的 token 首尾相接），算出 positions / out_loc
  3. 验证 packed batch 的 logits 与 mlx-lm 逐个请求单独计算的结果一致
  4. 跑几步 decode，观察 page table 和 KV 槽位如何增长
"""

import mlx.core as mx
import numpy as np

from minisgl_mlx.core import Batch, Req, SamplingParams
from minisgl_mlx.engine import Engine, EngineConfig
from minisgl_mlx.utils import load_tokenizer

MODEL = "mlx-community/Qwen3-0.6B-bf16"
engine = Engine(EngineConfig(model_path=MODEL, num_tokens_override=256, max_running_req=4))
tok = load_tokenizer(MODEL)
page_table = engine.page_table

print("\nModelConfig:", engine.model_config)
print("KV pool: 每层 K/V 形状", engine.kv_cache.k[0].shape, "层数", engine.kv_cache.num_layers)

# 为了让"槽位不连续"更直观，我们故意打乱空闲槽位的顺序（真实系统里，经过一段时间的分配/释放后就是这样）
free_slots = list(np.random.default_rng(0).permutation(engine.num_tokens))


def alloc(n):
    return np.array([free_slots.pop() for _ in range(n)], dtype=np.int32)


def prepare(batch: Batch, input_ids: np.ndarray) -> None:
    """scheduler._prepare_batch 的简化版。"""
    positions = np.concatenate([np.arange(r.cached_len, r.device_len) for r in batch.reqs]).astype(np.int32)
    rows = np.repeat([r.table_idx for r in batch.reqs], [r.extend_len for r in batch.reqs])
    batch.input_ids = mx.array(input_ids)
    batch.positions = mx.array(positions)
    batch.out_loc = mx.array(page_table[rows, positions])
    engine.attn_backend.prepare_metadata(batch)
    print(f"  input_ids = {input_ids.tolist()}")
    print(f"  positions = {positions.tolist()}")
    print(f"  out_loc   = {page_table[rows, positions].tolist()}   <- 新 token 的 KV 写到这些槽位")


# ---------------------------------------------------------------- prefill
prompts = ["The capital of France is", "1, 2, 3, 4,"]
reqs = []
for i, p in enumerate(prompts):
    ids = np.array(tok.encode(p), dtype=np.int32)
    req = Req(input_ids=ids, table_idx=i, cached_len=0, output_len=6, uid=i,
              sampling_params=SamplingParams(), cache_handle=None)  # type: ignore[arg-type]
    page_table[i, : len(ids)] = alloc(len(ids))
    reqs.append(req)

print("\n[prefill] 两个请求打包成一个 batch：")
batch = Batch(reqs=reqs, phase="prefill")
prepare(batch, np.concatenate([r.input_ids for r in reqs]))
with engine.ctx.forward_batch(batch):
    logits = engine.model(batch.input_ids)  # [2, vocab]：每个请求最后一个 token 的 logits
for r in reqs:
    r.complete_one()

# 和 mlx-lm（每个请求单独算、连续 KV cache）对比
from mlx_lm import load  # noqa: E402

ref_model, _ = load(MODEL)
for i, r in enumerate(reqs):
    ref = ref_model(mx.array(r.input_ids)[None])[0, -1]
    diff = mx.abs(ref.astype(mx.float32) - logits[i].astype(mx.float32)).max().item()
    print(f"  请求 {i}: 与 mlx-lm 的 logits 最大差 {diff:.4f}, argmax 一致: {mx.argmax(ref).item() == mx.argmax(logits[i]).item()}")
del ref_model

next_tokens = mx.argmax(logits, axis=-1).tolist()

# ---------------------------------------------------------------- decode
for step in range(5):
    for r, t in zip(reqs, next_tokens):
        r.append_host(t)
        page_table[r.table_idx, r.device_len - 1] = alloc(1)[0]  # 新 token 需要一个新槽位
    print(f"\n[decode step {step}] 每个请求只算 1 个新 token：")
    batch = Batch(reqs=reqs, phase="decode")
    prepare(batch, np.array(next_tokens, dtype=np.int32))
    out = engine.forward_batch(batch, engine.sampler.prepare(batch))  # 惰性
    next_tokens = out.tolist()  # .tolist() 触发真正的计算

for r, t in zip(reqs, next_tokens):
    r.append_host(t)
    print(f"\n请求 {r.uid}: {tok.decode(r.input_ids)!r}")
    # 只打印 KV 已经写进 pool 的部分 [0, cached_len)；最后采样出的 token 还没有被喂给模型
    print(f"  page table 第 {r.table_idx} 行: {page_table[r.table_idx, : r.cached_len].tolist()}")
print("\n=> 逻辑上连续的序列，KV 在物理上散落在 pool 各处 —— 这就是 paged KV cache")
