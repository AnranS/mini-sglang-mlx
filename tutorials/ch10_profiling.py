"""
第 10 章实验：性能分析工具箱 —— "这次推理为什么慢？"

    python tutorials/ch10_profiling.py roofline   # ① 理论下限 vs 实测：慢在访存、算力，还是额外开销？
    python tutorials/ch10_profiling.py phases     # ② CPU 侧每个阶段的耗时（overlap 开 / 关）
    python tutorials/ch10_profiling.py ops        # ③ 一步 decode 的 GPU 时间花在哪些算子上
    python tutorials/ch10_profiling.py donation   # ④ 检测 buffer 捐赠失败：每步耗时是否随 KV pool 变大而变大
    python tutorials/ch10_profiling.py memory     # ⑤ MLX 内存 / 缓存 / 系统 swap
    python tutorials/ch10_profiling.py timeline   # ⑥ 用 Instruments 测 GPU 真实利用率和空隙（需要 Xcode）
    python tutorials/ch10_profiling.py capture    # ⑦ 抓一段 Metal GPU trace，用 Xcode 打开看每个 kernel
    python tutorials/ch10_profiling.py all --plot docs/images   # 跑 ①~⑥ 并出图

加 --plot DIR 会把结果画成 PNG（light / dark 各一张）。画图需要 matplotlib：uv pip install matplotlib
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time
import types

import numpy as np

MODEL = "mlx-community/Qwen3-0.6B-bf16"


# ============================================================================
# 公共工具
# ============================================================================


def random_prompts(batch: int, length: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    return [rng.integers(0, 10000, length).tolist() for _ in range(batch)]


def build_llm(args, num_tokens: int, max_running: int = 8, **kw):
    from minisgl_mlx import LLM, SamplingParams

    llm = LLM(args.model, num_tokens_override=num_tokens, max_running_req=max(max_running, 8),
              max_extend_tokens=8192, **kw)
    llm.generate(["warmup"] * 4, SamplingParams(max_tokens=8))  # 编译 Metal kernel 等一次性开销
    return llm


def set_overlap(llm, on: bool) -> None:
    object.__setattr__(llm.config, "overlap_scheduling", on)  # 仅供实验：切换 frozen config


def measure_decode(llm, timer, batch: int, context: int, steps: int, overlap: bool):
    """跑 batch 个长度为 context 的请求、各生成 steps 个 token，返回 decode 阶段各环节的中位数耗时（ms）。"""
    from minisgl_mlx import SamplingParams

    set_overlap(llm, overlap)
    prompts = random_prompts(batch, context)
    llm.generate(prompts, SamplingParams(max_tokens=1))  # 先把 prompt 放进 radix cache，正式测量时 prefill 几乎为零
    timer.reset()
    llm.generate(prompts, SamplingParams(max_tokens=steps, ignore_eos=True))
    return timer.median_ms("decode")


def header(text: str) -> None:
    print(f"\n{'=' * 12} {text} {'=' * 12}")


PHASE_NAMES = {
    "schedule": "收消息 + 调度",
    "build": "构图",
    "submit": "提交 (async_eval)",
    "wait": "等待 GPU",
    "process": "处理结果",
}


# ============================================================================
# ① roofline
# ============================================================================


def cmd_roofline(args):
    from minisgl_mlx import SamplingParams
    from minisgl_mlx.profiling import (PhaseTimer, decode_lower_bound, measure_bandwidth, measure_matmul_tflops,
                                       prefill_lower_bound)

    header("① 理论下限（roofline）")
    bw, tf = measure_bandwidth(), measure_matmul_tflops()
    print(f"实测内存带宽 {bw:.0f} GB/s，bf16 矩阵乘 {tf:.1f} TFLOPS")

    batches = args.batch or [1, 4, 16, 64]
    llm = build_llm(args, num_tokens=max(batches) * (args.context + args.steps) + 16384, max_running=max(batches))
    timer = PhaseTimer().attach(llm)
    cfg, wbytes = llm.engine.model_config, llm.engine.model_bytes
    rows = []
    for B in batches:
        med = measure_decode(llm, timer, B, args.context, args.steps, overlap=True)
        bound = decode_lower_bound(cfg, wbytes, B, args.context + args.steps / 2, bw) * 1e3
        rows.append((f"decode  B={B}", bound, med["total"]))

    P = 4096  # prefill：单个 4096 token 的请求，关闭 overlap，用 submit + wait 作为 GPU 时间
    set_overlap(llm, False)
    timer.reset()
    llm.generate(random_prompts(1, P, seed=123), SamplingParams(max_tokens=1))
    pre = timer.median_ms("prefill")
    rows.append((f"prefill {P} tok", prefill_lower_bound(cfg, P, tf) * 1e3, pre["submit"] + pre["wait"]))

    print(f"\n{'场景':16s} {'理论下限':>10s} {'实测':>10s} {'达到下限':>8s}")
    for name, bound, meas in rows:
        print(f"{name:16s} {bound:8.2f}ms {meas:8.2f}ms {bound / meas:8.0%}")
    print("\n解读：比例低 -> 时间花在了'额外开销'上（提交/同步/kernel 启动/调度）；比例高 -> 已接近硬件上限，"
          "只能减少数据量（量化）或计算量。")
    llm.shutdown()
    if args.plot:
        plot_roofline(rows, bw, tf, args.plot)
    return rows


# ============================================================================
# ② phases
# ============================================================================


def cmd_phases(args):
    from minisgl_mlx.profiling import PhaseTimer

    header("② CPU 侧分阶段计时")
    batches = args.batch or [1, 64]
    llm = build_llm(args, num_tokens=max(batches) * (args.context + args.steps) + 16384, max_running=max(batches))
    timer = PhaseTimer().attach(llm)
    res = {}
    for B in batches:
        for ov in (False, True):
            med = measure_decode(llm, timer, B, args.context, args.steps, overlap=ov)
            med["schedule"] += med.pop("recv")
            res[(B, ov)] = med
            parts = "  ".join(f"{PHASE_NAMES[p]} {med[p]:5.2f}" for p in PHASE_NAMES)
            print(f"B={B:<3d} overlap={'开' if ov else '关'} | {parts} | 每步 {med['total']:5.2f} ms "
                  f"-> {B / med['total'] * 1e3:7.0f} tok/s")
    print("\n解读：overlap 关时，'等待 GPU' 以外的时间 GPU 都在空闲；overlap 开时，构图和处理结果与 GPU 计算重叠。")
    llm.shutdown()
    if args.plot:
        plot_phases(res, batches, args.plot)
    return res


# ============================================================================
# ③ ops（消融法）
# ============================================================================


def make_decode_batch(eng, B: int, ctx: int):
    """绕过调度器，手工拼一个 decode batch：B 个请求、每个已有 ctx 个 token 的 KV（槽位随机、互不重叠）。"""
    import mlx.core as mx

    from minisgl_mlx.core import Batch, Req, SamplingParams

    rng = np.random.default_rng(B)
    # 注意：这里直接占用 page table 的前 B 行和随机槽位，只适合在引擎空闲时做实验；
    # 往这些槽位写入的 KV 是假数据（可能覆盖 radix cache 里缓存的内容，只影响生成内容，不影响计时）
    slots = rng.choice(eng.num_tokens, B * (ctx + 1), replace=False).astype(np.int32).reshape(B, ctx + 1)
    eng.page_table[:B, : ctx + 1] = slots
    reqs = [Req(input_ids=np.zeros(ctx + 1, dtype=np.int32), table_idx=i, cached_len=ctx, output_len=1, uid=i,
                sampling_params=SamplingParams(), cache_handle=None) for i in range(B)]  # type: ignore[arg-type]
    batch = Batch(reqs=reqs, phase="decode")
    batch.input_ids = mx.zeros((B,), dtype=mx.int32)
    batch.positions = mx.array([ctx] * B, dtype=mx.int32)
    batch.out_loc = mx.array(slots[:, ctx])
    eng.attn_backend.prepare_metadata(batch)
    return batch


def override(obj, name, fn):
    """临时把 obj.name 换成 fn（写在实例 __dict__ 里，遮住原来的方法/子模块），返回恢复函数。"""
    had, old = name in obj.__dict__, obj.__dict__.get(name)

    def undo():
        if had:  # 原来就是实例属性（如 AttentionLayer.rotary）：放回原值
            object.__setattr__(obj, name, old)
        else:  # 原来是类上的方法或 nn.Module 的子模块：删掉遮挡即可
            obj.__dict__.pop(name)

    object.__setattr__(obj, name, fn)
    return undo


def cmd_ops(args):
    import mlx.core as mx

    header("③ 一步 decode 的 GPU 时间花在哪（消融法）")
    batches = args.batch or [1, 64]
    ctx = args.context
    llm = build_llm(args, num_tokens=max(batches) * (ctx + 1) + 16384, max_running=max(batches))
    eng, cfg = llm.engine, llm.engine.model_config
    model, layers, dt = eng.model, eng.model.model.layers, eng.dtype
    H, Hq, Hkv, D, V = cfg.hidden_size, cfg.num_qo_heads, cfg.num_kv_heads, cfg.head_dim, cfg.vocab_size

    def bench(fn, iters=30):
        for _ in range(5):
            mx.eval(fn())
        ts = []
        for _ in range(iters):
            t = time.perf_counter()
            mx.eval(fn())
            ts.append(time.perf_counter() - t)
        return float(np.median(ts)) * 1e3

    results = {}
    for B in batches:
        batch = make_decode_batch(eng, B, ctx)

        def step():  # 完整的一步 decode：模型 forward + greedy 采样
            with eng.ctx.forward_batch(batch):
                return mx.argmax(model(batch.input_ids), axis=-1)

        # "免费"的替身：返回形状正确的常量。关键：必须用 mx.depends 保留对输入的依赖！
        # 否则惰性求值会把替身上游的整段计算当成死代码跳过 —— 例如把 lm_head 换成常量后，
        # 输出不再依赖任何一层，整个 transformer 都不会被执行，"节省"就会被严重高估。
        cqkv, ch, clog = (mx.zeros(s, dtype=dt) for s in [(B, (Hq + 2 * Hkv) * D), (B, H), (B, V)])
        mx.eval(cqkv, ch, clog)
        kv = eng.kv_cache

        def free(const):
            return lambda x, *rest: mx.depends(const, x)

        def fake_store(k, v, out_loc, i):  # 不写 KV，但让 pool 依赖 k / v，上游的 k/v 投影仍然要算
            kv.k[i] = mx.depends(kv.k[i], k)
            kv.v[i] = mx.depends(kv.v[i], v)

        def fake_attention(q, i, meta):  # 不算 attention，但依赖 KV pool，KV 写入仍然要做
            return mx.depends(q, [kv.k[i], kv.v[i]])

        def multi(items):
            undos = [override(o, n, f) for o, n, f in items]
            return lambda: [u() for u in undos]

        ident = lambda x, *rest: x  # noqa: E731
        ablations = {
            "attention (paged kernel)": lambda: override(eng.attn_backend, "forward_decode", fake_attention),
            "写 KV (scatter)": lambda: override(kv, "store_kv", fake_store),
            "q/k/v 投影": lambda: multi([(ly.self_attn, "qkv_proj", free(cqkv)) for ly in layers]),
            "o 投影": lambda: multi([(ly.self_attn, "o_proj", free(ch)) for ly in layers]),
            "MLP": lambda: multi([(ly, "mlp", free(ch)) for ly in layers]),
            "q/k norm + RoPE": lambda: multi(
                [(ly.self_attn.attn, "rotary", ident) for ly in layers]
                + [(ly.self_attn, n, ident) for ly in layers for n in ("q_norm", "k_norm") if cfg.qk_norm]),
            "RMSNorm ×2（每层）": lambda: multi(
                [(ly, n, ident) for ly in layers for n in ("input_layernorm", "post_attention_layernorm")]),
            "lm_head + 采样": lambda: (override(model.model.embed_tokens, "as_linear", free(clog))
                                     if cfg.tie_word_embeddings else override(model, "lm_head", free(clog))),
        }
        base = bench(step)
        rows = []
        for name, patch in ablations.items():
            undo = patch()
            rows.append((name, base - bench(step)))
            undo()
        undos = [patch() for patch in ablations.values()]
        rest = bench(step)
        for undo in undos:
            undo()
        results[B] = dict(base=base, rows=rows, rest=rest)

        print(f"\nbatch={B}, context={ctx}：完整一步 {base:.2f} ms（同步执行，含构图、提交与同步开销）")
        print(f"  {'去掉的算子':24s} {'节省':>8s} {'占完整一步':>10s}")
        for name, saved in sorted(rows, key=lambda r: -r[1]):
            print(f"  {name:24s} {saved:6.2f}ms {saved / base:10.0%}")
        print(f"  {'全部去掉后仍剩下':24s} {rest:6.2f}ms {rest / base:10.0%}   <- embedding、残差加法、构图、提交与同步等固定开销")
    print("\n解读：'节省'是在真实的一步里把这个算子换成免费替身后快了多少，回答'优化它最多能赚多少'。"
          "各项之和不一定等于总时间（算子之间会互相掩盖等待）。")
    llm.shutdown()
    if args.plot:
        plot_ops(results, batches, ctx, args.plot)
    return results


# ============================================================================
# ④ donation
# ============================================================================


def overlap_loop_pattern_a(self, last_data):
    """第 7 章的"写法 A"：构图后立即提交，再等上一步（mlx-lm generate_step 的思路）。用 --pattern-a 对比。"""
    blocking = not (last_data is not None or self.prefill_manager.runnable or self.decode_manager.runnable)
    for msg in self.receive_msg(blocking=blocking):
        self._process_one_msg(msg)
    ongoing = None
    if (fi := self._schedule_next_batch()) is not None:
        ongoing = (fi, self._forward(fi))
        self._submit(ongoing)
    if last_data is not None:
        self._wait(last_data)
    self._process_last_data(last_data)
    return ongoing


def install_leaky_debug_hook(llm):
    """
    模拟一个真实可能出现的 bug：为了调试，在写 KV 之前"记一下"每层 KV pool 的第一行。
    kv.k[i][0:1] 是一个惰性的切片，它在计算图里引用着写入前的旧 pool ——
    于是 scatter 时旧 buffer 还有别的引用，无法捐赠，MLX 只能把整层 pool 复制一份再写。
    """
    import mlx.core as mx  # noqa: F401

    kv = llm.engine.kv_cache
    orig_store = kv.store_kv
    held = []

    def leaky_store(k, v, out_loc, i):
        held.append((kv.k[i][0:1], kv.v[i][0:1]))  # "调试快照"：只是一个没被求值的切片
        del held[: -2 * kv.num_layers]  # 只保留最近一步的
        orig_store(k, v, out_loc, i)

    kv.store_kv = leaky_store


def cmd_donation(args):
    import mlx.core as mx

    from minisgl_mlx.profiling import PhaseTimer

    header("④ buffer 捐赠检测：每步耗时 / 峰值内存是否随 KV pool 大小增长")
    pools = args.pool or [16384, 65536, 131072]
    B, ctx, steps = 8, 256, 32
    variants = ("ok", "leaky") + (("pattern_a",) if args.pattern_a else ())
    data = {v: [] for v in variants}
    for variant in variants:
        for n in pools:
            llm = build_llm(args, num_tokens=n)
            if variant == "leaky":
                install_leaky_debug_hook(llm)
            elif variant == "pattern_a":
                llm.overlap_loop = types.MethodType(overlap_loop_pattern_a, llm)
            timer = PhaseTimer().attach(llm)
            base = mx.get_active_memory()
            mx.reset_peak_memory()
            med = measure_decode(llm, timer, B, ctx, steps, overlap=True)
            extra = (mx.get_peak_memory() - base) / (1 << 20)
            gib = llm.engine.kv_cache.nbytes / (1 << 30)
            layer = llm.engine.kv_cache.k[0].nbytes / (1 << 20)
            data[variant].append((gib, med["total"]))
            name = {"ok": "正常", "leaky": "持有旧 pool 引用", "pattern_a": "写法 A"}[variant]
            print(f"{name:8s} KV pool {gib:5.2f} GiB -> 每步 {med['total']:6.2f} ms，峰值比常驻内存多 {extra:5.0f} MiB"
                  f"（一层 pool = {layer:.0f} MiB）")
            llm.shutdown()
            del llm
            mx.clear_cache()
    print("\n解读：正常情况下每步耗时和峰值内存都不随 KV pool 大小变化；一旦某处还引用着旧的 pool，"
          "每一步、每一层都会复制整层 pool：耗时随 pool 线性增长，峰值内存多出至少一层 pool 的大小。")
    if args.plot:
        plot_donation(data, args.plot)
    return data


# ============================================================================
# ⑤ memory
# ============================================================================


def cmd_memory(args):
    import mlx.core as mx

    from minisgl_mlx.profiling import PhaseTimer, format_bytes, memory_snapshot

    header("⑤ 内存：active / peak / cache / swap")
    for limit in args.cache_limits:
        llm = build_llm(args, num_tokens=64 * 1100 + 4096, max_running=64, attention_backend=args.attn,
                        mlx_cache_limit_gb=limit)
        timer = PhaseTimer().attach(llm)
        mx.clear_cache()
        mx.reset_peak_memory()
        before = memory_snapshot()
        measure_decode(llm, timer, 64, 1000, 64, overlap=True)
        after = memory_snapshot()
        print(f"\nattn={args.attn}, mlx_cache_limit={limit} GiB")
        for k in ("active", "peak", "cache", "swap"):
            print(f"  {k:7s} {format_bytes(before[k]):>11s} -> {format_bytes(after[k]):>11s}")
        layer_bytes = llm.engine.kv_cache.k[0].nbytes * 2
        if after["cache"] >= 0.9 * limit * (1 << 30):
            print("  ! 缓存已到上限：有大量尺寸各异的临时 buffer 无法复用（考虑把形状分桶）")
        if before["swap"] is not None and after["swap"] is not None and after["swap"] - before["swap"] > 256 << 20:
            print("  ! swap 在增长：内存压力过大，性能会断崖式下降")
        if after["peak"] - before["active"] > layer_bytes:
            print("  ! 峰值增量超过一层 KV pool：可能发生了 buffer 拷贝，或存在很大的临时数组")
        llm.shutdown()
        del llm
        mx.clear_cache()


# ============================================================================
# ⑥ timeline（Instruments）
# ============================================================================


def cmd_workload(args):
    """被 timeline 子命令通过 xctrace 启动：预热 -> sleep 0.5s（作为分隔标记）-> 正式 decode。"""
    from minisgl_mlx import SamplingParams

    llm = build_llm(args, num_tokens=args.batch[0] * (args.context + args.steps) + 4096, max_running=args.batch[0],
                    overlap_scheduling=args.overlap == 1)
    prompts = random_prompts(args.batch[0], args.context)
    llm.generate(prompts, SamplingParams(max_tokens=1))  # prompt 进入 radix cache，正式测量几乎只有 decode
    time.sleep(0.5)
    llm.generate(prompts, SamplingParams(max_tokens=args.steps, ignore_eos=True))


def cmd_workload_mlx_lm(args):
    """对照组：mlx-lm 单请求生成（同样的预热 -> sleep -> 正式测量结构）。"""
    from mlx_lm import load, stream_generate

    model, tok = load(args.model)
    prompt = random_prompts(1, args.context)[0]
    for _ in stream_generate(model, tok, prompt, max_tokens=16):
        pass
    time.sleep(0.5)
    for _ in stream_generate(model, tok, prompt, max_tokens=args.steps):
        pass


def cmd_timeline(args):
    from minisgl_mlx.profiling import export_gpu_intervals, gpu_timeline_stats, record_metal_trace

    header("⑥ GPU 时间线（Instruments Metal System Trace）")
    configs = [(B, ov) for B in (args.batch or [1, 64]) for ov in (1, 0)]
    if args.with_mlx_lm:
        configs.insert(0, ("mlx-lm", None))
    results = []
    with tempfile.TemporaryDirectory() as tmp:
        for B, ov in configs:
            trace = os.path.join(tmp, f"b{B}_ov{ov}.trace")
            common = ["--model", args.model, "--context", str(args.context), "--steps", str(args.steps)]
            if B == "mlx-lm":
                cmd, label = [sys.executable, os.path.abspath(__file__), "_workload_mlx_lm", *common], "mlx-lm  B=1"
            else:
                cmd = [sys.executable, os.path.abspath(__file__), "_workload", *common, "--batch", str(B),
                       "--overlap", str(ov)]
                label = f"B={B}  overlap {'开' if ov else '关'}"
            print(f"\n录制 {label} ...")
            record_metal_trace(cmd, trace)
            stats = gpu_timeline_stats(export_gpu_intervals(trace))
            # mlx-lm 的正式测量包含一次完整 prefill，我们的 workload 已把 prompt 放进 radix cache
            print(stats.describe(num_steps=args.steps))
            results.append((label, stats))
            if args.keep_trace:
                import shutil

                dst = os.path.abspath(os.path.basename(trace))
                shutil.copytree(trace, dst, dirs_exist_ok=True)
                print(f"trace 已保存到 {dst}，可以用 `open {dst}` 在 Instruments 中查看")
    if args.plot:
        plot_timeline(results, args.plot)
    return results


# ============================================================================
# ⑦ capture（Xcode GPU 帧捕获）
# ============================================================================


def cmd_capture(args):
    if os.environ.get("MTL_CAPTURE_ENABLED") != "1":  # 必须在进程启动时就设置，所以重新执行一次自己
        os.execve(sys.executable, [sys.executable] + sys.argv, {**os.environ, "MTL_CAPTURE_ENABLED": "1"})
    import shutil

    import mlx.core as mx

    from minisgl_mlx import SamplingParams

    header("⑦ Metal GPU trace 捕获")
    path = os.path.abspath(args.out)
    shutil.rmtree(path, ignore_errors=True)
    B = (args.batch or [4])[0]
    # 捕获会把用到的 buffer 都存下来，所以 KV pool 一定要小
    llm = build_llm(args, num_tokens=B * (args.context + 64) + 1024, max_running=B)
    start_at, state = 5, {"n": 0}
    orig_submit = llm._submit

    def submit(data):
        is_decode = data[0].batch.is_decode
        if is_decode:
            state["n"] += 1
            if state["n"] == start_at:
                mx.metal.start_capture(path)
        orig_submit(data)
        if is_decode and state["n"] == start_at + args.capture_steps - 1:
            mx.synchronize()
            mx.metal.stop_capture()

    llm._submit = submit
    llm.generate(random_prompts(B, args.context), SamplingParams(max_tokens=start_at + args.capture_steps + 4,
                                                                 ignore_eos=True))
    print(f"已捕获 {args.capture_steps} 步 decode -> {path}")
    print(f"用 Xcode 打开：open {path}  （在 Performance / Timeline 里可以看到每个 kernel 的耗时）")


# ============================================================================
# 画图
# ============================================================================


def plot_roofline(rows, bw, tf, outdir):
    from ch10_plots import PT, hbar, new_figure, save_both, title

    def draw(theme):
        n = len(rows)
        hpx = 110 + 44 * n
        fig, (ax,) = new_figure(theme, 860, hpx)
        fig.subplots_adjust(left=0.15, right=0.98, top=1 - 70 / hpx, bottom=40 / hpx)
        ax.set_xlim(0, 160)
        ax.set_ylim(n - 0.5, -0.75)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
        ax.set_yticks(range(n))
        ax.set_yticklabels([r[0] for r in rows], color=theme["ink2"], fontsize=9)
        ax.axvline(100, color=theme["ink2"], linewidth=1 * PT, zorder=2)
        ax.text(98.5, -0.62, "理论下限 →", fontsize=8, color=theme["ink2"], ha="right", va="bottom")
        ax.text(106, -0.62, "理论 / 实测（每步）", fontsize=8, color=theme["muted"], va="bottom")
        for i, (_, bound, meas) in enumerate(rows):
            eff = min(bound / meas * 100, 100)
            hbar(ax, i, 0, eff, theme["series"][0], theme)
            ax.text(eff + 1.5, i, f"{eff:.0f}%", va="center", fontsize=9, color=theme["ink"])
            ax.text(106, i, f"{bound:.2f} ms / {meas:.2f} ms", va="center", fontsize=8.5, color=theme["ink2"])
        title(fig, "实测速度达到理论下限的比例",
              f"decode 按带宽估计（权重 + KV 各读一遍，实测 {bw:.0f} GB/s）；prefill 按算力估计（实测 {tf:.1f} TFLOPS）", theme)
        return fig

    print("saved:", save_both(draw, outdir, "ch10-roofline"))


def plot_phases(res, batches, outdir):
    from ch10_plots import hbar, legend_row, new_figure, save_both, title

    def draw(theme):
        hpx = 300
        fig, axes = new_figure(theme, 980, hpx, ncols=len(batches))
        fig.subplots_adjust(left=0.1, right=0.98, top=1 - 118 / hpx, bottom=48 / hpx, wspace=0.28)
        for ax, B in zip(axes, batches):
            meds = [res[(B, False)], res[(B, True)]]
            xmax = max(m["total"] for m in meds) * 1.08
            ax.set_xlim(0, xmax)
            ax.set_ylim(1.6, -0.6)
            ax.set_yticks([0, 1])
            ax.set_yticklabels(["overlap 关", "overlap 开"], color=theme["ink2"], fontsize=9)
            ax.set_xlabel("每步耗时 (ms)", color=theme["muted"], fontsize=8.5)
            tput = [B / m["total"] * 1e3 for m in meds]
            ax.set_title(f"batch = {B}    {tput[0]:.0f} → {tput[1]:.0f} tok/s", loc="left", fontsize=10,
                         color=theme["ink"], pad=8)
            for row, med in enumerate(meds):
                left = 0.0
                names = list(PHASE_NAMES)
                last = max(i for i, p in enumerate(names) if med[p] > 0)
                for j, p in enumerate(names):
                    hbar(ax, row, left, med[p], theme["series"][j], theme, round_end=(j == last), gap=(j != last))
                    left += med[p]
        legend_row(fig, [(PHASE_NAMES[p], theme["series"][j]) for j, p in enumerate(PHASE_NAMES)], theme,
                   y=1 - 66 / hpx)
        title(fig, "decode 每一步，CPU 主循环的时间花在哪里",
              "Qwen3-0.6B，各阶段取中位数；overlap 开启后构图与处理结果和 GPU 计算重叠", theme)
        return fig

    print("saved:", save_both(draw, outdir, "ch10-phases"))


def plot_ops(results, batches, ctx, outdir):
    from ch10_plots import hbar, new_figure, save_both, title

    def draw(theme):
        names = [r[0] for r in sorted(results[batches[-1]]["rows"], key=lambda r: -r[1])] + ["全部去掉后仍剩下"]
        n = len(names)
        hpx = 150 + 34 * n
        fig, axes = new_figure(theme, 1000, hpx, ncols=len(batches))
        fig.subplots_adjust(left=0.17, right=0.98, top=1 - 100 / hpx, bottom=46 / hpx, wspace=0.12)
        for k, (ax, B) in enumerate(zip(axes, batches)):
            r = results[B]
            vals = dict(r["rows"])
            vals["全部去掉后仍剩下"] = r["rest"]
            xmax = max(vals.values()) * 1.3
            ax.set_xlim(0, xmax)
            ax.set_ylim(n - 0.5, -0.6)
            ax.set_yticks(range(n))
            ax.set_yticklabels(names if k == 0 else [], color=theme["ink2"], fontsize=8.5)
            ax.set_xlabel("ms / 步", color=theme["muted"], fontsize=8.5)
            ax.set_title(f"batch = {B}    完整一步 {r['base']:.2f} ms", loc="left", fontsize=10, color=theme["ink"], pad=8)
            for i, name in enumerate(names):
                v = max(vals[name], 0.0)
                color = theme["muted"] if name == "全部去掉后仍剩下" else theme["series"][0]
                hbar(ax, i, 0, v, color, theme, height_px=16)
                label = "≈ 0（低于测量噪声）" if vals[name] < 0.1 else f"{vals[name]:.2f}  ({vals[name] / r['base']:.0%})"
                ax.text(v + xmax * 0.02, i, label, va="center", fontsize=8, color=theme["ink2"])
        title(fig, "去掉某个算子，一步 decode 能快多少（消融法）",
              f"Qwen3-0.6B，context = {ctx}；蓝色 = 把该算子换成免费替身后节省的时间，灰色 = 所有算子都去掉后仍剩下的固定开销",
              theme)
        return fig

    print("saved:", save_both(draw, outdir, "ch10-ops"))


def plot_donation(data, outdir):
    from ch10_plots import PT, PX, legend_row, new_figure, save_both, style_axes, title

    def draw(theme):
        hpx = 380
        fig, (ax,) = new_figure(theme, 860, hpx)
        style_axes(ax, theme, grid_axis="y")
        fig.subplots_adjust(left=0.09, right=0.8, top=1 - 96 / hpx, bottom=58 / hpx)
        series = [("ok", "正常", theme["series"][0]),
                  ("leaky", "有 bug：调试代码持有旧 KV pool 的惰性引用", theme["series"][1])]
        ymax = max(v for s in data.values() for _, v in s) * 1.15
        ax.set_ylim(0, ymax)
        ax.set_xlim(0, max(g for g, _ in data["ok"]) * 1.05)
        ax.set_xlabel("KV pool 大小 (GiB)", color=theme["muted"], fontsize=8.5)
        ax.set_ylabel("每步耗时 (ms)", color=theme["muted"], fontsize=8.5)
        for key, name, color in series:
            xs, ys = zip(*data[key])
            ax.plot(xs, ys, color=color, linewidth=2 * PT, solid_capstyle="round", solid_joinstyle="round",
                    marker="o", markersize=8 * PT, markeredgecolor=theme["surface"], markeredgewidth=2 * PT, zorder=3)
            ax.annotate(f"{ys[-1]:.1f} ms", (xs[-1], ys[-1]), xytext=(10, 0), textcoords="offset points",
                        va="center", fontsize=9, color=theme["ink2"])
        legend_row(fig, [(n, c) for _, n, c in series], theme, y=1 - 72 / hpx)
        title(fig, "检测 buffer 捐赠失败：每步耗时是否随 KV pool 变大而变大",
              "batch = 8，context = 256；捐赠失败时，每一步每一层都要把整层 KV pool 复制一遍", theme)
        return fig

    print("saved:", save_both(draw, outdir, "ch10-donation"))


def plot_timeline(results, outdir, window_ms: float = 40.0):
    from matplotlib.patches import Rectangle

    from ch10_plots import _px_to_data, new_figure, save_both, title

    def draw(theme):
        n = len(results)
        hpx = 112 + 50 * n
        fig, (ax,) = new_figure(theme, 980, hpx)
        fig.subplots_adjust(left=0.15, right=0.86, top=1 - 76 / hpx, bottom=46 / hpx)
        ax.set_xlim(0, window_ms)
        ax.set_ylim(n - 0.5, -0.5)
        ax.set_yticks(range(n))
        ax.set_yticklabels([r[0] for r in results], color=theme["ink2"], fontsize=9)
        ax.set_xlabel("时间 (ms)", color=theme["muted"], fontsize=8.5)
        hdata = _px_to_data(ax, 0, 18)[1]  # 18px 高的色条
        for i, (_, st) in enumerate(results):
            iv = st.busy_ms_intervals
            t0 = iv[len(iv) // 2][0]  # 从分析窗口的中段截取一段
            for s, e in iv:
                if e < t0 or s > t0 + window_ms:
                    continue
                s, e = max(s, t0) - t0, min(e, t0 + window_ms) - t0
                ax.add_patch(Rectangle((s, i - hdata / 2), e - s, hdata, facecolor=theme["series"][0],
                                       edgecolor="none", zorder=3))
            ax.text(window_ms * 1.01, i, f"GPU 利用率 {st.utilization:.0%}", va="center", fontsize=9,
                    color=theme["ink2"])
        title(fig, "GPU 时间线：色块是 GPU 在执行，空白是 GPU 在等",
              f"Instruments Metal System Trace，decode 阶段截取 {window_ms:.0f} ms；利用率按整段 decode 统计", theme)
        return fig

    print("saved:", save_both(draw, outdir, "ch10-timeline"))


# ============================================================================
# main
# ============================================================================


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["roofline", "phases", "ops", "donation", "memory", "timeline", "capture",
                                        "all", "_workload", "_workload_mlx_lm"])
    p.add_argument("--model", default=MODEL)
    p.add_argument("--batch", type=int, nargs="*", default=None)
    p.add_argument("--context", type=int, default=1024)
    p.add_argument("--steps", type=int, default=128)
    p.add_argument("--pool", type=int, nargs="*", default=None, help="donation: KV pool 的 token 数")
    p.add_argument("--pattern-a", action="store_true", help="donation: 额外测第 7 章的写法 A（构图后立即提交）")
    p.add_argument("--attn", default="naive", help="memory: attention 后端")
    p.add_argument("--cache-limits", type=float, nargs="*", default=[4.0], help="memory: 要对比的 MLX 缓存上限 (GiB)")
    p.add_argument("--overlap", type=int, default=1, help=argparse.SUPPRESS)
    p.add_argument("--keep-trace", action="store_true", help="timeline: 把 .trace 保存到当前目录")
    p.add_argument("--with-mlx-lm", action="store_true", help="timeline: 额外录制 mlx-lm 单请求生成作为对照")
    p.add_argument("--out", default="decode.gputrace", help="capture: 输出路径")
    p.add_argument("--capture-steps", type=int, default=3)
    p.add_argument("--plot", default=None, help="把结果画成 PNG，保存到这个目录")
    args = p.parse_args()

    if args.plot:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    commands = {
        "roofline": cmd_roofline, "phases": cmd_phases, "ops": cmd_ops, "donation": cmd_donation,
        "memory": cmd_memory, "timeline": cmd_timeline, "capture": cmd_capture, "_workload": cmd_workload,
        "_workload_mlx_lm": cmd_workload_mlx_lm,
    }
    if args.command == "all":
        for name in ("roofline", "phases", "ops", "donation", "memory", "timeline"):
            commands[name](args)
    else:
        commands[args.command](args)


if __name__ == "__main__":
    main()
