"""
性能分析工具（第 10 章）。四类工具，对应"推理为什么慢"的四个层次：

    1. 理论下限   measure_bandwidth / measure_matmul_tflops / decode_lower_bound / prefill_lower_bound
                  —— 先算出"最快能有多快"，判断慢在哪一侧（访存？计算？额外开销？）
    2. CPU 侧阶段  PhaseTimer —— 不修改调度器代码，挂钩子统计每一轮循环中各阶段的耗时
    3. 内存        memory_snapshot —— MLX active / peak / cache 内存，以及系统 swap
    4. GPU 时间线  gpu_timeline_stats —— 解析 Instruments (xctrace) 的 Metal System Trace，
                  得到 GPU 真实的忙碌时间、利用率和空隙分布

用法见 tutorials/ch10_profiling.py 和 docs/10-profiling.md。
"""

from __future__ import annotations

import functools
import re
import subprocess
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Tuple

import mlx.core as mx
import numpy as np

if TYPE_CHECKING:
    from minisgl_mlx.models import ModelConfig
    from minisgl_mlx.scheduler import Scheduler

# ============================================================================
# 1. 理论下限（roofline）
# ============================================================================


def _bench_best(fn, iters: int = 20, warmup: int = 10) -> float:
    """返回多次测量中最快的一次（秒）。GPU 需要预热几轮才会升到最高频率，所以 warmup 不能太少。"""
    for _ in range(warmup):
        mx.eval(fn())
    best = float("inf")
    for _ in range(iters):
        t = time.perf_counter()
        mx.eval(fn())
        best = min(best, time.perf_counter() - t)
    return best


def measure_bandwidth(gib: float = 2.0) -> float:
    """
    实测"可达峰值"内存带宽（GB/s）：对一个大数组做 y = x * 2（读一遍 + 写一遍），取最好的一次。
    单次测量波动很大（实测同一台机器 500~1050 GB/s），取最好值才适合作为理论下限的依据。
    """
    cols = 1 << 16
    rows = int(gib * (1 << 30) / 2 / cols)
    x = mx.ones((rows, cols), dtype=mx.bfloat16)
    mx.eval(x)
    return 2 * x.nbytes / _bench_best(lambda: x * 2) / 1e9


def measure_matmul_tflops(n: int = 8192, dtype=mx.bfloat16) -> float:
    """实测矩阵乘的可达峰值算力（TFLOPS），同样取最好的一次。"""
    a = mx.random.normal((n, n)).astype(dtype)
    b = mx.random.normal((n, n)).astype(dtype)
    mx.eval(a, b)
    return 2 * n**3 / _bench_best(lambda: a @ b, iters=10, warmup=5) / 1e12


def linear_flops_per_token(cfg: ModelConfig) -> float:
    """每个 token 经过所有线性层（不含 lm_head）的 FLOPs：2 × 参数量。"""
    h, d = cfg.hidden_size, cfg.head_dim
    per_layer = h * cfg.num_qo_heads * d * 2 + h * cfg.num_kv_heads * d * 2 + 3 * h * cfg.intermediate_size
    return 2.0 * per_layer * cfg.num_layers


def kv_bytes_per_token(cfg: ModelConfig, dtype_size: int = 2) -> int:
    return 2 * cfg.num_layers * cfg.num_kv_heads * cfg.head_dim * dtype_size


def decode_lower_bound(cfg: ModelConfig, weight_bytes: int, batch: int, context: int, bandwidth_gbs: float) -> float:
    """一步 decode 的理论最短时间（秒）：至少要把权重读一遍 + 每个请求的全部 KV 读一遍。"""
    total = weight_bytes + batch * context * kv_bytes_per_token(cfg)
    return total / (bandwidth_gbs * 1e9)


def prefill_lower_bound(cfg: ModelConfig, num_tokens: int, tflops: float) -> float:
    """prefill num_tokens 个 token（单个请求、无前缀）的理论最短时间（秒），按算力估计。"""
    linear = linear_flops_per_token(cfg) * num_tokens
    # causal attention：QK^T 和 PV 各 2·T²/2·Hq·D FLOPs（只算下三角）
    attn = 2 * num_tokens**2 * cfg.num_qo_heads * cfg.head_dim * cfg.num_layers
    return (linear + attn) / (tflops * 1e12)


# ============================================================================
# 2. CPU 侧分阶段计时
# ============================================================================


@dataclass
class IterRecord:
    phase: str | None = None  # 本轮调度出的 batch 是 prefill / decode / None
    batch_size: int = 0
    num_tokens: int = 0
    times: Dict[str, float] = field(default_factory=dict)

    @property
    def total(self) -> float:
        return sum(self.times.values())


class PhaseTimer:
    """
    把计时器"挂"到一个 Scheduler（或 LLM）实例上，不需要修改调度器的代码：
    把实例上的若干方法替换成"计时 + 调用原方法"的包装，每一轮主循环产生一条 IterRecord。

        timer = PhaseTimer().attach(llm)
        llm.generate(...)
        print(timer.summary())

    各阶段的含义：
        recv      收消息（阻塞等待新请求的那次不计入）
        schedule  调度 + 分配 KV 槽位 + 准备元数据（_schedule_next_batch）
        build     构建计算图（_forward）
        submit    提交给 GPU（mx.async_eval）
        wait      等待 GPU 算完（mx.eval）
        process   处理结果（_process_last_data）

    非 overlap 模式下 submit + wait ≈ GPU 时间（再加上提交/同步的固定开销），其余阶段都是"GPU 空闲时的 CPU 时间"。
    overlap 模式下 build / process 与 GPU 计算重叠，wait 是"CPU 干完活后还要等 GPU 多久"。
    """

    PHASES = ("recv", "schedule", "build", "submit", "wait", "process")

    def __init__(self) -> None:
        self.records: List[IterRecord] = []
        self._cur: IterRecord | None = None

    def attach(self, sched: Scheduler) -> PhaseTimer:
        def timed(name: str, fn):
            @functools.wraps(fn)
            def wrapper(*args, **kwargs):
                if name == "recv" and kwargs.get("blocking", False):
                    return fn(*args, **kwargs)  # 空闲等待不算开销
                t = time.perf_counter()
                out = fn(*args, **kwargs)
                if self._cur is not None:
                    self._cur.times[name] = self._cur.times.get(name, 0.0) + time.perf_counter() - t
                if name == "schedule" and out is not None and self._cur is not None:
                    self._cur.phase = out.batch.phase
                    self._cur.batch_size = out.batch.size
                    self._cur.num_tokens = int(out.batch.positions.shape[0])
                return out

            return wrapper

        def loop(fn):
            @functools.wraps(fn)
            def wrapper(*args, **kwargs):
                self._cur = IterRecord()
                try:
                    return fn(*args, **kwargs)
                finally:
                    self.records.append(self._cur)
                    self._cur = None

            return wrapper

        sched.receive_msg = timed("recv", sched.receive_msg)
        sched._schedule_next_batch = timed("schedule", sched._schedule_next_batch)
        sched._forward = timed("build", sched._forward)
        sched._submit = timed("submit", sched._submit)
        sched._wait = timed("wait", sched._wait)
        sched._process_last_data = timed("process", sched._process_last_data)
        sched.overlap_loop = loop(sched.overlap_loop)
        sched.normal_loop = loop(sched.normal_loop)
        return self

    def reset(self) -> None:
        self.records = []

    def select(self, phase: str = "decode") -> List[IterRecord]:
        return [r for r in self.records if r.phase == phase]

    def median_ms(self, phase: str = "decode") -> Dict[str, float]:
        recs = self.select(phase)
        if not recs:
            return {}
        out = {p: float(np.median([r.times.get(p, 0.0) for r in recs])) * 1e3 for p in self.PHASES}
        out["total"] = float(np.median([r.total for r in recs])) * 1e3
        return out

    def summary(self) -> str:
        lines = []
        for phase in ("prefill", "decode"):
            recs = self.select(phase)
            if not recs:
                continue
            med = self.median_ms(phase)
            bs = int(np.median([r.batch_size for r in recs]))
            toks = int(np.median([r.num_tokens for r in recs]))
            parts = " | ".join(f"{p} {med[p]:6.2f}" for p in self.PHASES)
            lines.append(
                f"[{phase:7s}] {len(recs):4d} iters, median batch={bs} tokens={toks}\n"
                f"           median ms: {parts} | total {med['total']:6.2f}"
            )
        return "\n".join(lines)


# ============================================================================
# 3. 内存
# ============================================================================


def swap_used_bytes() -> int | None:
    try:
        out = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True).stdout
        m = re.search(r"used = ([\d.]+)M", out)
        return int(float(m.group(1)) * (1 << 20)) if m else None
    except OSError:
        return None


def memory_snapshot() -> Dict[str, int | None]:
    return {
        "active": mx.get_active_memory(),  # 正在被 mx.array 使用的内存
        "peak": mx.get_peak_memory(),  # 自上次 reset_peak_memory 以来的峰值
        "cache": mx.get_cache_memory(),  # MLX 缓存起来、等待复用的 buffer
        "swap": swap_used_bytes(),
    }


def format_bytes(n: int | None) -> str:
    return "n/a" if n is None else f"{n / (1 << 30):.2f} GiB"


# ============================================================================
# 4. GPU 时间线（Instruments / xctrace 的 Metal System Trace）
# ============================================================================


def record_metal_trace(cmd: List[str], output: str) -> None:
    """用 Instruments 的 Metal System Trace 模板运行 cmd，结果保存为 output（.trace）。"""
    subprocess.run(
        ["xctrace", "record", "--template", "Metal System Trace", "--output", output, "--launch", "--", *cmd],
        check=True,
    )


def export_gpu_intervals(trace: str, process_prefix: str = "python") -> List[Tuple[int, int]]:
    """
    从 .trace 中导出 metal-gpu-intervals 表，返回目标进程在 GPU 上执行的 (start_ns, end_ns) 列表。

    xctrace 导出的 XML 会把重复出现的值写成 <xxx ref="id"/>，引用前面某个 id="id" 的元素，
    所以解析时要维护一张 id -> (fmt, text) 的表。
    """
    import xml.etree.ElementTree as ET

    xpath = '/trace-toc/run[@number="1"]/data/table[@schema="metal-gpu-intervals"]'
    xml = subprocess.run(
        ["xctrace", "export", "--input", trace, "--xpath", xpath], capture_output=True, text=True, check=True
    ).stdout
    table: Dict[str, Tuple[str, str]] = {}

    def register(el) -> None:
        if "id" in el.attrib:
            table[el.attrib["id"]] = (el.attrib.get("fmt", ""), el.text or "")
        for child in el:
            register(child)

    def value(el) -> Tuple[str, str]:
        if "ref" in el.attrib:
            return table[el.attrib["ref"]]
        return el.attrib.get("fmt", ""), el.text or ""

    intervals = []
    for row in ET.fromstring(xml).iter("row"):
        cols = list(row)
        for c in cols:
            register(c)
        start, dur = int(value(cols[0])[1]), int(value(cols[1])[1])
        channel, depth, proc = value(cols[2])[0], value(cols[5])[0], value(cols[10])[0]
        if channel == "Compute" and depth == "0" and proc.lower().startswith(process_prefix.lower()):
            intervals.append((start, start + dur))
    return intervals


@dataclass
class TimelineStats:
    window_ms: float
    busy_ms: float
    num_intervals: int
    gaps_us: np.ndarray
    busy_ms_intervals: np.ndarray  # [N, 2]，相对窗口起点的 (start_ms, end_ms)，画时间线用

    @property
    def utilization(self) -> float:
        return self.busy_ms / self.window_ms if self.window_ms > 0 else 0.0

    def describe(self, num_steps: int | None = None, big_gap_us: float = 300.0) -> str:
        g = self.gaps_us
        big = g[g > big_gap_us]
        lines = [
            f"分析窗口 {self.window_ms:.1f} ms，GPU 忙碌 {self.busy_ms:.1f} ms -> 利用率 {self.utilization:.0%}",
            f"空隙: {len(g)} 个，中位数 {np.median(g) if len(g) else 0:.0f} us，"
            f"> {big_gap_us:.0f} us 的有 {len(big)} 个，合计 {big.sum() / 1e3:.1f} ms",
        ]
        if num_steps:
            lines.append(
                f"按 {num_steps} 步平均：每步 {self.window_ms / num_steps:.2f} ms，其中 GPU 忙碌 "
                f"{self.busy_ms / num_steps:.2f} ms、空闲 {(self.window_ms - self.busy_ms) / num_steps:.2f} ms"
            )
        return "\n".join(lines)


def gpu_timeline_stats(intervals: List[Tuple[int, int]], split_gap_ms: float = 300.0) -> TimelineStats:
    """
    合并重叠的区间，并只分析"最后一段连续工作"：
    把 > split_gap_ms 的空闲当作分隔（tutorial 会在正式测量前 sleep 一下来制造这个分隔），
    这样模型加载、预热等阶段不会混进统计。
    """
    merged: List[List[int]] = []
    for s, e in sorted(intervals):
        if merged and s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    if not merged:
        return TimelineStats(0.0, 0.0, 0, np.array([]), np.zeros((0, 2)))
    cut = 0
    for i in range(len(merged) - 1):
        if merged[i + 1][0] - merged[i][1] > split_gap_ms * 1e6:
            cut = i + 1
    m = merged[cut:]
    window = m[-1][1] - m[0][0]
    busy = sum(e - s for s, e in m)
    gaps = np.array([m[i + 1][0] - m[i][1] for i in range(len(m) - 1)], dtype=np.float64) / 1e3
    rel = (np.array(m, dtype=np.float64) - m[0][0]) / 1e6
    return TimelineStats(window / 1e6, busy / 1e6, len(m), gaps, rel)


def parameter_bytes(module: Any) -> int:
    from mlx.utils import tree_flatten

    return sum(v.nbytes for _, v in tree_flatten(module.parameters()))


__all__ = [
    "measure_bandwidth",
    "measure_matmul_tflops",
    "linear_flops_per_token",
    "kv_bytes_per_token",
    "decode_lower_bound",
    "prefill_lower_bound",
    "PhaseTimer",
    "IterRecord",
    "memory_snapshot",
    "swap_used_bytes",
    "format_bytes",
    "record_metal_trace",
    "export_gpu_intervals",
    "gpu_timeline_stats",
    "TimelineStats",
    "parameter_bytes",
]
