"""
Scheduler：整个推理系统的"心脏"，对应原版 minisgl/scheduler/scheduler.py。

每一轮循环做四件事：
    1. 收消息    新请求进入 PrefillManager 的等待队列，abort 的请求被释放
    2. 调度      prefill 优先：能凑出 prefill batch 就做 prefill，否则所有运行中的请求做一步 decode
    3. 执行      为 batch 分配 KV 槽位、准备元数据，交给 Engine 构建计算图并提交给 GPU
    4. 处理结果  把采样出的 token 追加到请求上，判断是否结束，结束的请求释放资源

overlap 调度（默认开启）把第 4 步和下一轮的第 1~3 步与 GPU 计算重叠起来，详见 docs/07-overlap.md。
"""

from __future__ import annotations

import time
from typing import Any, List, NamedTuple, NoReturn, Set, Tuple

import mlx.core as mx
import numpy as np

from minisgl_mlx.core import Batch, Req
from minisgl_mlx.engine import BatchSamplingArgs, Engine
from minisgl_mlx.message import AbortBackendMsg, DetokenizeMsg, ExitMsg, UserMsg
from minisgl_mlx.utils import init_logger, load_tokenizer

from .cache import CacheManager
from .config import SchedulerConfig
from .decode import DecodeManager
from .io import SchedulerIOMixin
from .prefill import ChunkedReq, PrefillManager
from .table import TableManager

logger = init_logger(__name__)


class ForwardInput(NamedTuple):
    batch: Batch
    sample_args: BatchSamplingArgs
    write_idx: mx.array  # [B] 每个请求的 table_idx：采样结果写回 token_pool 的位置


# (本轮输入, 本轮采样结果[惰性 mx.array])
ForwardData = Tuple[ForwardInput, mx.array]


class Scheduler(SchedulerIOMixin):
    def __init__(self, config: SchedulerConfig) -> None:
        self.config = config
        self.engine = Engine(config)

        self.table_manager = TableManager(config.max_running_req, self.engine.page_table)
        self.cache_manager = CacheManager(
            num_pages=self.engine.num_tokens // config.page_size,
            page_size=config.page_size,
            page_table=self.engine.page_table,
            type=config.cache_type,
        )
        self.decode_manager = DecodeManager(config.page_size)
        self.prefill_manager = PrefillManager(self.cache_manager, self.table_manager, self.decode_manager)

        self.finished_reqs: Set[Req] = set()
        self.tokenizer = load_tokenizer(self.engine.model_path)
        self.eos_token_ids = set(self.engine.model_config.eos_token_ids)
        if self.tokenizer.eos_token_id is not None:
            self.eos_token_ids.add(self.tokenizer.eos_token_id)
        self.token_pool = self.table_manager.token_pool
        self.prefill_budget = config.max_extend_tokens
        self.stats = _Stats()
        super().__init__(config)

    # ======================================================================
    # 主循环
    # ======================================================================

    def normal_loop(self) -> None:
        """不做 overlap 的版本：构图 -> 提交并等待 -> 处理结果，CPU 和 GPU 轮流干活。"""
        blocking = not (self.prefill_manager.runnable or self.decode_manager.runnable)
        for msg in self.receive_msg(blocking=blocking):
            self._process_one_msg(msg)

        data = None
        if (forward_input := self._schedule_next_batch()) is not None:
            data = (forward_input, self._forward(forward_input))
            self._submit(data)
            self._wait(data)
        self._process_last_data(data)

    def overlap_loop(self, last_data: ForwardData | None) -> ForwardData | None:
        """
        overlap 调度：GPU 算第 t 批的同时，CPU 处理第 t-1 批的结果、并构建第 t+1 批。

            CPU: [调度 t+1 | 构图 t+1] 等待 [提交 t+1] [处理 t 的结果]  [调度 t+2 | 构图 t+2] 等待 ...
            GPU: ======== 计算 t ========  ============== 计算 t+1 ============== ...

        与原版（CUDA）的一个重要区别：原版构图即提交（kernel launch 是异步的），这里必须等第 t 批
        算完后再提交第 t+1 批 —— 否则第 t 批对 KV pool 的读取还没结束，MLX 无法把 KV pool 的
        buffer "捐赠"给第 t+1 批的 scatter，只能把整个 KV pool 拷贝一遍（GB 级别！）。
        详见 docs/07-overlap.md 和 tutorials/ch07_overlap.py。
        """
        blocking = not (
            last_data is not None or self.prefill_manager.runnable or self.decode_manager.runnable
        )
        for msg in self.receive_msg(blocking=blocking):
            self._process_one_msg(msg)

        ongoing_data = None
        if (forward_input := self._schedule_next_batch()) is not None:
            # ① 构建第 t+1 批的计算图（纯 CPU 工作，此时 GPU 还在算第 t 批）
            ongoing_data = (forward_input, self._forward(forward_input))
        if last_data is not None:
            # ② 等第 t 批算完
            self._wait(last_data)
        if ongoing_data is not None:
            # ③ 提交第 t+1 批（不等它算完；但 MLX 在途任务较多时 async_eval 也会阻塞一会儿，见第 7、10 章）
            self._submit(ongoing_data)
        # ④ 处理第 t 批的结果（CPU），GPU 正在算第 t+1 批
        self._process_last_data(last_data)
        return ongoing_data

    def run_forever(self) -> NoReturn:
        if self.config.overlap_scheduling:
            data = None
            while True:
                data = self.overlap_loop(data)
        else:
            while True:
                self.normal_loop()

    def run_when_idle(self) -> None:
        self.stats.flush(self, force=True)
        logger.info("Scheduler is idle, waiting for new requests...")
        self.cache_manager.check_integrity()

    def shutdown(self) -> None:
        mx.synchronize()
        self.shutdown_io()
        self.engine.shutdown()

    # ======================================================================
    # 调度 & 执行
    # ======================================================================

    def _schedule_next_batch(self) -> ForwardInput | None:
        # 策略：prefill 优先（新请求尽快出第一个 token，降低 TTFT）
        batch = self.prefill_manager.schedule_next_batch(self.prefill_budget) or (
            self.decode_manager.schedule_next_batch()
        )
        return self._prepare_batch(batch) if batch is not None else None

    def _prepare_batch(self, batch: Batch) -> ForwardInput:
        """为 batch 分配 KV 槽位，并准备好 forward 需要的所有输入（CPU 上用 numpy 计算，再转 mx.array）。"""
        reqs = batch.reqs
        self.cache_manager.allocate_paged(reqs)

        positions = np.concatenate([np.arange(r.cached_len, r.device_len, dtype=np.int32) for r in reqs])
        rows = np.repeat(np.array([r.table_idx for r in reqs]), [r.extend_len for r in reqs])
        batch.positions = mx.array(positions)
        batch.out_loc = mx.array(self.engine.page_table[rows, positions])  # 新 token 的 KV 写到哪
        table_idx = mx.array([r.table_idx for r in reqs], dtype=mx.int32)

        if batch.is_prefill:  # prompt 在 CPU 上本来就有
            batch.input_ids = mx.array(np.concatenate([r.input_ids[r.cached_len : r.device_len] for r in reqs]))
        else:  # 上一步采样出的 token 还在 GPU 上（甚至还没算出来），直接从 token_pool 读
            batch.input_ids = self.token_pool[table_idx]

        self.engine.attn_backend.prepare_metadata(batch)
        self.stats.on_batch(batch, positions)
        return ForwardInput(batch, self.engine.sampler.prepare(batch), table_idx)

    def _submit(self, data: ForwardData) -> None:
        """把这一批的计算图提交给 GPU（单独成一个方法，方便 profiling.PhaseTimer 挂钩子计时）。"""
        mx.async_eval(data[1], self.token_pool)

    def _wait(self, data: ForwardData) -> None:
        """等待这一批的采样结果算完。"""
        mx.eval(data[1])

    def _forward(self, forward_input: ForwardInput) -> mx.array:
        batch, sample_args, write_idx = forward_input
        next_tokens = self.engine.forward_batch(batch, sample_args)
        # 采样结果写回 token_pool —— 仍然是惰性的，下一轮 decode 读它时自然形成计算图上的依赖
        self.token_pool[write_idx] = next_tokens
        self.decode_manager.filter_reqs(batch.reqs)
        return next_tokens

    def _process_last_data(self, last_data: ForwardData | None) -> None:
        if last_data is None:
            return
        batch, next_tokens = last_data[0].batch, last_data[1]
        tokens = next_tokens.tolist()  # 已经 eval 过；统一内存下读回 CPU 几乎零开销

        reply: List[DetokenizeMsg] = []
        new_finished_reqs: Set[Req] = set()
        with self.cache_manager.lazy_free_region():
            for req, token in zip(batch.reqs, tokens):
                if isinstance(req, ChunkedReq):
                    continue  # prompt 还没算完，这个 token 没有意义
                if req in self.finished_reqs:
                    # overlap 调度下，一个请求在第 t 批结束时已经被放进了第 t+1 批。
                    # 第 t+1 批的这个 token 是多余的，丢弃即可（资源在处理第 t 批时已经释放）。
                    continue
                req.append_host(token)
                # 注意：这里不能用 `not req.can_decode` 判断是否达到长度上限！overlap 调度下，
                # 处理第 t 批结果时 req.device_len 已经被第 t+1 批的 complete_one() 推进了一格，
                # 会提前一个 token 判定结束。host 侧 input_ids 的长度才是"真正处理过"的 token 数。
                finished = len(req.input_ids) >= req.max_device_len
                if not req.sampling_params.ignore_eos:
                    finished |= token in self.eos_token_ids
                reply.append(DetokenizeMsg(uid=req.uid, next_token=token, finished=finished))

                if finished:
                    self.decode_manager.remove_req(req)
                    self._free_req_resources(req)
                    new_finished_reqs.add(req)
                elif batch.is_prefill:
                    # prefill 刚结束：把 prompt 的 KV 插入前缀缓存，后续相同前缀的请求可以复用
                    self.cache_manager.cache_req(req, finished=False)

        self.finished_reqs = new_finished_reqs
        self.stats.on_tokens(len(reply))
        self.stats.flush(self)
        self.send_result(reply)

    # ======================================================================
    # 消息处理 & 资源释放
    # ======================================================================

    def _process_one_msg(self, msg: Any) -> None:
        if isinstance(msg, ExitMsg):
            raise KeyboardInterrupt
        elif isinstance(msg, UserMsg):
            input_len, max_seq_len = len(msg.input_ids), self.engine.max_seq_len
            if input_len == 0 or input_len >= max_seq_len:
                logger.warning(f"Request {msg.uid} dropped: input length {input_len}, max {max_seq_len}")
                eos = next(iter(self.eos_token_ids))
                return self.send_result([DetokenizeMsg(uid=msg.uid, next_token=eos, finished=True)])
            if msg.sampling_params.max_tokens > (max_output_len := max_seq_len - input_len):
                msg.sampling_params.max_tokens = max_output_len
                logger.warning(f"Adjust max_tokens to {max_output_len} for request {msg.uid}")
            self.prefill_manager.add_one_req(msg)
        elif isinstance(msg, AbortBackendMsg):
            req = self.prefill_manager.abort_req(msg.uid) or self.decode_manager.abort_req(msg.uid)
            if req is not None:
                logger.debug(f"Abort request {msg.uid}")
                self._free_req_resources(req)
        else:
            raise NotImplementedError(f"Unknown message type: {type(msg)}")

    def _free_req_resources(self, req: Req) -> None:
        self.table_manager.free(req.table_idx)
        self.cache_manager.cache_req(req, finished=True)


class _Stats:
    """周期性打印吞吐等指标（类似 SGLang 的 "Decode batch. #running-req ..." 日志）。"""

    INTERVAL = 5.0

    def __init__(self) -> None:
        self.last_flush = time.perf_counter()
        self.num_tokens = self.num_prefill_tokens = self.num_cached_tokens = 0
        self.num_decode_steps = 0
        self.idle = True

    def on_batch(self, batch: Batch, positions: np.ndarray) -> None:
        if self.idle:  # 从空闲恢复：重新开始计时，不把空闲时间算进吞吐
            self.idle, self.last_flush = False, time.perf_counter()
        if batch.is_prefill:
            self.num_prefill_tokens += len(positions)
            # 只统计"前缀缓存命中"：请求的第一个 chunk 满足 cached_len == 命中长度；
            # chunked prefill 后续 chunk 的 cached_len 来自自己上一个 chunk，不算命中
            self.num_cached_tokens += sum(
                r.cached_len for r in batch.reqs if r.cached_len == r.cache_handle.cached_len
            )
        else:
            self.num_decode_steps += 1
        logger.debug(
            f"{batch.phase:7s} batch: {batch.size:3d} reqs, {len(positions):5d} tokens"
            + (f", cached={[r.cached_len for r in batch.reqs]}" if batch.is_prefill else "")
        )

    def on_tokens(self, n: int) -> None:
        self.num_tokens += n

    def flush(self, scheduler: Scheduler, force: bool = False) -> None:
        """force=True 表示调度器即将进入空闲：把剩余统计打印出来，并在下次有活时重新计时。"""
        now = time.perf_counter()
        elapsed = max(now - self.last_flush, 1e-6)
        has_data = self.num_tokens > 0 or self.num_prefill_tokens > 0
        if has_data and (force or elapsed >= self.INTERVAL):
            cm = scheduler.cache_manager
            used = 1 - cm.available_size / (cm.num_pages * cm.page_size)
            logger.info(
                f"running={len(scheduler.decode_manager.running_reqs)}, "
                f"pending={len(scheduler.prefill_manager.pending_list)}, kv_usage={used:.1%}, "
                f"prefill={self.num_prefill_tokens / elapsed:.0f} tok/s "
                f"(prefix hit {self.num_cached_tokens} tok), gen={self.num_tokens / elapsed:.1f} tok/s"
            )
            self.last_flush = now
            self.num_tokens = self.num_prefill_tokens = self.num_cached_tokens = self.num_decode_steps = 0
        if force:
            self.idle = True
