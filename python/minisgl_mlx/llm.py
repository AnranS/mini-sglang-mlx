"""
离线批量推理接口，对应原版 minisgl/llm/llm.py。

LLM 直接继承 Scheduler，把"从 ZMQ 收消息 / 往 ZMQ 发结果"换成"从 Python 列表取请求 / 把结果存进字典"，
于是整套调度逻辑（continuous batching、chunked prefill、radix cache、overlap）在单进程里原样运行。
这是学习和调试调度器最方便的入口。

    from minisgl_mlx import LLM, SamplingParams
    llm = LLM("mlx-community/Qwen3-0.6B-bf16")
    outputs = llm.generate(["你好", "1+1="], SamplingParams(max_tokens=32))
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from minisgl_mlx.core import SamplingParams
from minisgl_mlx.message import DetokenizeMsg, UserMsg
from minisgl_mlx.scheduler import Scheduler, SchedulerConfig


class RequestAllFinished(Exception):
    pass


@dataclass
class RequestStatus:
    uid: int
    input_ids: List[int]
    output_ids: List[int]


class LLM(Scheduler):
    def __init__(self, model_path: str, **kwargs: Any) -> None:
        super().__init__(SchedulerConfig(model_path=model_path, offline_mode=True, **kwargs))
        self.pending_requests: List[Tuple[np.ndarray, SamplingParams]] = []
        self.status_map: Dict[int, RequestStatus] = {}
        self.counter = 0

    def _tokenize(self, prompt: str | List[int]) -> np.ndarray:
        ids = self.tokenizer.encode(prompt) if isinstance(prompt, str) else prompt
        return np.asarray(ids, dtype=np.int32)

    # ---------------------------------------------------- 替换掉 ZMQ 的收发

    def offline_receive_msg(self, blocking: bool = False) -> List[UserMsg]:
        if blocking and not self.pending_requests:
            raise RequestAllFinished()  # 调度器空闲且没有新请求：说明全部做完了
        # 一次只"投递"大约一个 prefill budget 的请求，模拟请求陆续到达
        results: List[UserMsg] = []
        budget = 0
        while self.pending_requests and budget < self.prefill_budget:
            input_ids, sp = self.pending_requests.pop(0)
            uid = self.counter
            self.counter += 1
            budget += len(input_ids)
            results.append(UserMsg(uid=uid, input_ids=input_ids, sampling_params=sp))
            self.status_map[uid] = RequestStatus(uid, input_ids.tolist(), [])
        return results

    def offline_send_result(self, reply: List[DetokenizeMsg]) -> None:
        for msg in reply:
            status = self.status_map[msg.uid]
            if not (msg.finished and msg.next_token in self.eos_token_ids):
                status.output_ids.append(msg.next_token)

    # ---------------------------------------------------- 对外接口

    def generate(
        self,
        prompts: Sequence[str] | Sequence[List[int]],
        sampling_params: SamplingParams | Sequence[SamplingParams] = SamplingParams(),
    ) -> List[Dict[str, Any]]:
        self.pending_requests, self.status_map, self.counter = [], {}, 0
        if isinstance(sampling_params, SamplingParams):
            sampling_params = [sampling_params] * len(prompts)
        for prompt, sp in zip(prompts, sampling_params, strict=True):
            # 复制一份，避免调度器修改 max_tokens 时影响到调用方
            self.pending_requests.append((self._tokenize(prompt), SamplingParams(**vars(sp))))
        try:
            self.run_forever()
        except RequestAllFinished:
            pass
        return [
            {"text": self.tokenizer.decode(s.output_ids), "token_ids": s.output_ids}
            for s in (self.status_map[i] for i in range(len(prompts)))
        ]

    def chat(self, messages_list: Sequence[List[Dict[str, str]]], sampling_params=SamplingParams(), **kw):
        """messages 格式与 OpenAI 相同；用 tokenizer 的 chat template 拼成 prompt。"""
        prompts = [
            self.tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True, **kw)
            for m in messages_list
        ]
        # chat template 已经包含了 BOS 等特殊 token，分词时不要再加一次（否则 Llama 会出现两个 BOS）
        token_ids = [self.tokenizer.encode(p, add_special_tokens=False) for p in prompts]
        return self.generate(token_ids, sampling_params)
