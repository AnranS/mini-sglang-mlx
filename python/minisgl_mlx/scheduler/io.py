"""调度器的收发逻辑：在线模式走 ZMQ，离线模式（LLM 类）由子类覆盖。对应原版 minisgl/scheduler/io.py。"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, List

from minisgl_mlx.message import BatchTokenizerMsg, DetokenizeMsg, decode, encode, unwrap
from minisgl_mlx.utils import ZmqPullQueue, ZmqPushQueue

if TYPE_CHECKING:
    from .config import SchedulerConfig


class SchedulerIOMixin:
    def __init__(self, config: SchedulerConfig) -> None:
        if config.offline_mode:
            self.receive_msg = self.offline_receive_msg
            self.send_result = self.offline_send_result
            return
        self._recv_from_tokenizer = ZmqPullQueue(config.zmq_backend_addr, create=True, decoder=decode)
        self._send_into_tokenizer = ZmqPushQueue(config.zmq_detokenizer_addr, create=False, encoder=encode)
        self.receive_msg = self._recv_msg
        self.send_result = self._reply_tokenizer

    def run_when_idle(self) -> None:
        raise NotImplementedError

    def offline_receive_msg(self, blocking: bool = False) -> List[Any]:
        raise NotImplementedError

    def offline_send_result(self, reply: List[DetokenizeMsg]) -> None:
        raise NotImplementedError

    def _recv_msg(self, blocking: bool = False) -> List[Any]:
        msgs: List[Any] = []
        if blocking:  # 没有任何活可干：阻塞等待新请求
            self.run_when_idle()
            msgs.extend(unwrap(self._recv_from_tokenizer.get()))
        while not self._recv_from_tokenizer.empty():
            msgs.extend(unwrap(self._recv_from_tokenizer.get()))
        return msgs

    def _reply_tokenizer(self, reply: List[DetokenizeMsg]) -> None:
        if len(reply) == 1:
            self._send_into_tokenizer.put(reply[0])
        elif len(reply) > 1:
            self._send_into_tokenizer.put(BatchTokenizerMsg(data=reply))  # type: ignore[arg-type]

    def shutdown_io(self) -> None:
        if hasattr(self, "_recv_from_tokenizer"):
            self._recv_from_tokenizer.stop()
            self._send_into_tokenizer.stop()
