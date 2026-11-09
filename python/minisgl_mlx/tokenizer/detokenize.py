"""
流式反分词（incremental detokenization），对应原版 minisgl/tokenizer/detokenize.py（源自 SGLang）。

为什么不能"每来一个 token 就 decode 这个 token"？
  1. BPE 中一个汉字/emoji 可能被拆成多个 byte-level token，单独 decode 会得到 "�"；
  2. 有些 tokenizer（如 SentencePiece）decode 时会根据上下文决定是否加空格，单独 decode 会丢空格。

做法：维护两个偏移量
    surr_offset ... read_offset ... len(decoded_ids)
    |<-- surrounding -->|<-- 新 token -->|
每次把 [surr_offset, 末尾) 整段 decode，减去 [surr_offset, read_offset) 的 decode 结果，
得到的差就是"新增文本"。若新增文本以 "�" 结尾，说明字符还不完整，先不推进偏移量。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Set

from minisgl_mlx.message import DetokenizeMsg


def _is_chinese_char(cp: int) -> bool:
    return (
        0x4E00 <= cp <= 0x9FFF
        or 0x3400 <= cp <= 0x4DBF
        or 0x20000 <= cp <= 0x2A6DF
        or 0x2A700 <= cp <= 0x2B73F
        or 0x2B740 <= cp <= 0x2B81F
        or 0x2B820 <= cp <= 0x2CEAF
        or 0xF900 <= cp <= 0xFAFF
        or 0x2F800 <= cp <= 0x2FA1F
    )


def find_printable_text(text: str) -> str:
    """返回可以安全输出的最长前缀（不会因为后续 token 而改变）。"""
    if text.endswith("\n"):
        return text
    if len(text) > 0 and _is_chinese_char(ord(text[-1])):
        return text
    if len(text) > 1 and _is_chinese_char(ord(text[-2])):
        return text[:-1]
    return text[: text.rfind(" ") + 1]


@dataclass
class DecodeStatus:
    decoded_ids: List[int]
    decoded_str: str
    read_offset: int
    surr_offset: int
    sent_offset: int  # 已经发送给用户的字符数


class DetokenizeManager:
    def __init__(self, tokenizer, eos_token_ids: Set[int]) -> None:
        self.decode_map: Dict[int, DecodeStatus] = {}
        self.tokenizer = tokenizer
        self.eos_token_ids = eos_token_ids

    def detokenize(self, msgs: List[DetokenizeMsg]) -> List[str]:
        read_ids: List[List[int]] = []
        surr_ids: List[List[int]] = []
        for msg in msgs:
            s = self.decode_map.setdefault(msg.uid, DecodeStatus([], "", 0, 0, 0))
            if not (msg.finished and msg.next_token in self.eos_token_ids):
                s.decoded_ids.append(msg.next_token)
            read_ids.append(s.decoded_ids[s.surr_offset :])
            surr_ids.append(s.decoded_ids[s.surr_offset : s.read_offset])

        read_texts = self.tokenizer.batch_decode(read_ids)
        surr_texts = self.tokenizer.batch_decode(surr_ids)

        outputs: List[str] = []
        for msg, read_str, surr_str in zip(msgs, read_texts, surr_texts, strict=True):
            s = self.decode_map[msg.uid]
            new_text = read_str[len(surr_str) :]
            if len(new_text) > 0 and not new_text.endswith("�"):
                output_str = s.decoded_str + new_text
                s.decoded_str = output_str
                s.surr_offset = s.read_offset
                s.read_offset = len(s.decoded_ids)
            elif msg.finished:  # 结束了：剩下的不管完不完整都吐出来
                output_str = s.decoded_str + new_text
            else:  # 字符还不完整：只输出确定不会再变的部分
                output_str = s.decoded_str + find_printable_text(new_text)
            outputs.append(output_str[s.sent_offset :])
            s.sent_offset = len(output_str)
            if msg.finished:
                del self.decode_map[msg.uid]
        return outputs
