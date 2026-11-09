from __future__ import annotations

from typing import List

import numpy as np

from minisgl_mlx.message import TokenizeMsg


class TokenizeManager:
    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer

    def tokenize(self, msgs: List[TokenizeMsg]) -> List[np.ndarray]:
        results: List[np.ndarray] = []
        for msg in msgs:
            if isinstance(msg.text, list):  # chat messages -> 用模型自带的 chat template 拼 prompt
                prompt = self.tokenizer.apply_chat_template(
                    msg.text, tokenize=False, add_generation_prompt=True
                )
                # chat template 里已经包含了 BOS 等特殊 token，不要再加一次
                ids = self.tokenizer.encode(prompt, add_special_tokens=False)
            else:
                ids = self.tokenizer.encode(msg.text)
            results.append(np.asarray(ids, dtype=np.int32))
        return results
