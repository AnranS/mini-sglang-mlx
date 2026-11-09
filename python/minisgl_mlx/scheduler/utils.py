from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from minisgl_mlx.core import SamplingParams

    from .prefill import ChunkedReq


@dataclass
class PendingReq:
    """还在排队、没被调度上去的请求（或者 chunked prefill 做到一半的请求）。"""

    uid: int
    input_ids: np.ndarray
    sampling_params: SamplingParams
    chunked_req: ChunkedReq | None = None

    @property
    def input_len(self) -> int:
        return len(self.input_ids)

    @property
    def output_len(self) -> int:
        return self.sampling_params.max_tokens
