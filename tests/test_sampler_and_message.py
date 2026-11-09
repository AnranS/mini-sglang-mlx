import mlx.core as mx
import numpy as np

from minisgl_mlx.core import Batch, Req, SamplingParams
from minisgl_mlx.engine import Sampler
from minisgl_mlx.message import BatchBackendMsg, UserMsg, decode, encode


def _batch(params):
    reqs = [
        Req(np.array([1, 2], dtype=np.int32), i, 0, 4, i, p, None)  # type: ignore[arg-type]
        for i, p in enumerate(params)
    ]
    return Batch(reqs=reqs, phase="decode")


def test_greedy_and_degenerate_sampling():
    sampler = Sampler(vocab_size=50)
    logits = mx.random.normal((4, 50))
    expect = mx.argmax(logits, axis=-1).tolist()
    params = [
        SamplingParams(),  # greedy
        SamplingParams(temperature=1.0, top_k=1),  # top_k=1 等价于 greedy
        SamplingParams(temperature=0.8, top_p=1e-6),  # 极小的 top_p 只保留最大的那个
        SamplingParams(temperature=1.0, top_k=1, top_p=0.5),
    ]
    batch = _batch(params)
    args = sampler.prepare(batch)
    for _ in range(5):
        assert sampler.sample(logits, args).tolist() == expect


def test_top_k_restricts_support():
    sampler = Sampler(vocab_size=10)
    logits = mx.array([[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]] * 64)
    batch = _batch([SamplingParams(temperature=5.0, top_k=3)] * 64)
    out = sampler.sample(logits, sampler.prepare(batch)).tolist()
    assert set(out) <= {7, 8, 9} and len(set(out)) > 1


def test_message_roundtrip():
    msg = BatchBackendMsg(data=[UserMsg(3, np.array([1, 2, 3], dtype=np.int32), SamplingParams(top_k=5))])
    out = decode(encode(msg))
    assert isinstance(out, BatchBackendMsg)
    assert out.data[0].uid == 3 and out.data[0].input_ids.tolist() == [1, 2, 3]
    assert out.data[0].sampling_params.top_k == 5
