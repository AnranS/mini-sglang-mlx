"""attention 后端正确性：与"逐请求取出连续 KV 再算 attention"的参考实现对比，不需要模型权重。"""

import mlx.core as mx
import numpy as np
import pytest

from minisgl_mlx.attention import paged_decode_attention


def reference_decode(q, k_pool, v_pool, indices_per_req, scale):
    outs = []
    for b, idx in enumerate(indices_per_req):
        k = k_pool[mx.array(idx)].transpose(1, 0, 2)[None]
        v = v_pool[mx.array(idx)].transpose(1, 0, 2)[None]
        outs.append(mx.fast.scaled_dot_product_attention(q[b : b + 1, :, None], k, v, scale=scale)[:, :, 0])
    return mx.concatenate(outs)


@pytest.mark.parametrize("hq,hkv,d", [(16, 8, 128), (32, 8, 128), (8, 8, 64), (16, 2, 256)])
@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16])
def test_paged_decode_kernel(hq, hkv, d, dtype):
    rng = np.random.default_rng(0)
    N, B = 4096, 7
    k_pool = mx.random.normal((N, hkv, d)).astype(dtype)
    v_pool = mx.random.normal((N, hkv, d)).astype(dtype)
    q = mx.random.normal((B, hq, d)).astype(dtype)
    lens = [1, 2, 17, 31, 32, 33, 700]  # 包含 "比 simdgroup 数还少" 的边界情况
    perm = rng.permutation(N).astype(np.int32)
    per_req, offset = [], 0
    for n in lens:  # 每个请求的 KV 散落在随机的、互不重叠的槽位上
        per_req.append(perm[offset : offset + n])
        offset += n
    indptr = mx.array(np.concatenate([[0], np.cumsum(lens)]).astype(np.int32))
    indices = mx.array(np.concatenate(per_req))
    scale = d**-0.5
    out = paged_decode_attention(q, k_pool, v_pool, indptr, indices, mx.array([scale], dtype=mx.float32))
    ref = reference_decode(q, k_pool, v_pool, per_req, scale)
    tol = 1e-4 if dtype == mx.float32 else 2e-2
    assert mx.allclose(out.astype(mx.float32), ref.astype(mx.float32), atol=tol, rtol=tol).item()
