import numpy as np

from minisgl_mlx.profiling import PhaseTimer, gpu_timeline_stats


class _FakeBatch:
    def __init__(self, phase, size):
        import mlx.core as mx

        self.phase, self.size = phase, size
        self.positions = mx.zeros((size,))


class _FakeInput:
    def __init__(self, phase, size):
        self.batch = _FakeBatch(phase, size)


class _FakeScheduler:
    """只实现 PhaseTimer 需要挂钩的那些方法。"""

    def __init__(self):
        self.receive_msg = lambda blocking=False: []

    def _schedule_next_batch(self):
        return _FakeInput("decode", 4)

    def _forward(self, x):
        return x

    def _submit(self, data):
        pass

    def _wait(self, data):
        pass

    def _process_last_data(self, data):
        pass

    def overlap_loop(self, last):
        self.receive_msg(blocking=False)
        fi = self._schedule_next_batch()
        data = (fi, self._forward(fi))
        self._submit(data)
        if last is not None:
            self._wait(last)
        self._process_last_data(last)
        return data

    def normal_loop(self):
        pass


def test_phase_timer_records_each_iteration():
    sched = _FakeScheduler()
    timer = PhaseTimer().attach(sched)
    data = None
    for _ in range(5):
        data = sched.overlap_loop(data)
    recs = timer.select("decode")
    assert len(recs) == 5 and all(r.batch_size == 4 for r in recs)
    assert set(recs[-1].times) == {"recv", "schedule", "build", "submit", "wait", "process"}
    assert "wait" not in recs[0].times  # 第一轮没有上一批可等
    assert set(timer.median_ms("decode")) == set(PhaseTimer.PHASES) | {"total"}


def test_gpu_timeline_stats_merges_and_splits():
    ms = 1_000_000  # ns
    # 一段"预热"，然后 > 300ms 的空闲，然后正式测量：每 4ms 一步，其中 GPU 忙 3ms（有两个重叠的区间）
    warmup = [(0, 5 * ms)]
    steps = [(1000 * ms + i * 4 * ms, 1000 * ms + i * 4 * ms + 3 * ms) for i in range(10)]
    overlap = [(1000 * ms + ms, 1000 * ms + 2 * ms)]
    st = gpu_timeline_stats(warmup + steps + overlap)
    assert st.num_intervals == 10  # 预热被切掉，重叠区间被合并
    assert np.isclose(st.window_ms, 39.0) and np.isclose(st.busy_ms, 30.0)
    assert np.allclose(st.gaps_us, 1000.0)
