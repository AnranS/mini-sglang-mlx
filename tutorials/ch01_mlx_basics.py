"""
第 1 章实验：MLX 的四个关键特性 —— 统一内存、惰性求值、异步求值、buffer 捐赠。

    python tutorials/ch01_mlx_basics.py
"""

import time

import mlx.core as mx
import numpy as np


def section(title):
    print(f"\n{'=' * 10} {title} {'=' * 10}")


# ---------------------------------------------------------------------------
section("1. 惰性求值：写下表达式时什么都没算")
a = mx.random.normal((4096, 4096))
mx.eval(a, a @ a)  # 先把 a 算出来，并预热矩阵乘法 kernel，避免把这些时间算进下面的测量
t = time.perf_counter()
b = a @ a + 1  # 只是在构建计算图
print(f"构图耗时   {(time.perf_counter() - t) * 1e3:7.3f} ms   (几乎为 0)")
t = time.perf_counter()
mx.eval(b)  # 真正提交给 GPU 并等待完成
print(f"eval 耗时  {(time.perf_counter() - t) * 1e3:7.3f} ms   (矩阵乘法真正在这里发生)")

# ---------------------------------------------------------------------------
section("2. 统一内存：CPU 和 GPU 共用同一块物理内存")
x = np.arange(8, dtype=np.int32)
y = mx.array(x)  # numpy -> mx：一次 memcpy，没有 PCIe 传输
z = (y * 2).tolist()  # mx -> python：GPU 算完后 CPU 直接读同一块内存
print("numpy -> mx -> GPU 计算 -> CPU 读回:", z)
print("=> 原版 mini-sglang 需要 pin_memory + non_blocking 拷贝的地方，这里都变得非常便宜")

# ---------------------------------------------------------------------------
section("3. async_eval：提交后立即返回，CPU 可以去做别的事")


def gpu_work():
    return (a @ a).sum()


def cpu_work():  # 模拟调度器的 CPU 开销
    s = 0
    for i in range(300_000):
        s += i
    return s


mx.eval(gpu_work())  # warmup
t = time.perf_counter()
for _ in range(10):
    mx.eval(gpu_work())
    cpu_work()
serial = time.perf_counter() - t

t = time.perf_counter()
for _ in range(10):
    out = gpu_work()
    mx.async_eval(out)  # 提交，不等待
    cpu_work()  # GPU 算的同时 CPU 干活
    mx.eval(out)  # 需要结果时再等
overlap = time.perf_counter() - t
print(f"串行: {serial * 1e3:.1f} ms,  重叠: {overlap * 1e3:.1f} ms  —— 这就是 overlap 调度的基本原理")

# ---------------------------------------------------------------------------
section("4. 原地更新与 buffer 捐赠 (donation)")
pool = mx.zeros((1_000_000, 8, 128), dtype=mx.bfloat16)  # 2 GB，模拟 KV pool
mx.eval(pool)
val = mx.ones((64, 8, 128), dtype=mx.bfloat16)
idx = mx.arange(64)
for i in range(3):  # 预热：第一次调用会编译 scatter kernel
    pool[idx] = val
    mx.eval(pool)

t = time.perf_counter()
for i in range(50):
    pool[idx + i * 64] = val  # scatter：旧 buffer 没有其他引用 -> 直接复用，真正的原地写
    mx.eval(pool)
print(f"可以捐赠时，每次 scatter: {(time.perf_counter() - t) / 50 * 1e3:.3f} ms")

t = time.perf_counter()
for i in range(50):
    # 注意不能写 keep = pool：那只是同一个 Python 对象的别名，scatter 后它也会指向新版本。
    # pool[0] 是一个惰性的"读"，它在计算图里引用着 scatter 之前的旧版本 pool。
    keep = pool[0]
    pool[idx + i * 64] = val  # 旧 buffer 还被 keep 引用 -> 不能捐赠 -> 只能把 2GB 整个拷贝一份
    mx.eval(pool)
    del keep
print(f"不能捐赠时，每次 scatter: {(time.perf_counter() - t) / 50 * 1e3:.3f} ms   <- 整块拷贝！")
print("=> 第 7 章会看到：overlap 调度如果写得不小心，就会不知不觉触发这种拷贝")
