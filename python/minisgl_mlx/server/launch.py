"""
启动整个服务，对应原版 minisgl/server/launch.py。一共三个进程：

    主进程        API server（FastAPI + uvicorn）或交互式 shell
    子进程 1      tokenizer / detokenizer worker
    子进程 2      scheduler（持有 MLX 模型和 KV cache，所有 GPU 计算都在这里）

原版每张 GPU 一个 scheduler 进程（TP rank）；Apple Silicon 只有一个 GPU，所以只有一个。
"""

from __future__ import annotations

import asyncio
import multiprocessing as mp
import sys

from minisgl_mlx.utils import init_logger

from .args import ServerArgs, parse_args


def _run_scheduler(args: ServerArgs, ack_queue: mp.Queue) -> None:
    from minisgl_mlx.scheduler import Scheduler

    scheduler = Scheduler(args)
    ack_queue.put("Scheduler is ready")
    try:
        scheduler.run_forever()
    except KeyboardInterrupt:
        init_logger(__name__).info("Scheduler exiting...")
        scheduler.shutdown()


def _run_tokenizer(args: ServerArgs, ack_queue: mp.Queue) -> None:
    from minisgl_mlx.tokenizer import tokenize_worker

    tokenize_worker(
        model_path=args.model_path,
        listen_addr=args.zmq_detokenizer_addr,
        backend_addr=args.zmq_backend_addr,
        frontend_addr=args.zmq_frontend_addr,
        ack_queue=ack_queue,
    )


def launch_server(run_shell: bool = False) -> None:
    from .api_server import app, create_frontend

    args, run_shell = parse_args(sys.argv[1:], run_shell)
    logger = init_logger(__name__, "launcher")

    # 先在主进程把模型下载好（命中缓存则什么也不做），避免两个子进程同时下载
    from minisgl_mlx.utils import resolve_model_path

    resolve_model_path(args.model_path)

    state = create_frontend(args)
    ctx = mp.get_context("spawn")
    ack_queue: mp.Queue = ctx.Queue()
    procs = [
        ctx.Process(target=_run_scheduler, args=(args, ack_queue), name="minisgl-scheduler", daemon=True),
        ctx.Process(target=_run_tokenizer, args=(args, ack_queue), name="minisgl-tokenizer", daemon=True),
    ]
    for p in procs:
        p.start()
    for _ in procs:
        logger.info(ack_queue.get())

    try:
        if run_shell:
            from .shell import run_shell as shell_main

            asyncio.run(shell_main(state))
        else:
            import uvicorn

            logger.info(f"API server is ready: http://{args.server_host}:{args.server_port}")
            uvicorn.run(app, host=args.server_host, port=args.server_port, log_level="warning")
    finally:
        state.shutdown()
        for p in procs:
            p.terminate()
        for p in procs:
            p.join(timeout=5)


if __name__ == "__main__":
    launch_server()
