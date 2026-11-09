import os

import pytest

TEST_MODEL = os.getenv("MINISGL_TEST_MODEL", "mlx-community/Qwen3-0.6B-bf16")


def _local_model_path():
    """只使用本地已缓存的模型，测试不触发下载。"""
    if os.path.isdir(TEST_MODEL):
        return TEST_MODEL
    try:
        from huggingface_hub import snapshot_download

        return snapshot_download(TEST_MODEL, local_files_only=True)
    except Exception:
        return None


@pytest.fixture(scope="session")
def model_path():
    path = _local_model_path()
    if path is None:
        pytest.skip(f"model {TEST_MODEL} not cached locally (set MINISGL_TEST_MODEL or download it first)")
    return path


@pytest.fixture(autouse=True)
def reset_global_ctx():
    import minisgl_mlx.core as core

    old = core._GLOBAL_CTX
    yield
    core._GLOBAL_CTX = old
