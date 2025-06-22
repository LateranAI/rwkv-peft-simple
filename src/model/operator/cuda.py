from src.configs.model import model_config
from src.configs.train import train_config
import torch
from torch.utils.cpp_extension import load
import functools
import threading
from typing import Callable

def RUN_CUDA_RWKV7g(*args, **kwargs):
    raise NotImplementedError('CUDA backend not available')

def RUN_RWKV7_STATE(*args, **kwargs):
    raise NotImplementedError('CUDA backend not available')

def RUN_RWKV7_INFCTX(*args, **kwargs):
    raise NotImplementedError('CUDA backend not available')

# ------------------------- 延迟初始化（装饰器写法） -------------------------

# 编译成功后保存真正的 CUDA kernel 调用实现
_run_impl: Callable | None = None

# 线程锁，防止多线程环境下重复编译
_compile_lock = threading.Lock()


def _initialize_backend():
    """在配置加载完毕后编译并加载 CUDA kernel。"""
    global _run_impl
    if _run_impl is not None:
        return  # 已初始化，无需重复执行

    with _compile_lock:  # 双检 + 线程安全
        if _run_impl is not None:
            return

    HEAD_SIZE = model_config.head_size_a
    if HEAD_SIZE <= 0:
        raise RuntimeError(
            "model_config.head_size_a 仍为0, 请在调用 CUDA 算子前确保已经正确加载模型配置。",
        )

    # 仅在 my_testing 包含 'x070' 时编译对应 kernel，与旧实现保持一致
    if 'x070' not in train_config.my_testing:
        raise NotImplementedError(
            f"当前 my_testing={train_config.my_testing} 未包含 'x070', 暂未实现对应 CUDA backend.",
        )

    CHUNK_LEN = 16
    flags = [
        '-res-usage',
        f'-D_C_={HEAD_SIZE}',
        f"-D_CHUNK_LEN_={CHUNK_LEN}",
        "--use_fast_math",
        "-O3",
        "-Xptxas -O3",
        "--extra-device-vectorization",
    ]

    # 编译并加载 CUDA 扩展
    load(
        name="wind_backstepping",
        sources=['src/model/cuda/wkv7_cuda.cu', 'src/model/cuda/wkv7_op.cpp'],
        is_python_module=False,
        verbose=True,
        extra_cuda_cflags=flags,
    )

    class _WindBackstepping(torch.autograd.Function):
        @staticmethod
        def forward(ctx, w, q, k, v, z, b):
            B, T, H, C = w.shape
            assert T % CHUNK_LEN == 0
            assert all(i.dtype == torch.bfloat16 for i in [w, q, k, v, z, b])
            assert all(i.is_contiguous() for i in [w, q, k, v, z, b])
            y = torch.empty_like(v)
            s = torch.empty(B, H, T // CHUNK_LEN, C, C, dtype=torch.float32, device=w.device)
            sa = torch.empty(B, T, H, C, dtype=torch.float32, device=w.device)
            torch.ops.wind_backstepping.forward(w, q, k, v, z, b, y, s, sa)
            ctx.save_for_backward(w, q, k, v, z, b, s, sa)
            return y

        @staticmethod
        def backward(ctx, dy):
            assert dy.dtype == torch.bfloat16 and dy.is_contiguous()
            w, q, k, v, z, b, s, sa = ctx.saved_tensors
            dw, dq, dk, dv, dz, db = [torch.empty_like(x) for x in [w, q, k, v, z, b]]
            torch.ops.wind_backstepping.backward(
                w, q, k, v, z, b, dy, s, sa, dw, dq, dk, dv, dz, db,
            )
            return dw, dq, dk, dv, dz, db

    def _run_impl_local(q, w, k, v, a, b):
        B, T, HC = q.shape
        q, w, k, v, a, b = [i.view(B, T, HC // 64, 64) for i in [q, w, k, v, a, b]]
        return _WindBackstepping.apply(w, q, k, v, a, b).view(B, T, HC)

    _run_impl = _run_impl_local

# 装饰器：保证在第一次调用前完成编译


def _ensure_compiled(func):
    """装饰器：在首次调用时完成 CUDA 编译。"""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if _run_impl is None:
            _initialize_backend()
        return func(*args, **kwargs)

    return wrapper


# 接口函数：签名保持不变，内部转调 _run_impl


@_ensure_compiled
def RUN_CUDA_RWKV7g(q, w, k, v, a, b):  # type: ignore
    if _run_impl is None:  # 理论上不会发生，保险检查
        raise RuntimeError("CUDA backend 尚未初始化完成")
    return _run_impl(q, w, k, v, a, b)

# ------------------------------------------------------------------------
