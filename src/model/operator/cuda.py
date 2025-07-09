"""
模块: src.model.operator.cuda

角色: 当选择 CUDA 后端 (model_config.op == 'cuda') 时提供 RWKV7 算子在 PyTorch 中的 Python 包装。
该文件的主要职责是通过 `torch.utils.cpp_extension.load` 动态编译并加载位于 src/model/cuda 目录的自定义 CUDA/C++ 扩展 (wind_backstepping)，并向外暴露 Python 侧的 Autograd Function `WindBackstepping` 以及便捷调用封装 `RUN_CUDA_RWKV7g / RUN_RWKV7_STATE / RUN_RWKV7_INFCTX`。

依赖:
1. src.configs.train.train_config – 决定是否启用 x070 测试模式及训练类型。
2. torch 与 torch.utils.cpp_extension – 编译/加载自定义 CUDA 扩展并实现 Autograd Function。
3. src/model/cuda/wkv7_cuda.cu & wkv7_op.cpp – 实际 CUDA 实现。

公共接口:
- RUN_CUDA_RWKV7g(q, w, k, v, a, b)
- RUN_RWKV7_STATE(...)
- RUN_RWKV7_INFCTX(...)

该模块仅在 `train_config.my_testing` 包含 "x070" 时才会真正编译并启用 CUDA 核函数，否则抛出 NotImplementedError。
"""
# rom src.configs.model import model_config
from src.configs.train import train_config
import torch
from torch.utils.cpp_extension import load

def RUN_CUDA_RWKV7g(*args, **kwargs):
    """占位函数: 当 CUDA Kernel 未编译时调用。
    抛出 NotImplementedError，以提示用户所选 op 后端不可用。"""
    raise NotImplementedError('CUDA backend not available')

def RUN_RWKV7_STATE(*args, **kwargs):
    """同上，占位函数。"""
    raise NotImplementedError('CUDA backend not available')

def RUN_RWKV7_INFCTX(*args, **kwargs):
    """同上，占位函数。"""
    raise NotImplementedError('CUDA backend not available')

# HEAD_SIZE = model_config.head_size_a
if 'x070' in train_config.my_testing:
    HEAD_SIZE = 64
    CHUNK_LEN = 512

    flags = ['-res-usage', f'-D_C_={HEAD_SIZE}', f"-D_CHUNK_LEN_={CHUNK_LEN}", "--use_fast_math", "-O3", "-Xptxas -O3", "--extra-device-vectorization"]
    load(name="wind_backstepping", sources=[f'src/model/cuda/wkv7_cuda.cu', 'src/model/cuda/wkv7_op.cpp'], is_python_module=False, verbose=True, extra_cuda_cflags=flags)

    class WindBackstepping(torch.autograd.Function):
        """基于自定义 CUDA 内核 wind_backstepping 的 RWKV7 注意力前向/反向实现。

        forward 输入:
            w, q, k, v, z, b : 形状 (B, T, H, C) 的 bfloat16 张量
        返回:
            y : 与 v 同形状的输出张量

        backward 会自动从 ctx.saved_tensors 取回前向所需的中间结果并调用底层 C++/CUDA 实现。
        """
        @staticmethod
        def forward(ctx, w,q,k,v,z,b):
            B,T,H,C = w.shape
            assert T%CHUNK_LEN == 0
            assert all(i.dtype==torch.bfloat16 for i in [w,q,k,v,z,b])
            assert all(i.is_contiguous() for i in [w,q,k,v,z,b])
            y = torch.empty_like(v)
            s = torch.empty(B,H,T//CHUNK_LEN,C,C, dtype=torch.float32,device=w.device)
            sa = torch.empty(B,T,H,C, dtype=torch.float32,device=w.device)
            torch.ops.wind_backstepping.forward(w,q,k,v,z,b, y,s,sa)
            ctx.save_for_backward(w,q,k,v,z,b,s,sa)
            return y
        @staticmethod
        def backward(ctx, dy):
            assert all(i.dtype==torch.bfloat16 for i in [dy])
            assert all(i.is_contiguous() for i in [dy])
            w,q,k,v,z,b,s,sa = ctx.saved_tensors
            dw,dq,dk,dv,dz,db = [torch.empty_like(x) for x in [w,q,k,v,z,b]]
            torch.ops.wind_backstepping.backward(w,q,k,v,z,b, dy,s,sa, dw,dq,dk,dv,dz,db)
            return dw,dq,dk,dv,dz,db

    def RUN_CUDA_RWKV7g(q,w,k,v,a,b):
        B,T,HC = q.shape
        q,w,k,v,a,b = [i.view(B,T,HC//64,64) for i in [q,w,k,v,a,b]]
        return WindBackstepping.apply(w,q,k,v,a,b).view(B,T,HC)

