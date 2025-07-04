"""
模块: src.model.operator.rwkvop

角色: 根据 `model_config.op` 动态路由 RWKV7 张量算子到不同后端实现 (fla / triton / cuda)。
该文件不实现任何算法逻辑，仅负责根据配置导入正确的 `RUN_CUDA_RWKV7g`, `RUN_RWKV7_STATE`, `RUN_RWKV7_INFCTX` 三元组，供上层模型调用。
"""

from src.configs.model import model_config

if model_config.op == 'fla':
    from .fla import RUN_CUDA_RWKV7g, RUN_RWKV7_STATE, RUN_RWKV7_INFCTX
elif model_config.op == 'triton':
    from .triton import RUN_CUDA_RWKV7g, RUN_RWKV7_STATE, RUN_RWKV7_INFCTX
else:
    from .cuda import RUN_CUDA_RWKV7g, RUN_RWKV7_STATE, RUN_RWKV7_INFCTX
