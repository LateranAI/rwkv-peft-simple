"""Operator router for RWKV7 backends."""

from src.configs.configs import model_config

if model_config.op == 'fla':
    from .fla import RUN_CUDA_RWKV7g, RUN_RWKV7_STATE, RUN_RWKV7_INFCTX
elif model_config.op == 'triton':
    from .triton import RUN_CUDA_RWKV7g, RUN_RWKV7_STATE, RUN_RWKV7_INFCTX
else:
    from .cuda import RUN_CUDA_RWKV7g, RUN_RWKV7_STATE, RUN_RWKV7_INFCTX
