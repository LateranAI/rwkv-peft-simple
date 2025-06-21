from src.configs.model import model_config
from src.configs.train import train_config
import torch

# default placeholders
def RUN_CUDA_RWKV7g(*args, **kwargs):
    raise NotImplementedError('FLA backend not available')

def RUN_RWKV7_STATE(*args, **kwargs):
    raise NotImplementedError('FLA backend not available')

def RUN_RWKV7_INFCTX(*args, **kwargs):
    raise NotImplementedError('FLA backend not available')


# FLA backend implementations
if 'x070' in train_config.my_testing:
    from rwkvfla.ops.rwkv7 import chunk_rwkv7

    if train_config.train_type == 'infctx':
        def RUN_RWKV7_INFCTX(r, k, v, w, a, b, s, HEAD_SIZE=64):
            B, T, HC = w.shape
            C = HEAD_SIZE
            H = HC // C
            r, w, k, v, a, b = [i.view(B, T, H, C) for i in [r, w, k, v, a, b]]
            o, state = chunk_rwkv7(r=r, w=w, k=k, v=v, a=a, b=b, scale=1.0,
                                   initial_state=s, output_final_state=True,
                                   head_first=False)
            return o, state
    elif train_config.train_type == 'state':
        def RUN_RWKV7_STATE(r, k, v, w, a, b, s, HEAD_SIZE=64):
            B, T, HC = w.shape
            C = HEAD_SIZE
            H = HC // C
            s = s.transpose(1, 2).expand(B, *s.shape)
            r, w, k, v, a, b = [i.view(B, T, H, C) for i in [r, w, k, v, a, b]]
            o, state = chunk_rwkv7(r=r, w=w, k=k, v=v, a=a, b=b, scale=1.0,
                                   initial_state=s, output_final_state=True,
                                   head_first=False)
            return o, state
    else:
        def RUN_CUDA_RWKV7g(r, w, k, v, a, b, HEAD_SIZE=64):
            B, T, HC = w.shape
            C = HEAD_SIZE
            H = HC // C
            r, w, k, v, a, b = [i.view(B, T, H, C) for i in [r, w, k, v, a, b]]
            o, _ = chunk_rwkv7(r=r, w=w, k=k, v=v, a=a, b=b, scale=1.0,
                               initial_state=None, output_final_state=False,
                               head_first=False)
            return o
