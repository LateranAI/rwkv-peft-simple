"""
文件名: ffn.py
所属路径: src/model/rwkv7

功能概述:
    提供 RWKV-7 模型的 ChannelMix 前馈网络 (Feed Forward) 实现, 与 TimeMix 形成互补。
    该文件定义三种 ChannelMix 变体 (普通 / FusedKernel / Infinite-Context),
    并通过工厂方法 `RWKV_Cmix_v7` 按需路由。

关键职责:
    • 在 `RWKV_CMix_x070` 中使用零填充移位构造相邻差分特征, 结合可训练比例参数 `x_k`。
    • 调用 `make_linear_ffn` 构建 LoRA 兼容的键值线性层。
    • (Fused) 版本使用 `rwkvfla.ops.channel_mixing_rwkv7` CUDA Kernel 提升吞吐。

输入 / 输出约定:
    forward 接受 (B, T, C) 张量并返回同形状张量; infctx 变体同时返回 ChannelMixState。
"""

import torch.nn as nn
from src.model.state import *
from src.model.peft.linear import make_linear_ffn
from src.configs.train import train_config
from src.configs.model import model_config

if model_config.fused_kernel:
    from rwkvfla.ops.rwkv7 import channel_mixing_rwkv7
else:
    channel_mixing_rwkv7 = None

def RWKV_Cmix_v7(*args, **kwargs):
    """工厂函数

    根据 `train_config.train_type` 和 `model_config.fused_kernel` 选择 ChannelMix 变体。
    """
    
    if train_config.train_type == 'infctx':
        return RWKV_CMix_x070_infctx(*args, **kwargs)
    elif model_config.fused_kernel:
        return RWKV_CMix_x070_fla(*args, **kwargs)
    else:
        return RWKV_CMix_x070(*args, **kwargs)

class RWKV_CMix_x070(nn.Module):
    """基础 ChannelMix 实现 (x070 版本)

    关键逻辑:
        1. 使用 `time_shift` 生成差分特征 xx, 与当前输入 x 结合产生 k。
        2. 通过 ReLU² 激活后映射到更高维度后再投影回隐层维度, 起到前馈作用。
    """
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

        with torch.no_grad():
            ratio_1_to_almost0 = 1.0 - (layer_id / args.n_layer)  # 1 to ~0
            ddd = torch.ones(1, 1, args.n_embd)
            for i in range(args.n_embd):
                ddd[0, 0, i] = i / args.n_embd
            self.x_k = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0**4))

        self.key = make_linear_ffn(args.n_embd, args.n_embd * 4, bias=False)
        self.value = make_linear_ffn(args.n_embd * 4, args.n_embd, bias=False)

        # !!! initialize if you are using RWKV_Tmix_x070 in your code !!!
        # self.key.weight.data.uniform_(-0.5/(args.n_embd**0.5), 0.5/(args.n_embd**0.5))
        # self.value.weight.data.zero_()

    def forward(self, x, attention_mask=None):
        """ChannelMix 前向。

        参数:
            x (torch.Tensor): [B, T, C]
            attention_mask (torch.FloatTensor | None): [B, ≤T]

        返回:
            out (torch.Tensor): [B, T, C]
        """
        if attention_mask is not None:
            x = x.mul(attention_mask[:, -x.shape[-2]:, None])
        xx = self.time_shift(x) - x
        
        k = x + xx * self.x_k
        k = torch.relu(self.key(k)) ** 2

        return self.value(k)
    
class RWKV_CMix_x070_fla(RWKV_CMix_x070):
    """ChannelMix FusedKernel 变体
    使用 `channel_mixing_rwkv7` CUDA Kernel 取代 Python 循环以获得更高效率。"""
    def __init__(self, args, layer_id):
        super().__init__(args, layer_id)
        del self.time_shift

    @torch.compile
    def forward(self, x, attention_mask=None):
        """Fused Kernel ChannelMix 前向 (同上形状, 计算加速)。"""
        if attention_mask is not None:
            x = x.mul(attention_mask[:, -x.shape[-2]:, None])
        x_prev = torch.zeros(x.size(0), x.size(2), device=x.device, dtype=x.dtype)
        output, _ = channel_mixing_rwkv7(x, x_prev, self.x_k, self.key.weight.t(), self.value.weight.t())
        return output

class RWKV_CMix_x070_infctx(RWKV_CMix_x070):
    """ChannelMix 无限上下文 (Infinite-Context) 变体
    在前向传播中维护并返回 `ChannelMixState` 以实现跨 Chunk 累积。"""
    def __init__(self, args, layer_id):
        super().__init__(args, layer_id)
        del self.time_shift

    def forward(self, x, last_state: ChannelMixState, attention_mask=None):
        """无限上下文 ChannelMix 前向。

        参数:
            x (torch.Tensor): [B, T, C]
            last_state (ChannelMixState): 含上一 token 的 shift_state (B, C)。
            attention_mask (torch.FloatTensor | None): [B, ≤T]

        返回:
            out (torch.Tensor): [B, T, C]
            new_state (ChannelMixState): 更新后的 shift_state
        """
        if attention_mask is not None:
            x = x.mul(attention_mask[:, -x.shape[-2]:, None])
        xx = torch.concat((last_state.shift_state.unsqueeze(1), x[:, :-1]), dim=1) - x  
        
        k = x + xx * self.x_k
        k = torch.relu(self.key(k)) ** 2

        return self.value(k),ChannelMixState(x[:, -1])