"""
文件名: block.py
所属路径: src/model/rwkv7

功能概述:
    定义 RWKV-7 中的语言模型 Block 结构, 由 TimeMix (注意力) 与 ChannelMix (前馈) 两部分组成。
    Block 对外暴露统一前向接口, 并根据是否处于增量推理 (infctx) 状态路由至不同实现。

关键职责:
    • 调用子模块 RWKV_Tmix_v7 / RWKV_Cmix_v7 完成注意力与前馈运算。
    • 在首层添加额外的 LayerNorm (`ln0`) 以对输入 Embedding 进行归一化。
    • 在 infctx 模式下处理并返回 BlockState 以保存跨 Chunk 的历史状态。
"""

import torch.nn as nn
from .ffn import RWKV_Cmix_v7
from .att import RWKV_Tmix_v7
from src.model.state import BlockState
from src.configs.train import train_config

class Block(nn.Module):
    """RWKV 模型的基础计算块 (Block)

    参数:
        args: Namespace ‑ 全局模型/训练超参数
        layer_id: int  ‑ 当前块编号 (0-based)

    前向接口:
        forward_normal : 用于常规训练 / 推理
        forward_infctx  : 用于无限上下文推理, 需维护 BlockState
    """
    def __init__(self, args, layer_id):
        """初始化 RWKV Block

        功能说明：
            组装一个完整的 Block，包括两层 LayerNorm、TimeMix 与 ChannelMix 子模块，
            以及在第 0 层时额外添加的输入归一化层 `ln0`。

        输入参数：
            args (Namespace): 模型超参数集合。
            layer_id (int): 当前块编号。
        """
        super().__init__()
        self.args = args
        self.layer_id = layer_id

        self.ln1 = nn.LayerNorm(args.n_embd)
        self.ln2 = nn.LayerNorm(args.n_embd)

        if self.layer_id == 0:
            self.ln0 = nn.LayerNorm(args.n_embd)

        self.att = RWKV_Tmix_v7(args, layer_id)  
        self.ffn = RWKV_Cmix_v7(args, layer_id)


    # def forward(self, x, v_first):
    #     if self.layer_id == 0:
    #         x = self.ln0(x)

    #     x_attn, v_first = self.att(self.ln1(x), v_first)
    #     x = x + x_attn

    #     x = x + self.ffn(self.ln2(x))
    #     return x, v_first
    @property
    def _use_infctx(self):
        """判断当前是否处于无限上下文模式 (train_config.train_type == 'infctx')"""
        if train_config.train_type == 'infctx':
            return True
        else:
            return False

    def forward(self, *args, **kwargs):
        """路由前向传播

        当 `_use_infctx` 为 True 时，调用 `forward_infctx`，否则走 `forward_normal`。
        参数 / 返回值与对应具体实现一致。
        """
        if self._use_infctx:
            return self.forward_infctx(*args, **kwargs)
        return self.forward_normal(*args, **kwargs)

    def forward_normal(self, x, v_first, attention_mask = None):
        """常规前向。

        参数:
            x (torch.Tensor): [B, T, C] 当前隐表示。
            v_first (torch.Tensor): [B, T, C] 第一层保存的 v 向量。
            attention_mask (torch.FloatTensor | None): [B, ≤T] — 可选掩码。

        返回:
            x_out (torch.Tensor): [B, T, C] — 经过 Block 后的表示。
            v_first (torch.Tensor): [B, T, C] — 可能更新的 v_first。
        """
        if self.layer_id == 0:
            x = self.ln0(x)

        x_attn, v_first = self.att(self.ln1(x), v_first, attention_mask = attention_mask)
        x = x + x_attn

        x = x + self.ffn(self.ln2(x), attention_mask = attention_mask)
        return x, v_first

    def forward_infctx(self, x, v_first, last_state: BlockState, attention_mask = None):
        """无限上下文前向。

        参数:
            x (torch.Tensor): [B, T, C]
            v_first (torch.Tensor): [B, T, C]
            last_state (BlockState): 保存上一块 TimeMix / ChannelMix 状态。
            attention_mask (torch.FloatTensor | None): [B, ≤T]

        返回:
            x_out (torch.Tensor): [B, T, C]
            v_first (torch.Tensor): [B, T, C]
            new_state (BlockState): 更新后的状态
        """
        if self.layer_id == 0:
            x = self.ln0(x)

        x_attn, v_first, att_state = self.att(self.ln1(x), v_first, last_state.time_mix_state, attention_mask = attention_mask)
        x = x + x_attn

        ffn_out ,ffn_state = self.ffn(self.ln2(x), last_state.channel_mix_state, attention_mask = attention_mask)

        x = x + ffn_out
        return x, v_first, BlockState(att_state, ffn_state)