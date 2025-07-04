"""
文件名: model.py
所属路径: src/model/rwkv7

功能概述:
    汇总定义 RWKV-7 顶层语言模型 `RWKV7`。该模块实例化 Embedding、多个 Block 以及输出头 (Head)。

    forward 接口同时支持:
        • 常规训练 / 推理 (`forward_normal`)
        • 无限上下文增量推理 (`forward_infctx`)

    其中无限上下文模式会维护每个 Block 的 `BlockState` (shift & wkv) 以在长文本推理中持续累积信息。

依赖说明:
    - deepspeed: 用于显存优化的 Checkpointing。
    - src.model.state.BlockStateList: 管理跨层状态张量。
    - src.configs.train.train_config: 判断当前 train_type 以决定 forward 路由。
"""

########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################
from torch.utils.checkpoint import checkpoint as torch_checkpoint
import torch
import torch.nn as nn
import deepspeed
from src.model.state import BlockStateList
from .block import Block
from src.configs.train import train_config

class RWKV7(nn.Module):
    """RWKV 第七代语言模型顶层封装

    构造参数:
        args: Namespace ‑ 模型超参数, 包含 vocab_size, n_embd, n_layer, ctx_len 等。

    前向接口:
        forward_normal (默认):
            输入: idx (LongTensor[B, T])
            输出: logits (FloatTensor[B, T, vocab_size])

        forward_infctx (增量):
            输入: idx, last_shift_states, last_wkv_states
            输出: (logits, new_shift_states, new_wkv_states)
    """
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.emb = nn.Embedding(args.vocab_size, args.n_embd)

        self.blocks = nn.ModuleList([Block(args, i) for i in range(args.n_layer)])

        self.ln_out = nn.LayerNorm(args.n_embd)
        self.head = nn.Linear(args.n_embd, args.vocab_size, bias=False)

    @property
    def _use_infctx(self):
        """判断是否使用无限上下文模式"""
        return train_config.train_type == 'infctx'

    def forward(self, *args, **kwargs):
        if self._use_infctx:
            return self.forward_infctx(*args, **kwargs)
        return self.forward_normal(*args, **kwargs)

    def forward_normal(self, idx, attention_mask = None):
        """普通训练 / 推理前向传播。

        参数:
            idx (torch.LongTensor): [B, T] — 词 ID 序列, T ≤ args.ctx_len。
            attention_mask (torch.FloatTensor | None): [B, ≤T] — 可选 0/1 掩码。

        返回:
            logits (torch.FloatTensor): [B, T, vocab_size]
        """
        args = self.args

        B, T = idx.size()
        assert T <= args.ctx_len, "Cannot forward, model ctx_len is exhausted."

        x = self.emb(idx)
        v_first = torch.empty_like(x)

        for block in self.blocks:
            if args.grad_cp == 1:
                if args.train_type == 'state' or args.peft !='none':
                    x, v_first = torch_checkpoint(block, x, v_first , attention_mask, use_reentrant=False)
                else:
                    x, v_first = deepspeed.checkpointing.checkpoint(block, x, v_first, attention_mask)
            else:
                x, v_first = block(x, v_first, attention_mask)

        x = self.ln_out(x)
        x = self.head(x)

        return x

    def forward_infctx(self, idx,  last_shift_states: torch.Tensor,
            last_wkv_states: torch.Tensor, attention_mask = None):
        """无限上下文 (infctx) 前向传播。

        参数:
            idx (torch.LongTensor): [B, T] 当前输入 Chunk。
            last_shift_states (torch.Tensor): [N, 2, B, C] 前一次 Block 移位状态。
            last_wkv_states  (torch.Tensor): [N, B, H, C//H, C//H] 前一次 Block WKV 状态。
            attention_mask (torch.FloatTensor | None): [B, ≤T] 掩码。

        返回:
            logits (torch.FloatTensor): [B, T, vocab_size]
            new_shift_states (torch.Tensor): 与 last_shift_states 同形状 — 更新后状态。
            new_wkv_states  (torch.Tensor): 与 last_wkv_states 同形状 — 更新后状态。
        """
        args = self.args
        B, T = idx.size()
        assert T <= args.chunk_ctx, "Cannot forward, model ctx_len is exhausted."
        C = args.n_embd
        H =  args.dim_att // args.head_size_a
        assert C==H*args.head_size_a
        
        x = self.emb(idx)
        new_states = BlockStateList.empty(args.n_layer, B, args.n_embd, H,
                                        x.device, x.dtype)

        v_first = torch.empty_like(x)
        
        for i, (block, block_state) in enumerate(zip(self.blocks,
            BlockStateList(last_shift_states, last_wkv_states))):
            if args.grad_cp == 1 and i > 0:# and i < len(self.blocks)-1 :
                x, v_first, new_block_state = torch_checkpoint(block, x, v_first, block_state, attention_mask, use_reentrant=False)

            else:
                x, v_first, new_block_state = block(x,v_first,block_state, attention_mask)
    
            new_states[i] = new_block_state 

        x = self.ln_out(x)
        x = self.head(x)

        return x, new_states.shift_states, new_states.wkv_states
    
    # def forward(self, idx):
    #     args = self.args
    #     B, T = idx.size()
    #     assert T <= args.ctx_len, "Cannot forward, model ctx_len is exhausted."

    #     x = self.emb(idx)
    #     v_first = torch.empty_like(x)

    #     for block in self.blocks:
    #         if args.grad_cp == 1:
    #             if args.train_type == 'state' or args.peft !='none':
    #                 x, v_first = torch_checkpoint(block, x, v_first ,use_reentrant=False)
    #             else:
    #                 x, v_first = deepspeed.checkpointing.checkpoint(block, x, v_first)
    #         else:
    #             x, v_first = block(x, v_first)

    #     x = self.ln_out(x)
    #     x = self.head(x)

    #     return x

