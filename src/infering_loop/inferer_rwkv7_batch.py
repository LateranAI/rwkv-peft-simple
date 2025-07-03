"""inferer_rwkv7_batch.py

基于 `src/model/rwkv7/model.py` 的批量推理封装。

主要目标：
1. 提供与训练同构的推理模型，天然支持 `tokens.shape == (B, T)` 的批量输入。
2. 在 `train_config.train_type == 'state'` 模式下，额外返回每层的 `att_kv` 状态，
   形状 `(B, H, N, N)`，以兼容数据集里"state-decoding"任务对隐藏状态的需求。

实现思路：
* 继承 & 改写 `RWKV_Tmix_x070_State`，使其 `forward` 同时返回 `state`；
* 重新封装 `BlockWithState`，将其 `att` 部分替换为改写后的类；
* 构造 `RWKV7_StateInfer` 模型，`forward` 输出 `(logits, states)`，其中
  `states` 是 `List[Tensor]`，长度 = `n_layer`，每个张量形状 `(B, H, N, N)`。

依赖：要求 `model_config.op` 已正确配置，使 `RUN_RWKV7_STATE` 可用。
"""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.rwkv7.model import RWKV7  # 训练用完整模型
from src.model.rwkv7.att import RWKV_Tmix_x070_State
from src.model.operator.rwkvop import RUN_RWKV7_STATE
from src.configs.train import train_config
from src.configs.model import model_config

__all__ = [
    "RWKV7_StateInfer",
]


# ---------------------------------------------------------------------------
# Step 1: 改写 Attention，使其返回 state
# ---------------------------------------------------------------------------

class _RWKV_Tmix_x070_StateReturn(RWKV_Tmix_x070_State):
    """在保持原逻辑的同时，把 time_state 返回出来。"""

    def forward(self, x: torch.Tensor, v_first: torch.Tensor, attention_mask: torch.Tensor | None = None):
        # 复制自 RWKV_Tmix_x070_State.forward，唯一区别：接收并返回 `state`
        B, T, C = x.size()
        H = self.n_head

        if attention_mask is not None:
            x = x.mul(attention_mask[:, -x.shape[-2]:, None])
        xx = self.time_shift(x) - x

        # element-wise mix
        xr, xw, xk, xv, xa, xg = self.addcmul_kernel(x, xx)

        # linear projections -------------------------------------------------
        r = self.receptance(xr)
        w = -F.softplus(-(self.w0 + torch.tanh(xw @ self.w1) @ self.w2)) - 0.5  # (-∞, -0.5)
        k = self.key(xk)
        v = self.value(xv)

        if self.layer_id == 0:
            v_first = v
        else:
            v = v + (v_first - v) * torch.sigmoid(self.v0 + (xv @ self.v1) @ self.v2)

        a = torch.sigmoid(self.a0 + (xa @ self.a1) @ self.a2)
        g = torch.sigmoid(xg @ self.g1) @ self.g2

        # scaled & normalized k ---------------------------------------------
        kk = k * self.k_k
        kk_norm = F.normalize(kk.view(B, T, H, -1), dim=-1, p=2.0).view(B, T, C)
        k = k * (1 + (a - 1) * self.k_a)

        # 核心 CUDA / Triton / FLA kernel ------------------------------------
        x_out, state = RUN_RWKV7_STATE(
            r, k, v, w, -kk_norm, kk_norm * a, self.time_state
        )  # x_out: (B,T,C) ; state: (B,H,N,N)

        # layer norm & output proj ----------------------------------------
        x_out = self.ln_x(x_out.view(B * T, C)).view(B, T, C)

        x_out = x_out + (
            (r.view(B, T, H, -1) * k.view(B, T, H, -1) * self.r_k)
            .sum(dim=-1, keepdim=True)
            * v.view(B, T, H, -1)
        ).view(B, T, C)
        x_out = self.output(x_out * g)
        return x_out, v_first, state  # <-- 关键：返回 state


# ---------------------------------------------------------------------------
# Step 2: 替换 Block 以输出 state
# ---------------------------------------------------------------------------

from src.model.rwkv7.block import Block as _OrigBlock


class _BlockWithState(_OrigBlock):
    """Block 改写版，使 forward 额外返回 `state`."""

    def __init__(self, args, layer_id: int):
        super().__init__(args, layer_id)
        # 强制使用返回版 attention（仅在 state 模式下生效）
        if train_config.train_type == "state":
            self.att = _RWKV_Tmix_x070_StateReturn(args, layer_id)

    def forward(self, x: torch.Tensor, v_first: torch.Tensor, attention_mask: torch.Tensor | None = None):
        # 仅实现 normal forward；infctx 模式不在此封装范围
        if self.layer_id == 0:
            x = self.ln0(x)

        x_attn, v_first, state = self.att(self.ln1(x), v_first, attention_mask)
        x = x + x_attn
        x = x + self.ffn(self.ln2(x), attention_mask)
        return x, v_first, state


# ---------------------------------------------------------------------------
# Step 3: 整体批量推理模型
# ---------------------------------------------------------------------------

class RWKV7_StateInfer(nn.Module):
    """可批量推理并返回 per-layer state 的封装模型。"""

    def __init__(self, args):
        """参数与 `RWKV7` 原构造函数保持一致。"""
        super().__init__()
        self.args = args

        # --- 复制 RWKV7 结构但替换 Block ----------------------------------
        self.emb = nn.Embedding(args.vocab_size, args.n_embd)
        self.blocks = nn.ModuleList([_BlockWithState(args, i) for i in range(args.n_layer)])
        self.ln_out = nn.LayerNorm(args.n_embd)
        self.head = nn.Linear(args.n_embd, args.vocab_size, bias=False)

    # ------------------------------------------------------------------
    def forward(
        self, idx: torch.Tensor, *, attention_mask: torch.Tensor | None = None, return_states: bool = False
    ) -> Tuple[torch.Tensor, List[torch.Tensor] | None]:
        """批量前向。

        Parameters
        ----------
        idx              : LongTensor, 形状 (B, T)
        attention_mask    : FloatTensor, 形状 (B, T)，可选
        return_states     : 若为 True，返回每层的 att_kv state。

        Returns
        -------
        logits  : (B, T, vocab) 张量。
        states   : None 或 List[Tensor]，长度 = n_layer，
                   每个元素形状 (B, H, N, N)。
        """
        B, T = idx.size()
        args = self.args
        assert T <= args.ctx_len, "Cannot forward, model ctx_len is exhausted."

        x = self.emb(idx)
        v_first = torch.empty_like(x)

        collected_states: List[torch.Tensor] = []

        for blk in self.blocks:
            x, v_first, state = blk(x, v_first, attention_mask)
            if return_states:
                collected_states.append(state)  # (B, H, N, N)

        x = self.ln_out(x)
        logits = self.head(x)

        return logits, collected_states if return_states else None


# ---------------------------------------------------------------------------
# 便捷工厂
# ---------------------------------------------------------------------------

def build_state_infer_model(args, device: str | torch.device = "cuda") -> RWKV7_StateInfer:
    """构造并移动到设备。"""
    model = RWKV7_StateInfer(args)
    model.to(device)
    model.eval()
    return model 