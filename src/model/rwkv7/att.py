"""
文件名: att.py
所属路径: src/model/rwkv7

功能概述:
    RWKV-7 模型 TimeMix 注意力模块的实现。该文件提供三个 TimeMix 变体:
        1. RWKV_Tmix_x070            ‑ 适用于常规训练 / 推理
        2. RWKV_Tmix_x070_State      ‑ 适用于 State-Tuning 训练模式
        3. RWKV_Tmix_x070_infctx     ‑ 适用于无限上下文 (Infinite Context) 推理

主要职责:
    • 依据全局 `train_config.train_type` 在工厂方法 `RWKV_Tmix_v7` 中动态路由 TimeMix 类。
    • 在 `RWKV_Tmix_x070` 中实现 LoRA-参数化的时间衰减、门控、增量值残差及 CUDA Fused Kernel 调用。
    • 为 State-Tuning / Infinite-Context 引入持久状态 (TimeMixState) 的前向推理逻辑。

关键依赖:
    - src.model.peft.linear.make_linear_att : 构造可插拔的线性层 (支持 LoRA / P-Tuning)。
    - src.model.operator.rwkvop : 提供 `RUN_CUDA_RWKV7g`, `RUN_RWKV7_STATE`, `RUN_RWKV7_INFCTX` CUDA Kernel。
    - src.model.state : TimeMixState 数据结构, 用于保存跨 Chunk 的状态。

输入 / 输出约定:
    forward 接口统一接受形如 (B, T, C) 的张量, 并根据模式返回:
        - 普通模式: (output, v_first)
        - State 模式: 同上
        - infctx 模式: (output, v_first, TimeMixState)
"""

import math

import torch.nn as nn
from src.model.state import *

from src.model.peft.linear import make_linear_att
from src.model.operator.rwkvop import RUN_CUDA_RWKV7g, RUN_RWKV7_STATE, RUN_RWKV7_INFCTX
from torch.nn import functional as F

from src.configs.model import model_config
from src.configs.train import train_config

if model_config.fused_kernel:
    from rwkvfla.ops.rwkv7 import fused_addcmul_rwkv7
    from rwkvfla.modules.layernorm import GroupNorm as FusedGroupNorm
else:
    fused_addcmul_rwkv7 = None
    FusedGroupNorm = None

def RWKV_Tmix_v7(*args, **kwargs):
    """工厂函数

    根据当前 `train_config.train_type` 返回合适的 TimeMix 实例。

    参数:
        *args, **kwargs: 透传给 TimeMix 类构造函数。

    返回:
        nn.Module: TimeMix 变体实例 (RWKV_Tmix_x070 / RWKV_Tmix_x070_State / RWKV_Tmix_x070_infctx)。
    """
    
    if train_config.train_type == 'state':
        return RWKV_Tmix_x070_State(*args, **kwargs)
    elif train_config.train_type == 'infctx':
        return RWKV_Tmix_x070_infctx(*args, **kwargs)
    else:
        return RWKV_Tmix_x070(*args, **kwargs)
    
class RWKV_Tmix_x070(nn.Module):
    """基础 TimeMix 实现 (x070 版本)

    关键逻辑:
        1. 构造多组可训练的 LoRA 权重 (w0/w1/w2, a0/a1/a2, v0/v1/v2, 等) 用于控制衰减、学习率及门控。
        2. 使用 `time_shift` 产生相邻 Token 差分特征, 再通过 addcmul_kernel (PyTorch / Fused) 组合生成多路输入。
        3. 调用 `RUN_CUDA_RWKV7g` 完成高效的注意力累加。
        4. 通过 GroupNorm 进行多头归一化, 并最终映射到输出线性层。

    构造参数:
        args : Namespace  ‑ 由 config 解析得到的模型/训练超参数集合。
        layer_id : int    ‑ 当前块所在层序号, 用于计算层级相关的衰减比例。
    """
    def __init__(self, args, layer_id):
        """初始化 TimeMix 模块

        功能说明：
            构建 TimeMix 所需所有可训练参数与子层，包括多路 LoRA 权重、归一化层及线性映射。

        关键逻辑：
            1. 依据 `layer_id` 计算衰减与比例参数，初始化可训练参数 `x_*`、`w*`、`a*`、`v*`、`g*` 等。
            2. 调用 `make_linear_att` 生成 LoRA 兼容的线性层 (receptance/key/value/output)。
            3. 根据 `model_config.fused_kernel` 决定 addcmul 实现（PyTorch or Fused）。

        输入参数：
            args (Namespace): 全局模型超参数，常见字段示例：
                • n_embd (int) — 隐层维度，例如 1024
                • n_layer (int) — 模型层数，例如 24
                • dim_att (int) — 注意力维度，默认与 n_embd 相同
                • head_size_a (int) — 每头维度，例如 64 / 128
            layer_id (int): 当前块序号，从 0 开始。

        返回值：
            None

        副作用：
            向本模块注册大量 `nn.Parameter` 和子层对象，占用显存。
        """
        super().__init__()
        self.args = args
        self.layer_id = layer_id
        self.my_testing = args.my_testing

        self.head_size = args.head_size_a
        self.n_head = args.dim_att // self.head_size
        assert args.dim_att % self.n_head == 0
        H = self.n_head
        N = self.head_size
        C = args.n_embd

        if model_config.fused_kernel:
            self.addcmul_kernel = self.fused_addcmul
        else:
            self.addcmul_kernel = self.torch_addcmul

        with torch.no_grad():
            # 针对单层模型 (args.n_layer == 1) 需避免除零错误，此时将比率视为 0
            if args.n_layer > 1:
                ratio_0_to_1 = layer_id / (args.n_layer - 1)  # 0 到 1
            else:
                ratio_0_to_1 = 0.0
            ratio_1_to_almost0 = 1.0 - (layer_id / args.n_layer)  # 1 to ~0
            ddd = torch.ones(1, 1, C)
            for i in range(C):
                ddd[0, 0, i] = i / C

            self.x_r = nn.Parameter(1.0 - torch.pow(ddd, 0.2 * ratio_1_to_almost0))
            self.x_w = nn.Parameter(1.0 - torch.pow(ddd, 0.9 * ratio_1_to_almost0))
            self.x_k = nn.Parameter(1.0 - (torch.pow(ddd, 0.9 * ratio_1_to_almost0) + 0.4 * ratio_0_to_1))
            self.x_v = nn.Parameter(1.0 - (torch.pow(ddd, 0.4 * ratio_1_to_almost0) + 0.6 * ratio_0_to_1))
            self.x_a = nn.Parameter(1.0 - torch.pow(ddd, 0.9 * ratio_1_to_almost0))
            self.x_g = nn.Parameter(1.0 - torch.pow(ddd, 0.2 * ratio_1_to_almost0))

            def ortho_init(x, scale):
                with torch.no_grad():
                    shape = x.shape
                    if len(shape) == 2:
                        gain = math.sqrt(shape[0] / shape[1]) if shape[0] > shape[1] else 1
                        nn.init.orthogonal_(x, gain=gain * scale)
                    elif len(shape) == 3:
                        gain = math.sqrt(shape[1] / shape[2]) if shape[1] > shape[2] else 1
                        for i in range(shape[0]):
                            nn.init.orthogonal_(x[i], gain=gain * scale)
                    else:
                        assert False
                    return x

            # D_DECAY_LORA = 64
            D_DECAY_LORA = max(32, int(round(  (1.8*(C**0.5))  /32)*32)) # suggestion
            self.w1 = nn.Parameter(torch.zeros(C, D_DECAY_LORA))
            self.w2 = nn.Parameter(ortho_init(torch.zeros(D_DECAY_LORA, C), 0.1))
            decay_speed = torch.ones(C)
            for n in range(C):
                decay_speed[n] = -7 + 5 * (n / (C - 1)) ** (0.85 + 1.0 * ratio_0_to_1 ** 0.5)
            self.w0 = nn.Parameter(decay_speed.reshape(1,1,C) + 0.5) # !!! 0.5 comes from F.softplus !!!

            # D_AAA_LORA = 64
            D_AAA_LORA = max(32, int(round(  (1.8*(C**0.5))  /32)*32)) # suggestion
            self.a1 = nn.Parameter(torch.zeros(C, D_AAA_LORA))
            self.a2 = nn.Parameter(ortho_init(torch.zeros(D_AAA_LORA, C), 0.1))
            self.a0 = nn.Parameter(torch.zeros(1,1,C))

            # D_MV_LORA = 32
            D_MV_LORA = max(32, int(round(  (1.3*(C**0.5))  /32)*32)) # suggestion
            if self.layer_id!=0:
                self.v1 = nn.Parameter(torch.zeros(C, D_MV_LORA))
                self.v2 = nn.Parameter(ortho_init(torch.zeros(D_MV_LORA, C), 0.1))
                self.v0 = nn.Parameter(torch.zeros(1,1,C)+1.0)

            # D_GATE_LORA = 128
            D_GATE_LORA = max(32, int(round(  (0.6*(C**0.8))  /32)*32)) # suggestion
            if C==1024:
                D_GATE_LORA = 128
            # Note: for some data, you can reduce D_GATE_LORA or even remove this gate
            self.g1 = nn.Parameter(torch.zeros(C, D_GATE_LORA))
            self.g2 = nn.Parameter(ortho_init(torch.zeros(D_GATE_LORA, C), 0.1))

            self.k_k = nn.Parameter(torch.ones(1,1,C)*0.85)
            self.k_a = nn.Parameter(torch.ones(1,1,C))
            self.r_k = nn.Parameter(torch.zeros(H,N))

            self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
            self.receptance = make_linear_att(C, C, bias=False)
            self.key = make_linear_att(C, C, bias=False)
            self.value = make_linear_att(C, C, bias=False)
            self.output = make_linear_att(C, C, bias=False)
            if model_config.fused_kernel:
                self.ln_x = FusedGroupNorm(H, C, eps=(1e-5)*(args.head_size_divisor**2), bias=True) # !!! notice eps value !!!
            else:
                self.ln_x = nn.GroupNorm(H, C, eps=(1e-5)*(args.head_size_divisor**2))


            # !!! initialize if you are using RWKV_Tmix_x070 in your code !!!
            # self.receptance.weight.data.uniform_(-0.5/(C**0.5), 0.5/(C**0.5))
            # self.key.weight.data.uniform_(-0.05/(C**0.5), 0.05/(C**0.5))
            # self.value.weight.data.uniform_(-0.5/(C**0.5), 0.5/(C**0.5))
            # self.output.weight.data.zero_()

    def torch_addcmul(self, x, xx):
        """PyTorch addcmul 实现

        功能说明：
            对输入张量 `x` 与差分特征 `xx` 按照可训练比例 `x_*` 进行 element-wise addcmul，
            生成六路门控输入 xr/xw/xk/xv/xa/xg。

        输入参数：
            x (Tensor[B, T, C]): 当前特征表示。
            xx (Tensor[B, T, C]): 差分特征 (time_shift(x) - x)。

        返回值：
            Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]: 六路张量，形状均为 (B, T, C)。

        副作用： 无。
        """
        xr = x + xx * self.x_r
        xw = x + xx * self.x_w
        xk = x + xx * self.x_k
        xv = x + xx * self.x_v
        xa = x + xx * self.x_a
        xg = x + xx * self.x_g
        return xr, xw, xk, xv, xa, xg
    
    def fused_addcmul(self, x, xx):
        """Fused CUDA addcmul 实现

        功能说明：
            调用 `fused_addcmul_rwkv7` CUDA Kernel，在 GPU 端一次性计算 xr/xw/xk/xv/xa/xg，
            提升吞吐量。

        参数与返回值同 `torch_addcmul`，但依赖额外 GPU Kernel。

        副作用： 调用外部 CUDA Kernel，引入流同步开销。"""
        return fused_addcmul_rwkv7(x, xx, self.x_r, self.x_w, self.x_k, self.x_v, self.x_a, self.x_g)

    @torch.compile
    def forward(self, x, v_first, attention_mask=None):
        """基础 TimeMix 前向 (常规 / 训练)。

        参数:
            x (torch.Tensor): [B, T, C] 当前层输入。
            v_first (torch.Tensor): [B, T, C] 第一层存储的 v 向量 (用于 residual)。
            attention_mask (torch.FloatTensor | None): [B, ≤T] — 掩码。

        返回:
            out (torch.Tensor): [B, T, C]
            v_first (torch.Tensor): 可能更新的 v_first
        """
        # 1. 记录形状，后续运算多次依赖 (B=batch, T=序列长度, C=特征维)
        B, T, C = x.size()          # x: [B, T, C]
        H = self.n_head            # 头数 H, 每头维度 N=C/H

        # 2. 可选 attention_mask: 仅保留最近 T 步的 0/1 掩码并广播到特征维度
        if attention_mask is not None:
            x = x.mul(attention_mask[:, -x.shape[-2]:, None])  # [B,T,C]

        # 3. 右移一位构造相邻差分特征 xx, 与原始 x 组合产生门控输入
        xx = self.time_shift(x) - x  # [B,T,C]

        # 4. 使用 addcmul_kernel 生成六路变体 (xr/xw/xk/..) 形状同 [B,T,C]
        xr, xw, xk, xv, xa, xg = self.addcmul_kernel(x, xx)

        # 5. 经过多路线性映射得到 r,w,k,v 向量
        r = self.receptance(xr)  # [B,T,C]
        # 时间衰减参数 w 经过 softplus 限定范围 (-∞,-0.5]
        w = -F.softplus(-(self.w0 + torch.tanh(xw @ self.w1) @ self.w2)) - 0.5  # [1,1,C] 广播至 [B,T,C]
        k = self.key(xk)         # [B,T,C]
        v = self.value(xv)       # [B,T,C]

        # 6. 第一层缓存 v_first, 其余层对 v 进行残差校正 (value residual)
        if self.layer_id == 0:
            v_first = v  # 缓存首层 v
        else:
            v = v + (v_first - v) * torch.sigmoid(self.v0 + (xv @ self.v1) @ self.v2)

        # 7. a: in-context LR, g: 输出门控向量
        a = torch.sigmoid(self.a0 + (xa @ self.a1) @ self.a2)
        g = torch.sigmoid(xg @ self.g1) @ self.g2  # [B,T,C]

        # 8. 归一化键向量以提升数值稳定性, 并对 k 应用学习率 a
        kk = k * self.k_k                 # [B,T,C]
        kk = F.normalize(kk.view(B,T,H,-1), dim=-1, p=2.0).view(B,T,C)
        k = k * (1 + (a-1) * self.k_a)

        # 9. mask 应用于 v 以屏蔽 pad token 的值
        if attention_mask is not None:
            v = v * attention_mask[:, -v.shape[-2]:, None]

        # 10. 调用 CUDA Kernel 进行递归注意力累积，返回加权值 x
        x = RUN_CUDA_RWKV7g(r, w, k, v, -kk, kk*a)  # [B,T,C]

        # 11. 分组归一化，先展平 batch 与 time 维再 reshape 回
        x = self.ln_x(x.view(B * T, C)).view(B, T, C)

        # 12. 输出残差融合 & 最终线性映射
        x = x + ((r.view(B,T,H,-1)*k.view(B,T,H,-1)*self.r_k).sum(dim=-1, keepdim=True) * v.view(B,T,H,-1)).view(B,T,C)
        x = self.output(x * g)
        return x, v_first
  

class RWKV_Tmix_x070_State(RWKV_Tmix_x070):
    """TimeMix 状态微调 (State-Tuning) 变体

    在基础 TimeMix 上引入 `time_state` 参数, 使模型能够在训练阶段学习可持久化的时间相关状态, 
    以实现少量参数微调 (PEFT) 的效果。
    """
    def __init__(self, args, layer_id):
        """初始化 State-Tuning TimeMix 模块

        功能说明：
            在基础 TimeMix 上额外创建可学习参数 `time_state`，用于少量参数微调 (PEFT)。

        输入参数同 `RWKV_Tmix_x070.__init__`。

        副作用： 向模块注册 `self.time_state`。
        """
        super().__init__(args, layer_id)
        with torch.no_grad():
            #for State-tuning
            self.time_state = nn.Parameter(torch.zeros(self.n_head, self.head_size, self.head_size))

            # !!! initialize if you are using RWKV_Tmix_x070 in your code !!!
            # self.receptance.weight.data.uniform_(-0.5/(C**0.5), 0.5/(C**0.5))
            # self.key.weight.data.uniform_(-0.05/(C**0.5), 0.05/(C**0.5))
            # self.value.weight.data.uniform_(-0.5/(C**0.5), 0.5/(C**0.5))
            # self.output.weight.data.zero_()

    @torch.compile
    def forward(self, x, v_first, attention_mask=None):
        """State-Tuning TimeMix 前向, 输入/输出与基础版本一致。"""
        B, T, C = x.size()
        H = self.n_head

        if attention_mask is not None:
            x = x.mul(attention_mask[:, -x.shape[-2]:, None])
        xx = self.time_shift(x) - x

        xr, xw, xk, xv, xa, xg = self.addcmul_kernel(x, xx)

        r = self.receptance(xr)
        w = -F.softplus(-(self.w0 + torch.tanh(xw @ self.w1) @ self.w2)) - 0.5 # soft-clamp to (-inf, -0.5)
        k = self.key(xk)
        v = self.value(xv)
        if self.layer_id == 0:
            v_first = v # store the v of the first layer
        else:
            v = v + (v_first - v) * torch.sigmoid(self.v0 + (xv @ self.v1) @ self.v2) # add value residual
        a = torch.sigmoid(self.a0 + (xa @ self.a1) @ self.a2) # a is "in-context learning rate"
        g = torch.sigmoid(xg @ self.g1) @ self.g2

        kk = k * self.k_k
        kk = F.normalize(kk.view(B,T,H,-1), dim=-1, p=2.0).view(B,T,C)
        k = k * (1 + (a-1) * self.k_a)

        x , _ = RUN_RWKV7_STATE(r,k,v,w,-kk, kk*a,self.time_state)
        x = self.ln_x(x.view(B * T, C)).view(B, T, C)

        x = x + ((r.view(B,T,H,-1)*k.view(B,T,H,-1)*self.r_k).sum(dim=-1, keepdim=True) * v.view(B,T,H,-1)).view(B,T,C)
        x = self.output(x * g)
        return x, v_first
    

class RWKV_Tmix_x070_infctx(RWKV_Tmix_x070):
    """TimeMix 无限上下文 (Infinite-Context) 变体

    在推理阶段通过 `TimeMixState` 维持跨 Chunk 的注意力累积状态, 从而突破固定 ctx_len 限制。
    """
    def __init__(self, args, layer_id):
        """初始化 Infinite-Context TimeMix 模块

        仅调用父类构造，保持参数一致。
        """
        super().__init__(args, layer_id)

    def forward(self, x, v_first, last_state: TimeMixState, attention_mask=None):
        """无限上下文 TimeMix 前向。

        参数:
            x (torch.Tensor): [B, T, C]
            v_first (torch.Tensor): [B, T, C]
            last_state (TimeMixState):
                shift_state — [B, C]
                wkv_state   — [H, N, N] or [B, H, C//H, C//H] 具体取决于实现
            attention_mask (torch.FloatTensor | None): [B, ≤T]

        返回:
            out (torch.Tensor): [B, T, C]
            v_first (torch.Tensor): [B, T, C]
            new_state (TimeMixState): 更新后的状态
        """
        B, T, C = x.size()
        H = self.n_head

        if attention_mask is not None:
            x = x.mul(attention_mask[:, -x.shape[-2]:, None])
        
        shift_state = last_state.shift_state
        wkv_state = last_state.wkv_state.clone().contiguous() 

        xx = torch.concat((shift_state.unsqueeze(1), x[:, :-1]), dim=1) - x


        xr, xw, xk, xv, xa, xg = self.addcmul_kernel(x, xx)

        #print(f'x shape = {x.shape}')

        shift_state = x[:,-1,:]

        r = self.receptance(xr)
        w = -F.softplus(-(self.w0 + torch.tanh(xw @ self.w1) @ self.w2)) - 0.5 # soft-clamp to (-inf, -0.5)
        k = self.key(xk)
        v = self.value(xv)
        if self.layer_id == 0:
            v_first = v # store the v of the first layer
        else:
            v = v + (v_first - v) * torch.sigmoid(self.v0 + (xv @ self.v1) @ self.v2) # add value residual
        a = torch.sigmoid(self.a0 + (xa @ self.a1) @ self.a2) # a is "in-context learning rate"
        g = torch.sigmoid(xg @ self.g1) @ self.g2

        kk = k * self.k_k
        kk = F.normalize(kk.view(B,T,H,-1), dim=-1, p=2.0).view(B,T,C)
        k = k * (1 + (a-1) * self.k_a)

        x , wkv_state = RUN_RWKV7_INFCTX(r,k,v,w,-kk, kk*a,wkv_state)

        x = self.ln_x(x.view(B * T, C)).view(B, T, C)

        x = x + ((r.view(B,T,H,-1)*k.view(B,T,H,-1)*self.r_k).sum(dim=-1, keepdim=True) * v.view(B,T,H,-1)).view(B,T,C)
        x = self.output(x * g)
        
        return x, v_first, TimeMixState(shift_state,wkv_state)