########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################
"""rwkv-peft-simple | model.light_rwkv
本模块封装了 RWKV 模型在 PyTorch Lightning 训练框架中的顶层逻辑。

功能角色：
    1. 管理 `RWKV7` 网络结构与训练/推理过程。
    2. 构造分组优化器与学习率比例，以支持 LayerWise-LR、权重衰减与 DeepSpeed offload。
    3. 针对不同 `train_type`（常规 / infctx）动态生成 `forward` 与 `training_step` 实现。

依赖关系：
    - PyTorch / Lightning
    - DeepSpeed（可选）
    - 项目内部模块：`src.model.rwkv7.*`, `src.model.state.BlockStateList`, `src.configs.*`

对外公共接口：
    - RWKV (LightningModule)

类型：核心训练逻辑层。
"""
import os

from torch.utils.checkpoint import checkpoint as torch_checkpoint
#from adam_mini import Adam_mini

import math, gc, importlib
import torch
from src.configs.train import train_config
from src.configs.model import model_config
# torch._C._jit_set_profiling_executor(True)
# torch._C._jit_set_profiling_mode(True)
import torch.nn as nn
import lightning as pl
from lightning.pytorch.strategies import DeepSpeedStrategy

from src.model.state import BlockStateList

if importlib.util.find_spec('deepspeed'):
    from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam

from lightning_utilities.core.rank_zero import rank_zero_info

rank_zero_info(f'RWKV_MY_TESTING {train_config.my_testing}')

from src.model.rwkv7.model import RWKV7 as RWKVModel

if train_config.train_type == 'infctx':
    class L2Wrap(torch.autograd.Function):
        @staticmethod
        def forward(ctx, loss, y, token_amount):
            ctx.save_for_backward(y)
            ctx.token_amount = token_amount
            return loss

        @staticmethod
        def backward(ctx, grad_output): #这个函数会不会影响batch和grad_accu的一致性？感觉上会。梯度累积时，factor变大了。但是只有loss缩放，这里的正则化项反而没有缩放
            y = ctx.saved_tensors[0]
            # to encourage the logits to be close to 0
            if ctx.token_amount == 0:
                return (grad_output, None, None)
            factor = 1e-4 / ctx.token_amount #这一行类似crossentropy在token上平均。
            maxx, ids = torch.max(y, -1, keepdim=True)
            gy = torch.zeros_like(y)
            if os.environ.get("WN_FIX_L2WRAP"): #实现batch等价性
                # maxx[maxx<3.]=0. #防止对已经较小的logits值下拉，只对大于阈值的往下拉
                gy.scatter_(-1, ids, maxx * factor * grad_output)
            else:
                gy.scatter_(-1, ids, maxx * factor)
            return (grad_output, gy, None)
else:
    class L2Wrap(torch.autograd.Function):
        @staticmethod
        def forward(ctx, loss, y):
            ctx.save_for_backward(y)
            return loss

        @staticmethod
        def backward(ctx, grad_output):
            y = ctx.saved_tensors[0]
            # to encourage the logits to be close to 0
            factor = 1e-4 / (y.shape[0] * y.shape[1])
            maxx, ids = torch.max(y, -1, keepdim=True)
            gy = torch.zeros_like(y)
            gy.scatter_(-1, ids, maxx * factor)
            return (grad_output, gy)


class RWKV(pl.LightningModule):
    """LightningModule 封装的 RWKV 模型。

    关键职责：
        1. 初始化底层 `RWKV7` 网络并根据 `model_config` 设置损失函数。
        2. 构建分组学习率与权重衰减策略的优化器，支持 DeepSpeed Offload。
        3. 针对 `train_type` 的不同，派生合适的 `forward` / `training_step` 实现。

    参数:
        args (Namespace): 由 CLI 或配置文件解析得到的超参数集合。

    属性:
        model (RWKV7):     底层可微分网络结构。
        criterion (nn.Module): 交叉熵损失或融合损失，依据 `model_config.fused_kernel`。
    """
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.model = RWKVModel(args)
        if model_config.fused_kernel:
            from rwkvfla.modules import FusedCrossEntropyLoss
            self.criterion = FusedCrossEntropyLoss(inplace_backward=True)
        else:
            FusedCrossEntropyLoss = None
            self.criterion = nn.CrossEntropyLoss()

    def configure_optimizers(self):
        args = self.args
        
        lr_decay = set()
        lr_1x = set()
        lr_2x = set()
        lr_3x = set()
        for n, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            if (("_w1" in n) or ("_w2" in n)) and (args.layerwise_lr > 0):
                lr_1x.add(n)
            elif (("time_mix" in n) or ("time_maa" in n)) and (args.layerwise_lr > 0):
                if args.my_pile_stage == 2:
                    lr_2x.add(n)
                else:
                    lr_1x.add(n)
            elif (("time_decay" in n) or ("time_daaaa" in n)) and (args.layerwise_lr > 0):
                if args.my_pile_stage == 2:
                    lr_3x.add(n)
                else:
                    lr_2x.add(n)
            elif ("time_faaaa" in n) and (args.layerwise_lr > 0):
                if args.my_pile_stage == 2:
                    lr_2x.add(n)
                else:
                    lr_1x.add(n)
            elif ("time_first" in n) and (args.layerwise_lr > 0):
                lr_3x.add(n)
            elif (len(p.squeeze().shape) >= 2) and (args.weight_decay > 0):
                lr_decay.add(n)
            else:
                lr_1x.add(n)

        lr_decay = sorted(list(lr_decay))
        lr_1x = sorted(list(lr_1x))
        lr_2x = sorted(list(lr_2x))
        lr_3x = sorted(list(lr_3x))

        param_dict = {n: p for n, p in self.model.named_parameters()}
        
        if args.layerwise_lr > 0:
            if args.my_pile_stage == 2:
                optim_groups = [
                    {"params": [param_dict[n] for n in lr_1x], "weight_decay": 0.0, "my_lr_scale": 1.0},
                    {"params": [param_dict[n] for n in lr_2x], "weight_decay": 0.0, "my_lr_scale": 5.0},# test: 2e-3 / args.lr_init},
                    {"params": [param_dict[n] for n in lr_3x], "weight_decay": 0.0, "my_lr_scale": 5.0},# test: 3e-3 / args.lr_init},
                ]
            else:
                optim_groups = [
                    {"params": [param_dict[n] for n in lr_1x], "weight_decay": 0.0, "my_lr_scale": 1.0},
                    {"params": [param_dict[n] for n in lr_2x], "weight_decay": 0.0, "my_lr_scale": 2.0},
                    {"params": [param_dict[n] for n in lr_3x], "weight_decay": 0.0, "my_lr_scale": 3.0},
                ]
        else:
            optim_groups = [{"params": [param_dict[n] for n in lr_1x], "weight_decay": 0.0, "my_lr_scale": 1.0}]

        if args.weight_decay > 0:
            optim_groups += [{"params": [param_dict[n] for n in lr_decay], "weight_decay": args.weight_decay, "my_lr_scale": 1.0}]
            # if args.optim=='adam_mini':
            #     return Adam_mini(self, lr=self.args.lr_init, betas=self.args.betas, eps=self.args.adam_eps, weight_decay=0, model_sharding=True, n_feature=args.n_embd, n_head=args.n_embd//64, lora_r=8)
            if self.deepspeed_offload:
                return DeepSpeedCPUAdam(optim_groups, lr=self.args.lr_init, betas=self.args.betas, eps=self.args.adam_eps, bias_correction=True, adamw_mode=True, amsgrad=False)
            return FusedAdam(optim_groups, lr=self.args.lr_init, betas=self.args.betas, eps=self.args.adam_eps, bias_correction=True, adam_w_mode=True, amsgrad=False)
        else:
            # if args.optim=='adam_mini':
            #     return Adam_mini(self, lr=self.args.lr_init, betas=self.args.betas, eps=self.args.adam_eps, weight_decay=0, model_sharding=True, n_feature=args.n_embd, n_head=args.n_embd//64, lora_r=8)
            if self.deepspeed_offload:
                return DeepSpeedCPUAdam(optim_groups, lr=self.args.lr_init, betas=self.args.betas, eps=self.args.adam_eps, bias_correction=True, adamw_mode=False, weight_decay=0, amsgrad=False)
            return FusedAdam(optim_groups, lr=self.args.lr_init, betas=self.args.betas, eps=self.args.adam_eps, bias_correction=True, adam_w_mode=False, weight_decay=0, amsgrad=False)
        # return ZeroOneAdam(optim_groups, lr=self.args.lr_init, betas=self.args.betas, eps=self.args.adam_eps, bias_correction=True, weight_decay=0, amsgrad=False, cuda_aware=False)

    @property
    def deepspeed_offload(self) -> bool:
        strategy = self.trainer.strategy
        if isinstance(strategy, DeepSpeedStrategy):
            cfg = strategy.config["zero_optimization"]
            return cfg.get("offload_optimizer") or cfg.get("offload_param")
        return False

    if train_config.train_type == 'infctx':
        def forward(self, idx,  last_shift_states: torch.Tensor,
                last_wkv_states: torch.Tensor, attention_mask=None):
            """无限上下文 (infctx) 前向传播。

            参数说明:
                idx (torch.LongTensor): [B, T] — 词 ID 序列, B 为批大小, T 为当前 Chunk 长度。
                last_shift_states (torch.Tensor): [N, 2, B, C] — 上一次 Block 移位状态, N 为层数, C 为嵌入维度。
                last_wkv_states  (torch.Tensor): [N, B, H, C//H, C//H] — 累积的 WKV 状态 (见 `BlockStateList`).
                attention_mask (torch.FloatTensor | None): [B, ≤T] — 可选遮罩, 元素为 0/1。

            返回:
                logits (torch.FloatTensor): [B, T, vocab_size]
                new_shift_states (torch.Tensor): 与 `last_shift_states` 相同形状, 更新后的状态。
                new_wkv_states  (torch.Tensor): 与 `last_wkv_states` 相同形状, 更新后的状态。
            """
            return self.model(idx, last_shift_states, last_wkv_states, attention_mask)
    else:
        def forward(self, idx, attention_mask=None):
            """常规前向传播 (非 infctx)。

            参数:
                idx (torch.LongTensor): [B, T]
                attention_mask (torch.FloatTensor | None): [B, ≤T]

            返回:
                logits (torch.FloatTensor): [B, T, vocab_size]
            """
            return self.model(idx, attention_mask)

    if train_config.train_type == 'infctx':
        def training_step(self, batch, batch_idx):
            """infctx 训练一步。

            batch 结构: Tuple[idx, targets]。
                idx     — LongTensor[B, T]
                targets — LongTensor[B, T] (填充位置标记为 -100)

            内部张量:
                states.shift_states — [N, 2, B, C]
                states.wkv_states  — [N, B, H, C//H, C//H]
            """
            args = self.args
            T_train = args.chunk_ctx 
            idx, targets= batch

            B, T = idx.shape
            C = args.n_embd
            H =  args.dim_att // args.head_size_a
            assert C==H*args.head_size_a
            states = BlockStateList.create(args.n_layer, B, C, H, idx.device,
                self.model.emb.weight.dtype)

            def checkpointed_step(idx, targets, prev_loss, last_shift_states,
                                last_wkv_states, prev_token_amount):
                logits, new_shift_states, new_wkv_states = self(idx, last_shift_states, last_wkv_states)
                current_token_amount = (targets!=-100).sum() #这样是不是更合适？
                current_token_amount = idx.shape[1]

                loss = self.criterion(logits.view(-1, logits.size(-1)), targets.reshape(-1))
                if current_token_amount != 0:
                    loss = L2Wrap.apply(loss, logits, current_token_amount)
                new_token_amount = prev_token_amount+current_token_amount
                if new_token_amount>0:
                    new_loss = prev_loss * (prev_token_amount / new_token_amount) + loss * (
                        current_token_amount / new_token_amount)
                else:
                    new_loss = prev_loss

                return new_loss, new_shift_states, new_wkv_states, new_token_amount
            
            total_loss = torch.tensor(0.,dtype=self.model.emb.weight.dtype).requires_grad_()
            token_amount = 0
            i = 0
            for i in range(math.ceil(T / T_train)):

                total_loss,new_shift_states, new_wkv_states,token_amount = torch_checkpoint(
                    checkpointed_step,
                    idx[:, i * T_train:(i + 1) * T_train],
                    targets[:, i * T_train:(i + 1) * T_train],
                    total_loss,
                    states.shift_states,
                    states.wkv_states,
                    token_amount,
                    use_reentrant=False
                )

                states = BlockStateList(new_shift_states.clone().detach(), new_wkv_states.clone().detach())
            
            return total_loss
    else:
        def training_step(self, batch, batch_idx):
            """常规训练一步 (非 infctx)。

            batch 可能为两种结构:
                1. data_type == 'sft': (idx, targets, mask)
                   mask 形状 [B, T] — 0/1 掩码。
                2. 其他: (idx, targets)

            返回: torch.Tensor 标量 loss。
            """
            args = self.args
            if args.data_type=='sft':
                idx, targets, mask = batch
                logits = self(idx, mask)            
                
            else:
                idx, targets = batch
                logits = self(idx)
            loss = self.criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
            
            return L2Wrap.apply(loss, logits)
    
    
    def training_step_end(self, batch_parts):
        pass

    def generate_init_weight(self):
        print(
            f"""
############################################################################
#
# Init model weight (slow for large models)...
#
############################################################################
"""
        )
        m = {}
        for n in self.model.state_dict():
            p = self.model.state_dict()[n]
            shape = p.shape

            gain = 1.0
            scale = 1.0
            if (
                "ln_" in n
                or ".ln" in n
                or "time_" in n
                or "_mask" in n
                or "pos_emb" in n
                or ".mask." in n
                or n.endswith("_w")
                or n.endswith("_w1")
                or n.endswith("_w2")
                or n.endswith("_bias")
                or (".weight" not in n)
            ):
                if 'ln_x.weight' in n:
                    layer_scale = (1+int(n.split('.')[1])) / self.args.n_layer
                    m[n] = (p * 0.0) + (layer_scale ** 0.7)
                else:
                    m[n] = p
            else:
                if n == "emb.weight":
                    scale = -1 * self.args.lr_init
                else:
                    if shape[0] > shape[1]:
                        gain = math.sqrt(shape[0] / shape[1])

                    zero = [".att.output.", ".ffn.value.", ".ffn.receptance.", ".ffnPre.value.", ".ffnPre.receptance.", "head_q.", '.oo.', '.rr.']

                    for kk in zero:
                        if kk in n:
                            scale = 0
                    if n == "head.weight":
                        scale = 0.5
                    if "head_k." in n:
                        scale = 0.1
                    if "head_q." in n:
                        scale = 0

                print(f"{str(shape[0]).ljust(5)} {str(shape[1]).ljust(5)} {str(scale).ljust(4)} {n}")

                if self.args.accelerator.upper() == "GPU":
                    m[n] = torch.empty((shape[0], shape[1]), device="cuda")
                else:
                    m[n] = torch.empty((shape[0], shape[1]))

                if scale == 0:
                    nn.init.zeros_(m[n])
                elif scale < 0:
                    nn.init.uniform_(m[n], a=scale, b=-scale)
                else:
                    nn.init.orthogonal_(m[n], gain=gain * scale)

            m[n] = m[n].cpu()
            if train_config.precision == "fp16":
                m[n] = m[n].half()
            elif train_config.precision == "bf16":
                m[n] = m[n].bfloat16()

            # if n == "emb.weight":
            #     print(m[n])

        gc.collect()
        torch.cuda.empty_cache()
        return m
