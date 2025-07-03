########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################
import os
import types

from torch.utils.checkpoint import checkpoint as torch_checkpoint
# from adam_mini import Adam_mini

import math, gc, importlib
import torch
from src.configs.train import train_config
from src.configs.model import model_config
# torch._C._jit_set_profiling_executor(True)
# torch._C._jit_set_profiling_mode(True)
import torch.nn as nn
import lightning as pl
from lightning.pytorch.strategies import DeepSpeedStrategy
from lightning_utilities.core.rank_zero import rank_zero_info

from src.model.rwkv7.adaptor import Adaptor

if importlib.util.find_spec('deepspeed'):
    from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam

rank_zero_info(f'RWKV_MY_TESTING {train_config.my_testing}')

from src.model.rwkv7.model import RWKV7



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


class StateDecoder(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.model = Adaptor(args)
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
                    {"params": [param_dict[n] for n in lr_2x], "weight_decay": 0.0, "my_lr_scale": 5.0},
                    # test: 2e-3 / args.lr_init},
                    {"params": [param_dict[n] for n in lr_3x], "weight_decay": 0.0, "my_lr_scale": 5.0},
                    # test: 3e-3 / args.lr_init},
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
            optim_groups += [
                {"params": [param_dict[n] for n in lr_decay], "weight_decay": args.weight_decay, "my_lr_scale": 1.0}]
            # if args.optim == 'adam_mini':
            #     return Adam_mini(self, lr=self.args.lr_init, betas=self.args.betas, eps=self.args.adam_eps,
            #                      weight_decay=0, model_sharding=True, n_feature=args.n_embd, n_head=args.n_embd // 64,
            #                      lora_r=8)
            if self.deepspeed_offload:
                return DeepSpeedCPUAdam(optim_groups, lr=self.args.lr_init, betas=self.args.betas,
                                        eps=self.args.adam_eps, bias_correction=True, adamw_mode=True, amsgrad=False)
            return FusedAdam(optim_groups, lr=self.args.lr_init, betas=self.args.betas, eps=self.args.adam_eps,
                             bias_correction=True, adam_w_mode=True, amsgrad=False)
        else:
            # if args.optim == 'adam_mini':
            #     return Adam_mini(self, lr=self.args.lr_init, betas=self.args.betas, eps=self.args.adam_eps,
            #                      weight_decay=0, model_sharding=True, n_feature=args.n_embd, n_head=args.n_embd // 64,
            #                      lora_r=8)
            if self.deepspeed_offload:
                return DeepSpeedCPUAdam(optim_groups, lr=self.args.lr_init, betas=self.args.betas,
                                        eps=self.args.adam_eps, bias_correction=True, adamw_mode=False, weight_decay=0,
                                        amsgrad=False)
            return FusedAdam(optim_groups, lr=self.args.lr_init, betas=self.args.betas, eps=self.args.adam_eps,
                             bias_correction=True, adam_w_mode=False, weight_decay=0, amsgrad=False)
        # return ZeroOneAdam(optim_groups, lr=self.args.lr_init, betas=self.args.betas, eps=self.args.adam_eps, bias_correction=True, weight_decay=0, amsgrad=False, cuda_aware=False)

    @property
    def deepspeed_offload(self) -> bool:
        strategy = self.trainer.strategy
        if isinstance(strategy, DeepSpeedStrategy):
            cfg = strategy.config["zero_optimization"]
            return cfg.get("offload_optimizer") or cfg.get("offload_param")
        return False

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    # if train_config.train_type == 'only_idx':
    #     def forward(self, idx, attention_mask=None):
    #         return self.model(idx, attention_mask)
    # elif train_config.train_type == 'only_state':
    #     def forward(self, last_shift_states: torch.Tensor,
    #                 last_wkv_states: torch.Tensor, attention_mask=None):
    #         return self.model(last_shift_states, last_wkv_states, attention_mask)
    # else:
    #     def forward(self, idx, last_shift_states: torch.Tensor,
    #                 last_wkv_states: torch.Tensor, attention_mask=None):
    #         return self.model(idx, last_shift_states, last_wkv_states, attention_mask)

    def training_step(self, batch, batch_idx):
        x, y, state_like_x, last_shift_states, last_wkv_states = batch

        if self.args.train_type == "sd_only_idx":
            logits = self(state_like_x)
        elif self.args.train_type == "sd_only_state":
            logits = self(x, last_shift_states, last_wkv_states)
        else:
            logits = self(state_like_x, last_shift_states, last_wkv_states)
        loss = self.criterion(logits.view(-1, logits.size(-1)), y.view(-1))
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
                    layer_scale = (1 + int(n.split('.')[1])) / self.args.n_layer
                    m[n] = (p * 0.0) + (layer_scale ** 0.7)
                else:
                    m[n] = p
            else:
                if n == "emb.weight":
                    scale = -1 * self.args.lr_init
                else:
                    if shape[0] > shape[1]:
                        gain = math.sqrt(shape[0] / shape[1])

                    zero = [".att.output.", ".ffn.value.", ".ffn.receptance.", ".ffnPre.value.", ".ffnPre.receptance.",
                            "head_q.", '.oo.', '.rr.']

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

