"""
文件说明:
    该脚本用于将不同形式的权重微调增量（LoRA/PiSSA 等）合并到基础 RWKV 权重中，输出可直接用于推理的完整模型权重文件。

功能角色:
    - 工具层脚本: 仅在训练流程结束后使用，不参与训练或验证循环。
    - 支持模式: 普通 LoRA、PiSSA，两者均可选择量化方式 (4bit / nf4 / fp4 / int8 / none)。

核心逻辑依赖:
    - torch: 负责权重加载、矩阵运算及保存。
    - bitsandbytes.functional: 提供多种量化与反量化函数。

关键步骤概览:
    1. 解析命令行参数，确定权重位置与合并方式;
    2. 加载基础权重以及 LoRA 增量权重 (及 PiSSA 初始 LoRA 权重);
    3. 遍历权重字典，将 LoRA 参数按公式 `W = W_base - W_init + B @ A` 合并;
    4. 根据用户选择，对最终权重执行可选量化并存储到输出文件。

输入参数说明(典型值):
    --type: "pissa" / "lora" (默认 pissa)
    --base_model: "rwkv_base.pth"
    --lora_init: "lora_init.pth" (PiSSA 专用，可选)
    --lora_checkpoint: "lora_delta.pth"
    --output: "merged.pth"
    --quant: "none" / "4bit" / "nf4" / "fp4" / "int8"
    --lora_alpha: 16 (常见范围 8~32)

返回值:
    无直接返回值，侧重副作用: 将合并后的权重保存至 `--output` 指定路径。

副作用说明:
    - 在 GPU 上进行张量运算时将占用显存资源;
    - 会在输出路径写入新的权重文件，可能覆盖同名文件。
"""
from collections import OrderedDict
import os
import sys
from typing import Dict
import typing
import torch
import bitsandbytes as bnb
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--type", default="pissa", type=str)
parser.add_argument("--base_model", default="", type=str)
parser.add_argument("--lora_init", default="none", type=str)
parser.add_argument("--lora_checkpoint", default="", type=str)
parser.add_argument("--output", default="", type=str)
parser.add_argument("--quant", default="none", type=str)
parser.add_argument("--device", default="cuda", type=str)
parser.add_argument("--lora_alpha", default=16, type=int)
args = parser.parse_args()
device= args.device
base_model = args.base_model
init_lora= args.lora_init
lora= args.lora_checkpoint
output= args.output
quant= args.quant
lora_alpha = args.lora_alpha

with torch.no_grad():
    w: Dict[str, torch.Tensor] = torch.load(base_model, map_location='cpu')
    # merge LoRA-only slim checkpoint into the main weights
    w_lora: Dict[str, torch.Tensor] = torch.load(lora, map_location='cpu')

    if args.type=='pissa':
        w_init_lora: Dict[str, torch.Tensor] = torch.load(init_lora, map_location='cpu')
    for k in w_lora.keys():
        w[k] = w_lora[k]
    output_w: typing.OrderedDict[str, torch.Tensor] = OrderedDict()
    # merge LoRA weights
    keys = list(w.keys())
    for k in keys:
        if k.endswith('.weight'):
            prefix = k[:-len('.weight')]
            lora_A = prefix + '.lora_A'
            lora_B = prefix + '.lora_B'
            init_lora_A = 'model.' + prefix + '.init_lora_A'
            init_lora_B = 'model.' + prefix + '.init_lora_B'
            if lora_A in keys:
                assert lora_B in keys
                print(f'merging {lora_A} and {lora_B} into {k}')
                assert w[lora_B].shape[1] == w[lora_A].shape[0]
                lora_r = w[lora_B].shape[1]
                w[k] = w[k].to(device=device)
                w[lora_A] = w[lora_A].to(device=device)
                w[lora_B] = w[lora_B].to(device=device)
                
                if args.type=='pissa':
                    w_init_lora[init_lora_A] = w_init_lora[init_lora_A].to(device=device)
                    w_init_lora[init_lora_B] = w_init_lora[init_lora_B].to(device=device)
                    if quant=='4bit':
                        qw,qs = bnb.functional.quantize_4bit(w[k]- w_init_lora[init_lora_B] @ w_init_lora[init_lora_A])
                        w[k] = (bnb.functional.dequantize_4bit(qw,quant_state=qs)).to(dtype=torch.bfloat16)
                    elif quant == 'nf4':
                        qw,qs = bnb.functional.quantize_nf4(w[k]- w_init_lora[init_lora_B] @ w_init_lora[init_lora_A])
                        w[k] = (bnb.functional.dequantize_nf4(qw,quant_state=qs)).to(dtype=torch.bfloat16)
                    elif quant == 'fp4':
                        qw,qs = bnb.functional.quantize_fp4(w[k]- w_init_lora[init_lora_B] @ w_init_lora[init_lora_A])
                        w[k] = (bnb.functional.dequantize_fp4(qw,quant_state=qs)).to(dtype=torch.bfloat16)
                    elif quant == 'int8':
                        qw,qs = bnb.functional.quantize(w[k]- w_init_lora[init_lora_B] @ w_init_lora[init_lora_A])
                        w[k] = (bnb.functional.dequantize(qw,state=qs)).to(dtype=torch.bfloat16)
                    else:
                        w[k] = (w[k]- w_init_lora[init_lora_B] @ w_init_lora[init_lora_A]).to(dtype=torch.bfloat16)
                    w[k] +=  w[lora_B] @ w[lora_A]
                else:
                    if quant=='4bit':
                        qw,qs = bnb.functional.quantize_4bit(w[k])
                        w[k] = (bnb.functional.dequantize_4bit(qw,quant_state=qs)).to(dtype=torch.bfloat16)
                    elif quant=='nf4':
                        qw,qs = bnb.functional.quantize_nf4(w[k])
                        w[k] = (bnb.functional.dequantize_nf4(qw,quant_state=qs)).to(dtype=torch.bfloat16)
                    elif quant=='fp4':
                        qw,qs = bnb.functional.quantize_fp4(w[k])
                        w[k] = (bnb.functional.dequantize_fp4(qw,quant_state=qs)).to(dtype=torch.bfloat16)
                    elif quant=='int8':
                        qw,qs = bnb.functional.quantize(w[k])
                        w[k] = (bnb.functional.dequantize(qw,state=qs)).to(dtype=torch.bfloat16)
                    w[k] += w[lora_B] @ w[lora_A] * (lora_alpha / lora_r)
                output_w[k] = w[k].to(device='cpu', copy=True)
                del w[k]
                del w[lora_A]
                del w[lora_B]
                continue

        if 'lora' not in k:
            print(f'retaining {k}')
            output_w[k] = w[k].clone()
            del w[k]
    torch.save(output_w, output)