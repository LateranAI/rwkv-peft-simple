"""
文件说明:
    该脚本用于将 DiSHA 形式的参数高效组块 (Group Block Matrix Multiplication, GBMM) 增量权重与基础 RWKV 权重进行合并，得到可推理的完整模型权重文件。

功能角色:
    - 工具层脚本，定位于模型后处理阶段，仅在 DiSHA 微调结束后调用。
    - 支持可选量化 (4bit / nf4 / fp4 / int8 / none) 以减小输出权重体积。

核心依赖:
    - torch: 权重加载与保存、张量计算。
    - bitsandbytes.functional: 量化与反量化。
    - einops.rearrange: 用于张量维度变换以适配 DiSHA 的分块结构。

关键步骤概览:
    1. 解析命令行参数，确定基础权重、DiSHA 增量权重、输出位置以及量化方式;
    2. 将三者加载到 CPU/GPU 内存;
    3. 针对每个 `*.weight` 参数，若存在对应 `.disha` 增量，则根据维度 (2-D/3-D) 做分块加法或矩阵乘法累加;
    4. 可选对结果进行量化处理;
    5. 保存合并后的权重至输出路径。

输入参数说明:
    --base_model: 基础模型权重文件;
    --peft_checkpoint: DiSHA 增量权重文件;
    --output: 输出权重文件路径;
    --quant: 量化类型 (默认 none);
    --device: 运算设备 (默认 cuda)。

副作用说明:
    - 根据 `--device` 选择可能占用显存资源;
    - 会在输出路径生成/覆盖权重文件。
"""
from collections import OrderedDict
import os
import sys
from typing import Dict
import typing
import torch
import bitsandbytes as bnb
from argparse import ArgumentParser
from einops import rearrange

parser = ArgumentParser()
parser.add_argument("--base_model", default="", type=str)
parser.add_argument("--peft_checkpoint", default="", type=str)
parser.add_argument("--output", default="", type=str)
parser.add_argument("--quant", default="none", type=str)
parser.add_argument("--device", default="cuda", type=str)
args = parser.parse_args()
device = args.device
base_model = args.base_model
peft = args.peft_checkpoint
output = args.output
quant = args.quant

with torch.no_grad():
    w: Dict[str, torch.Tensor] = torch.load(base_model, map_location="cpu")

    w_peft: Dict[str, torch.Tensor] = torch.load(peft, map_location="cpu")

    for k in w_peft.keys():
        w[k] = w_peft[k]
    output_w: typing.OrderedDict[str, torch.Tensor] = OrderedDict()

    keys = list(w.keys())
    for k in keys:
        if k.endswith(".weight"):
            prefix = k[: -len(".weight")]
            gbmm = prefix + ".disha"

            if gbmm in keys:
                w[k] = w[k].to(device=device)
                w[gbmm] = w[gbmm].to(device=device)
                dim = w[gbmm].dim()
                if quant == "4bit":
                    qw, qs = bnb.functional.quantize_4bit(w[k])
                    w[k] = (bnb.functional.dequantize_4bit(qw, quant_state=qs)).to(
                        dtype=torch.bfloat16
                    )
                elif quant == "nf4":
                    qw, qs = bnb.functional.quantize_nf4(w[k])
                    w[k] = (bnb.functional.dequantize_nf4(qw, quant_state=qs)).to(
                        dtype=torch.bfloat16
                    )
                elif quant == "fp4":
                    qw, qs = bnb.functional.quantize_fp4(w[k])
                    w[k] = (bnb.functional.dequantize_fp4(qw, quant_state=qs)).to(
                        dtype=torch.bfloat16
                    )
                elif quant == "int8":
                    qw, qs = bnb.functional.quantize(w[k])
                    w[k] = (bnb.functional.dequantize(qw, state=qs)).to(
                        dtype=torch.bfloat16
                    )
                if dim == 3:
                    b, r, _ = w[gbmm].shape
                    disha = (
                        rearrange(w[k], "(a r1) (b r2) -> a b r1 r2", r1=r, r2=r)
                        @ w[gbmm]
                        + w[gbmm]
                    )
                    w[k] += rearrange(disha, "a b r1 r2 ->(a r1) (b r2) ")
                if dim == 2:
                    r, b = w[gbmm].shape

                    in_features = w[k].size(-1)
                    if in_features % r != 0:
                        last_size = in_features % r
                        n_block = in_features // r
                        n_block_size = n_block * r
                        w[k][:, :n_block_size] = (
                            (
                                w[k][:, :n_block_size]
                                .reshape(-1, n_block, r)
                                .permute(1, 2, 0)
                                + w[gbmm]
                            )
                            .permute(2, 0, 1)
                            .reshape(*w[k][:, :n_block_size].shape)
                        )
                        w[k][:, n_block_size:] = (
                            w[k][:, n_block_size:]
                            + (w[gbmm].transpose(0, 1))[:, :last_size]
                        )
                    else:
                        t = (
                            w[k].reshape(-1, w[k].size(-1) // r, r).permute(1, 2, 0)
                            + w[gbmm]
                        )
                        w[k] = t.permute(2, 0, 1).reshape(*w[k].shape)
                output_w[k] = w[k].to(device="cpu", copy=True)
                del w[k]
                del w[gbmm]
                continue

        if "disha" not in k:
            print(f"retaining {k}")
            output_w[k] = w[k].clone()
            del w[k]
    torch.save(output_w, output)
