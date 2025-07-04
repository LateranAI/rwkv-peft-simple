"""
文件说明:
    该脚本用于将 LoRA 微调增量权重合并回基础 RWKV 模型权重，生成可直接用于推理的完整权重文件。

功能角色:
    - 工具层脚本，定位于模型后处理阶段，不参与训练循环。
    - 仅处理标准 LoRA (A, B) 低秩适配矩阵，不包含 PiSSA 特殊逻辑。

核心依赖:
    - torch: 权重加载与保存、矩阵运算。

关键步骤概览:
    1. 解析命令行参数，支持 `--use-gpu` 选项决定运算设备;
    2. 将基础模型权重与 LoRA 增量权重加载到内存;
    3. 遍历所有 `*.weight` 参数，按公式 `W = W + B @ A * (alpha / r)` 进行合并;
    4. 将合并后的权重保存至输出路径。

输入参数说明:
    --use-gpu: 是否使用 GPU 进行运算 (可选);
    <lora_alpha>: LoRA 缩放因子，常见取值 8~32;
    <base_model.pth>: 基础权重文件路径;
    <lora_checkpoint.pth>: LoRA 增量权重文件路径;
    <output.pth>: 输出文件路径。

副作用说明:
    - 根据 `--use-gpu` 选项可能占用 GPU 显存;
    - 在输出路径写入新的权重文件。
"""
from collections import OrderedDict
import os
import sys
from typing import Dict
import typing
import torch

if '-h' in sys.argv or '--help' in sys.argv:
    print(f'Usage: python3 {sys.argv[0]} [--use-gpu] <lora_alpha> <base_model.pth> <lora_checkpoint.pth> <output.pth>')

if sys.argv[1] == '--use-gpu':
    device = 'cuda'
    lora_alpha, base_model, lora, output = float(sys.argv[2]), sys.argv[3], sys.argv[4], sys.argv[5]
else:
    device = 'cpu'
    lora_alpha, base_model, lora, output = float(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4]


with torch.no_grad():
    w: Dict[str, torch.Tensor] = torch.load(base_model, map_location='cpu')
    # merge LoRA-only slim checkpoint into the main weights
    w_lora: Dict[str, torch.Tensor] = torch.load(lora, map_location='cpu')
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
            if lora_A in keys:
                assert lora_B in keys
                print(f'merging {lora_A} and {lora_B} into {k}')
                assert w[lora_B].shape[1] == w[lora_A].shape[0]
                lora_r = w[lora_B].shape[1]
                w[k] = w[k].to(device=device)
                w[lora_A] = w[lora_A].to(device=device)
                w[lora_B] = w[lora_B].to(device=device)
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
