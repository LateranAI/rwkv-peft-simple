"""
文件说明:
    该脚本针对 PiSSA 训练策略，将 LoRA 增量权重与其初始化权重以及基础 RWKV 权重进行差分合并，得到可直接推理的完整权重文件。

功能角色:
    - 工具层脚本: 仅在 PiSSA 微调完成后使用。
    - 处理三组权重: 基础权重、PiSSA 初始化 LoRA 权重、PiSSA 训练后增量权重。

核心依赖:
    - torch: 权重加载与保存、矩阵运算。

关键步骤概览:
    1. 解析命令行参数，支持 `--use-gpu` 选项;
    2. 加载三组权重至 CPU/GPU;
    3. 对每个可学习矩阵执行 `W = (W - B_init @ A_init) + B @ A` 完成差分合并;
    4. 保存新权重至输出文件。

输入参数说明:
    --use-gpu: 是否使用 GPU 进行运算;
    <base_model.pth>: 基础模型权重;
    <lora_init.pth>: PiSSA 初始化 LoRA 权重;
    <lora_checkpoint.pth>: PiSSA 微调后增量权重;
    <output.pth>: 输出文件路径。

副作用说明:
    - 可能占用 GPU 资源;
    - 会在输出路径生成/覆盖权重文件。
"""
from collections import OrderedDict
import os
import sys
from typing import Dict
import typing
import torch

if '-h' in sys.argv or '--help' in sys.argv:
    print(f'Usage: python3 {sys.argv[0]} [--use-gpu] <base_model.pth> <lora_init.pth> <lora_checkpoint.pth> <output.pth>')

if sys.argv[1] == '--use-gpu':
    device = 'cuda'
    base_model, init_lora, lora, output = sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5]
else:
    device = 'cpu'
    base_model, init_lora, lora, output = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]


with torch.no_grad():
    w: Dict[str, torch.Tensor] = torch.load(base_model, map_location='cpu')
    # merge LoRA-only slim checkpoint into the main weights
    w_lora: Dict[str, torch.Tensor] = torch.load(lora, map_location='cpu')
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
            init_lora_A = prefix + '.init_lora_A'
            init_lora_B = prefix + '.init_lora_B'
            if lora_A in keys:
                assert lora_B in keys
                print(f'merging {lora_A} and {lora_B} into {k}')
                assert w[lora_B].shape[1] == w[lora_A].shape[0]
                lora_r = w[lora_B].shape[1]
                w[k] = w[k].to(device=device)
                w[lora_A] = w[lora_A].to(device=device)
                w[lora_B] = w[lora_B].to(device=device)
                w_init_lora[init_lora_A] = w_init_lora[init_lora_A].to(device=device)
                w_init_lora[init_lora_B] = w_init_lora[init_lora_B].to(device=device)
                w[k] = (w[k]- w_init_lora[init_lora_B] @ w_init_lora[init_lora_A]).to(dtype=torch.bfloat16)
                w[k] +=  w[lora_B] @ w[lora_A]
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