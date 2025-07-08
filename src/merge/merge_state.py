"""
文件说明:
    该脚本用于将『状态微调』(State Tuning) 产生的权重增量合并到基础 RWKV 模型权重中。

功能角色:
    - 工具层脚本，仅在 State Tuning 训练完成后调用。

核心依赖:
    - torch: 权重加载与保存、张量操作。

关键步骤概览:
    1. 解析命令行参数，确定基础权重、State Checkpoint、输出路径;
    2. 加载两组权重到内存;
    3. 直接用 State Checkpoint 中的键覆盖基础权重对应键;
    4. 保存合并后的权重到输出路径。

输入参数说明:
    --base_model: 基础模型权重文件;
    --state_checkpoint: State Tuning 产生的增量权重文件;
    --output: 输出权重路径;
    --device: 运算设备 (默认 cuda)。

副作用说明:
    - 占用一定显存/内存;
    - 会覆盖输出路径同名文件。
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
parser.add_argument("--base_model", default="", type=str)
parser.add_argument("--state_checkpoint", default="", type=str)
parser.add_argument("--output", default="", type=str)

parser.add_argument("--device", default="cuda", type=str)

args = parser.parse_args()
device = args.device
base_model = args.base_model
state = args.state_checkpoint
output = args.output


with torch.no_grad():
    w: Dict[str, torch.Tensor] = torch.load(base_model, map_location="cpu")

    w_state: Dict[str, torch.Tensor] = torch.load(state, map_location="cpu")

    for k in w_state.keys():
        print(k)
        w[k] = w_state[k]

    for k in w.keys():
        print(k)

    torch.save(w, output)
