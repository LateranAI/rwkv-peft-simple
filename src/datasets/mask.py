import torch
import numpy as np

"""
mask.py
=====================================================
提供针对自然语言标记序列的 **损失掩码 (loss mask)** 生成函数。

背景
------
在指令微调与对话数据集中，常见模式是 <User /> 与 <Assistant />
轮流出现。训练时我们通常只对 assistant 段计算语言模型损失，
而忽略 user 段。`create_mask` / `generate_mask` 通过两种策略在
token 级别生成 0/1 掩码，供上层 Dataset 在 ``__getitem__`` 内调用。

公开字典 `mask_fn_dict` 便于根据 `loss_mask` 配置字段动态选择。
"""

def create_mask(seq, token1, token2, min_len):
    """基于 token1 / token2 边界生成 0/1 损失掩码。"""
    # 找到所有特殊标记的索引
    indices1 = []
    for i in range(min_len - len(token1) + 1):
        if np.array_equal(seq[i:i + len(token1)], token1):
            indices1.append(i)
    indices2 = []

    for i in range(min_len - len(token2) + 1):
        if np.array_equal(seq[i:i + len(token2)], token2):
            indices2.append(i)
    mask = torch.zeros(seq.shape)
    # assert len(indices2)!=0 and len(indices1)!=0
    select = 0
    for i in range(min_len):
        if i in indices1:
            select = 0
        elif i in indices2:
            select = 1
        mask[i] = select
    if torch.sum(mask) == 0:
        mask[:min_len - 1] = 1
    return mask[1:]

def generate_mask(seq, token1, token2, min_len):
    """另一种实现：通过 while-loop 顺序遍历生成掩码。"""
    mask = torch.zeros(seq.shape)  # 初始化mask列表，默认全为0
    current_mask_value = 0  # 初始状态下，所有位置的mask值为0

    i = 0
    while i < min_len:
        if seq[i:i + len(token1)] == token1:
            current_mask_value = 0
            for j in range(len(token1)):
                mask[i + j] = current_mask_value
            i += len(token1)
        elif seq[i:i + len(token2)] == token2:
            current_mask_value = 1
            for j in range(len(token2)):
                mask[i + j] = current_mask_value
            i += len(token2)
        else:
            mask[i] = current_mask_value
            i += 1

    if torch.sum(mask) == 0:
        mask[:min_len - 1] = 1
    return mask[1:]


mask_fn_dict = {
    "qa": create_mask,
    "se": generate_mask
}