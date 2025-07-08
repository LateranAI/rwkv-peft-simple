"""rwkv-peft-simple | trick.lrs
本模块位于 `src/model/trick`，属于训练技巧（trick）层。

功能角色：
    1. 提供学习率调度策略函数 `cos_decay` 与 `wsd`，供训练循环或优化器调用。
    2. 两个函数均纯粹基于数学公式计算，不涉及框架状态，因此可独立复用。

依赖关系：
    - 仅依赖 Python 标准库 `math`，无第三方依赖。

对外公共接口：
    - cos_decay(initial_lr, final_lr, current_step, total_steps)
    - wsd(initial_lr, final_lr, current_step, total_steps, warmup_steps=100)

类型：
    独立工具模块；调用方只需按需导入函数。
"""
import math


def cos_decay(initial_lr, final_lr, current_step, total_steps):

    if current_step >= total_steps:
        lr = final_lr
    else:
        lr = final_lr + 0.5 * (initial_lr - final_lr) * (
            1 + math.cos(math.pi * current_step / total_steps)
        )
    return lr


def wsd(initial_lr, final_lr, current_step, total_steps, warmup_steps=100):

    if warmup_steps <= 0:
        warmup_steps = 100
    if current_step < warmup_steps:

        return initial_lr * current_step / max(1, warmup_steps)
    else:

        effective_step = current_step - warmup_steps
        effective_total = total_steps - warmup_steps

        if effective_step >= effective_total:
            return final_lr

        cosine_decay = 0.5 * (1 + math.cos(math.pi * effective_step / effective_total))
        decayed_lr = final_lr + (initial_lr - final_lr) * cosine_decay
        return decayed_lr
