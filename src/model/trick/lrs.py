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
    """
    Compute the cosine decayed learning rate with a final learning rate.

    Args:
        initial_lr (float): Initial learning rate.
        final_lr (float): Final learning rate.
        current_step (int): Current training step.
        total_steps (int): Total number of training steps in one epoch.

    Returns:
        float: Decayed learning rate.
    """
    if current_step>=total_steps:
        lr = final_lr
    else:
        lr = final_lr + 0.5 * (initial_lr - final_lr) * (1 + math.cos(math.pi * current_step / total_steps))
    return lr

def wsd(initial_lr, final_lr, current_step, total_steps, warmup_steps=100):
    """
    Compute the learning rate using cosine annealing with a warmup phase.

    Warmup phase:
        For the first warmup_steps, the learning rate increases linearly from 0 to initial_lr.
    Cosine annealing phase:
        From warmup_steps to total_steps, the learning rate decays from initial_lr to final_lr
        following the cosine annealing schedule.

    Args:
        initial_lr (float): The target learning rate after warmup (also the starting learning rate for decay).
        final_lr (float): The final learning rate after total_steps.
        current_step (int): Current training step.
        total_steps (int): Total number of training steps.
        warmup_steps (int): Number of steps used for the warmup phase.

    Returns:
        float: The computed learning rate.
    """
    if warmup_steps<=0:
        warmup_steps = 100
    if current_step < warmup_steps:
        # Warmup phase: linearly increase LR from 0 to initial_lr.
        return initial_lr * current_step / max(1, warmup_steps)
    else:
        # Adjust step count for cosine annealing phase.
        effective_step = current_step - warmup_steps
        effective_total = total_steps - warmup_steps
        
        if effective_step >= effective_total:
            return final_lr
        
        # Compute cosine annealing decay.
        cosine_decay = 0.5 * (1 + math.cos(math.pi * effective_step / effective_total))
        decayed_lr = final_lr + (initial_lr - final_lr) * cosine_decay
        return decayed_lr