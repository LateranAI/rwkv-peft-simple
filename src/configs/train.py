from __future__ import annotations

from dataclasses import dataclass
import tomllib
from typing import Tuple

import torch
from loguru import logger
from lightning import seed_everything

@dataclass
class TrainConfig:
    random_seed: int = -1
    epoch_steps: int = 0
    epoch_count: int = 0
    epoch_begin: int = 0
    epoch_save: int = 0
    ctx_len: int = 0
    micro_bsz: int = 0
    accelerator: str = "gpu"
    strategy: str = "auto"
    devices: int = 1
    num_nodes: int = 1
    precision: str = "bf16"
    accumulate_grad_batches: int = 1
    lr_init: float = 0.0
    lr_final: float = 0.0
    warmup_steps: int = 0
    beta1: float = 0.0
    beta2: float = 0.0
    adam_eps: float = 0.0
    weight_decay: float = 0.0
    weight_decay_final: float = 0.0
    grad_cp: int = 0
    layerwise_lr: int = 0
    optim: str = "none"
    lr_schedule: str = "cos"
    gradient_clip_val: float = 1.0
    peft: str = "none"
    train_parts: list[str] | None = None
    lora_config: dict | None = None
    pissa_config: dict | None = None
    disha_config: dict | None = None
    dataload: str = "pad"
    chunk_ctx: int = 0
    loss_mask: str = "pad"
    mask_id: dict | None = None
    data_shuffle: int = 1
    avg_loss: int = 0
    sft_field: list[str] | None = None
    sft_split: str = "train"
    ds_bucket_mb: int = 0
    my_sample_len: int = 0
    my_qa_mask: int = 0
    my_random_steps: int = 0
    my_ffn_shift: int = 0
    my_att_shift: int = 0
    my_testing: str = "x070"
    fla: bool = False
    train_type: str = "none"
    load_partial: int = 0
    quant: str = "none"
    real_bsz: int | None = None
    betas: tuple[float, float] = ()

    def check(self):
        self.betas = (self.beta1, self.beta2)

        if self.random_seed >= 0:
            logger.warning(f"GLOBAL SEED {self.random_seed} THIS WILL AFFECT MULTIGPU SAMPLING")
            seed_everything(self.random_seed)

        if self.warmup_steps < 0:
            self.warmup_steps = 30

        self.real_bsz = int(self.num_nodes) * int(self.devices) * self.micro_bsz
        self.epoch_steps = 40320 // self.real_bsz
        assert self.epoch_steps * self.real_bsz == 40320

        if self.precision == "fp32":
            torch.backends.cudnn.allow_tf32 = False
            torch.backends.cuda.matmul.allow_tf32 = False
        else:
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cuda.matmul.allow_tf32 = True

        if "32" in self.precision:
            self.precision = 32
        elif self.precision == "fp16":
            self.precision = 16
        else:
            self.precision = "bf16"

    def show(self):
        logger.info(f"TrainConfig: {self.__dict__}")


train_config = TrainConfig()


def load_config(path: str) -> TrainConfig:
    global train_config
    with open(path, "rb") as f:
        data = tomllib.load(f)
    for k, v in data.items():
        setattr(train_config, k, v)
    return train_config
