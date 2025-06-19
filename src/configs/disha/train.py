import torch
from loguru import logger

from lightning import seed_everything


class TrainConfig:
    """训练循环、优化器及数据相关配置。

    该配置**不包含**模型结构（参见 ``ModelConfig``）或磁盘路径（参见 ``FileConfig``）。
    默认值与 `train.py` 中 ArgumentParser 保持一致，确保完整覆盖。
    """

    def __init__(self) -> None:
        # 随机种子 & 环境
        self.random_seed: int = -1

        # 训练 epoch / step 控制
        self.epoch_steps: int = 200
        self.epoch_count: int = 1
        self.epoch_begin: int = 0
        self.epoch_save: int = 1

        # 批大小 & 并行设置
        self.ctx_len: int = 1024
        self.micro_bsz: int = 8
        self.accelerator: str = "gpu"
        self.strategy: str = "auto"
        self.devices: int = 1
        self.num_nodes: int = 1
        self.precision: str = "bf16"  # fp32 / tf32 / fp16 / bf16
        self.accumulate_grad_batches: int = 1

        # 优化器参数
        self.lr_init: float = 2e-5
        self.lr_final: float = 2e-5
        self.warmup_steps: int = 0
        self.beta1: float = 0.9
        self.beta2: float = 0.99
        self.adam_eps: float = 1e-8
        self.weight_decay: float = 0.0
        self.weight_decay_final: float = -1.0
        self.grad_cp: int = 0
        self.layerwise_lr: int = 1
        self.optim: str = "none"

        # 学习率日程 & Clip
        self.lr_schedule: str = "cos"
        self.gradient_clip_val: float = 1.0

        # 训练模式 & PEFT
        self.peft: str = "none"  # lora / pissa / DiSHA
        self.train_parts: list[str] = ["time", "ln"]
        self.lora_config: dict = {"lora_load": "", "lora_r": 8, "lora_alpha": 32, "lora_dropout": 0.01}
        self.pissa_config: dict = {"pissa_load": "", "pissa_init": "", "pissa_r": 8, "svd_niter": 4}
        self.disha_config: dict = {"mode": "bone", "load": "", "r": 64}

        # 数据加载 & 掩码
        self.dataload: str = "pad"
        self.chunk_ctx: int = 512
        self.loss_mask: str = "pad"  # pad / qa / se
        self.mask_id: dict = {"mask0": "0", "mask1": "1"}
        self.data_shuffle: int = 1
        self.avg_loss: int = 0

        # SFT
        self.sft_field: list[str] | None = None
        self.sft_split: str = "train"

        # Pile & 其它特殊模式
        self.ds_bucket_mb: int = 200
        self.my_sample_len: int = 0
        self.my_qa_mask: int = 0
        self.my_random_steps: int = 0

        self.my_ffn_shift: int = 1  # 冗余，保持与 ModelConfig 同名字段但可独立修改
        self.my_att_shift: int = 1
        self.my_testing: str = "x052"

        # FLA / 训练类型
        self.fla: bool = False
        self.train_type: str = "none"  # state / infctx / none

        # 其他 Debug / 实验性开关
        self.load_partial: int = 0

        # 量化（文件路径见 FileConfig.quant_path）
        self.quant: str = "none"

        # Auto
        self.real_bsz = None

    def check(self):
        if self.random_seed >= 0:
            logger.warning(f"GLOBAL SEED {self.random_seed} THIS WILL AFFECT MULTIGPU SAMPLING")
            seed_everything(self.random_seed)

        if self.warmup_steps < 0:
            self.warmup_steps = 30

        self.real_bsz = int(self.num_nodes) * int(self.devices) * self.micro_bsz
        self.epoch_steps = 40320 // self.real_bsz
        assert self.epoch_steps * self.real_bsz == 40320
        
        samples_per_epoch = self.epoch_steps * self.real_bsz
        tokens_per_epoch = samples_per_epoch * self.ctx_len

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


# 单例实例
train_config = TrainConfig()
