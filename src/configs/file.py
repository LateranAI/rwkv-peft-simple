"""
文件名: file.py
所属路径: src/configs

功能概述:
    定义数据与文件相关的配置项 (FileConfig)，包含模型/数据路径、WandB 设置及阶段控制参数等。
    提供加载 TOML 配置文件到全局单例 `file_config` 的辅助函数 `load_config`。

关键角色:
    • FileConfig — 数据类，封装训练过程所需的文件与路径超参数，并包含 `check` / `show` 方法。
    • load_config — 将 TOML 文件映射到 FileConfig 全局实例。

依赖模块:
    - dataclasses.dataclass
    - tomllib (Python 3.11 for TOML 解析)
    - loguru.logger / lightning_utilities.rank_zero

适用场景:
    训练 / 推理脚本在启动前读取 configs/*/file.toml，将字段写入此数据类，再由模型加载器使用。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import os
import tomllib
from loguru import logger
from lightning_utilities.core.rank_zero import rank_zero_info

@dataclass
class FileConfig:
    """文件路径 & 数据相关配置

    功能说明:
        作为项目全局单例，负责保存与磁盘、数据集、检查点相关的路径及控制字段。调用 `check()` 可执行路径有效性校验。

    关键字段示例 (典型值):
        model_path (str): 模型权重路径，例如 "./models/rwkv-1.5b.pth"。
        proj_dir  (str): 训练输出目录，例如 "./runs/run-001"。
        data_file (str): 数据集前缀路径，例如 "./data/openwebtext"。
        data_type (str): 数据类型，支持 "utf-8" / "binidx" 等。

    副作用:
        `check()` 会在必要时创建目录，并打印 / 记录警告信息。
    """
    model_path: str = ""
    wandb: str = ""
    proj_dir: str = ""
    data_file: str = ""
    data_type: str = ""
    vocab_size: int = 0
    my_exit: int = 0
    my_exit_tokens: int = 0
    magic_prime: int = 0
    lora_load: str = ""
    pissa_load: str = ""
    pissa_init: str = ""
    disha_load: str = ""
    quant_path: str = ""
    my_pile_version: int = 0
    my_pile_stage: int = 0
    my_pile_shift: int = 0
    my_pile_edecay: int = 0
    my_timestamp: str = field(default_factory=lambda: datetime.today().strftime("%Y-%m-%d-%H-%M-%S"))
    my_pile_prev_p: int | None = None
    load_model: str = ""
    epoch_begin: int = 0

    def check(self):
        """检查文件路径的有效性。"""
        if not os.path.exists(self.proj_dir):
            os.makedirs(self.proj_dir)

        # if self.model_path and not os.path.exists(self.model_path):
        #     logger.warning(f"Model file not found: {self.model_path}")

        if self.data_file:
            if not os.path.exists(f"{self.data_file}.bin"):
                logger.warning(f"Data bin file not found: {self.data_file}.bin")
            if not os.path.exists(f"{self.data_file}.idx"):
                logger.warning(f"Data idx file not found: {self.data_file}.idx")

        if self.my_pile_stage >= 2:  # find latest saved model
            list_p = []
            rank_zero_info(f"proj_dir: {self.proj_dir}")
            for p in os.listdir(self.proj_dir):
                if p.startswith("rwkv") and p.endswith(".pth"):
                    p = ((p.split("-"))[1].split("."))[0]
                    if p != "final":
                        if p == "init":
                            p = -1
                        else:
                            p = int(p)
                        list_p += [p]
            list_p.sort()
            if not list_p:
                # 如果没有找到任何历史模型，则不进行任何操作
                pass
            else:
                max_p = list_p[-1]
                if len(list_p) > 1:
                    self.my_pile_prev_p = list_p[-2]  # in case max_p is corrupted
                if max_p == -1:
                    self.load_model = f"{self.proj_dir}/rwkv-init.pth"
                else:
                    self.load_model = f"{self.proj_dir}/rwkv-{max_p}.pth"
                self.epoch_begin = max_p + 1

        # 检查 PEFT 相关文件
        for name, path in [
            ("LoRA", self.lora_load),
            ("PISSA", self.pissa_load),
            ("PISSA init", self.pissa_init),
            ("DiSHA", self.disha_load),
            ("Quant", self.quant_path),
        ]:
            if path and not os.path.exists(path):
                logger.warning(f"{name} file not found: {path}")

        valid_data_types = ["utf-8", "utf-16le", "numpy", "binidx", "dummy", "uint16", "sft", "jsonl"]
        if self.data_type not in valid_data_types:
            logger.error(f"Invalid data_type: {self.data_type}. Must be one of {valid_data_types}")

        logger.info("✓ File config validation completed")

    def show(self):
        logger.info(f"FileConfig: {self.__dict__}")


file_config = FileConfig()


def load_config(path: str) -> FileConfig:
    """Load configuration from TOML file into the global instance."""
    global file_config
    with open(path, "rb") as f:
        data = tomllib.load(f)
    for k, v in data.items():
        setattr(file_config, k, v)
    return file_config
