from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import os
import tomllib
from loguru import logger

@dataclass
class FileConfig:
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
    epoch_begin: int | None = None

    def check(self):
        """检查文件路径的有效性。"""
        if not os.path.exists(self.proj_dir):
            os.makedirs(self.proj_dir)

        if self.model_path and not os.path.exists(self.model_path):
            logger.warning(f"Model file not found: {self.model_path}")

        if self.data_file and not os.path.exists(self.data_file):
            logger.warning(f"Data file not found: {self.data_file}")

        if self.my_pile_stage >= 2:  # find latest saved model
            list_p = []
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
