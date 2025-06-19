from datetime import datetime

from loguru import logger
import os

class FileConfig:
    def __init__(self):
        """专门用于保存与文件路径相关的配置。

        仅放与磁盘上具体文件 / 目录相关的字段，其他超参请放入专门的配置模块。
        字段默认值保持与 ``train.py`` 中 ArgumentParser 所采用的默认值一致，
        方便在不同模块之间共享配置。
        """
        self.model_path: str = ""
        self.wandb: str = ""

        # 训练输出 & 项目目录
        self.proj_dir: str = "out"  # 与 --proj_dir 对应

        # 数据相关
        self.data_file: str = ""      # 与 --data_file 对应，数据集文件路径
        self.data_type: str = "sft"  # 与 --data_type 对应，数据格式(txt/binidx/sft 等)

        self.vocab_size: int = 0       # 与 --vocab_size，对应词表大小（存储于文件）

        self.my_exit: int = 99999999
        self.my_exit_tokens: int = 0
        self.magic_prime: int = 0

        # PEFT / LoRA / PISSA / DiSHA 等微调组件所需文件
        self.lora_load: str = ""       # LoRA 权重文件路径，--lora_config.lora_load
        self.pissa_load: str = ""      # PISSA 权重文件路径，--pissa_config.pissa_load
        self.pissa_init: str = ""      # PISSA 初始化矩阵，--pissa_config.pissa_init
        self.disha_load: str = ""      # DiSHA 权重文件路径，--disha_config.load

        # 其它可能用到的文件路径
        self.quant_path: str = ""      # 量化模型文件，如果 --quant != "none"

        self.my_pile_version: int = 1
        self.my_pile_stage: int = 0
        self.my_pile_shift: int = -1
        self.my_pile_edecay: int = 0

        # Auto
        self.my_timestamp = datetime.today().strftime("%Y-%m-%d-%H-%M-%S")
        self.my_pile_prev_p = None
        self.load_model = ""
        self.epoch_begin = None

    def check(self):
        """检查文件路径的有效性。"""
        # 检查必要的文件是否存在
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
            ("Quant", self.quant_path)
        ]:
            if path and not os.path.exists(path):
                logger.warning(f"{name} file not found: {path}")
        
        # 检查数据类型
        valid_data_types = ["utf-8", "utf-16le", "numpy", "binidx", "dummy", "uint16", "sft", "jsonl"]
        if self.data_type not in valid_data_types:
            logger.error(f"Invalid data_type: {self.data_type}. Must be one of {valid_data_types}")
            
        logger.info("✓ File config validation completed")

    def show(self):
        logger.info(f"FileConfig: {self.__dict__}")

file_config = FileConfig()
