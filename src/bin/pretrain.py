import os, warnings, datetime
import numpy as np
import torch
from loguru import logger
import lightning as pl

from src.configs.file import file_config
from src.configs.model import model_config
from src.configs.train import train_config

if "deepspeed" in train_config.strategy: import deepspeed


def pretrain():
    """预训练主函数。"""
    logger.info("Starting RWKV pretraining...")
    
    # 创建输出目录
    if not os.path.exists(file_config.proj_dir):
        os.makedirs(file_config.proj_dir)

    # 基础环境设置
    np.set_printoptions(precision=4, suppress=True, linewidth=200)
    warnings.filterwarnings("ignore", ".*Consider increasing the value of the `num_workers` argument*")
    warnings.filterwarnings("ignore", ".*The progress bar already tracks a metric with the*")

    # 配置检查和显示
    logger.info("Validating configurations...")
    file_config.check()
    model_config.check()
    train_config.check()
    
    # 显示配置摘要
    file_config.show()
    model_config.show()
    train_config.show()




if __name__ == "__main__":
    pretrain()
