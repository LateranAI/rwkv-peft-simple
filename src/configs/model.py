"""
文件名: model.py
所属路径: src/configs

功能概述:
    定义模型结构与算子相关的超参数 (ModelConfig)。
    通过 `load_config` 从 TOML 文件加载，并暴露全局单例 `model_config` 供模型构造时使用。

关键角色:
    • ModelConfig — 数据类，保存 Layer 数、Embedding 维度、Attention 尺寸等。
    • load_config — TOML -> ModelConfig。

依赖模块:
    - dataclasses.dataclass
    - tomllib
    - loguru.logger
"""

from __future__ import annotations

from dataclasses import dataclass
import tomllib
from loguru import logger


@dataclass
class ModelConfig:
    """模型结构配置

    功能说明:
        保存 RWKV 模型的层数、隐藏维度、Attention / FFN 尺寸等关键超参数。
        `check()` 会自动推断缺省值 (如 dim_att = n_embd)。

    典型字段范围:
        n_layer (int): 6 ~ 48
        n_embd  (int): 256 ~ 4096
        fused_kernel (bool): 是否启用 Fused RWKV7 CUDA kernel。
    """
    n_layer: int = 0
    n_embd: int = 0
    dim_att: int = 0
    dim_ffn: int = 0
    pre_ffn: int = 0
    head_qk: int = 0
    tiny_att_dim: int = 0
    tiny_att_layer: int = 0
    head_size_a: int = 0
    head_size_divisor: int = 0
    my_pos_emb: int = 0
    my_ffn_shift: int = 0
    my_att_shift: int = 0
    dropout: float = 0.0
    quant: str = "none"
    op: str = "cuda"
    fused_kernel: bool = False

    def check(self):
        if self.dim_att <= 0:
            self.dim_att = self.n_embd
        if self.dim_ffn <= 0:
            self.dim_ffn = int((self.n_embd * 3.5) // 32 * 32)

    def show(self):
        logger.info(f"ModelConfig: {self.__dict__}")


model_config = ModelConfig()


def load_config(path: str) -> ModelConfig:
    global model_config
    with open(path, "rb") as f:
        data = tomllib.load(f)
    for k, v in data.items():
        setattr(model_config, k, v)
    return model_config
