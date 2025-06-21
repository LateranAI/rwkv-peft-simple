from __future__ import annotations

from dataclasses import dataclass
import tomllib
from loguru import logger

@dataclass
class ModelConfig:
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
