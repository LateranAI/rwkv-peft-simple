from loguru import logger


class ModelConfig:
    """模型结构与超参数配置。

    该配置仅保存**直接影响模型前向/推理结构**的字段，
    与训练循环、优化器、文件路径等无关。
    字段默认值与 `train.py` 中 ArgumentParser 默认值保持一致，
    以保证不同模块之间参数同步且易于维护。
    """

    def __init__(self) -> None:
        # 网络层级与宽度
        self.n_layer: int = 6
        self.n_embd: int = 256

        # Attention & FFN 维度
        self.dim_att: int = 0  # 0 表示与 n_embd 相同
        self.dim_ffn: int = 0  # 0 表示自动 3.5 × n_embd 并向下取 32 的倍数

        # 结构 trick / 可选组件
        self.pre_ffn: int = 0      # 是否用 FFN 代替首个 Attention
        self.head_qk: int = 0      # HeadQK trick
        self.tiny_att_dim: int = 0 # Tiny Attention 维度
        self.tiny_att_layer: int = -999  # Tiny Attention 所在层

        # RWKV7 新增 / 其他 trick
        self.head_size_a: int = 64
        self.head_size_divisor: int = 8
        self.my_pos_emb: int = 0
        self.my_ffn_shift: int = 1
        self.my_att_shift: int = 1

        # Dropout
        self.dropout: float = 0.0

        # 量化与运算后端
        self.quant: str = "none"  # 量化模式
        self.op: str = "cuda"      # 后端算子实现 cuda / cpu
        self.fused_kernel: bool = False  # 是否启用 fla fused kernel

    def check(self):
        if self.dim_att <= 0:
            self.dim_att = self.n_embd
        if self.dim_ffn <= 0:
            self.dim_ffn = int((self.n_embd * 3.5) // 32 * 32)

    def show(self):
        logger.info(f"ModelConfig: {self.__dict__}")


# 单例实例，供其他模块直接 import 使用
model_config = ModelConfig()
