#################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
#################################################################

"""
dataset_pt.py
=====================================================
Lightning DataModule 与 Dataset 的实现，服务于 **预训练 (pretrain)** 等
以 *pt* 为主的训练任务。

功能职责
----------
通过 `MMapIndexedDataset` 读取 `.bin`/`.idx` 双文件格式的语料，并使用
*magic_prime* 与黄金比例随机游走算法生成训练窗口，保证跨 epoch 及多卡
训练的随机性与均匀覆盖。

模块类型
~~~~~~~~
核心数据层；直接被 `trainer.py` 调用，是训练流程的关键依赖。

关键依赖
~~~~~~~~
- `src.configs.*` 读取模型 / 训练 / 文件配置
- `src.datasets.binidx.MMapIndexedDataset` 高效随机访问语料
- `src.infering_loop.inferer.RWKV_x070` 推理得到隐藏状态
"""

import math
from types import SimpleNamespace

import lightning
import torch
from pytorch_lightning.utilities import rank_zero_info
from torch.utils.data import Dataset, DataLoader

from src.configs.file import file_config
from src.configs.model import model_config
from src.configs.train import train_config
from src.datasets.binidx import MMapIndexedDataset


def get_data_by_l_version(trainer: lightning.Trainer):
    """根据 Lightning **Trainer** 获取数据模块实例。

    参数
    ------
    trainer : lightning.Trainer
        当前训练器，暂未使用，只为保持调用接口兼容。

    返回
    ------
    MyDataModule
        封装了 Dataset 与 DataLoader 的 LightningDataModule。
    """
    return MyDataModule()

class MyDataModule(lightning.LightningDataModule):
    """围绕 :class:`MyDataset` 的 DataModule 封装。

    关键职责
    ----------
    1. 在 ``setup`` 中实例化 :class:`MyDataset` 并将 **epoch / rank** 信息
       传递进去，保证分布式训练时样本索引一致。
    2. 在 ``train_dataloader`` 中返回单 GPU DataLoader；此处 **不使用
       DistributedSampler**，因为 Lightning 会自动处理。
    """
    def __init__(self):
        super().__init__()
        self.args = SimpleNamespace(**{**vars(model_config), **vars(train_config), **vars(file_config)})
        self.train_data = None

    def setup(self, stage=None):
        self.train_data = MyDataset()
        self.train_data.real_epoch = self.trainer.current_epoch
        self.train_data.rank = self.trainer.global_rank
        self.train_data.world_size = self.trainer.world_size
        self.train_data.setup(self.trainer.global_rank, self.trainer.world_size,
                              int(self.args.devices), self.args.data_shuffle)

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            shuffle=False,
            pin_memory=True,
            batch_size=train_config.micro_bsz,
            num_workers=1,
            persistent_workers=False,
            drop_last=True,
        )


class MyDataset(Dataset):
    """基于 **binidx** 语料的随机窗口 Dataset。

    采样策略
    ----------
    * 使用 *magic_prime* + 黄金比例混沌跳转生成样本起点，保证跨 epoch
      随机性且 covering 全数据。
    * 每次 ``__getitem__`` 产生长度 ``ctx_len`` 的输入序列 *x* 与对应
      右移标签 *y*。

    重要属性
    ----------
    rank : int
        当前进程编号 (DDP)。
    world_size : int
        总进程数 (DDP)。
    samples_per_epoch : int
        每个 epoch 的全局样本数 (= ``epoch_steps × real_bsz``)。
    """
    def __init__(self):
        self.rank = 0
        self.real_epoch = 0
        self.world_size = 0
        self.index_manager = None

        args = SimpleNamespace(**{**vars(model_config), **vars(train_config), **vars(file_config)})
        self.args = args
        self.vocab_size = args.vocab_size
        rank_zero_info(
            f"Current vocab size = {self.vocab_size} (make sure it's correct)")

        self.data = MMapIndexedDataset(args.data_file)
        self.data_size = len(
            self.data._bin_buffer) // self.data._index._dtype_size
        rank_zero_info(f"Data has {self.data_size} tokens.")

        self.samples_per_epoch = args.epoch_steps * args.real_bsz
        assert self.samples_per_epoch == 40320

        dataset_slot = self.data_size // args.ctx_len

        assert is_prime(args.magic_prime)
        assert args.magic_prime % 3 == 2
        assert 0.9 < args.magic_prime / dataset_slot <= 1

    def setup(self, rank, world_size, devices, shuffle):
        self.rank = rank
        self.world_size = world_size
        self.index_manager = GlobalIndexManager(rank=rank, device_num=devices, shuffle=shuffle)

    def __len__(self):
        """Return the total number of training samples for *all* ranks.

        Under distributed training, PyTorch-Lightning will automatically
        create a :class:`~torch.utils.data.DistributedSampler` which
        shards this dataset across every rank. Therefore the value we
        return here must be the *global* sample count, otherwise each
        rank would see only a fraction of the intended ``epoch_steps``
        and the progress bar would show fewer steps (e.g. 5 instead of
        32). ``real_bsz`` already equals ``micro_bsz × world_size`` so
        using it here guarantees that every rank still gets
        ``epoch_steps`` iterations per epoch.
        """
        return self.args.epoch_steps * self.args.real_bsz

    def __getitem__(self, idx):
        """按索引生成一个训练样本。

        参数
        ------
        idx : int
            当前 rank 内部的局部索引 (Lightning 会映射全局索引)。

        返回
        ------
        Tuple[torch.Tensor, torch.Tensor]
            *x* — 输入 token 序列, LongTensor[ctx_len]
            *y* — 标签 token 序列, LongTensor[ctx_len]
        """
        args = self.args
        rank = self.rank
        world_size = self.world_size
        # print(f"epoch {epoch} idx {idx} rank {rank}/{world_size}")
        epoch = self.real_epoch

        ctx_len = args.ctx_len
        req_len = ctx_len + 1
        magic_prime = args.magic_prime

        ii = 1 + epoch * self.samples_per_epoch + (idx * world_size) + rank

        factor = (math.sqrt(5) - 1) / 2
        factor = int(magic_prime * factor)
        i = ((factor * ii * ii * ii) % magic_prime) * ctx_len
        # print(f"epoch {epoch} idx {idx} rank {rank}/{world_size} ii {ii} pos {round(i / self.data_size, 3)}")

        dix = self.data.get(idx=0, offset=i, length=req_len)

        # 常规训练路径，直接构造张量
        x = torch.tensor(
            dix[:-1],
            dtype=torch.float32 if self.data.token_unit_type_code == 2 else torch.long,
        )
        y = torch.tensor(
            dix[1:],
            dtype=torch.float32 if self.data.token_unit_type_code == 2 else torch.long,
        )

        return x, y


class GlobalIndexManager:
    """在 *非随机 (shuffle=False)* 模式下，为多 GPU 训练生成确定性索引。

    当 **shuffle=True** 时此管理器被旁路，直接返回传入索引。
    """
    def __init__(self, rank=0, device_num=1, shuffle=True):
        self.current_idx = 0
        self.rank = rank
        self.device_num = device_num
        self.shuffle = shuffle

    def get_next_idx(self, idx_t):
        if self.shuffle:
            idx = idx_t
        else:
            idx = self.current_idx * self.device_num + self.rank
            self.current_idx += 1
        return idx


def is_prime(n):
    """判断整数是否为质数 (朴素 6k±1 检测)。

    返回
    ------
    bool
        ``True`` 表示 *n* 为质数。
    """
    if n <= 1:
        return False
    if n <= 3:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True