########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################
"""
dataset_sft.py
=====================================================
面向 **指令微调 / SFT** 任务的数据加载器实现。

特点
------
* 兼容三种 `data_type`:
  1. ``sft``   : 直接调用 `rwkv_sft.sft_dataset` 返回三元张量；
  2. ``jsonl`` : 支持单句 jsonl，每行含 ``text`` 字段；
  3. ``binidx``: 兼容大规模 token 语料，沿用 `binidx` 双文件格式。
* 通过 `mask_fn_dict` 灵活生成损失掩码，支持只在回答段计算 loss。
* `GlobalIndexManager` 保证确定性的 multi-device sample dispatch。

模块暴露
~~~~~~~~
* `get_vocab_size()` : 动态获取词表大小供模型初始化。
* `get_data_by_l_version()` : 适配旧版接口的快捷工厂。
"""
from types import SimpleNamespace

import torch.nn.functional as F

import torch
import lightning as L
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from lightning_utilities.core.rank_zero import rank_zero_info
from torchgen.native_function_generation import self_to_out_signature

from .binidx import MMapIndexedDataset
from rwkv.utils import PIPELINE
from src.datasets.rwkv_sft import sft_dataset

from .mask import mask_fn_dict
from ..configs.model import model_config

pipeline = PIPELINE('rwkv7', "rwkv_vocab_v20230424")

from src.configs.train import train_config
from src.configs.file import file_config


def get_vocab_size() -> int:
    """读取一次 :class:`MyDataset` 以推断词表大小。

    返回
    ------
    int
        数据集中 `vocab_size` 字段，用于模型初始化。
    """
    train_data = MyDataset()
    temp = train_data.vocab_size
    del train_data
    return int(temp)


def get_data_by_l_version(trainer: L.Trainer):
    return MyDataModule()


class GlobalIndexManager:
    """与 `dataset_pt` 中同名类相同，保证非 shuffle 时多卡索引互补。"""
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


class MyDataModule(L.LightningDataModule):
    """Lightning DataModule 封装 — 服务于 SFT / jsonl / binidx 三种数据格式。"""
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
        # must set shuffle=False, persistent_workers=False (because worker is in another thread)
        return DataLoader(
            self.train_data,
            shuffle=train_config.data_shuffle,
            pin_memory=True,
            batch_size=train_config.micro_bsz,
            num_workers=1,
            persistent_workers=False,
            drop_last=True
        )


class MyDataset(Dataset):
    """多数据格式统一 Dataset。

    根据 ``args.data_type`` 分流：
    * ``sft``   — 调用 :func:`rwkv_sft.sft_dataset`
    * ``jsonl`` — 逐行读取文本，并用 `pipeline.encode` 分词
    * ``binidx``— 兼容大规模二进制语料
    """
    def __init__(self):
        self.args = SimpleNamespace(**{**vars(model_config), **vars(train_config), **vars(file_config)})

        self.rank = 0
        self.real_epoch = 0
        self.world_size = 0
        self.index_manager = None
        self.vocab_size = self.args.vocab_size
        if self.args.data_type == "sft":
            self.data = sft_dataset(self.args)
        elif self.args.data_type == "jsonl":
            import jsonlines
            with jsonlines.open(file_config.data_file) as file:
                self.data = list(file)

        elif self.args.data_type == "binidx":
            self.data = MMapIndexedDataset(file_config.data_file)
            self.data_size = len(self.data._bin_buffer) // self.data._index._dtype_size
            rank_zero_info(f"Data has {self.data_size} tokens.")

    def setup(self, rank, world_size, devices, shuffle):
        self.rank = rank
        self.world_size = world_size
        self.index_manager = GlobalIndexManager(rank=rank, device_num=devices, shuffle=shuffle)

    def __len__(self):
        """Global sample count for a full epoch.

        See detailed rationale in ``dataset_pt.py`` – we need the global
        sample count here so that every rank receives ``epoch_steps``
        batches after the DistributedSampler splits the dataset.
        """
        return self.args.epoch_steps * self.args.real_bsz

    def __getitem__(self, idx):
        """生成一条训练样本，对不同数据源执行不同逻辑。"""
        idx = self.index_manager.get_next_idx(idx_t=idx) if self.index_manager else idx
        args = self.args
        rank = self.rank
        epoch = self.real_epoch
        world_size = self.world_size

        if args.data_type == "sft":
            inputs, labels, attn_mask = self.data[0][idx], self.data[1][idx], self.data[2][idx]
            labels = torch.roll(labels, shifts=-1, dims=-1)

            return inputs, labels, attn_mask
        elif args.data_type == "jsonl":
            ctx_len = args.ctx_len
            req_len = ctx_len + 1
            ctx = self.data[idx]['text']
            token = torch.tensor(pipeline.encode(ctx))
            token_len = len(token)
            min_len = min(token_len, req_len)
            if req_len < token_len:
                token = token[:req_len]
                pad_len = 0
            else:
                pad_len = req_len - token_len

            # dix = F.pad(token, (pad_len, 0), value=0)
            dix = F.pad(token, (0, pad_len), value=0)
            label = F.pad(token, (0, pad_len), value=-100)
            x = dix[:-1]
            y = label[1:]

        else:
            ctx_len = args.ctx_len
            req_len = ctx_len + 1
            data = self.data

            if args.data_type == "binidx":
                if args.dataload == 'pad':
                    dix, min_len = data.pad(idx=idx, length=req_len)
                elif args.dataload == 'only':
                    dix = data.only(idx=idx, length=req_len).astype(int)

            x = torch.tensor(dix[:-1], dtype=torch.long)
            dix[min_len:] = -100
            y = torch.tensor(dix[1:], dtype=torch.long)

        mask_fn = mask_fn_dict.get(args.loss_mask)

        if mask_fn is not None:
            t1 = pipeline.encode('User:')
            t2 = pipeline.encode('Assistant:')
            y = mask_fn(dix, t1, t2, min_len)
        return x, y
