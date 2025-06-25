#################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
#################################################################

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
    return MyDataModule()

class MyDataModule(lightning.LightningDataModule):
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
            # shuffle=train_config.data_shuffle,
            shuffle=False,
            pin_memory=True,
            batch_size=train_config.micro_bsz,
            num_workers=1,
            # persistent_workers=False,
            persistent_workers=True,
            drop_last=True
        )


class MyDataset(Dataset):
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

        x = torch.tensor(dix[:-1], dtype=torch.float32 if self.data.token_unit_type_code == 2 else torch.long)
        y = torch.tensor(dix[1:], dtype=torch.float32  if self.data.token_unit_type_code == 2 else torch.long)

        return x, y


class GlobalIndexManager:
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