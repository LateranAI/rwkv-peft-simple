#################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
#################################################################

import math
import types
from types import SimpleNamespace

import lightning
import torch
from pytorch_lightning.utilities import rank_zero_info
from torch.utils.data import Dataset, DataLoader
from torch.utils.data._utils.collate import default_collate

from src.configs.file import file_config
from src.configs.model import model_config
from src.configs.train import train_config
from src.datasets.binidx import MMapIndexedDataset
from src.infering_loop.inferer import RWKV_x070


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
        # 当属于 state decoding 任务(`sd` 前缀)时, Dataset 会返回 GPU 张量, 这时必须关闭 pin_memory
        # 否则 PinMemoryThread 会尝试对已在 CUDA 设备上的张量执行 pin 操作, 触发
        # `RuntimeError: cannot pin 'torch.cuda.HalfTensor' only dense CPU tensors can be pinned`
        pin_mem = not ("sd" in self.args.train_type)

        # 在 state-decoding 任务下，Dataset 内部包含无法序列化的推理模型实例，
        # 因此必须避免多进程 DataLoader，否则在创建工作进程时会触发
        # "Tried to serialize object ... which does not have a __getstate__" 的报错。
        workers = 0 if "sd" in self.args.train_type else 1

        return DataLoader(
            self.train_data,
            shuffle=False,
            pin_memory=pin_mem,
            batch_size=train_config.micro_bsz,
            num_workers=workers,
            persistent_workers=workers > 0,
            drop_last=True,
            collate_fn=sd_collate_fn,
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

        if "sd" in self.args.train_type:
            args_s = types.SimpleNamespace()
            args_s.MODEL_NAME = "/public/home/ssjxzkz/Projects/block-blm/assets/weights/rwkv_s"
            # args_s.MODEL_NAME = "/mnt/g/Projects/MachineLearning/block-blm/assets/weights/rwkv_s"
            args_s.n_layer = 6
            args_s.n_embd = 256
            args_s.vocab_size = 256
            args_s.head_size = 64

            self.infer_model = RWKV_x070(args_s)


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

        x, y, state_like_x, last_shift_states, last_wkv_states = None, None, None, None, None

        if "sd" in args.train_type:
            y = torch.tensor(
                dix[:-1],
                dtype=torch.float32 if self.data.token_unit_type_code == 2 else torch.long,
            )

            # 如果属于 state decoding 任务，将 token 列表直接送入推理模型，获得隐藏状态作为输入特征
            token_list = dix[:-1].tolist()  # List[int]

            with torch.no_grad():
                _, states = self.infer_model(token_list, None)
                # states是一个List[Tensor]
                # 其中[0, 3, 6...]是att_x_prev(emb_d), [1, 4, 7...]是att_kv(num_heads, head_k_dim, head_v_dim), [2, 5, 8]是ffn_x_prev(emb_d)
                num_layer = len(states) // 3

            if args.train_type in ["sd_only_idx", "sd_both"]:
                # 仅保留每层的 att_kv(索引 1,4,7,...) 并堆叠，形状应为 (num_layer, num_head, head_k_dim, head_v_dim)

                att_kv_states = [states[i * 3 + 1] for i in range(num_layer)]  # 每层的 att_kv
                state_like_x = torch.stack(att_kv_states, dim=0)  # Tensor: (L, H, N, N)

                # 将前两维(L, H)展平为时间维, 后两维(N, N)展平为特征维
                L, H, N_k, N_v = state_like_x.shape
                state_like_x = state_like_x.reshape(L * H, N_k * N_v).half()  # (L*H, N_k*N_v)

                del states
                torch.cuda.empty_cache()

                # 调整到 ctx_len
                ctx_len = args.ctx_len
                time_steps, feat_dim = state_like_x.shape
                if time_steps < ctx_len:
                    # 通过复制 x 自身来补足长度，而非使用零张量填充
                    repeat_times = math.ceil(ctx_len / time_steps)
                    state_like_x = state_like_x.repeat((repeat_times, 1))  # 在时间维上复制
                    state_like_x = state_like_x[-ctx_len:]  # 截取补足后的最后 ctx_len 步
                else:
                    state_like_x = state_like_x[:ctx_len]
            else:
                x = torch.zeros(ctx_len, dtype=torch.long)

            # 对于需要状态输入的训练类型，构造 last_shift_states 与 last_wkv_states
            if args.train_type in ["sd_only_state", "sd_both"]:
                # 1. 构造 last_shift_states ---------------------------------------------------
                #    形状: (num_layer, 2, emb_d)
                att_x_prev_list = [states[i * 3 + 0] for i in range(num_layer)]  # (C,)
                ffn_x_prev_list = [states[i * 3 + 2] for i in range(num_layer)]  # (C,)

                last_shift_states = torch.stack([
                    torch.stack([att_x_prev_list[i], ffn_x_prev_list[i]], dim=0)
                    for i in range(num_layer)
                ], dim=0).half()  # (L, 2, C)

                # 2. 构造 last_wkv_states -----------------------------------------------------
                #    形状: (num_layer, num_head, D, D)
                att_kv_list = [states[i * 3 + 1] for i in range(num_layer)]  # (H, D, D)
                last_wkv_states = (
                    torch.stack(att_kv_list, dim=0).to(dtype=torch.bfloat16)
                )  # (L, H, D, D)

                # 释放显存
                del states
                torch.cuda.empty_cache()

        else:
            # 常规训练路径，直接构造张量
            x = torch.tensor(
                dix[:-1],
                dtype=torch.float32 if self.data.token_unit_type_code == 2 else torch.long,
            )

        if args.train_type in ["state", "infctx", "none"]:
            y = torch.tensor(
                dix[1:],
                dtype=torch.float32 if self.data.token_unit_type_code == 2 else torch.long,
            )

        return x, y, state_like_x, last_shift_states, last_wkv_states


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

# -------------------------------------------------------------
# 自定义 collate，用于 state-decoding 任务对张量维度做一次整理
# -------------------------------------------------------------


def sd_collate_fn(batch):
    """针对 (*x*, *y*, *state_like_x*, *last_shift_states*, *last_wkv_states*)
    的五元组样本进行批处理, 并将状态张量调整为 Adaptor 所需的
    (L, 2, B, C) / (L, B, H, D, D) 形状。
    """

    # 按元素拆分
    xs, ys, slxs, shift_lst, wkv_lst = zip(*batch)

    # 使用 default_collate 处理普通张量 (允许 None)
    x = default_collate([v for v in xs]) if xs[0] is not None else None
    y = default_collate([v for v in ys])

    state_like_x = (
        default_collate([v for v in slxs]) if slxs[0] is not None else None
    )

    # 处理 shift / wkv 两类状态
    if shift_lst[0] is not None:
        shift = torch.stack(shift_lst, dim=0)  # (B, L, 2, C)
        shift = shift.permute(1, 2, 0, 3).contiguous()  # (L, 2, B, C)
    else:
        shift = None

    if wkv_lst[0] is not None:
        wkv = torch.stack(wkv_lst, dim=0)  # (B, L, H, D, D)
        wkv = wkv.permute(1, 0, 2, 3, 4).contiguous()  # (L, B, H, D, D)
    else:
        wkv = None

    return x, y, state_like_x, shift, wkv