"""rwkv-peft-simple | model.state
本模块位于 `src/model`，定义 RWKV 模型在前向推理及训练过程中的 **状态容器**。

功能角色：
    1. 提供若干轻量级数据结构，用于保存每一层 TimeMix / ChannelMix 的中间状态。
    2. 通过 `BlockStateList` 封装所有层的状态，支持快速索引、批量创建与清空。

依赖关系：
    - 仅依赖 `torch` (张量创建与类型系统)。

对外公共接口：
    - TimeMixState
    - ChannelMixState
    - BlockState
    - BlockStateList (create / empty / __getitem__ / __setitem__)

类型：
    核心逻辑中间层；供 `light_rwkv.py` 与各 Block 实现引用。
"""
import torch
######state
class TimeMixState:
    """TimeMixState 保存单个 TimeMix 模块的状态。

    属性:
        shift_state (torch.Tensor): 由前一 token 产生的移位向量，形状为 (B, C)。
        wkv_state  (torch.Tensor):  用于 WKV 递归计算的缓存，形状为 (B, H, C//H, C//H)。
    """
    def __init__(self, shift_state: torch.Tensor, wkv_state: torch.Tensor):
        self.shift_state = shift_state
        self.wkv_state = wkv_state


class ChannelMixState:
    """ChannelMixState 保存 ChannelMix 模块的移位缓存。"""
    def __init__(self, shift_state: torch.Tensor):
        self.shift_state = shift_state


class BlockState:
    """BlockState 将同一层的 TimeMixState 与 ChannelMixState 打包成统一对象。"""
    def __init__(self, time_mix_state: TimeMixState,
                 channel_mix_state: ChannelMixState):
        self.time_mix_state = time_mix_state
        self.channel_mix_state = channel_mix_state

class BlockStateList:
    """BlockStateList 管理模型所有层的状态列表。

    设计要点:
        - 内部张量采用预分配方式以减少显存碎片。
        - 提供静态方法 `create` 与 `empty` 进行零初始化或仅分配内存。
        - 支持通过下标操作符访问或更新某层状态。
    """

    def __init__(self, shift_states, wkv_states):
        self.wkv_states = wkv_states
        self.shift_states = shift_states

    @staticmethod
    def create(N, B, C, H, device, dtype):
        """创建零初始化的 BlockStateList。

        参数说明:
            N (int):    层数 (n_layer)
            B (int):    批大小 (batch)
            C (int):    特征维度 (embedding dim)
            H (int):    注意力头数 (C // head_size_a)
            device:     目标设备, 通常为 CUDA / CPU
            dtype:      张量数据类型, 与模型权重保持一致
        """
        result = BlockStateList.empty(N, B, C, H, device, dtype)
        result.wkv_states[:] = 0
        result.wkv_states[:] = 0
        result.shift_states[:] = 0
        return result

    @staticmethod
    def empty(N, B, C, H, device, dtype):
        """仅分配内存而不初始化内容的 BlockStateList (内部使用)。"""
        wkv_states = torch.empty((N, B, H, C//H, C//H),
                                 device=device,
                                 dtype=torch.bfloat16)
        shift_states = torch.empty((N, 2, B, C), device=device, dtype=dtype)
        return BlockStateList(shift_states, wkv_states)

    def __getitem__(self, layer: int):
        """按层索引并返回 BlockState。"""
        return BlockState(
            TimeMixState(self.shift_states[layer, 0], self.wkv_states[layer]),
            ChannelMixState(self.shift_states[layer, 1]))

    def __setitem__(self, layer: int, state: BlockState):
        """按层写入指定 BlockState 到内部张量。"""
        self.shift_states[layer, 0] = state.time_mix_state.shift_state
        self.wkv_states[layer] = state.time_mix_state.wkv_state
        self.shift_states[layer, 1] = state.channel_mix_state.shift_state
