########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################
import types

from torch.utils.checkpoint import checkpoint as torch_checkpoint
import torch
import torch.nn as nn
import deepspeed

from src.model.state import BlockStateList
from src.model.rwkv7.block import Block


class Adaptor(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        if args.train_type in ["sd_only_idx", "sd_both"]:
            raw_dim = (args.head_size_a if hasattr(args, 'head_size_a') and args.head_size_a > 0 else args.head_size) ** 2
            # self.emb = nn.Linear(raw_dim, args.n_embd * 4)
            self.emb = nn.Sequential(
                nn.Linear(raw_dim, args.n_embd * 4),
                nn.LeakyReLU(),
                nn.Linear(args.n_embd * 4, args.n_embd),
                nn.LeakyReLU(),
                nn.Linear(args.n_embd, args.n_embd * 4),
                nn.LeakyReLU(),
                nn.Linear(args.n_embd * 4, args.n_embd),
            )
        else:
            self.emb = nn.Embedding(args.vocab_size, args.n_embd)

        self.blocks = nn.ModuleList([Block(args, i) for i in range(args.n_layer)])

        self.ln_out = nn.LayerNorm(args.n_embd)
        self.head = nn.Linear(args.n_embd, args.vocab_size, bias=False)

    def forward(self, *args, **kwargs):
        if self.args.train_type == "sd_only_idx":
            return self.forward_only_idx(*args, **kwargs)
        elif self.args.train_type == "sd_only_state":
            return self.forward_only_state(*args, **kwargs)
        elif self.args.train_type == "sd_both":
            return self.forward_both(*args, **kwargs)
        else:
            raise NotImplementedError

    def forward_only_idx(self, state, attention_mask=None):
        args = self.args

        # B, T = idx.size()
        # assert T <= args.ctx_len, "Cannot forward, model ctx_len is exhausted."

        x = self.emb(state)
        v_first = torch.empty_like(x)

        for block in self.blocks:
            if args.grad_cp == 1:
                if args.train_type == 'state' or args.peft != 'none':
                    x, v_first = torch_checkpoint(block, x, v_first, attention_mask, use_reentrant=False)
                else:
                    x, v_first = deepspeed.checkpointing.checkpoint(block, x, v_first, attention_mask)
            else:
                x, v_first = block(x, v_first, attention_mask)

        x = self.ln_out(x)
        x = self.head(x)

        return x

    def forward_only_state(self, idx, last_shift_states: torch.Tensor,
                     last_wkv_states: torch.Tensor, attention_mask=None):
        args = self.args
        B, T = idx.size()
        assert T <= args.ctx_len, "Cannot forward, model ctx_len is exhausted."

        C = args.n_embd
        H = args.dim_att // args.head_size_a
        assert C == H * args.head_size_a

        x = self.emb(idx)
        new_states = BlockStateList.empty(args.n_layer, B, args.n_embd, H,
                                          x.device, x.dtype)

        v_first = torch.empty_like(x)

        for i, (block, block_state) in enumerate(zip(self.blocks, BlockStateList(last_shift_states, last_wkv_states))):
            if args.grad_cp == 1 and i > 0:  # and i < len(self.blocks)-1 :
                x, v_first, new_block_state = torch_checkpoint(block, x, v_first, block_state, attention_mask,
                                                               use_reentrant=False)

            else:
                x, v_first, new_block_state = block(x, v_first, block_state, attention_mask)

            new_states[i] = new_block_state

        x = self.ln_out(x)
        x = self.head(x)

        return x

    def forward_both(self, idx, last_shift_states: torch.Tensor,
                     last_wkv_states: torch.Tensor, attention_mask=None):
        args = self.args
        B, T = idx.size()
        assert T <= args.ctx_len, "Cannot forward, model ctx_len is exhausted."
        C = args.n_embd
        H = args.dim_att // args.head_size_a
        assert C == H * args.head_size_a

        x = self.emb(idx)
        new_states = BlockStateList.empty(args.n_layer, B, args.n_embd, H,
                                          x.device, x.dtype)

        v_first = torch.empty_like(x)

        for i, (block, block_state) in enumerate(zip(self.blocks,
                                                     BlockStateList(last_shift_states, last_wkv_states))):
            if args.grad_cp == 1 and i > 0:  # and i < len(self.blocks)-1 :
                x, v_first, new_block_state = torch_checkpoint(block, x, v_first, block_state, attention_mask, use_reentrant=False)

            else:
                x, v_first, new_block_state = block(x, v_first, block_state, attention_mask)

            new_states[i] = new_block_state

        x = self.ln_out(x)
        x = self.head(x)

        return x


