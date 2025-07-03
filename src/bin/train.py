import os
import sys
import warnings

# Set a unique triton cache dir for each rank to avoid race conditions
# Try to find a rank identifier from common environment variables
rank_keys = ["PL_GLOBAL_RANK", "RANK", "LOCAL_RANK", "SLURM_PROCID", "OMPI_COMM_WORLD_RANK"]
rank = None
for key in rank_keys:
    if key in os.environ:
        rank = os.environ[key]
        break

# Set a unique triton cache dir for each rank if a rank is found
if rank is not None:
    user = os.environ.get("USER", "user")
    pid = os.getpid()
    # Using /tmp is generally safer for multi-node, shared-filesystem setups
    cache_dir = f"/tmp/triton_{user}_rank_{rank}_pid_{pid}"
    os.environ["TRITON_CACHE_DIR"] = cache_dir
    os.makedirs(cache_dir, exist_ok=True)

import deepspeed
import pytorch_lightning
import torch
from loguru import logger
from types import SimpleNamespace

import numpy as np
from lightning import Trainer
from lightning_utilities.core.rank_zero import rank_zero_only

from src.configs.file import file_config, load_config as load_file_config
from src.configs.model import model_config, load_config as load_model_config
from src.configs.train import train_config, load_config as load_train_config
from src.training_loop.args_type import TrainingArgs
from src.training_loop.trainer import train_callback
from src.datasets.dataset_pt import get_data_by_l_version


@rank_zero_only
def show_configs():
    args = SimpleNamespace(**{**vars(model_config), **vars(train_config), **vars(file_config)})
    logger.info(
        f"""
    ############################################################################
    #
    # RWKV-7 {str(args.precision).upper()} on {args.num_nodes}x{args.devices} {args.accelerator.upper()}, bsz {args.num_nodes}x{args.devices}x{args.micro_bsz}={args.real_bsz}, {args.strategy} {'with grad_cp' if args.grad_cp > 0 else ''}
    #
    # Data = {file_config.data_file} ({file_config.data_type}), ProjDir = {file_config.proj_dir}
    #
    # Epoch = {args.epoch_begin} to {args.epoch_begin + args.epoch_count - 1} (will continue afterwards), save every {args.epoch_save} epoch
    #
    # Each "epoch" = {args.epoch_steps} steps, {args.epoch_steps * args.real_bsz} samples, {args.epoch_steps * args.real_bsz * args.ctx_len} tokens
    #
    # Model = {model_config.n_layer} n_layer, {model_config.n_embd} n_embd, {args.ctx_len} ctx_len
    #
    # Adam = lr {args.lr_init} to {args.lr_final}, warmup {args.warmup_steps} steps, beta {args.betas}, eps {args.adam_eps}
    #
    # Found torch {torch.__version__}, recommend latest torch
    # Found deepspeed {deepspeed.__version__}, recommend latest deepspeed
    # Found pytorch_lightning {pytorch_lightning.__version__}, recommend 1.9.5
    #
    ############################################################################
    """

    )



def main(config_dir: str):
    load_file_config(os.path.join(config_dir, "file.toml"))
    load_model_config(os.path.join(config_dir, "model.toml"))
    load_train_config(os.path.join(config_dir, "train.toml"))

    from src.model.peft.peft_loading import load_peft_model

    if "deepspeed" in train_config.strategy:
        pass

    np.set_printoptions(precision=4, suppress=True, linewidth=200)
    warnings.filterwarnings("ignore", ".*Consider increasing the value of the `num_workers` argument*")
    warnings.filterwarnings("ignore", ".*The progress bar already tracks a metric with the*")

    file_config.check()
    model_config.check()
    train_config.check()

    show_configs()

    _args, model = load_peft_model()

    trainer = Trainer(
        accelerator=train_config.accelerator,
        strategy=train_config.strategy,
        devices=train_config.devices,
        num_nodes=train_config.num_nodes,
        precision=train_config.precision,
        logger=False,
        callbacks=[train_callback(train_config)],
        max_epochs=train_config.epoch_count,
        check_val_every_n_epoch=TrainingArgs.check_val_every_n_epoch,
        num_sanity_val_steps=TrainingArgs.num_sanity_val_steps,
        log_every_n_steps=TrainingArgs.log_every_n_steps,
        enable_checkpointing=TrainingArgs.enable_checkpointing,
        accumulate_grad_batches=TrainingArgs.accumulate_grad_batches,
        gradient_clip_val=TrainingArgs.gradient_clip_val,
    )

    train_data = get_data_by_l_version(trainer)
    trainer.fit(model, train_data)

if __name__ == "__main__":
    config_root = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "../../configs/prepare")
    config_root = os.path.abspath(config_root)
    main(config_root)
