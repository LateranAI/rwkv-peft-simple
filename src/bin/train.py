import os
import warnings
import datetime
import numpy as np
import torch
import lightning as pl
from lightning import Trainer

from src.configs.configs import file_config
from src.configs.configs import model_config
from src.configs.configs import train_config
from src.rwkvt.peft.peft_loading import load_peft_model
from src.rwkvt.lightning_train.trainer import train_callback
from src.datasets.dataset_pt import get_data_by_l_version

if "deepspeed" in train_config.strategy:
    import deepspeed


def main():
    np.set_printoptions(precision=4, suppress=True, linewidth=200)
    warnings.filterwarnings("ignore", ".*Consider increasing the value of the `num_workers` argument*")
    warnings.filterwarnings("ignore", ".*The progress bar already tracks a metric with the*")

    file_config.check(); model_config.check(); train_config.check()
    file_config.show(); model_config.show(); train_config.show()

    train_config.my_timestamp = datetime.datetime.today().strftime("%Y-%m-%d-%H-%M-%S")
    train_config.real_bsz = int(train_config.num_nodes) * int(train_config.devices) * train_config.micro_bsz

    args, model = load_peft_model()

    trainer = Trainer(
        accelerator=train_config.accelerator,
        strategy=train_config.strategy,
        devices=train_config.devices,
        num_nodes=train_config.num_nodes,
        precision=train_config.precision,
        logger=False,
        callbacks=[train_callback(train_config)],
        max_epochs=train_config.epoch_count,
        check_val_every_n_epoch=train_config.check_val_every_n_epoch,
        num_sanity_val_steps=train_config.num_sanity_val_steps,
        log_every_n_steps=train_config.log_every_n_steps,
        enable_checkpointing=False,
        accumulate_grad_batches=train_config.accumulate_grad_batches,
        gradient_clip_val=train_config.gradient_clip_val,
    )

    train_data = get_data_by_l_version(trainer)
    trainer.fit(model, train_data)


if __name__ == "__main__":
    main()
