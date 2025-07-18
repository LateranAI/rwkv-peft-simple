"""
文件名: trainer.py
所属路径: src/training_loop

功能概述:
    定义 Lightning Callback `train_callback` 以及辅助函数 `my_save`、`generate_init_weight`。
    该模块负责在训练过程中动态调整学习率 / 权重衰减、记录日志、保存 checkpoint，
    并与 wandb 及 Streamlit frontend 进行简单对接 (通过 JSON 文件写 loss 曲线)。

关键函数/类:
    - my_save: 根据文件名策略将权重保存到本地或 S3。
    - train_callback: Lightning Callback，实现 batch/epoch 钩子以控制 LR 调度、记录指标、保存模型。
    - generate_init_weight: 在无预训练权重时生成并保存一次初始权重文件。

依赖:
    - PyTorch Lightning, torch
    - src.configs.*
    - src.model.trick.lrs for LR scheduler utilities
"""

import os, math, time, datetime, subprocess
from types import SimpleNamespace

import torch
from lightning_utilities.core.rank_zero import rank_zero_only
import lightning as pl
import json

from src.configs.file import file_config
from src.configs.model import model_config
from src.configs.train import train_config
from src.model.trick.lrs import wsd, cos_decay


def my_save(args, trainer, dd, ff):
    if "14b-run1" in ff:
        fn = ff.split("/")[-1]
        fff = "/dev/shm/" + fn
        torch.save(dd, fff)
        subprocess.Popen(f" aws s3 mv {fff} s3://rwkv-14b-4k/{fn} --quiet", shell=True)
    elif ("world/14b" in ff) or ("world/7b" in ff):
        aa = ff.split("/")[1]
        fn = ff.split("/")[-1]
        fff = f"/dev/shm/{aa}-{fn}"
        torch.save(dd, fff)
        subprocess.Popen(
            f" aws s3 mv {fff} s3://rwkv-world/{aa}-{fn} --quiet", shell=True
        )
    else:
        torch.save(dd, ff)


class train_callback(pl.Callback):
    def __init__(self, args):
        super().__init__()
        self.args = SimpleNamespace(
            **{**vars(model_config), **vars(train_config), **vars(file_config)}
        )

        self.loss_file = os.path.join(self.args.proj_dir, "loss_data.json")

        if (
            (not torch.distributed.is_available())
            or (not torch.distributed.is_initialized())
            or torch.distributed.get_rank() == 0
        ):
            if os.path.exists(self.loss_file):
                os.remove(self.loss_file)

    def write_data(self, loss_data, t_cost, kt_s):

        with open(self.loss_file, "a") as f:
            json.dump({"loss": float(loss_data), "t_cost": t_cost, "kt_s": kt_s}, f)
            f.write("\n")

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        args = self.args

        real_step = trainer.global_step + args.epoch_begin * args.epoch_steps

        w_step = args.warmup_steps
        lr = args.lr_init

        token_per_step = args.ctx_len * args.real_bsz

        if args.my_exit_tokens != 0:
            target_tokens = abs(args.my_exit_tokens)
            warmup_tokens = max(0, w_step * token_per_step)

            total_steps_eff = max(1, (target_tokens - warmup_tokens) // token_per_step)

            if args.lr_schedule == "wsd":
                lr_tmp = wsd(
                    args.lr_init,
                    args.lr_final,
                    current_step=real_step,
                    total_steps=total_steps_eff,
                    warmup_steps=w_step,
                )
            else:
                lr_tmp = cos_decay(
                    args.lr_init,
                    args.lr_final,
                    current_step=real_step,
                    total_steps=total_steps_eff,
                )

            if args.my_exit_tokens > 0:
                lr = lr_tmp
            else:
                lr = (lr + lr_tmp) / 2

            real_tokens = real_step * token_per_step
            if real_tokens - warmup_tokens >= target_tokens:
                if (trainer.is_global_zero) or ("deepspeed_stage_3" in args.strategy):
                    final_path = f"{args.proj_dir}/rwkv-final.pth"
                    my_save(args, trainer, pl_module.state_dict(), final_path)
                    print(f"\n✅ End of training. Model saved to: {final_path}\n")
                exit(0)

        elif w_step > 0 and trainer.global_step < w_step:
            lr = args.lr_init * (0.01 + 0.99 * trainer.global_step / w_step)

        wd_now = args.weight_decay

        for param_group in trainer.optimizers[0].param_groups:
            if param_group["weight_decay"] > 0:
                param_group["weight_decay"] = wd_now

            param_group["lr"] = lr * param_group["my_lr_scale"]

        trainer.my_lr = lr
        trainer.my_wd = wd_now

        if trainer.global_step == 0:
            if trainer.is_global_zero:
                trainer.my_loss_sum = 0
                trainer.my_loss_count = 0
                trainer.my_log = open(args.proj_dir + "/train_log.txt", "a")
                trainer.my_log.write(
                    f"NEW RUN {args.my_timestamp}\n{vars(self.args)}\n"
                )
                try:
                    print(f"\n{trainer.strategy.config}\n")
                    trainer.my_log.write(f"{trainer.strategy.config}\n")
                except:
                    pass
                trainer.my_log.flush()
                if len(args.wandb) > 0:
                    print("Login to wandb...")
                    import wandb

                    wandb.init(
                        project=args.wandb,
                        name=f"{args.vocab_size} ctx{args.ctx_len} L{args.n_layer} D{args.n_embd}"
                        + " "
                        + args.my_timestamp,
                        config=args,
                        save_code=False,
                    )
                    trainer.my_wandb = wandb

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        args = self.args
        token_per_step = args.ctx_len * args.real_bsz
        real_step = trainer.global_step + args.epoch_begin * args.epoch_steps

        loss = outputs["loss"]
        if int(args.devices) > 1:
            torch.distributed.all_reduce(loss, op=torch.distributed.ReduceOp.SUM)

        if trainer.is_global_zero:
            t_now = time.time_ns()
            kt_s = 0
            t_cost = 0
            try:
                t_cost = (t_now - trainer.my_time_ns) / 1e9
                kt_s = token_per_step / t_cost / 1000
                t_cost = 1.0 / t_cost
                self.log("REAL it/s", t_cost, prog_bar=True, on_step=True)
                self.log("Kt/s", kt_s, prog_bar=True, on_step=True)
            except:
                pass
            trainer.my_time_ns = t_now
            trainer.my_loss = loss * trainer.accumulate_grad_batches / int(args.devices)
            trainer.my_loss_sum += trainer.my_loss
            trainer.my_loss_count += 1
            trainer.my_epoch_loss = trainer.my_loss_sum / trainer.my_loss_count
            self.log("lr", trainer.my_lr, prog_bar=True, on_step=True)
            self.log("sum_loss", trainer.my_epoch_loss, prog_bar=True, on_step=True)
            self.log("loss", trainer.my_loss, prog_bar=True, on_step=True)

            if trainer.accumulate_grad_batches != None:
                args.avg_loss += trainer.my_loss / trainer.accumulate_grad_batches
                if (batch_idx + 1) % trainer.accumulate_grad_batches == 0:
                    if len(args.wandb) > 0:
                        lll = {
                            "loss": args.avg_loss,
                            "lr": trainer.my_lr,
                            "wd": trainer.my_wd,
                            "Gtokens": real_step * token_per_step / 1e9,
                        }
                        if kt_s > 0:
                            lll["kt/s"] = kt_s
                        trainer.my_wandb.log(lll, step=int(real_step))
                    self.write_data(args.avg_loss, t_cost, kt_s)
                    args.avg_loss = 0
            else:
                if len(args.wandb) > 0:
                    lll = {
                        "loss": trainer.my_loss,
                        "lr": trainer.my_lr,
                        "wd": trainer.my_wd,
                        "Gtokens": real_step * token_per_step / 1e9,
                    }
                    if kt_s > 0:
                        lll["kt/s"] = kt_s
                    trainer.my_wandb.log(lll, step=int(real_step))
                self.write_data(trainer.my_loss, t_cost, kt_s)

    def on_train_epoch_start(self, trainer, pl_module):
        args = self.args
        dataset = trainer.train_dataloader.dataset
        assert "MyDataset" in str(dataset)
        dataset.global_rank = trainer.global_rank
        dataset.real_epoch = int(args.epoch_begin + trainer.current_epoch)
        dataset.world_size = trainer.world_size

    def on_train_epoch_end(self, trainer, pl_module):
        args = self.args
        to_save_dict = {}
        if (trainer.is_global_zero) or ("deepspeed_stage_3" in args.strategy):
            if (
                args.epoch_save > 0 and trainer.current_epoch % args.epoch_save == 0
            ) or (trainer.current_epoch == args.epoch_count - 1):
                if args.data_type == "wds_img":
                    raw_dict = pl_module.state_dict()
                    for k in raw_dict:
                        if k.startswith("encoder.") or k.startswith("decoder."):
                            to_save_dict[k] = raw_dict[k]
                else:

                    to_save_dict = {
                        k.replace("model.", ""): v
                        for k, v in pl_module.state_dict().items()
                    }

                if args.train_type == "state":
                    peft_dict = {}
                    for name, state in to_save_dict.items():
                        if "state" in name:
                            peft_dict[name] = state
                    to_save_dict = peft_dict

                if args.peft != "none":
                    peft_dict = {}
                    for name, state in to_save_dict.items():
                        if len(args.load_model) == 0:
                            if "emb" in name or "head" in name or "ln" in name:
                                peft_dict[name] = state
                        for part in args.train_parts:
                            if part in name:
                                peft_dict[name] = state
                        if args.peft == "pissa" and ("lora" in name):
                            peft_dict[name] = state
                        elif args.peft in name:
                            peft_dict[name] = state

                    to_save_dict = peft_dict

                try:
                    my_save(
                        args,
                        trainer,
                        to_save_dict,
                        f"{args.proj_dir}/rwkv-{args.epoch_begin + trainer.current_epoch}.pth",
                    )
                except Exception as e:
                    print("Error\n\n", e, "\n\n")

        if trainer.is_global_zero:
            trainer.my_log.write(
                f"{args.epoch_begin + trainer.current_epoch} {trainer.my_epoch_loss:.6f} {math.exp(trainer.my_epoch_loss):.4f} {trainer.my_lr:.8f} {datetime.datetime.now()} {trainer.current_epoch}\n"
            )
            trainer.my_log.flush()

            trainer.my_loss_sum = 0
            trainer.my_loss_count = 0
            if (args.epoch_begin + trainer.current_epoch) >= args.my_exit:
                exit(0)


@rank_zero_only
def generate_init_weight(model, init_weight_name):
    mm = model.generate_init_weight()

    if model.args.my_pile_stage == 1:
        if len(model.args.load_model) > 0:
            print(f"Combine weights from {model.args.load_model}...")
            load_dict = torch.load(model.args.load_model, map_location="cpu")
            for k in load_dict:
                try:
                    assert k in mm
                except:
                    print("missing", k)
                    exit(0)
                src = load_dict[k]
                try:
                    mm[k] = src.reshape(mm[k].shape)
                except:
                    tmp = mm[k].squeeze().clone()
                    print(k, src.shape, "-->", mm[k].shape)
                    ss = src.shape[0]
                    dd = tmp.shape[0]
                    for i in range(dd):
                        pos = i / dd * ss
                        if pos >= ss - 1:
                            tmp[i] = src[ss - 1]
                        else:
                            p0 = int(math.floor(pos))
                            ii = pos - p0
                            tmp[i] = src[p0] * (1 - ii) + src[p0 + 1] * (ii)
                    mm[k] = tmp.reshape(mm[k].shape)
                    sss = src.squeeze().float().cpu().numpy()
                    print(sss[:10], "...", sss[-10:])
                    mmm = mm[k].squeeze().float().cpu().numpy()
                    print(mmm[:10], "...", mmm[-10:])

    print(f"Save to {init_weight_name}...")
    torch.save(mm, init_weight_name)

    if model.args.my_pile_stage == 1:
        print("Done. Now go for stage 2.")
        exit(0)
