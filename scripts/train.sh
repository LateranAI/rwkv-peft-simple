source ~/Projects/rwkv-peft-simple/.venv/bin/activate
cd ~/Projects/rwkv-peft-simple
export PYTHONPATH="~/Projects/rwkv-peft-simple:$PYTHONPATH"
export WANDB_MODE=offline

python ~/Projects/rwkv-peft-simple/src/bin/train.py configs/pretrain