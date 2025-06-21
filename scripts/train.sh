source /public/home/ssjxzkz/Projects/rwkv-peft-simple/.venv/bin/activate
cd /public/home/ssjxzkz/Projects/rwkv-peft-simple
export PYTHONPATH="/public/home/ssjxzkz/Projects/rwkv-peft-simple:$PYTHONPATH"
export WANDB_MODE=offline

python /public/home/ssjxzkz/Projects/rwkv-peft-simple/src/bin/train.py configs/prepare