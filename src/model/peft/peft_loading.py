"""
文件名: peft_loading.py
所属路径: src/model/peft

功能概述:
    负责根据训练/推理配置加载基础 RWKV 模型并按需注入 PEFT (LoRA / DiSHA / PISSA / State-Tuning) 权重。

    主要流程:
        1. 解析全局 configs (train/file/model) 合并为 args。
        2. 创建模型 (RWKV or StateDecoder) 并根据 train_type / peft / quant 决定哪些参数可训练。
        3. 加载基础 checkpoint、PEFT checkpoint、量化权重, 并处理多种 edge case (形状不一致、strict=False 等)。
        4. 返回 (args, model) 供上层训练脚本使用。

    该模块在多 GPU 训练/推理环境中保证只在 rank-0 打印日志并安全写文件。
"""

import os
from types import SimpleNamespace

import torch
from src.model.light_rwkv import RWKV
from lightning_utilities.core.rank_zero import rank_zero_info, rank_zero_only

from src.model.state_decoder_discarded import StateDecoder
from src.training_loop.trainer import generate_init_weight
from src.model.peft.linear import LORA_CONFIG
from src.configs.train import train_config
from src.configs.file import file_config
from src.configs.model import model_config

# -------------------------------------------------------------
# util: 判断当前进程是否为 rank-0（分布式未初始化则视为 rank-0）
# -------------------------------------------------------------

def _is_rank_zero() -> bool:
    """检测当前进程是否为 rank-0 (在未初始化分布式时亦视为 rank-0)。"""
    return (not torch.distributed.is_available()) or (not torch.distributed.is_initialized()) or torch.distributed.get_rank() == 0

def load_peft_model():
    """按配置构建并加载 PEFT 模型

    返回:
        Tuple[args, nn.Module]: 合并后的配置对象及加载完成的模型实例。
    """
    args = SimpleNamespace(**(vars(train_config) | vars(file_config) | vars(model_config)))

    freeze = False
    if args.peft == 'lora':
        assert args.lora_config['lora_r'] > 0, "LoRA should have its `r` > 0"
        LORA_CONFIG["r"] = args.lora_config['lora_r']
        LORA_CONFIG["alpha"] = args.lora_config['lora_alpha']
        LORA_CONFIG["dropout"] = args.lora_config['lora_dropout']
        # LORA_CONFIG["parts"] = set(str(args.lora_config['lora_parts']).split(','))
    if args.peft == 'pissa':
        assert args.pissa_config['pissa_r'] > 0, "LoRA should have its `r` > 0"
        LORA_CONFIG["r"] = args.pissa_config['pissa_r']
        # LORA_CONFIG["parts"] = set(str(args.pissa_config['pissa_parts']).split(','))
    if args.quant != 'none':
        LORA_CONFIG["quant"] = True
    if args.peft == 'disha':
        from src.model.peft.linear import DiSHA_CONFIG
        DiSHA_CONFIG["mode"] = args.disha_config['mode']
        DiSHA_CONFIG["r"] = args.disha_config['r']

    if not "sd" in args.train_type:
        model = RWKV(args)
    else:
        model = StateDecoder(args)

    rank_zero_info(model)

    if args.train_type == 'state':
        model.requires_grad_(False)
        freeze = True
        for name, module in model.named_modules():
            for pname, param in module.named_parameters():
                if 'state' in pname:
                    param.requires_grad = True
            break
    if args.peft != 'none':
        model.requires_grad_(False)
        freeze = True
        if len(args.load_model) == 0:
            for name, module in model.named_modules():
                if any(n.startswith("emb.") for n, _ in module.named_parameters()):
                    for pname, param in module.named_parameters():
                        if 'emb.weight' == pname:
                            param.requires_grad = True
                if any(n.startswith("head.") for n, _ in module.named_parameters()):
                    for pname, param in module.named_parameters():
                        if 'head.weight' == pname:
                            
                            param.requires_grad = True
                if 'ln' in name:
                    for param in module.parameters():
                        param.requires_grad = True
                break

        for name, module in model.named_modules():  # part train
            for pname, param in module.named_parameters():
                for part in args.train_parts:
                    if part in pname:
                        
                        param.requires_grad = True
            break

        if args.peft == 'lora' or args.peft == 'pissa':
            rank_zero_info(f'  {args.peft} additionally training module {name}')
            for name, module in model.named_modules():
                if any(n.startswith("lora_") for n, _ in module.named_parameters()):
                    for pname, param in module.named_parameters():
                        param.requires_grad = 'lora_' in pname
        if args.peft == 'disha':
            for name, module in model.named_modules():
                for pname, param in module.named_parameters():
                    if 'disha' in pname:
                        param.requires_grad = True
                break

    if len(args.load_model) == 0 or args.my_pile_stage == 1:  # shall we build the initial weights?
        init_weight_name = f"{args.proj_dir}/rwkv-init.pth"
        generate_init_weight(model, init_weight_name)  # save initial weights
        args.load_model = init_weight_name
    else:
        rank_zero_info(f"########## Loading {args.load_model}... ##########")
        state_dict = torch.load(args.load_model, map_location="cpu", weights_only=True)
        new_state_dict = {f"model.{k}": v for k, v in state_dict.items()}
        # 若检查点中的 emb.weight 与当前模型形状不一致，删除该键以避免 shape mismatch
        for key in list(new_state_dict.keys()):
            if key.endswith('emb.weight'):
                if key in model.state_dict() and new_state_dict[key].shape != model.state_dict()[key].shape:
                    rank_zero_info(f"[load_peft_model] 跳过形状不匹配的 {key}: checkpoint {tuple(new_state_dict[key].shape)} != model {tuple(model.state_dict()[key].shape)}")
                    del new_state_dict[key]
        try:
            # 默认按照 freeze 参数决定 strict 行为
            model.load_state_dict(new_state_dict, strict=(not freeze))
        except RuntimeError as e:
            # 当出现多余 / 缺失权重时，回退到非严格模式继续加载
            rank_zero_info(
                f"[load_peft_model] 非严格加载回退: {e}\n将使用 strict=False 重新加载 state_dict。"
            )
            model.load_state_dict(new_state_dict, strict=False)
        # model.load_state_dict(torch.load(
        #     args.load_model, map_location="cpu", weights_only=True), strict=(not freeze))

    # Load peft checkpoint
    # multi-GPU training
    if args.peft == 'disha':
        if os.path.isfile(args.disha_config['load']):
            model.load_state_dict(torch.load(args.disha_config['load'], map_location="cpu", weights_only=True),
                                  strict=False)
    elif args.peft == 'lora':
        if os.path.isfile(args.lora_config['lora_load']):
            model.load_state_dict(torch.load(args.lora_config['lora_load'], map_location="cpu", weights_only=True),
                                  strict=False)
    elif args.peft == 'pissa':
        if int(args.devices) == 1 and os.path.isfile(f'{args.proj_dir}/init_pissa.pth'):
            assert os.path.isfile(f'{args.proj_dir}/init_pissa.pth') == False
        if os.path.isfile(f'{args.proj_dir}/init_pissa.pth') and int(args.devices) > 1 and args.pissa_config['pissa_load'] == "":
            pissa_init = torch.load(
                f'{args.proj_dir}/init_pissa.pth', map_location="cpu", weights_only=True)
            rank_zero_info(f"########## Load Init PISSA... ##########")
            for name, m in model.named_modules():
                if hasattr(m, "pissa_load") and callable(getattr(m, "pissa_load")):
                    m.pissa_load(
                        pissa_init[f'{name}.init_lora_A'], pissa_init[f'{name}.init_lora_B'])

        if args.pissa_config['pissa_load'] == "" and not os.path.isfile(f'{args.proj_dir}/init_pissa.pth'):
            init_dict = {}
            rank_zero_info(f"########## Init PISSA... ##########")
            for name, m in model.named_modules():
                if hasattr(m, "pissa_init") and callable(getattr(m, "pissa_init")):
                    m.pissa_init(args.pissa_config['svd_niter'])
                    init_dict[f'{name}.init_lora_A'] = m.lora_A.data
                    init_dict[f'{name}.init_lora_B'] = m.lora_B.data
            if _is_rank_zero():
                torch.save(init_dict, f'{args.proj_dir}/init_pissa.pth')
        if os.path.isfile(args.pissa_config['pissa_load']):
            model.load_state_dict(torch.load(args.pissa_config['pissa_load'], map_location="cpu", weights_only=True),
                                  strict=False)
            pissa_init = torch.load(
                args.pissa_config['pissa_init'], map_location="cpu", weights_only=True)
            rank_zero_info(f"########## Load PISSA... ##########")
            for name, m in model.named_modules():
                if hasattr(m, "pissa_load") and callable(getattr(m, "pissa_load")):
                    m.pissa_load(
                        pissa_init[f'{name}.init_lora_A'], pissa_init[f'{name}.init_lora_B'])

    if args.quant != 'none':
        rank_zero_info(f"########## Quant... ##########")
        for name, m in model.named_modules():
            if hasattr(m, "quant") and callable(getattr(m, "quant")):
                m.quant(args.quant)

    return args, model