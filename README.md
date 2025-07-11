# rwkv-peft-simple: A Refined Framework for RWKV Training and PEFT

`rwkv-peft-simple` is a streamlined, user-friendly, and highly modular framework for training and fine-tuning RWKV models. It places a strong emphasis on **clarity, modularity, and ease of use**, providing a clean and well-structured codebase ideal for both research and practical applications. The framework is built on PyTorch Lightning to abstract away boilerplate training code and supports various Parameter-Efficient Fine-Tuning (PEFT) methods like LoRA and PiSSA out of the box.

## Core Philosophy

This project is not just a fork; it is a **conceptual rebuild** of the RWKV training ecosystem. The guiding principles are:
- **Modularity**: Every component (data loading, model architecture, training logic, PEFT methods) is isolated in its own dedicated module with clear interfaces.
- **Clarity over Cuteness**: The code prioritizes readability and clear intent over overly complex or "clever" implementations.
- **Configuration-Driven**: All aspects of training, from model size to learning rates to PEFT strategies, are controlled via simple `.toml` configuration files.
- **Extensibility**: The modular design makes it easy to add new PEFT methods, model components, or data loaders without modifying the core training pipeline.

---

## Architecture Overview

The framework follows a clear, layered architecture, with dependencies flowing from the high-level application layer down to the low-level computational kernels.

```mermaid
graph TD
    subgraph " "
        direction LR
        A["Application Layer<br>(src/bin)"]
        B["Training Control Layer<br>(src/training_loop)"]
        C["Model & Framework Interface<br>(src/model)"]
        D["Data Loading & Processing<br>(src/datasets)"]
        E["Configuration Layer<br>(src/configs)"]
        F["Utilities & Offline Tools<br>(src/utils)"]
        G["PEFT Weight Merging<br>(src/merge)"]
    end

    A --> B
    A --> C
    B --> C
    C --> D
    C --> F
    B --> E
    C --> E

    style A fill:#cde4ff,stroke:#8ab4f8
    style B fill:#d2e3fc,stroke:#8ab4f8
    style C fill:#d2e3fc,stroke:#8ab4f8
    style D fill:#e8f0fe,stroke:#8ab4f8
    style E fill:#e8f0fe,stroke:#8ab4f8
    style F fill:#e8f0fe,stroke:#8ab4f8
    style G fill:#e8f0fe,stroke:#8ab4f8
```

### Module Descriptions

- **`src/configs`**: **Centralized Configuration**. Defines simple, global objects that hold all configuration parameters loaded from `.toml` files.
- **`src/datasets`**: **Data Loading & Processing**. Handles loading data for pre-training and supervised fine-tuning (SFT). Features an efficient binary data loader (`binidx.py`) for large datasets.
- **`src/utils`**: **Utilities**. Contains a runtime tokenizer and sampling engine (`tokenizer.py`) for inference, and an offline tool (`json2binidx_tool`) to convert raw `.jsonl` data into the required binary format.
- **`src/model`**: **The Core Model Engine**. This is the heart of the project. It defines the RWKV architecture (`rwkv7`), provides a dynamic computation backend (`operator`), implements PEFT strategies (`peft`), and integrates with PyTorch Lightning (`light_rwkv.py`).
- **`src/training_loop`**: **Training Process Control**. Contains PyTorch Lightning Callbacks that manage the training dynamics, including learning rate scheduling, logging (to console, files, and W&B), and model checkpointing.
- **`src/bin`**: **High-Level Entrypoints**. Provides the main executable scripts, `train.py` and `eval.py`, which parse configs and launch the training or evaluation process.
- **`src/merge`**: **PEFT Weight Merging**. A collection of command-line scripts to merge the trained PEFT weights (e.g., LoRA) back into the base model to create a single, deployable checkpoint.
- **`src/infering_loop`**: **Inference Engine**. A standalone, lightweight inference engine (`inferer.py`) that is decoupled from the training framework, designed for efficient text generation.

---

## How to Run

### 1. Data Preparation

Before training, your dataset must be in `.jsonl` format (one JSON object per line, with a "text" key). Use the provided tool to convert it to the required `.bin`/`.idx` format.

```bash
# Example for a standard RWKV model
python src/utils/json2binidx_tool/tools/preprocess_data.py \
  --input /path/to/your/dataset.jsonl \
  --output-prefix /path/to/your/dataset \
  --vocab src/utils/json2binidx_tool/20B_tokenizer.json \
  --dataset-impl mmap \
  --tokenizer-type HFTokenizer \
  --append-eod
```

### 2. Configuration

All training settings are managed in `.toml` files located in the `configs/` directory. You can create a new configuration set (e.g., `configs/my_experiment/`) or modify an existing one (e.g., `configs/lora/`).

- **`file.toml`**: Specifies file paths for data, output directories, etc.
- **`model.toml`**: Defines the model architecture (layers, embedding size) and PEFT strategy.
- **`train.toml`**: Controls the training process (learning rate, batch size, epochs, etc.).

### 3. Launching Training

Use the `train.py` script to start a training run. You must specify the configuration directory name.

```bash
# Example for launching a LoRA fine-tuning run
python src/bin/train.py lora
```

The script will automatically:
1.  Load the configuration from `configs/lora/`.
2.  Instantiate the model and apply the LoRA configuration.
3.  Set up the data loader.
4.  Initialize the PyTorch Lightning `Trainer` with the appropriate callbacks.
5.  Start the training process.

Logs and checkpoints will be saved to the output directory specified in `file.toml`.

### 4. Merging PEFT Weights

After fine-tuning, you can merge the LoRA weights back into the base model.

```bash
# Example for merging LoRA weights
python src/merge/merge.py \
  --lora_path /path/to/your/lora_checkpoint.pth \
  --base_model_path /path/to/base/model.pth \
  --output_path /path/to/merged_model.pth
```

This will produce a single `merged_model.pth` file that can be used for deployment and inference.

<h1 align="center">
  <p><img src="assert/logo.jpg" alt="RWKV-PEFT" width="60px"  style="vertical-align: middle; margin-right: 10px;"/>RWKV-PEFT</p>
</h1>

\[ English | [中文](README_zh.md) \]

RWKV-PEFT is the official implementation for efficient parameter fine-tuning of RWKV models, supporting various advanced fine-tuning methods across multiple hardware platforms.

# Recent updates
## Support v7 & Code adjustment
 - 1.Removed `--fla` and added `--op cuda/fla/triton`. In RWKV7, you can choose from three different operators, with CUDA recommended by default. If you want to fine-tune using state tuning, please enable `--op fla` and set `--train_type state`.
 - 2.Renamed Bone to DiSHA(Note: The rank parameter in DiSHA is only half of that in LoRA. Under the same parameter setting, DiSHA(r) = 2 * LoRA(r)).:  
``` disha_config='{"mode":"bone","load":"","r":64}' ```  
You can still choose either `bone` or `bat` in the `mode` field.
- 3.The model code is now clearer and easier to migrate. Check the `rwkvt` file for details.
- 4.Removed the basic visualization training. A dedicated program will support visualization training in the future.
- 5.Added lr_schedule, with cos_decay as the default. You can also use cosine annealing by setting --lr_schedule wsd.
``` --my_testing "x070" ```
## SFT
Relevant parameters, detailed usage reference: scripts/run_sft.sh  
- data_file 'meta-math/MetaMathQA' #You can directly choose the Hugging Face path, or you can choose your own JSON path.  
- data_type sft #Select data type  
- sft_field query response #Perform retrieval based on the question-and-answer format in the JSON.  
- sft_split "train" #Set the number of data to load: "train" loads all the data, while "train[:1000]" loads only the first 1000 samples.  
```
--data_type sft --sft_field query response --sft_split "train"
```
## Specific settings for SFT
### RWKV-PEFT/src/rwkv_datasets/SFTdataset.py
```
tokenizer_path = 'RWKV/rwkv-5-world-3b' #Choose a tokenizer (select the official tokenizer)
IGNORE_INDEX = -100 #Padding (do not modify)
EOT_TOKEN = "\x17" #Set the stop token(s) you need

# Modify the corresponding prompt according to your requirements
PROMPT = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    )
```
> [!TIP]
> Downloading Hugging Face data may time out in China, so you need to add:   
>```HF_ENDPOINT="https://hf-mirror.com" sh scripts/run_sft.sh```

## DiSHA: Dimension-Sharding Adaptation of Large Language Models with Fast Convergence and Fast Computation [Paper](https://arxiv.org/pdf/2409.15371)
The paper has been updated. DiSHA(Bone) is now a simple and efficient basic PEFT method that is faster and uses less VRAM than LoRA, converges faster, and performs better than PiSSA. 
scripts:  
DiSHA(Bone):``` disha_config='{"mode":"bone","load":"","r":64}' ``` 
DiSHA(Bat):``` disha_config='{"mode":"bat","load":"","r":64}' ```


# Installation

> [!IMPORTANT]
> Installation is mandatory.

```bash
git clone https://github.com/JL-er/RWKV-PEFT.git
cd RWKV-PEFT
pip install -r requirements.txt
```

## Web Run
> [!TIP]
> Coming Soon!

## Table of Contents
- [Hardware Requirements](#hardware-requirements)
- [Quick Start](#quick-start)
- [Main Features](#main-features)
- [Detailed Configuration](#detailed-configuration)
- [GPU Support](#gpu-support)
- [Citation](#citation)

## Hardware Requirements

### RWKV-7 Models

Below is the RWKV-7 model fine-tuned video memory requirement data, tested with RTX 4090 (24GB video memory) + 64GB RAM, based on the following parameter configurations:

- Training precision: BF16
- `--strategy deepspeed_stage_1`
- `--ctx_len 1024`
- `--micro_bsz 1`
- `--lora_r 64` or `disha_config='{"mode":"bone","r":32}'`

| Model Parameters | State Tuning | LoRA | DiSHA | PiSSA |
|------------------|--------------|------|-------|-------|
| RWKV7-0.1B       | 2.6 GB       | 2.7 GB  | 2.7 GB   | 2.6 GB   |
| RWKV7-0.4B       | 3.1 GB       | 3.4 GB  | 3.1 GB   | 3.4 GB   |
| RWKV7-1.5B       | 5.3 GB       | 5.6 GB  | 5.6 GB   | 5.6 GB   |
| RWKV7-3B         | 8.2 GB       | 8.8 GB  | 8.8 GB   | 8.8 GB   |

<details>
<summary>🔍 <b>Click to view the VRAM requirements for quantized training of RWKV-7 models</b> </summary>

### INT8 VRAM Requirements

| Model Parameters | State Tuning | LoRA | DiSHA | PiSSA |
|------------------|--------------|------|-------|-------|
| RWKV7-0.1B       | 2.4 GB       | 2.5 GB  | 2.5 GB   | 2.5 GB   |
| RWKV7-0.4B       | 2.9 GB       | 2.9 GB  | 2.9 GB   | 3.0 GB   |
| RWKV7-1.5B       | 4.1 GB       | 4.6 GB  | 4.5 GB   | 4.6 GB   |
| RWKV7-3B         | 5.7 GB       | 6.7 GB  | 6.7 GB   | 6.7 GB   |

### NF4 VRAM Requirements

| Model Parameters | State Tuning | LoRA | DiSHA | PiSSA |
|------------------|--------------|------|-------|-------|
| RWKV7-0.1B       | 2.5 GB       | 2.4 GB  | 2.4 GB   | 2.4 GB   |
| RWKV7-0.4B       | 2.8 GB       | 2.7 GB  | 2.7 GB   | 2.7 GB   |
| RWKV7-1.5B       | 3.7 GB       | 3.9 GB  | 3.9 GB   | 3.9 GB   |
| RWKV7-3B         | 4.7 GB       | 5.7 GB  | 5.7 GB   | 5.7 GB   |

</details>

<details>
<summary>🔍 <b>Click to view the VRAM requirements of RWKV-6 models</b> </summary>


The following shows memory usage when using an RTX 4090 (24GB VRAM) + 64GB RAM (with parameters: `--strategy deepspeed_stage_1 --ctx_len 1024 --micro_bsz 1 --lora_r 64`):

|   Model Size   | Full Finetuning | LoRA/PISSA | QLoRA/QPISSA | State Tuning |
|---------------|-----------------|------------|--------------|--------------|
| RWKV6-1.6B    | OOM            | 7.4 GB      | 5.6 GB        | 6.4 GB        |
| RWKV6-3B      | OOM            | 12.1 GB     | 8.2 GB        | 9.4 GB        |
| RWKV6-7B      | OOM            | 23.7 GB*    | 14.9 GB**     | 18.1 GB       |

Note:
* OOM when batch size is 8
** Requires 19.5GB VRAM when batch size is 8

</details>

## Quick Start

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run example script:
```bash
sh scripts/run_lora.sh
```
Note: Please refer to the RWKV official tutorial for detailed data preparation


## Main Features

- **Multiple Fine-tuning Methods**: Supports LoRA, PISSA, Bone, State Tuning, etc.
- **Quantized Training**: Supports INT8/NF4 quantization for significant VRAM reduction
- **Flexible Data Loading**: Supports various data sampling strategies 
- **Memory Optimization**: Multiple DeepSpeed strategies available
- **Loss Masking**: Supports loss masking for QA dialogue and padding
- **Infinite Context Training**: Supports infctx training mode, utilizing RWKV's constant memory usage advantage to train with "infinite" context under limited resources
- **Multi-Hardware Support**: RWKV-PEFT officially supports NVIDIA, AMD, Moore Threads, Musa, Iluvatar CoreX, and other hardware platforms. Ascend NPU implementation will be available later. Note: Currently we only support issues for NVIDIA hardware
- **RWKV-FLA Efficient Training**: rwkv-fla is a Triton-based linear attention operator that can run efficiently on hardware without CUDA support

## Detailed Configuration

### 1. PEFT Method Selection
```bash
--peft disha --disha_config $disha_config
```

### 2. Training Parts Selection
```bash
--train_parts ["time", "ln"]
```
- Available parts: emb, head, time, ln
- Default training: time, ln (small parameter ratio)

### 3. Quantized Training
```bash
--quant int8/nf4
```

### 4. Infinite Length Training (infctx)
```bash
--train_type infctx --chunk_ctx 512 --ctx_len 2048
```
- ctx_len: Target training length
- chunk_ctx: Slice length, must be smaller than ctx_len

### 5. Data Loading Strategy
```bash
--dataload pad
```
- get: Default random sampling (RWKV-LM style)
- pad: Fixed-length padding sampling
- only: Single data sampling (only supports bsz=1)

### 6. DeepSpeed Strategy
```bash
--strategy deepspeed_stage_1
```
Available strategies:
- deepspeed_stage_1: Preferred option
- deepspeed_stage_2/3: For large models or full fine-tuning
- deepspeed_stage_2_offload
- deepspeed_stage_3_offload

### 7. FLA Operator
By default, RWKV-PEFT uses custom CUDA kernels for wkv computation.
However, you can use `--op fla` to enable the Triton kernel:
```
--op fla
```

## GPU Support

- NVIDIA: CUDA
- Intel, Moore Threads, Musa, Iluvatar CoreX: FLA, which means you need to pass `--fla`
- Ascend: CANN (soon)

## Citation

If you find this project helpful, please cite our work:
```bib
@misc{kang2025dishadimensionshardingadaptationlarge,
      title={DiSHA: Dimension-Sharding Adaptation of Large Language Models with Fast Convergence and Fast Computation}, 
      author={Jiale Kang},
      year={2025},
      eprint={2409.15371},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2409.15371}, 
}