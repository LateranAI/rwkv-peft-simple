# 项目重构计划 (REFACTOR_PLAN.md)

**版本:** 1.0
**日期:** 2024-07-29
**核心目标:** 提升代码库的现代化水平、可维护性和可测试性，同时保持功能与原始版本完全等价。

---

## **总览：三大重构方向**

经过对所有模块的全面审查，我们识别出三个主要的重构方向。这些方向将指导我们进行一系列具体的代码改进，使框架更健壮、更优雅、更符合现代Python软件工程的最佳实践。

1.  **中心化配置与依赖注入 (Centralized Config & Dependency Injection)**:
    *   **问题**: 当前存在两套并行的配置系统 (`src/configs` vs `src/training_loop/args_type.py`)，并且广泛使用全局单例模式，导致隐式依赖和可测试性差。
    *   **方案**: 统一为单一的、不可变的配置对象，并通过依赖注入模式在应用层中显式传递，彻底消除全局状态。

2.  **回归框架最佳实践 (Embrace Framework's Best Practices)**:
    *   **问题**: 项目中存在大量"重复造轮子"的现象，例如自定义的回调和手动的学习率调度，而 `PyTorch Lightning` 框架本身已提供更强大、更标准的解决方案。
    *   **方案**: 全面采用 `Lightning` 的原生功能，如 `ModelCheckpoint` 回调和标准的 `LRScheduler` 机制，移除自定义实现，使代码更"地道"(idiomatic)。

3.  **增强模块内聚性与职责分离 (Enhance Cohesion & Separation of Concerns)**:
    *   **问题**: 存在少数"上帝对象"(God Objects)，如 `train_callback` 和 `RWKV_x060_Lightning`，它们承担了过多的职责，违反了单一职责原则。
    *   **方案**: 将这些大而全的类拆分为更小的、职责单一的组件（如 `OptimizerFactory`），并优化模块间的耦合关系。

---

## **详细重构任务列表**

### **第一优先级：配置系统重构**

*   **任务 1.1: 统一配置对象**
    *   **文件**: 新建 `src/configs/main.py` (或类似名称)
    *   **操作**:
        *   定义一个顶层的、嵌套的 `dataclass AppConfig`，它包含 `FileConfig`, `ModelConfig`, `TrainConfig` 的实例。
        *   将该 `dataclass` 设置为 `frozen=True`，使其成为一个不可变对象，增加线程安全性。
        *   创建一个单一的加载函数 `load_app_config(config_dir: str) -> AppConfig`，它负责读取所有 `.toml` 文件并返回一个完整的 `AppConfig` 实例。
    *   **理由**: 建立唯一的配置真理来源。

*   **任务 1.2: 移除冗余配置**
    *   **文件**: `src/training_loop/args_type.py`
    *   **操作**: **彻底删除**该文件及其定义的 `TrainingArgs` 类。
    *   **理由**: 消除配置系统的二义性和代码冗余。

*   **任务 1.3: 应用依赖注入**
    *   **文件**: `src/bin/train.py` 及所有下游模块。
    *   **操作**:
        *   在 `train.py` 的 `main` 函数中调用 `load_app_config` 创建 `config` 实例。
        *   将 `config` 对象或其子对象 (`config.model`) **显式地**作为参数传递给所有需要它的类（如 `RWKV7LightningModule`）和函数。
        *   移除所有对旧的全局配置单例（`file_config`, `model_config`, `train_config`）的 `import` 和使用。
    *   **理由**: 将隐式依赖变为显式依赖，极大提升代码的可读性和可测试性。

### **第二优先级：训练循环与回调重构**

*   **任务 2.1: 实现标准学习率调度器**
    *   **文件**: 重命名 `src/model/trick/` 为 `src/model/optimizers/`，并在其中创建 `schedulers.py`。
    *   **操作**:
        *   将 `lrs.py` 中的逻辑封装到一个新的 `class MyCustomLRScheduler(torch.optim.lr_scheduler._LRScheduler)` 中。
        *   在 `RWKV7LightningModule` 的 `configure_optimizers` 中，返回这个新的调度器实例，并配置其更新间隔为 `"step"`。
        *   移除 `train_callback` 中 `on_train_batch_start` 的手动学习率更新逻辑。
    *   **理由**: 使用框架的标准机制，更稳定、更可预测。

*   **任务 2.2: 使用原生 `ModelCheckpoint`**
    *   **文件**: `src/training_loop/trainer.py` 和 `src/model/light_rwkv.py`。
    *   **操作**:
        *   在 `src/bin/train.py` 中，实例化并使用 `pytorch_lightning.callbacks.ModelCheckpoint`。
        *   移除 `train_callback` 中 `on_train_epoch_end` 的自定义保存逻辑和 `my_save` 辅助函数。
        *   如需实现"仅保存PEFT参数"的逻辑，应在 `RWKV7LightningModule` 中重写 `on_save_checkpoint` 钩子函数来实现。
    *   **理由**: 拥抱框架，减少维护成本，利用经过充分测试的健壮功能。

*   **任务 2.3: 拆分 `train_callback`**
    *   **文件**: `src/training_loop/trainer.py`
    *   **操作**:
        *   将 `train_callback` 重命名为 `LoggingCallback`。
        *   移除其所有与学习率调度和模型保存相关的代码，使其只专注于**日志记录**。
    *   **理由**: 遵循单一职责原则，使回调逻辑更清晰。

### **第三优先级：模型层与命名规范重构**

*   **任务 3.1: 为 `LightningModule` 减负**
    *   **文件**: `src/model/light_rwkv.py` 和 `src/model/optimizers/factory.py` (新建)。
    *   **操作**:
        *   创建 `OptimizerFactory` 类，将 `configure_optimizers` 中复杂的优化器和参数组（如分层学习率）的创建逻辑移入其中。
        *   `configure_optimizers` 现在只负责调用这个工厂类。
        *   将 `peft_loading.py` 的逻辑内联为 `RWKV7LightningModule` 的一个私有方法 `_init_peft_model`，简化调用栈。
    *   **理由**: 提升 `LightningModule` 的内聚性，使其更专注于训练步骤的定义。

*   **任务 3.2: 统一命名规范**
    *   **变量名**: 在所有非模型核心层的文件中，将模糊的缩写（如 `bsz`, `lr`, `ctx`）替换为更具描述性的全名（`batch_size`, `learning_rate`, `context_length`）。
    *   **类名**:
        *   `RWKV` -> `RWKVLightningModule` (当前命名过于通用，容易与 `RWKV7` 核心模型混淆，应明确其作为 Lightning 接口的职责)。
        *   `MyDataModule` -> `PretrainDataModule` / `SFTDataModule`
    *   **函数名**:
        *   `get_data_by_l_version` -> `create_datamodule`
    *   **理由**: 提升代码库的整体可读性和专业性。

*   **任务 3.3: 优化数据集模块**
    *   **文件**: `src/datasets/*.py`
    *   **操作**:
        *   在 `SFTDataModule` 中，使用**策略模式**（字典分发）来取代 `if/elif/else`，使其更容易扩展新的数据源。
        *   创建一个 `BaseDataModule`，将 `PretrainDataModule` 和 `SFTDataModule` 中共享的逻辑（如dataloader实例化）上移。
    *   **理由**: 提高代码的复用性和扩展性。

---

## **验收标准**

重构后的代码必须满足以下所有条件才能被接受：

1.  **功能等价**: 使用相同的配置和种子，训练的 `loss` 曲线以及最终模型的评估指标必须与原始版本严格一致（绝对误差 < 1e-7）。
2.  **代码精简**: 项目总代码行数应有明显下降。文件数量不增加，模块数量不增加。
3.  **测试通过**: 所有现有的单元测试和集成测试必须全部通过。
4.  **结构清晰**: 模块职责更分明，依赖关系更清晰，代码的可读性和可维护性得到显著提升。 