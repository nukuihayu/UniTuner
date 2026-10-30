# 训练器使用教程

[English](../en/training/README.md) · [第一次训练](../getting-started.md) · [参数手册](parameters.md)

图像训练按本页操作；Wan 与 MiniMax-H3 的数据准备、缓存和预览见[视频训练教程](video.md)。

<details>
<summary>本页导航</summary>

- [选择训练方法](#选择训练方法)
- [流匹配：从数据到训练](#流匹配从数据到训练)
- [图像修复与中间时间](#图像修复与中间时间)
- [选择训练层](#选择训练层)
- [参数如何配合](#参数如何配合)
- [公共训练参数](#公共训练参数)
- [执行策略](#执行策略)
- [训练预览](#训练预览)
- [保存与恢复](#保存与恢复)
- [多数据集训练](#多数据集训练)
- [独立命令配置、统一提示词与 cuDNN](#独立命令配置统一提示词与-cudnn)
- [推理与发布](#推理与发布)
- [常见填写问题](#常见填写问题)

</details>

## 选择训练方法

先区分训练目标和更新范围：

| 选择 | 配置 | 用途 |
| --- | --- | --- |
| 图像/视频生成微调 | `method.kind = "flow-matching"` | 主体、风格、编辑和视频条件学习 |
| 一步图像修复 | `method.kind = "restoration"` | 已对齐 LQ/HQ，或主动合成 LQ |
| 适配器 | `train.parameter_mode = "adapter"` + `[train.adapter]` | 输出 LoRA／LoHa／LoKr |
| 全参数 | `train.parameter_mode = "full"`，删除 `[train.adapter]` | 输出训练后的基模，显存和磁盘需求更高 |

例如“修复 LoRA”就是 restoration + adapter，两项独立选择。模型是否支持该任务见[模型表](models.md#模型选择)。

## 流匹配：从数据到训练

1. 从[示例](../examples/README.md)选择准确家族和变体，替换模型及数据路径。
2. 在同一 TOML 填 `[model]`、`[dataset.groups.NAME]`、`[train]`、`[method]`、`[execution]`、`[output]`。
3. 需要观察效果时添加 `[preview]`；设置固定提示词和种子。
4. 按顺序运行校验、缓存、训练。

```bash
uv run --no-dev unituner validate train.toml
uv run --no-dev unituner cache train.toml --device cuda
uv run --no-dev unituner train train.toml --device cuda
```

缓存命令只校验编码所需的参数；不会要求训练专用的优化器、LoRA 或损失参数。单独缓存可复制 image-cache、wan-cache 或 minimax-h3-cache 示例。已有缓存且不需要预览的训练可不加载 VAE/文本编码器；图像修复仍需实时读取图像和 VAE。

## 图像修复与中间时间

复制[修复教程配置](../examples/tutorial-restoration.toml)，用 source 指向 HQ、low_quality_source 指向同名 LQ，删除不需要的合成退化表。固定时间时填写 midtimestep 并保持 calibrate=false；自动校准时填写 calibrate=true，不再指定 sigma。具体填写见[修复教程](../restoration-tutorial.md)。

## 选择训练层

```toml
[train]
include_layers = ["layers.*"]
exclude_layers = ["layers.0.*"]
```

上例使用 Z-Image 原生命名。其他家族先核对其模块名称，再填写 glob；不是 ComfyUI 导出 LoRA 键。空 include_layers 选择所有合格目标，exclude_layers 在其后排除。通常先不限制，确认流程后再做局部实验。

full 使用相同的选择入口；它决定哪些底模参数可训练。adapter 用它限制适配器安装目标。匹配不到合格目标时应修正模式，不要改成无关的导出前缀。

## 参数如何配合

### 有效更新与批量

单组、完整批量时，有效批量约为 batch_size × accumulation_steps。尾批可能不足，多组 batch_size 也可不同，因此实际更新按消耗样本数归一化。日志中的 update 只在优化器成功更新后增加；修复 GAN 的生成器/判别器属于同一个逻辑更新。

将 accumulation_steps 从 1 改成 4 会让一次 update 包含四个 microbatch；不是免费把训练步数扩大四倍。保存、预览、学习率预热和 max_updates 都按有效更新计数。

### 学习率与预热

```toml
[train]
learning_rate = 5e-5
weight_decay = 0.01
max_updates = 10000
accumulation_steps = 4
max_grad_norm = 1.0
lr_schedule = "cosine"
warmup_updates = 100

[train.optimizer]
betas = [0.9, 0.999]
eps = 1e-8
backend = "auto"
```

learning_rate 为基础幅度；warmup_updates 控制前期线性升温，必须小于 max_updates。constant 在预热后保持学习率，linear/cosine 在总更新范围内衰减。权重衰减和梯度裁剪通常先保持默认，出现不稳定时结合分损失和梯度范数定位。

### LoRA、LoHa、LoKr

```toml
[train.adapter]
algorithm = "lora"
rank = 32
alpha = 32.0
```

普通 LoRA 使用 alpha/rank 缩放。先保持两者相等，单独比较 rank 的容量影响；切换算法后使用新 run，不直接将旧优化器状态当作新算法续训。推理配置必须与导出算法、rank、alpha 一致。

## 公共训练参数

| 优先级 | 必须理解的字段 | 填写建议 |
| --- | --- | --- |
| 路径 | model.weights/vae/text_encoder，source，cache_directory | 使用已存在的绝对路径更容易排错 |
| 任务 | model.family/variant/version，method.kind，parameter_mode | 与实际权重及目标任务匹配 |
| 数据量 | resolution、batch_size、repeats | 尺寸影响激活开销；repeats 调整组的相对频率 |
| 优化 | learning_rate、max_updates、accumulation_steps | 先短跑，再根据预览增加预算 |
| 容量 | algorithm、rank、alpha | 保持其他变量不变进行对照 |
| 观察 | preview、tensorboard | 固定预览条件，同时看总损失和分损失 |
| 产物 | name、save_every_updates | 保存成本高时增大周期 |

每个字段的完整默认值、范围和可选值见[填写手册](parameters.md)。默认参数一般无需重复写入 TOML；详细示例展开它们用于查阅和调参。

## 执行策略

推荐先显式确认计算精度，再处理显存：CUDA BF16 配方用 precision=bf16，CPU 测试用 fp32。auto 按权重推断；量化底模默认可能解析为 BF16。

显存不足依次比较小 batch、block 重算、冻结权重卸载或 adapter 底模量化。full 不使用冻结底模量化。编译和 cuDNN benchmark 都有启动成本，用稳定阶段的耗时比较，见[执行教程](execution.md)。

## 训练预览

图像流匹配使用 `[preview]` + `[[preview.samples]]`。修复使用 `[preview] num_images`，从真实数据轮换生成 LQ/恢复/HQ 对比，不填写图像生成 samples。Wan/H3 分别使用 preview.videos 和 preview.audio_video，复制对应 preview 示例。

every_updates 是有效更新周期；末步始终预览。at_start=true 仅在从头训练时生成基线。预览保持训练 RNG 和数据游标，不改变后续训练样本顺序。

预览和保存周期可不同。例如每 100 步观察、每 1000 步保存；预览慢时先降低图像数或频率，而不是改变训练批量。

## 保存与恢复

```toml
[output]
name = "portrait-lora"
save_every_updates = 1000
tensorboard = "runs/tensorboard"
# directory = "runs/portrait-lora"
# max_resume_checkpoints = 1
# max_weight_checkpoints = 0
# save_precision = "fp32"
```

省略 directory 自动生成 `runs/portrait-lora_<时间戳>/`。显式目录会复用；重新从头训练会替换同名旧产物，使用 `--resume` 才保留并继续训练状态。

| 位置 | 内容 | 如何使用 |
| --- | --- | --- |
| weights/NAME-00001000.safetensors | 第 1000 步 LoRA 或基模 | 比较不同训练版本、推理或分发 |
| weights/NAME.safetensors | 最近一次正式权重 | 固定推理路径，随保存更新 |
| weights/NAME-00001000.encoder.safetensors | 可选修复 encoder adapter | 与同一步正式权重配套使用 |
| resume/NAME.state-00001000.safetensors | 模型/方法、优化器、RNG、数据游标 | 只用于 --resume |
| logs/、previews/ | 完整指标与预览 | 排错和选择训练版本 |

每次达到保存周期及末步同时保存权重和独立 resume。默认只留最近一份 resume，正式权重全部保留。两个 max_* 参数设正整数保留最近 N 份/组，0 不限；encoder 随权重组清理。完整保存成功后才删除旧版本。

```bash
uv run --no-dev unituner train train.toml --device cuda --resume runs/portrait-lora/resume/portrait-lora.state-00001000.safetensors
```

续训到 max_updates 总数。保留相同底模与数据；影响数值的配置会进行身份校验。save_precision 仅控制正式权重导出，不降低续训文件精度。磁盘紧张时可研究 checkpoint_storage=frozen-reference，它恢复时要求完全相同的冻结底模。

## 多数据集训练

一个训练配置可包含多组数据。公共尺寸、batch 和 prompt 放在 `[dataset.defaults]`，每个 `[dataset.groups.NAME]` 覆盖自己的路径、重复数或提示词。每组必须有独立缓存；名称只使用 ASCII 字母、数字、下划线且不能以数字开头。修复组必须共用校准尺寸。

完整写法和混合频率示例见[数据集填写教程](datasets.md)及 [data-groups.toml](../examples/data-groups.toml)。

## 独立命令配置、统一提示词与 cuDNN

cache 只需要编码资产和数据；mid-timestep 只需要 VAE、配对图片和校准选项；generate/restore 使用独立推理配置。它们不复用完整训练校验。也可不提供 TOML，使用对应命令的 CLI 字段；用 `--help` 查看字段，`--set` 用于任意嵌套键。

统一提示词填 dataset.defaults.prompt 或 group.prompt。cuDNN/TF32 控制放在 `[runtime]`，可以由缓存、训练与推理各自设置。CUDA 卷积默认启用 cudnn_benchmark=true；需要重复性时使用 cudnn_deterministic=true，并通常关闭 benchmark，见[执行策略](execution.md)。

### TensorBoard

只填 output.tensorboard 根目录。省略 output.directory 时 run 使用训练名和时间戳；显式目录使用其固定 run 名。恢复训练接着记录，不需另填 TensorBoard 名称。

```bash
tensorboard --logdir runs/tensorboard --host 0.0.0.0 --port 6006
```

常看 loss/main 或 loss/generator、修复各分损失、loss/discriminator、梯度范数、lr、每更新样本数、耗时与显存。总损失是带权目标，分损失按各自原始尺度显示；不同配方之间不要直接用总损失大小判断质量。

## 推理与发布

adapter 的正式权重使用对应模型的 ComfyUI 加载键；选择匹配基模的 LoRA 加载器。full 权重作为新底模。修复 encoder 是独立 VAE 增量，UniTuner restore 要与同轮 LoRA 一起填写。完整命令见[推理](../inference.md)。

## 常见填写问题

| 现象 | 先检查 |
| --- | --- |
| 字段 unknown/extra | 是否写错表路径，或把推理配置交给 validate |
| 缓存身份不匹配 | 图片、提示词、编码器或预处理变化后重建缓存 |
| 权重形状不匹配 | 家族、variant/version、VAE、文本编码器是否匹配 |
| 找不到 tokenizer | 本地 source 是否直接包含分词文件，offline=true 时缓存是否完整 |
| CUDA OOM | 看 peak 显存；降低 batch/尺寸，尝试 block 重算或卸载 |
| loss 下降但图像变差 | 看固定预览和各分损失，降低学习率或缩短训练，不只比较总 loss |

---

[文档首页](../README.md) · [参数参考](parameters.md) · [故障排查](../troubleshooting.md)
