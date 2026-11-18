# 执行策略与显存管理

简体中文 · [English](../en/training/execution.md) · [文档首页](../README.md)

显存主要用于模型权重、优化器状态、梯度和前向激活。不同策略针对不同部分：重算减少激活保存，权重卸载减少设备上的冻结权重，BF16 AMP 改变部分运算精度。先明确是哪一部分放不下，再选择策略。

<details>
<summary>本页导航</summary>

- [如何选择策略](#如何选择策略)
- [默认行为](#默认行为)
- [训练前权重量化](#训练前权重量化)
- [激活重算](#激活重算)
- [注意力激活卸载](#注意力激活卸载)
- [冻结块卸载](#冻结块卸载)
- [BF16 AMP 与发布精度](#bf16-amp-与发布精度)
- [推理时使用 INT8 注意力](#推理时使用-int8-注意力)
- [组合限制与恢复训练](#组合限制与恢复训练)
- [cuDNN 与 TF32 调优](#cudnn-与-tf32-调优)

</details>

> [!IMPORTANT]
> 驻留模式由你显式选择：resident 不会在 OOM 后自动切换到卸载。H2D 自动容量只在所选模式内工作；`compile = "auto"` 按设备选择，不按剩余显存选择。

## 如何选择策略

| 当前情况 | 优先尝试 | 代价与限制 |
| --- | --- | --- |
| 模型和激活都能放下 | 默认 resident，不重算 | 用作吞吐与显存对照 |
| 权重能放下，前向／反向激活过大 | `recompute = "block"` | 重算注意力与前馈，减少保存的激活 |
| 注意力保存的激活占用大 | `attention_activation_offload = true` | 增加主机内存和 D2H/H2D 传输 |
| 冻结权重存储或传输占用过大 | `quantization = "float8_e4m3fn"` 等 | 逐层量化；计算临时反量化，只支持 adapter |
| 适配器训练的冻结权重放不下 | `residency = "frozen-blocks"` | 自动调整共享缓存与异步预取；要求逐块重算 |
| 流匹配可接受混合精度计算 | `precision = "bf16"` | CUDA AMP；可训练参数与优化器 masters 仍为 FP32 |

这些策略可在支持范围内组合，但不会保证单步更快。固定模型、输入尺寸和 batch，分别比较稳定运行后的耗时和总显存；首次编译耗时应单独记录。

## 默认行为

默认参数如下，通常无需全部写进配置：

```toml
[execution]
precision = "auto"
compile = false
residency = "resident"
recompute = "none"
recompute_interval = 1
attention_activation_offload = false
quantization = "none"
```

默认不开启编译。先用默认设置完成一次训练，再尝试 `compile = true` 比较稳定运行后的速度；首次运行和尺寸变化可能触发编译。`compile = "auto"` 在 CUDA 上启用编译，在 CPU 上保持关闭。

修复训练启用编译后，首次运行可能需要更长准备时间。是否提速取决于模型、分辨率和设备，应比较完整更新的耗时与峰值显存。

以下代码块是**修改片段**，应合并到现有 `[execution]`，不要重复追加同名配置段。

## 训练前权重量化

在浮点基模的 adapter 配置中添加：

```toml
[execution]
quantization = "float8_e4m3fn"
quantization_rotation_size = 256
quantization_group_size = 16
```

`quantization` 选择权重存储布局，`precision` 选择浮点计算精度，两者独立。CUDA 训练在 GPU 逐层量化，之后与其他布局一样临时反量化并做浮点乘法。旋转／分组大小只在相应布局使用；完整格式表和不兼容尺寸的处理见[量化基模训练](quantized-base.md)。

已有量化原生文件可直接加载；`quantization = "none"` 保留文件布局，不会反量化整个模型。源文件与量化设置参与续训身份。层保护是预设策略，适配器的 `include_layers`／`exclude_layers` 仅决定训练哪些层，不会覆盖量化保护。全参数训练要求浮点权重。

文本缓存和文本推理使用独立的 `[model.text]` 精度／量化设置，不继承 `execution.quantization`。详见[文本编码精度与量化](text-encoding.md)。

## 激活重算

开启重算可以减少训练激活的显存占用，代价是增加计算时间：

```toml
[execution]
recompute = "block"
recompute_interval = 1
```

Resident 可以增大 `recompute_interval`，只重算部分 DiT 块：1 为全部，2 约为一半，4 约为四分之一。该间隔不影响 VAE 和 DINO。冻结权重卸载要求间隔为 1。

图像恢复可以独立保留 VAE 和 DINO 的重算，用剩余显存减少 DiT 重算。先从以下配置比较完整更新的耗时和峰值显存：

```toml
[execution]
residency = "resident"
compile = true
recompute = "block"
recompute_interval = 2
restoration_recompute = "block"
```

显存仍有余量时可以尝试间隔 4；进一步设置 `recompute = "none"` 只关闭 DiT 重算，显式的 `restoration_recompute = "block"` 仍保护 VAE 和 DINO。默认 `restoration_recompute = "inherit"` 延续统一开关的行为。此字段仅用于图像恢复，不影响流匹配训练。保留现有注意力激活卸载设置，单独比较重算策略；用最大 bucket 跑完反向和优化器更新后确认余量。此设置不会自动根据空闲显存调整间隔或在 OOM 后重试。

## 注意力激活卸载

```toml
[execution]
attention_activation_offload = true
```

将部分注意力激活暂存到 CPU，反向计算时再取回，以降低 GPU 显存占用。需要 CUDA，并会增加主机内存和传输开销。

可与重算和冻结权重卸载组合。启用后检查主机内存及训练速度，避免 CPU 内存或传输带宽成为瓶颈。

## 冻结块卸载

```toml
[execution]
residency = "frozen-blocks"
```

仅用于 CUDA adapter 训练。冻结的块权重保留在 CPU，在前向和反向重算时传入 GPU。适配器和块外嵌入／输出层仍在计算设备上，因此这不是整个模型的 CPU offload。

该模式要求 `recompute = "block"`、`recompute_interval = 1`；省略重算设置时会自动补齐。GPU 缓存容量会自动调整，通常无需手动设置。建议先确认最大的训练尺寸能完成一次更新；自动调整不能保证任意更大尺寸都能运行。

通常省略 `frozen_weight_cache_mb` 和 `use_pinned_memory_for_block_swap`。确需限制冻结权重缓存时，前者以 MiB 设置额外保留容量；0 只保留当前计算块所需空间。它不是总显存上限，激活和优化器仍会占用显存。主机内存紧张时可将后者设为 `false`，但传输可能变慢。


## BF16 AMP 与发布精度

```toml
[execution]
precision = "bf16"
```

BF16／FP16 可用于 CUDA 流匹配和图像修复。默认值、FP16 溢出处理及修复特征精度见[精度配置](precision.md)。

若仅希望缩小发布文件，使用 `output.save_precision = "bf16"` 或 `"fp16"`。发布精度与训练计算精度是独立设置，续训状态保留训练 dtype。

## 推理时使用 INT8 注意力

安装 Kitchen 后，在生成配置中显式选择：

```bash
uv sync --locked
```

```toml
[sampling]
attention_backend = "kitchen-int8"
```

图像训练预览在 `[[preview.samples]]` 中设置同一字段；独立 `restore` 放在配置根节点；Wan／H3 使用各自 sampling。此后端只作用于 denoiser 推理，VAE 与文本编码不变。

默认使用 `"torch"`。`kitchen-int8` 仅用于推理，需要 CUDA 和受支持的模型形状；不支持时请改回 `"torch"`。切换注意力后端可能改变生成结果。

## 组合限制与恢复训练

`trainable-blocks` 尚不支持。FP16 使用 AMP 与 FP32 主参数，并通过固定 loss scale 缩放损失。无需校准的量化权重支持冻结基模的 LoRA／LoHa／LoKr 训练，计算使用临时反量化后的浮点权重，见[量化基模训练](quantized-base.md)。Full 训练采用 resident；frozen-blocks 仅限 adapter，且要求整块重算。

更改精度、量化或编译设置后，旧续训状态可能不兼容。比较不同设置时使用新的输出目录，并固定模型、数据和种子。固定种子不能保证不同设备或设置得到完全一致的结果。


## cuDNN 与 TF32 调优

CUDA 卷积默认使用以下 runtime 设置，可按需覆盖：

```toml
[runtime]
cudnn_benchmark = true
cudnn_deterministic = false
```

cudnn_enabled 默认 true。cuda_allow_tf32 默认 false，可为支持的 FP32 矩阵运算显式开启；它不把 BF16 改成 TF32。benchmark 启动时搜索算法，bucket 尺寸变化较多时搜索成本可能更高。需要重复性时使用 cudnn_deterministic=true，通常同时关闭 benchmark。这些是后端选项，不放在 train/optimizer，也可用于缓存和推理。

每次只比较一种策略：记录基线更新耗时和 peak 显存；激活放不下先启用 block 重算；冻结权重占用高再比较卸载/量化；eager 稳定后再测试 compile。详细字段和限制见[参数手册](parameters.md)。

---

[文档首页](../README.md) · [参数参考](parameters.md) · [故障排查](../troubleshooting.md)
