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
- [VBAR 驻留](#vbar-驻留)
- [BF16 AMP 与发布精度](#bf16-amp-与发布精度)
- [推理时使用 INT8 注意力](#推理时使用-int8-注意力)
- [组合限制与恢复训练](#组合限制与恢复训练)
- [cuDNN 与 TF32 调优](#cudnn-与-tf32-调优)

</details>

> [!IMPORTANT]
> 驻留模式由你显式选择：resident 不会在 OOM 后自动切换到卸载。H2D 自动容量与 VBAR 页管理只在所选模式内工作；`compile = "auto"` 按设备选择，不按剩余显存选择。

## 如何选择策略

| 当前情况 | 优先尝试 | 代价与限制 |
| --- | --- | --- |
| 模型和激活都能放下 | 默认 resident，不重算 | 用作吞吐与显存对照 |
| 权重能放下，前向／反向激活过大 | `recompute = "block"` | 重算注意力与前馈，减少保存的激活 |
| 注意力保存的激活占用大 | `attention_activation_offload = true` | 增加主机内存和 D2H/H2D 传输 |
| 冻结权重存储或传输占用过大 | `quantization = "float8_e4m3fn"` 等 | 逐层量化；计算临时反量化，只支持 adapter |
| 适配器训练的冻结权重放不下 | `residency = "frozen-blocks"` | 自动调整共享缓存与异步预取；要求逐块重算 |
| 希望使用虚拟内存管理冻结权重驻留 | `residency = "vbar"` | 按设备容量管理物理页；需要 NVIDIA CUDA 和 vbar extra |
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

默认使用 eager 直接执行，没有首次编译耗时。通过 `compile` 比较编译训练性能。显式 `true` 使用 CUDA Inductor；`"auto"` 在 CUDA 使用 Inductor，在 CPU 使用 eager。编译作用于训练计算，预览和独立推理使用原生评估路径。比较稳定吞吐后再启用编译；编译可能改变浮点舍入和更新轨迹。

图像恢复训练在 CUDA 同时启用 `compile` 和恢复网络 block 重算时，还会编译 VAE 解码器中支持重算的无状态卷积残差块和上采样块，将前向与重算中的逐元素运算融合，减少中间激活。解码器保持冻结和 eval 模式，梯度仍可回传到 latent；编码器 LoRA 正常训练。设置 `encoder_rank = 0`（同时设置 `losses.lrr = 0`）时不安装编码器重算，解码器优化仍生效，像素损失的梯度仍通过冻结解码器回传到 DiT。首次训练和部分尺寸变化可能触发编译。无梯度预览、校准不走该编译路径；注意力保持原生执行，Qwen 时序状态块不参与此优化。

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

使用整块非重入重计算，在反向时重算注意力和前馈，保留随机数状态：

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

CUDA 下，将 SDPA 为 backward 保存的非叶激活复制到 pinned CPU 内存，反向按需取回。它不会卸载整个注意力模块，也不会搬走优化器状态。

可以与 resident 的 none／block、frozen-blocks 以及受支持的 VBAR 重算组合。预览不启用这些 hooks。主机内存和传输带宽可能成为新的瓶颈。

## 冻结块卸载

```toml
[execution]
residency = "frozen-blocks"
```

仅用于 CUDA adapter 训练。冻结的块权重保留在 CPU，在前向和反向重算时传入 GPU。适配器和块外嵌入／输出层仍在计算设备上，因此这不是整个模型的 CPU offload。

省略 recompute 时，冻结块/VBAR 自动使用必要的 block 重算；显式 `recompute="none"` 仍会被训练入口拒绝。默认自动使用 pinned CPU masters、单级异步预取和整个模型共享的权重缓存。无需设置交换块数、ring 大小或预取深度。首次 backward 后，根据设备可用显存、实际 allocator 峰值及精度转换空间调整缓存容量，并为连续大张量分配保留余量；前后向交界与后续更新会复用尚在 GPU 的权重。显存充足时会缓存全部冻结块，停止相应权重传输。容量不足以保留全部块时，自动分散选择常驻块，并为流式块预留两个最大块的缓冲空间；预取可跨过这些受保护的常驻块，提前加载后续缺失块。其余缓冲按最近使用顺序复用。无需仿照 musubi 手工指定交换块数。缓存只在前向起点检查外部显存压力并缩容，避免块内临时分配造成缓存抖动。该策略不会自动切换 residency 或启用激活卸载，也不保证未见过的更大 bucket 一定能运行。

通常省略 `frozen_weight_cache_mb` 和 `use_pinned_memory_for_block_swap`。有明确容量需求时，前者以 MiB 指定保留预算，另加一个最大冻结块的当前计算空间；0 限制为一个块的存储，不能同时保留另一个同尺寸预取块。该预算不限制总显存，也不包含活跃图必须保留的额外权重、激活、优化器和精度转换。pinned masters 占用不可分页的主机内存，后者可显式设为 false 关闭。

仅在最后一次图读取结束且 CUDA 事件满足依赖后复用缓冲区，CPU masters 始终不变。要求整块重算，不能与 full 组合。多个未完成图可能临时超过缓存预算；不会为了维持预算覆写活跃权重。

## VBAR 驻留

安装对应依赖：

```bash
uv sync --locked --extra media --extra text --extra vbar
```

然后修改配置：

```toml
[execution]
residency = "vbar"
```

VBAR 使用 comfy-aimdo 管理冻结权重的虚拟驻留，复用已驻留页，缺页时从不可变 CPU 源传入，并自动执行前向／反向预取。它与 frozen-blocks 的普通 CUDA 缓存不同，不使用 `frozen_weight_cache_mb`。

预取会检查可用显存；如果实际 allocator 分配重试增加，在反向完成后扩大预取余量，减少失败的推测性分配，无需手动设置阈值。需求分配本身无法容纳时仍会报错。H2D 与 VBAR 的吞吐取决于计算、传输和驻留比例，应使用实际训练尺寸比较。

该策略仅用于 NVIDIA CUDA adapter，可与 compile 或 BF16 AMP 组合。CLI 会在 Torch 导入前初始化 AIMDO；Python 调用者必须自行保持该顺序。

VBAR 页不计入 Torch 的 `max_memory_allocated`。测量时还需查看 `ExecutionInstallation.vbar_statistics().loaded_bytes`；主机侧同时查看 `pinned_staging_bytes` 与 `registered_source_bytes`。Torch 计数下降不等于总显存下降。

## BF16 AMP 与发布精度

```toml
[execution]
precision = "bf16"
```

BF16／FP16 AMP 支持 CUDA 流匹配和图像修复。默认自动选择、权重存储、FP16 loss scaling 和 VGG／DINO 默认 BF16 的规则，见[精度配置](precision.md)。可训练主参数保持 FP32，反向与优化器更新在前向 autocast 之外执行。

若仅希望缩小发布文件，使用 `output.save_precision = "bf16"` 或 `"fp16"`。发布精度与训练计算精度是独立设置，续训状态保留训练 dtype。

## 推理时使用 INT8 注意力

安装 Kitchen 后，在生成配置中显式选择：

```bash
uv sync --locked --extra media --extra text --extra kitchen
```

```toml
[sampling]
attention_backend = "kitchen-int8"
```

图像训练预览在 `[[preview.samples]]` 中设置同一字段；独立 `restore` 放在配置根节点；Wan／H3 使用各自 sampling。此后端只作用于 denoiser 推理，VAE 与文本编码不变。

默认 `"torch"` 使用原生 SDPA。Kitchen INT8 不提供此处所需的训练梯度；显式在梯度路径调用会报错。CPU、未支持的 head 形状／布局等情况不会静默回退，INT8 输出也可能与 Torch 不同。

## 组合限制与恢复训练

`trainable-blocks` 尚不支持。FP16 使用 AMP 与 FP32 主参数，并通过固定 loss scale 缩放损失。无需校准的量化权重支持冻结基模的 LoRA／LoHa／LoKr 训练，计算使用临时反量化后的浮点权重，见[量化基模训练](quantized-base.md)。Full 训练采用 resident；frozen-blocks／VBAR 仅限 adapter。VBAR 和 frozen-blocks 均要求整块重算。

编译、LoRA 联合计算及部分精度边界属于续训身份。实验策略时建议创建独立输出目录，从相同底模开始比较；不要假定修改策略后仍能按旧 checkpoint 的同一数值轨迹恢复。固定种子也不能单独保证 CUDA 逐字节复现。

传输流、权重 lease、VBAR 暂存与编译边界的细节见[执行实现说明](execution-internals.md)。

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
