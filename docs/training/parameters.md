# 配置参数与填写手册

[English](../en/training/parameters.md) · [训练教程](README.md)

下表默认值来自当前配置定义。默认值不代表适用于所有模型的推荐配方。`null` 表示省略该字段，TOML 不支持填写 null。表只在对应任务需要时添加；“必填”指添加该表后该字段必填。将 NAME 换成 main 等 ASCII 标识符。

首次填写只需改路径、模型变体、数据尺寸、学习率、rank、更新数和保存周期。其他默认参数可以省略；详细示例把常用可调项展开以便学习。

## 按职责查参数

| 范围 | 参数位置 |
| --- | --- |
| 模型与数据 | [模型](#model) · [文本](#modeltext) · [数据组](#datasetgroupsname) |
| 训练 | [更新](#train) · [适配器](#trainadapter) · [优化器](#trainoptimizer) · [方法](#method) |
| 执行与产物 | [执行](#execution) · [输出](#output) · [预览](#preview) · [运行时](#runtime) |
| 修复 | [修复方法](#methodrestoration) · [损失](#losses) · [退化](#degradationconfig) |
| 辅助目标 | [LayerSync](#methodlayersync) · [TREAD](#methodtread) · [REPA](#methodrepa) · [小波](#methodwavelet) |

命令选择与覆盖规则见 [TOML 与 CLI](../configuration.md)，报错处理见[故障排查](../troubleshooting.md)。

## TOML 与 CLI 填写规则

字符串加引号，布尔值写 true/false，数值不加引号，数组写方括号。相对路径以命令工作目录为基准；Windows 可用正斜杠或单引号路径。修改已有表中的字段，不重复追加同名表。

```toml
[train]
learning_rate = 5e-5
accumulation_steps = 4

[output]
name = "portrait-lora"
# directory = "runs/portrait-lora"
# tensorboard = "runs/tensorboard"
```

```bash
uv run unitunner train train.toml --device cuda --train.learning-rate 5e-5
uv run unitunner train train.toml --device cuda --set 'dataset.groups.main.batch_size=1'
```

## `model`

| 参数 | 类型／可选值 | 默认值 | 用途与填写方式 |
| --- | --- | --- | --- |
| `family` | "z-image", "moss-image", "flux", "flux2", "flux3", "qwen-image", "qwen-image-2-1", "krea2", "wan", "minimax-h3" | `必填` | 模型家族；先按实际权重选择，再填写对应 variant/version。 |
| `variant` | string | `必填` | 权重变体，如 base/turbo、dev、original；不能仅改名称来切换底模。 |
| `version` | "dev", "klein-4b", "klein-9b", "2.1", "2.2" / null | `null` | 仅 FLUX.2 与 Wan 填写；分别选择 dev/klein-4b/klein-9b 或 2.1/2.2。 |
| `vae` | string / null | `null` | 匹配该家族的本地 VAE 权重；缓存、预览和推理需要。 |
| `text_encoder` | string / null | `null` | 本地文本编码器文件或目录；分词器来源单独配置。 |
| `assets` | string / null | `null` | 辅助资产根目录；FLUX 的 CLIP 位于 text_encoder/，Wan2.1 I2V 的图像编码器位于 image_encoder/model.safetensors。 |
| `text` | object | `子表默认值` | 文本编码精度、量化和分词器来源子表；编码器权重路径仍在 model.text_encoder。 |
| `minimax_h3` | object / null | `null` | H3 专属子表；模型表填编码资产，数据表填音频处理，方法表填时间与条件策略。 |
| `text_source_format` | "transformers", "qwen" / null | `null` | 文本权重键格式 transformers/qwen；仅支持该选项的家族使用。 |
| `weights` | string | `必填` | 本地权重文件、分片索引或支持的模型目录；替换为实际路径。 |

## `model.text`

| 参数 | 类型／可选值 | 默认值 | 用途与填写方式 |
| --- | --- | --- | --- |
| `tokenizer` | object | `子表默认值` | 使用下方 tokenizer 表结构；clip_tokenizer 供 FLUX CLIP，processor 供 Qwen 编辑。 |
| `clip_tokenizer` | object / null | `null` | 使用下方 tokenizer 表结构；clip_tokenizer 供 FLUX CLIP，processor 供 Qwen 编辑。 |
| `processor` | object / null | `null` | 使用下方 tokenizer 表结构；clip_tokenizer 供 FLUX CLIP，processor 供 Qwen 编辑。 |
| `max_length` | integer / null | `null` | 文本 token 上限；只对支持覆盖的家族填写，其他家族使用固定预算。 |
| `precision` | "fp32", "fp16", "bf16" / null | `null` | 计算精度；训练 auto 按底模解析，CUDA 可显式 bf16，CPU 流程用 fp32。 |
| `quantization` | "none", "fp8", "convrot-int8", "float8_e4m3fn", "float8_e5m2", "mxfp8", "nvfp4", "int8_tensorwise", "int8_rowwise", "int8_convrot", "convrot_w4a4", "convrot_w4a8", "asym_w4a8_int8", "w6a8_int8" | `"none"` | 冻结基模/文本权重的存储量化布局；none 关闭，底模量化仅用于 adapter。 |
| `quantization_rotation_size` | integer；≥ 4 | `256` | 支持旋转布局的旋转块大小；使用该布局要求的合法尺寸。 |
| `quantization_group_size` | integer；≥ 4 | `16` | 分组量化大小；必须满足该格式及权重形状的整除要求。 |

## `model.text.tokenizer`

| 参数 | 类型／可选值 | 默认值 | 用途与填写方式 |
| --- | --- | --- | --- |
| `source` | string / null | `null` | 分词器本地目录或 Hub 仓库 ID；省略使用该模型默认来源。 |
| `subfolder` | string / null | `null` | 来源中的相对子目录，如 tokenizer；本地目录已直接指向分词器时通常省略。 |
| `revision` | string / null | `null` | Hub 来源的分支/标签/commit；正式实验可固定 commit，离线需本地已有该版本。 |
| `offline` | boolean | `false` | true 仅读取本地目录/缓存；false 允许下载分词器文件，不自动下载编码器权重。 |

## `dataset.defaults`

| 参数 | 类型／可选值 | 默认值 | 用途与填写方式 |
| --- | --- | --- | --- |
| `batch_size` | integer；> 0 | `1` | 每个 microbatch 的样本数；显存不足先减此项。 |
| `repeats` | integer；> 0 | `1` | 该组每轮样本重复次数；提高小数据组的相对出现频率，不改变 max_updates。 |
| `resolution` | [宽, 高] 或 [[宽, 高], ...] | `[1024, 1024]` | 单档面积预算，或每张图按列表各档生成训练样本；每档独立分桶及缓存子目录。列表须非空、不能重复；多尺度用于生成/编辑训练。 |
| `enable_bucket` | boolean | `true` | 按宽高比分桶；关闭时使用固定 resolution 裁剪。 |
| `bucket_no_upscale` | boolean | `false` | 避免将小图放大到较大 bucket；小图可能落入更小的桶。 |
| `prompt` | string / null | `null` | 数据组中覆盖所有同名 txt；空字符串也会覆盖。采样配置中为生成提示词。 |
| `caption_extension` | string / null | `null` | 提示词侧文件扩展名，包含点，如 .txt；未设 prompt 时使用。 |

## `dataset.groups.NAME`

| 参数 | 类型／可选值 | 默认值 | 用途与填写方式 |
| --- | --- | --- | --- |
| `batch_size` | integer；> 0 | `1` | 每个 microbatch 的样本数；显存不足先减此项。 |
| `repeats` | integer；> 0 | `1` | 该组每轮样本重复次数；提高小数据组的相对出现频率，不改变 max_updates。 |
| `resolution` | [宽, 高] 或 [[宽, 高], ...] | `[1024, 1024]` | 单档面积预算，或每张图按列表各档生成训练样本；每档独立分桶及缓存子目录。列表须非空、不能重复；多尺度用于生成/编辑训练。 |
| `enable_bucket` | boolean | `true` | 按宽高比分桶；关闭时使用固定 resolution 裁剪。 |
| `bucket_no_upscale` | boolean | `false` | 避免将小图放大到较大 bucket；小图可能落入更小的桶。 |
| `prompt` | string / null | `null` | 数据组中覆盖所有同名 txt；空字符串也会覆盖。采样配置中为生成提示词。 |
| `caption_extension` | string / null | `null` | 提示词侧文件扩展名，包含点，如 .txt；未设 prompt 时使用。 |
| `reference_sources` | [string, …] | `[]` | 有序参考图目录列表；与目标同名主体匹配，仅供支持参考条件的变体使用。 |
| `control_source` | string / null | `null` | Wan 对齐控制目录；其他家族不使用。 |
| `mask_source` | string / null | `null` | FLUX Fill 掩码目录；文件主体需匹配目标。 |
| `layer_sources` | [string, …] | `[]` | Qwen layered 的前景层目录列表，与合成目标按主体对应。 |
| `kind` | "image", "video", "restoration" | `"image"` | 任务或输入类型；数据用 image/video/restoration，训练方法用 flow-matching/restoration。 |
| `video` | object / null | `null` | 视频窗口与帧率子表；仅 kind=video 时添加。 |
| `wan` | object / null | `null` | Wan 条件帧子表；单帧 I2V 使用 single_frame。 |
| `minimax_h3` | object / null | `null` | H3 专属子表；模型表填编码资产，数据表填音频处理，方法表填时间与条件策略。 |
| `source` | string | `必填` | 数据源目录或该模型要求的记录文件；改动内容后重建缓存。 |
| `cache_directory` | string | `必填` | 该组独立的缓存目录；不同组不能指向同一路径。 |
| `low_quality_source` | string / null | `null` | 修复输入 LQ 目录，与 source 的 HQ 按文件主体配对；不同时开启合成退化。 |
| `degradation` | object / null | `null` | 合成 LQ 的两阶段退化子表；与同组 low_quality_source 互斥。 |
| `horizontal_flip_probability` | number；≥ 0, ≤ 1 | `0.5` | 修复训练成对水平翻转概率 0..1；方向敏感任务填 0。 |

## `dataset.groups.NAME.video`

| 参数 | 类型／可选值 | 默认值 | 用途与填写方式 |
| --- | --- | --- | --- |
| `target_frames` | [integer, …] | `[81]` | 训练窗口目标帧数列表；Wan 为 1+4k，H3 为 17n+5。 |
| `extraction` | "head", "chunk", "slide", "uniform", "full" | `"head"` | 视频窗口提取 head/chunk/slide/uniform/full；决定从同一视频生成哪些窗口。 |
| `stride` | integer；> 0 | `1` | 视频滑动/提取的步长，必须为正整数。 |
| `samples` | integer；> 0 | `1` | 视频 uniform 模式采样窗口数；preview.samples 则为图像采样配置列表。 |
| `max_frames` | integer；> 0 | `129` | 视频处理帧数上限；Wan 仍须 1+4k，H3 至少 5。 |
| `source_fps` | number / null | `null` | fractional 重采样的显式源帧率；timestamps 模式省略此项。 |
| `target_fps` | number；> 0 | `16.0` | 缓存重采样目标帧率；H3 必须 24.0。 |
| `fps_resampling` | "fractional", "timestamps" | `"fractional"` | fractional 使用分数索引；timestamps 使用解码时间戳，H3 要求后者。 |

## `cache`

| 参数 | 类型／可选值 | 默认值 | 用途与填写方式 |
| --- | --- | --- | --- |
| `batch_size` | integer；> 0 | `1` | 缓存编码批大小，独立于 dataset 的训练 batch_size。目前仅 Z-Image 支持大于 1：连续同尺寸图片批量编码 VAE 与文本，尾批可少于此值。其他家族保持 1。 |

```toml
[cache]
batch_size = 16
```

也可仅在缓存命令中设置 `--cache.batch-size 16`。共享训练 TOML 可保留此表，不改变训练 microbatch。批大小和批次源内容计入缓存身份，批量结果可能有浮点舍入差异；修改后重建缓存。显存不足时降低该值。

## `train`

| 参数 | 类型／可选值 | 默认值 | 用途与填写方式 |
| --- | --- | --- | --- |
| `parameter_mode` | "adapter", "full" | `"adapter"` | adapter 训练适配器；full 更新底模并删除 train.adapter。 |
| `adapter` | object / null | `null` | 适配器设置；训练在 train.adapter 填算法，推理在 adapter 填 weights 和 config。 |
| `include_layers` | [string, …] | `[]` | 原生层/模块/参数 glob；空数组选择所有合格目标，不使用 ComfyUI 导出键。 |
| `exclude_layers` | [string, …] | `[]` | 在 include_layers 之后排除匹配目标，排除规则优先。 |
| `learning_rate` | number；≥ 0 | `0.0001` | AdamW 基础学习率；先固定数据与 rank，再根据预览和损失调整。 |
| `weight_decay` | number；≥ 0 | `0.0001` | AdamW 解耦权重衰减；0 关闭，不是学习率。 |
| `optimizer` | object | `子表默认值` | AdamW 实现与数值选项子表，常规训练使用默认值。 |
| `max_updates` | integer；> 0 | `1000` | 有效优化器更新总数；梯度累积的 microbatch 不单独计数。 |
| `accumulation_steps` | integer；> 0 | `1` | 每次有效更新累计的 microbatch 数；降低单批显存需求但会增加一次更新耗时。 |
| `max_grad_norm` | number；≥ 0 | `1.0` | 全局梯度范数裁剪阈值；0 关闭。 |
| `seed` | integer | `42` | 随机种子；固定后便于比较配方，续训由状态文件恢复 RNG。 |
| `lr_schedule` | "constant", "linear", "cosine" | `"constant"` | constant 固定、linear 线性衰减、cosine 余弦衰减；均支持 warmup。 |
| `warmup_updates` | integer；≥ 0 | `0` | 学习率线性预热的有效更新数；必须小于 max_updates。 |

## `train.adapter`

| 参数 | 类型／可选值 | 默认值 | 用途与填写方式 |
| --- | --- | --- | --- |
| `algorithm` | "lora", "loha", "lokr" | `"lora"` | 适配器算法 lora/loha/lokr；推理时与训练保持一致。 |
| `fused` | boolean | `true` | 融合适用的 CUDA LoRA 缩放与残差相加，保留中间精度舍入；同时允许编译器联合优化冻结底模与 LoRA 分支。通常保持默认。 |
| `rank` | integer；> 0 | `32` | 适配器容量，必须为正整数；增大会增加参数、优化器内存和文件体积。 |
| `alpha` | number；> 0 | `32.0` | 适配器缩放参数；普通 LoRA 使用 alpha/rank，常以 alpha=rank 起步。 |

## `train.optimizer`

| 参数 | 类型／可选值 | 默认值 | 用途与填写方式 |
| --- | --- | --- | --- |
| `betas` | [number, number]；length ≥ 2, length ≤ 2 | `[0.9, 0.999]` | AdamW 一阶/二阶矩衰减系数，两个数均在 [0,1)。 |
| `eps` | number；> 0 | `1e-08` | AdamW 分母稳定项，必须大于 0；通常保留 1e-8。 |
| `backend` | "auto", "fused", "foreach", "single" | `"auto"` | auto 自动选择优化器实现；fused/foreach/single 显式指定后端。 |

## `method`

| 参数 | 类型／可选值 | 默认值 | 用途与填写方式 |
| --- | --- | --- | --- |
| `kind` | "flow-matching", "restoration" | `"flow-matching"` | 任务或输入类型；数据用 image/video/restoration，训练方法用 flow-matching/restoration。 |
| `layersync` | object / null | `null` | 流匹配层间特征自监督；填写 weight 与学生/教师层索引。 |
| `repa` | object / null | `null` | 流匹配特征对齐；需要特征编码器、真实摘要与独立特征缓存。 |
| `tread` | object / null | `null` | 流匹配 token 路由；填写丢弃比例和有效层区间。 |
| `wavelet` | object / null | `null` | 小波目标子表；wavelet 字段选择 haar/db4，其余字段设置频带与时间权重。 |
| `guidance` | number / null | `null` | 支持该字段的 FLUX 变体训练 guidance embedding；不是推理 CFG。 |
| `restoration` | object / null | `null` | 修复时间、VAE encoder adapter 和感知模型子表；仅 restoration 方法添加。 |
| `qwen_image` | object / null | `null` | Qwen layered 监督选项；其他模型不填写。 |
| `minimax_h3` | object / null | `null` | H3 专属子表；模型表填编码资产，数据表填音频处理，方法表填时间与条件策略。 |

## `method.restoration`

| 参数 | 类型／可选值 | 默认值 | 用途与填写方式 |
| --- | --- | --- | --- |
| `midtimestep` | integer；≥ 1, ≤ 998 | `244` | 修复训练的整数时间 1..998，默认 244；不同模型时间表对应不同 sigma。 |
| `sigma` | number / null | `null` | 修复噪声水平，严格在 (0,1)；显式设置时不同时开启 calibrate。 |
| `calibrate` | boolean | `false` | 训练开始前从数据估计修复 sigma；固定时间配方保持 false。 |
| `calibration_pairs` | integer；> 0 | `32` | 自动校准使用的配对数量；非默认值要求 calibrate=true。 |
| `encoder_rank` | integer；≥ 0 | `4` | VAE encoder adapter 的 rank；0 禁用，推理时也不再填写 encoder_weights。 |
| `posterior` | "mean", "sample" / null | `null` | 训练 VAE 后验 mean/sample；省略遵循各家族策略，推理/预览使用均值。 |
| `histogram_bins` | integer；≥ 2 | `32` | 颜色直方图损失的桶数，至少 2；仅 histogram 权重大于 0 时影响训练。 |
| `dino_weights` | string / null | `null` | 本地 DINOv3 ConvNeXt 权重；dv3d 或 gan 大于 0 时必填。 |
| `vgg_weights` | string / null | `null` | 本地 VGG19 权重；fdl 大于 0 时必填。 |
| `feature_precision` | "bf16", "fp32" | `"bf16"` | 感知特征网络精度；支持 bf16/fp32，与主模型计算精度分开。 |
| `dino_widths` | [integer, …]；length ≥ 4, length ≤ 4 | `[192, 384, 768, 1536]` | DINO ConvNeXt 四阶段通道数；默认匹配 Large，只在匹配其他结构时更改。 |
| `dino_depths` | [integer, …]；length ≥ 4, length ≤ 4 | `[3, 3, 27, 3]` | DINO 四阶段深度；必须与 dino_weights 结构一致。 |
| `vgg_widths` | [integer, …]；length ≥ 5, length ≤ 5 | `[64, 128, 256, 512, 512]` | VGG 五阶段通道数；必须与 vgg_weights 结构一致。 |

## `losses`

| 参数 | 类型／可选值 | 默认值 | 用途与填写方式 |
| --- | --- | --- | --- |
| `lrr` | number；≥ 0 | `5.0` | latent 修复回归损失系数；0 关闭。 |
| `dv3d` | number；≥ 0 | `3.0` | DINOv3 特征距离损失系数；大于 0 需 DINO 权重。 |
| `l1` | number；≥ 0 | `0.5` | 输出与 HQ 的像素 L1 损失系数；0 关闭。 |
| `gan` | number；≥ 0 | `0.5` | 生成器对抗损失系数；大于 0 启用判别器并需要 DINO 权重。 |
| `fdl` | number；≥ 0 | `0.5` | VGG 频率分布损失系数；大于 0 需 VGG 权重。 |
| `histogram` | number；≥ 0 | `0.0` | 颜色直方图损失系数；无需额外特征模型。 |

## `execution`

| 参数 | 类型／可选值 | 默认值 | 用途与填写方式 |
| --- | --- | --- | --- |
| `precision` | "auto", "fp32", "fp16", "bf16" | `"auto"` | 计算精度；训练 auto 按底模解析，CUDA 可显式 bf16，CPU 流程用 fp32。 |
| `fp16_loss_scale` | number；≥ 1.0 | `128.0` | FP16 损失缩放倍数；仅 FP16 路径需要。 |
| `compile` | boolean / "auto" | `false` | false 直接执行；true 使用 CUDA 编译，auto 按设备选择；首次编译额外耗时。 |
| `recompute` | "none", "block" | `"none"` | none 保存激活；block 按块重算以降低激活显存，增加计算量。 |
| `recompute_interval` | integer；> 0 | `1` | 块重算间隔；冻结块卸载策略要求逐块重算时保持 1。 |
| `attention_activation_offload` | boolean | `false` | 将保存的注意力激活卸载到主机，节省显存但增加传输和内存需求。 |
| `residency` | "resident", "frozen-blocks", "vbar", "trainable-blocks" | `"resident"` | resident 常驻；frozen-blocks/vbar 用于冻结权重管理；trainable-blocks 为保留值，当前不支持。 |
| `use_pinned_memory_for_block_swap` | boolean / null | `null` | 块交换是否用锁页主机内存；省略由策略决定，注意主机 RAM 占用。 |
| `frozen_weight_cache_mb` | integer / null | `null` | 冻结块 CUDA 缓存预算，单位 MiB；省略时自动，0 最小保留，仍需当前块工作区。 |
| `quantization` | "none", "fp8", "convrot-int8", "float8_e4m3fn", "float8_e5m2", "mxfp8", "nvfp4", "int8_tensorwise", "int8_rowwise", "int8_convrot", "convrot_w4a4", "convrot_w4a8", "asym_w4a8_int8", "w6a8_int8" | `"none"` | 冻结基模/文本权重的存储量化布局；none 关闭，底模量化仅用于 adapter。 |
| `quantization_rotation_size` | integer；≥ 4 | `256` | 支持旋转布局的旋转块大小；使用该布局要求的合法尺寸。 |
| `quantization_group_size` | integer；> 0 | `16` | 分组量化大小；必须满足该格式及权重形状的整除要求。 |

## `output`

| 参数 | 类型／可选值 | 默认值 | 用途与填写方式 |
| --- | --- | --- | --- |
| `directory` | string | `"runs"` | 产物目录；省略自动建立 runs/<name>_<时间戳>，显式路径会复用。 |
| `name` | string | `"model"` | 正式权重文件的名称前缀，也是日志及 TensorBoard run 命名的基础。 |
| `log_every_seconds` | number；> 0 | `60.0` | 后台/重定向摘要间隔，默认 60 秒；终端原位刷新，完整指标仍逐步写入文件。 |
| `save_every_updates` | integer；> 0 | `100` | 每 N 次有效更新同时导出正式权重与独立续训状态；末步也保存。 |
| `max_resume_checkpoints` | integer；≥ 0 | `1` | 最多保留最近 N 份续训状态；默认 1，0 表示全部。 |
| `max_weight_checkpoints` | integer；≥ 0 | `0` | 最多保留最近 N 组正式权重；默认 0 全保留，encoder 随组清理。 |
| `tensorboard` | string / null | `null` | 事件根目录；只填目录，训练器自动建立带训练名的 run 子目录。 |
| `save_precision` | "fp32", "fp16", "bf16" | `"fp32"` | 正式权重导出精度 fp32/bf16/fp16；不改变续训状态的训练 dtype。 |
| `checkpoint_storage` | "complete", "frozen-reference" | `"complete"` | complete 包含完整续训张量；frozen-reference 省略重复冻结参数，恢复需完全相同底模。 |

## `preview`

| 参数 | 类型／可选值 | 默认值 | 用途与填写方式 |
| --- | --- | --- | --- |
| `every_updates` | integer；> 0 | `100` | 每 N 次有效更新预览，末步也预览；只影响观察频率。 |
| `at_start` | boolean | `false` | 从头训练时先生成基线预览；续训不重复初始预览。 |
| `samples` | [object, …] | `[]` | 视频 uniform 模式采样窗口数；preview.samples 则为图像采样配置列表。 |
| `videos` | [object, …] | `[]` | Wan 视频预览数组表，填写原生视频采样参数。 |
| `audio_video` | [object, …] | `[]` | H3 音视频预览数组表，包含 inputs 与 sampling。 |
| `num_images` | integer；> 0 | `1` | 每轮修复预览抽取的图像数量；按固定种子顺序轮换数据。 |

## `preview.samples`

| 参数 | 类型／可选值 | 默认值 | 用途与填写方式 |
| --- | --- | --- | --- |
| `prompt` | string | `""` | 本次推理或预览使用的提示词。 |
| `negative_prompt` | string / null | `null` | 采样负提示词；是否支持以及 CFG 语义由模型变体决定。 |
| `reference_images` | [string, …] | `[]` | 推理参考图路径列表，顺序有意义；按模型变体要求填写。 |
| `conditioning_image` | string / null | `null` | 单张条件图路径；如 FLUX Kontext/Fill 或 Wan I2V，遵循模型约束。 |
| `mask_image` | string / null | `null` | 推理掩码图；仅用于支持局部编辑的模型变体。 |
| `condition_posterior` | "sample", "mean" | `"sample"` | FLUX 条件图 VAE 后验 sample/mean；其他模型不一定开放该项。 |
| `conditioning_resize` | "preferred", "original", "output" | `"preferred"` | 推理条件图缩放策略 preferred/original/output；output 仅支持对应 Qwen 模型。 |
| `output_layers` | integer / null | `null` | Qwen layered 输出前景层数量；仅此变体使用。 |
| `caption_language` | "en", "zh" / null | `null` | Qwen layered 的图层描述语言 en/zh。 |
| `width` | integer；≥ 16 | `1024` | 输出宽度，像素；一般 16 对齐，Qwen2.1/H3 使用 32 对齐。 |
| `height` | integer；≥ 16 | `1024` | 输出高度，像素；与宽度一起决定采样显存需求。 |
| `seed` | integer | `42` | 采样噪声种子；比较训练版本时保持不变。 |
| `attention_backend` | "torch", "kitchen-int8" | `"torch"` | torch 使用原生注意力；kitchen-int8 需要相应运行环境，仅用于支持的推理。 |
| `steps` | integer / null | `null` | 采样迭代数；省略使用变体默认，Turbo 与非蒸馏模型不要套用同一数值。 |
| `shift` | number / null | `null` | 采样时间表移位；含义因模型而异，省略用原生默认。 |
| `guidance` | number / null | `null` | 该模型原生采样 guidance/CFG；省略使用变体默认。 |
| `training_shift` | number；> 0 | `1.0` | MOSS 训练时间表对应的采样移位倍率；其他家族不随意设置。 |
| `uncond_skip_layers` | [integer, …] / null | `null` | MOSS 无条件分支跳过的零起始块列表；需匹配实际模型深度。 |
| `slg_layers` | [integer, …] | `[]` | MOSS SLG 跳层列表；仅适用模型支持时填写。 |
| `slg_scale` | number；≥ 0 | `0.0` | SLG 引导强度；0 关闭。 |
| `slg_start` | number；≥ 0, ≤ 1 | `0.0` | SLG 生效的采样进度起点，0..1。 |
| `slg_stop` | number；≥ 0, ≤ 1 | `1.0` | SLG 生效的采样进度终点，须大于等于 slg_start。 |

## `runtime`

| 参数 | 类型／可选值 | 默认值 | 用途与填写方式 |
| --- | --- | --- | --- |
| `cuda_allow_tf32` | boolean | `false` | 允许支持 GPU 上的 TF32 矩阵运算；可能改变数值轨迹。 |
| `cudnn_enabled` | boolean | `true` | 启用 cuDNN 卷积后端；通常保持 true。 |
| `cudnn_benchmark` | boolean | `true` | 默认搜索卷积算法；首次运行及新尺寸有搜索开销，可显式关闭。 |
| `cudnn_deterministic` | boolean | `false` | 限制为确定性 cuDNN 算法；需要重复性时开启，可能降低速度。 |

## `method.layersync`

| 参数 | 类型／可选值 | 默认值 | 用途与填写方式 |
| --- | --- | --- | --- |
| `weight` | number；> 0 | `1.0` | 该辅助目标在总目标中的权重；0 禁用其贡献。 |
| `student_layer` | integer；≥ 0 | `0` | LayerSync 学生块的原生零起始索引。 |
| `teacher_layer` | integer；≥ 0 | `1` | LayerSync 教师块的原生零起始索引。 |

## `method.tread`

| 参数 | 类型／可选值 | 默认值 | 用途与填写方式 |
| --- | --- | --- | --- |
| `drop_fraction` | number；> 0, < 1 | `0.5` | TREAD 丢弃 token 比例，作用于指定块区间；先用小值比较效果。 |
| `start_layer` | integer；≥ 0 | `0` | TREAD 起始块索引，零起始。 |
| `end_layer` | integer；> 0 | `1` | TREAD 结束块边界，须满足方法与模型的层区间限制。 |

## `method.repa`

| 参数 | 类型／可选值 | 默认值 | 用途与填写方式 |
| --- | --- | --- | --- |
| `weight` | number；> 0 | `1.0` | 该辅助目标在总目标中的权重；0 禁用其贡献。 |
| `layer` | integer；≥ 0 | `0` | REPA 对齐的原生块零起始索引。 |
| `encoder_dim` | integer；> 0 | `必填` | REPA 特征维度；必须与实际特征编码器输出匹配。 |
| `encoder_sha256` | string；pattern ^[0-9a-f]{64}$ | `必填` | REPA 编码器文件的真实 SHA256；示例全零必须替换。 |
| `cache_directory` | string | `必填` | 该组独立的缓存目录；不同组不能指向同一路径。 |
| `encoder_weights` | string / null | `null` | 修复 encoder adapter 或 REPA 特征编码器路径，按所在表的任务选择。 |
| `encoder_device` | string | `"cpu"` | REPA 特征编码设备 cpu/cuda；影响缓存阶段显存。 |

## `method.wavelet`

加权 DWT 频带 MSE。图像恢复时比较解码 RGB 输出与 detach 的 HQ，不应用时间步衰减；见[图像恢复](restoration.md)。

| 参数 | 类型／可选值 | 默认值 | 用途与填写方式 |
| --- | --- | --- | --- |
| `weight` | number；> 0 | `1.0` | 该辅助目标在总目标中的权重；0 禁用其贡献。 |
| `wavelet` | "haar", "db4" | `"db4"` | 小波目标子表；wavelet 字段选择 haar/db4，其余字段设置频带与时间权重。 |
| `levels` | integer；≥ 1, ≤ 8 | `3` | 小波分解层数，1..8；更深层包含更低空间频率。 |
| `ll` | number；≥ 0 | `0.1` | 小波 LL 低频分量损失权重。 |
| `lh` | number；≥ 0 | `0.01` | 小波 LH 细节分量损失权重。 |
| `hl` | number；≥ 0 | `0.01` | 小波 HL 细节分量损失权重。 |
| `hh` | number；≥ 0 | `0.05` | 小波 HH 高频细节分量损失权重。 |
| `ll_level_threshold` | integer / null | `-1` | 选择计入 LL 损失的分解层门限；-1 使用实现的默认选择。 |
| `normalize_bands` | boolean | `false` | 是否归一化频带损失尺度；对比配方时保持一致。 |
| `timestep_cutoff` | number；≥ 0, ≤ 1 | `0.7` | 小波目标的时间权重分界，0..1。 |
| `timestep_transition_width` | number；> 0 | `0.4` | 小波时间权重的平滑过渡宽度，必须大于 0。 |

## `method.qwen_image`

| 参数 | 类型／可选值 | 默认值 | 用途与填写方式 |
| --- | --- | --- | --- |
| `remove_base` | boolean | `false` | Qwen layered 是否移除合成基图监督；仅 layered 流匹配使用。 |

## `model.minimax_h3`

| 参数 | 类型／可选值 | 默认值 | 用途与填写方式 |
| --- | --- | --- | --- |
| `audio_vae` | string | `必填` | H3 本地音频 VAE 权重。 |
| `text_config` | string | `必填` | H3 文本编码器结构配置文件，需与权重匹配。 |
| `text_processor` | string / null | `null` | H3 本地 processor 目录；未填时可用该家族默认来源。 |
| `text_device` | "cpu", "cuda" / null | `null` | H3 文本编码设备；仅在需要单独放置编码器时设置。 |
| `video_topology` | object / null | `null` | H3 非标准视频 VAE 结构覆盖；标准权重一般省略。 |
| `audio_topology` | object / null | `null` | H3 非标准音频 VAE 结构覆盖；标准权重一般省略。 |

## `method.minimax_h3`

| 参数 | 类型／可选值 | 默认值 | 用途与填写方式 |
| --- | --- | --- | --- |
| `sigma_min` | number；≥ 0, ≤ 1 | `0.0` | H3 训练 sigma 采样下界。 |
| `sigma_max` | number；≥ 0, ≤ 1 | `1.0` | H3 训练 sigma 采样上界。 |
| `focus_min` | number；≥ 0, ≤ 1 | `0.0` | H3 重点时间区间下界。 |
| `focus_max` | number；≥ 0, ≤ 1 | `1.0` | H3 重点时间区间上界。 |
| `focus_probability` | number；≥ 0, ≤ 1 | `0.0` | H3 从重点时间区间抽样的概率。 |
| `video_shift` | number；≥ 0.01, ≤ 100 | `12.0` | H3 视频时间表移位，默认 12.0；训练和采样使用匹配配方。 |
| `audio_shift` | number；≥ 0.01, ≤ 100 | `3.0` | H3 音频时间表移位，默认 3.0。 |
| `visual_condition_clean` | number；≥ 0, ≤ 1 | `0.999` | H3 视觉条件清洁度；默认 0.999。 |
| `audio_condition_clean` | number；≥ 0, ≤ 1 | `1.0` | H3 音频条件清洁度；默认 1.0。 |
| `audio_loss_weight` | number；≥ 0 | `1.0` | H3 音频损失相对权重，默认 1.0。 |

## `dataset.groups.NAME.wan.single_frame`

| 参数 | 类型／可选值 | 默认值 | 用途与填写方式 |
| --- | --- | --- | --- |
| `clean_sources` | [string, …]；length ≥ 1, length ≤ 2 | `必填` | Wan 单帧 I2V 的干净条件图目录列表。 |
| `clean_indices` | [integer, …]；length ≥ 1, length ≤ 2 | `必填` | 与 clean_sources 顺序对应的条件帧索引。 |
| `target_index` | integer；≥ 0, < 1024 | `必填` | 目标帧索引；应与数据记录和条件索引对应。 |

## `dataset.groups.NAME.minimax_h3`

| 参数 | 类型／可选值 | 默认值 | 用途与填写方式 |
| --- | --- | --- | --- |
| `cache_seed` | integer | `0` | H3 缓存生成种子，固定缓存随机条件。 |
| `allow_experimental_duration` | boolean | `false` | 允许 H3 实验性时长；常规训练保持 false。 |
| `audio_decode` | object | `子表默认值` | H3 音频解码采样率、声道与时间连续性子表。 |
| `audio_pad_tolerance` | integer；≥ 0 | `800` | H3 音频对齐填充容差，使用模型原生单位与限制。 |
| `audio_max_repair_samples` | integer；≥ 0 | `6400` | H3 允许修复的最大音频采样点数量；不是训练样本数。 |

## `dataset.groups.NAME.minimax_h3.audio_decode`

| 参数 | 类型／可选值 | 默认值 | 用途与填写方式 |
| --- | --- | --- | --- |
| `sample_rate` | integer；> 0 | `必填` | 解码音频采样率；H3 原生为 32000 Hz。 |
| `channels` | integer；≥ 1, ≤ 2 | `必填` | 解码音频声道数；H3 原生为双声道。 |
| `max_discontinuity_seconds` | number；≥ 0 | `0.05` | 允许的音频时间戳不连续容差，单位秒。 |

## `DegradationConfig`

仅用于合成超分 LQ，参见[场景与操作顺序](datasets.md#合成超分退化)。

| 字段 | 类型／可选值 | 默认值 | 含义 |
| --- | --- | --- | --- |
| `enabled` | boolean | `false` | 启用该组的合成退化；真实 LQ 配对任务不添加此表。 |
| `scales` | [number, …] | unset | 可选的显式倍率列表，支持 1～16 的小数；等概率抽取，优先于 scale_range。 |
| `scale_range` | [number, number] | `[1.0, 8.0]` | 连续倍率区间，按对数均匀分布采样；上下界在 1～16 内。 |
| `resize_modes` | ["area", "bilinear", "bicubic", "nearest", …] | `["area", "bilinear", "bicubic"]` | bilinear/bicubic 降采样启用抗混叠；放大时有其他候选则不选 area。nearest 仅显式启用。 |
| `first` | object | `子表` | 第一阶段的模糊、缩放、噪声和压缩分布。 |
| `second` | object | `子表` | 第二阶段的独立模糊、缩放、噪声和压缩分布。 |
| `final_sinc_probability` | number | `0.1` | 最后应用 sinc 滤波的概率，0..1。 |
| `gray_probability` | number | `0.0` | 退化转灰度的概率；噪声子表中表示灰度噪声概率，0..1。 |
| `color_jitter_probability` | number | `0.0` | 启用色彩扰动的概率，0..1；0 关闭。 |
| `brightness` | [number, number] | `[0.5, 1.5]` | 亮度倍率范围 [下界, 上界]，用于已启用的色彩扰动。 |
| `contrast` | [number, number] | `[0.5, 1.5]` | 对比度倍率范围 [下界, 上界]。 |
| `saturation` | [number, number] | `[0.0, 1.5]` | 饱和度倍率范围 [下界, 上界]。 |
| `hue` | [number, number] | `[-0.1, 0.1]` | 色相扰动范围 [下界, 上界]。 |
| `scenario_probabilities` | [number, number, number] | `[0.25, 0.35, 0.4]` | 普通照片 / 网络转载 / 低照度概率，非负且和为 1。 |
| `grain_strength` | [number, number] | `[0.0, 1.5]` | 残余颗粒增益，0～4；[0,0] 关闭。 |
| `grain_scale_range` | [number, number] | `[1.0, 4.0]` | 噪声生成网格相对输出的缩小倍率，1～16 内按对数均匀分布采样；仅缩放残差。 |
| `smear_strength` | [number, number] | `[0.0, 0.8]` | 连续纹理平滑混合强度，0～1；按实际降采样倍率 d 乘以 min(1,2/d)。 |
| `sharpen_strength` | [number, number] | `[0.0, 0.4]` | 连续交付锐化强度，范围 0～1；[0,0] 关闭。 |
| `smear_detail_protection` | number | `0.6` | 局部对比度保护，0～1；0 恢复均匀混合。 |
| `compression_blur_probability` | number | `0.25` | 在 LQ 网格先编码后模糊；第二阶段显式关闭模糊或压缩时禁用。 |

## `StageConfig`

采集阶段：模糊 → 缩放 → 可选噪声/低照度降噪 → 可选网络转载编码。

| 字段 | 类型／可选值 | 默认值 | 含义 |
| --- | --- | --- | --- |
| `blur` | object | `子表` | 模糊核与参数分布子表。 |
| `blur_probability` | number | `0.8` | 本阶段应用模糊的概率，0..1。 |
| `resize_probabilities` | [number, number, number] | `[0.0, 0.6, 0.4]` | 放大/缩小/不变的概率，依此顺序，非负且和为 1。 |
| `resize_range` | [number, number] | `[0.75, 1.0]` | 缩放倍率范围，需为正且覆盖 1。 |
| `noise` | object | `子表` | 高斯/泊松/ISO 噪声分布子表。 |
| `compression` | object | `子表` | JPEG/WebP/JPEG2000 压缩分布子表。 |
| `noise_probability` | number | `0.8` | 采集阶段添加噪声的概率。 |
| `compression_probability` | number | `0.75` | 仅网络转载场景执行的中间编码概率。 |

## `SecondStageConfig`

交付阶段随机排列缩放、局部模糊与噪声，再执行目标 LQ 缩放及可选 sinc/锐化。噪声可推迟至放大回 HQ 后，最终编码可位于放大前或后。

| 字段 | 类型／可选值 | 默认值 | 含义 |
| --- | --- | --- | --- |
| `blur` | object | `子表` | 模糊核与参数分布子表。 |
| `blur_probability` | number | `0.15` | 本阶段应用模糊的概率，0..1。 |
| `resize_probabilities` | [number, number, number] | `[0.1, 0.2, 0.7]` | 放大/缩小/不变的概率，依此顺序，非负且和为 1。 |
| `resize_range` | [number, number] | `[0.9, 1.1]` | 缩放倍率范围，需为正且覆盖 1。 |
| `noise` | object | `子表` | 高斯/泊松/ISO 噪声分布子表。 |
| `compression` | object | `子表` | JPEG/WebP/JPEG2000 压缩分布子表。 |
| `noise_probability` | number | `0.1` | 额外残差噪声概率；触发后等概率在交付阶段或放大回 HQ 后添加一次。 |
| `compression_probability` | number | `0.9` | 所有场景最终编码概率；仍需放大时等概率位于放大前或后，已到 HQ 网格时在该网格编码。 |

## `BlurConfig`

sinc、运动和失焦概率之和不得超过 1，其余概率使用六类混合核。场景强度将 sigma、噪声与压缩损伤关联起来。

| 字段 | 类型／可选值 | 默认值 | 含义 |
| --- | --- | --- | --- |
| `sigma` | [number, number] | `[0.4, 2.8]` | 1024 短边基准 sigma；乘以 clamp(sqrt(短边 / 1024), 0.5, 1.5)。 |
| `probabilities` | [number, number, number, number, number, number] | `[0.5, 0.25, 0.1, 0.05, 0.07, 0.03]` | 非负概率向量且总和为 1；噪声依次 Gaussian/Poisson/ISO，压缩依次 JPEG/WebP/JPEG2000，模糊依次 Gaussian/generalized/plateau 的各向同性与各向异性。 |
| `generalized_beta` | [number, number] | `[0.5, 4.0]` | 广义高斯模糊 beta 范围，正数且区间跨过 1。 |
| `plateau_beta` | [number, number] | `[1.0, 2.0]` | 平台核模糊 beta 范围，正数且区间跨过 1。 |
| `sinc_probability` | number | `0.05` | 模糊核选择 sinc 的概率，0..1。 |
| `motion_probability` | number | `0.12` | 居中直线运动模糊核概率。 |
| `defocus_probability` | number | `0.1` | 圆形孔径失焦核概率。 |

## `SecondBlurConfig`

核选择规则与 BlurConfig 一致，但 sigma 使用 LQ 局部像素单位。

| 字段 | 类型／可选值 | 默认值 | 含义 |
| --- | --- | --- | --- |
| `sigma` | [number, number] | `[0.15, 0.65]` | LQ 网格像素单位的 sigma，不随 HQ 分辨率放大。 |
| `probabilities` | [number, number, number, number, number, number] | `[0.5, 0.25, 0.1, 0.05, 0.07, 0.03]` | 非负概率向量且总和为 1；噪声依次 Gaussian/Poisson/ISO，压缩依次 JPEG/WebP/JPEG2000，模糊依次 Gaussian/generalized/plateau 的各向同性与各向异性。 |
| `generalized_beta` | [number, number] | `[0.5, 4.0]` | 广义高斯模糊 beta 范围，正数且区间跨过 1。 |
| `plateau_beta` | [number, number] | `[1.0, 2.0]` | 平台核模糊 beta 范围，正数且区间跨过 1。 |
| `sinc_probability` | number | `0.03` | 模糊核选择 sinc 的概率，0..1。 |
| `motion_probability` | number | `0.0` | LQ 像素尺度运动模糊核概率。 |
| `defocus_probability` | number | `0.0` | LQ 像素尺度圆形失焦核概率。 |

## `NoiseConfig`

概率顺序为 Gaussian/Poisson/ISO。低照度采集固定使用线性光 shot/read 噪声，并关联保边与色度降噪。场景标准差倍率：普通 0.35、转载 0.5、低照度 1.0。

| 字段 | 类型／可选值 | 默认值 | 含义 |
| --- | --- | --- | --- |
| `probabilities` | [number, number, number] | `[0.35, 0.15, 0.5]` | 非负概率向量且总和为 1；噪声依次 Gaussian/Poisson/ISO，压缩依次 JPEG/WebP/JPEG2000，模糊依次 Gaussian/generalized/plateau 的各向同性与各向异性。 |
| `gaussian_sigma` | [number, number] | `[1.0, 9.0]` | 高斯噪声 sigma 范围，按 8-bit 强度单位填写。 |
| `poisson_scale` | [number, number] | `[0.05, 0.5]` | 固定 255 级参考的泊松残差幅度，不随图片颜色数量变化。 |
| `iso_shot` | [number, number] | `[0.0002, 0.006]` | 线性光域信号相关方差系数。 |
| `iso_read` | [number, number] | `[5e-06, 0.00015]` | 线性光域信号无关读出方差。 |
| `iso_color_gain` | [number, number] | `[0.92, 1.15]` | ISO 各颜色通道增益范围。 |
| `exposure` | [number, number] | `[0.75, 1.0]` | 恢复曝光前的线性采集曝光倍率；范围 (0,1]。 |
| `gray_probability` | number | `0.25` | 退化转灰度的概率；噪声子表中表示灰度噪声概率，0..1。 |

## `SecondNoiseConfig`

交付网格或最终 HQ 尺寸上的微弱残差噪声，沿用采集阶段的场景强度倍率。

| 字段 | 类型／可选值 | 默认值 | 含义 |
| --- | --- | --- | --- |
| `probabilities` | [number, number, number] | `[0.7, 0.1, 0.2]` | 非负概率向量且总和为 1；噪声依次 Gaussian/Poisson/ISO，压缩依次 JPEG/WebP/JPEG2000，模糊依次 Gaussian/generalized/plateau 的各向同性与各向异性。 |
| `gaussian_sigma` | [number, number] | `[0.2, 2.0]` | 高斯噪声 sigma 范围，按 8-bit 强度单位填写。 |
| `poisson_scale` | [number, number] | `[0.02, 0.15]` | 固定 255 级参考的泊松残差幅度，不随图片颜色数量变化。 |
| `iso_shot` | [number, number] | `[1e-05, 0.0002]` | 线性光域信号相关方差系数。 |
| `iso_read` | [number, number] | `[1e-07, 4e-06]` | 线性光域信号无关读出方差。 |
| `iso_color_gain` | [number, number] | `[0.95, 1.05]` | ISO 各颜色通道增益范围。 |
| `exposure` | [number, number] | `[1.0, 1.0]` | 恢复曝光前的线性采集曝光倍率；范围 (0,1]。 |
| `gray_probability` | number | `0.25` | 退化转灰度的概率；噪声子表中表示灰度噪声概率，0..1。 |

## `CompressionConfig`

概率顺序为 JPEG/WebP/JPEG2000，仅用于网络转载的中间编码；JPEG2000 默认关闭。

| 字段 | 类型／可选值 | 默认值 | 含义 |
| --- | --- | --- | --- |
| `probabilities` | [number, number, number] | `[0.85, 0.15, 0.0]` | 非负概率向量且总和为 1；噪声依次 Gaussian/Poisson/ISO，压缩依次 JPEG/WebP/JPEG2000，模糊依次 Gaussian/generalized/plateau 的各向同性与各向异性。 |
| `jpeg_quality` | [number, number] | `[40.0, 95.0]` | JPEG 质量范围，0..100；数值越低压缩越强。 |
| `recompression_probability` | number | `0.25` | Extra shifted-origin codec pass probability. |
| `webp_quality` | [number, number] | `[60.0, 98.0]` | WebP 质量范围，0..100。 |
| `jpeg2000_quality` | [number, number] | `[250.0, 900.0]` | JPEG2000 原生 OpenCV 质量范围，0..1000。 |

## `SecondCompressionConfig`

所有场景的最终编码。网络转载使用完整强度分布；普通/低照度场景将压缩强度减半。

| 字段 | 类型／可选值 | 默认值 | 含义 |
| --- | --- | --- | --- |
| `probabilities` | [number, number, number] | `[0.8, 0.2, 0.0]` | 非负概率向量且总和为 1；噪声依次 Gaussian/Poisson/ISO，压缩依次 JPEG/WebP/JPEG2000，模糊依次 Gaussian/generalized/plateau 的各向同性与各向异性。 |
| `jpeg_quality` | [number, number] | `[30.0, 95.0]` | JPEG 质量范围，0..100；数值越低压缩越强。 |
| `recompression_probability` | number | `0.25` | Extra shifted-origin codec pass probability. |
| `webp_quality` | [number, number] | `[45.0, 98.0]` | WebP 质量范围，0..100。 |
| `jpeg2000_quality` | [number, number] | `[200.0, 900.0]` | JPEG2000 原生 OpenCV 质量范围，0..1000。 |

## `restore` 独立命令

使用独立 TOML 和对应命令。图像生成 sampling 表使用上方 preview.samples 的图像采样字段。

| 字段 | 类型／可选值 | 默认值 | 填写方式 |
| --- | --- | --- | --- |
| `runtime` | object | `子表默认值` | 共享 CUDA/cuDNN 后端选项子表；各命令可独立设置。 |
| `model` | object | `必填` | 模型选择与本地权重路径子表；校准命令只填 family 与 vae。 |
| `input` | string | `必填` | 待修复图像文件；输出保留经过方向校正后的输入尺寸。 |
| `output` | string | `必填` | 推理输出文件路径；图像通常为 PNG，视频按对应命令配置。 |
| `prompt` | string | `""` | 本次推理或预览使用的提示词。 |
| `attention_backend` | "torch", "kitchen-int8" | `"torch"` | torch 使用原生注意力；kitchen-int8 需要相应运行环境，仅用于支持的推理。 |
| `sigma` | number / null | `null` | 修复噪声水平，严格在 (0,1)；显式设置时不同时开启 calibrate。 |
| `adapter` | object / null | `null` | 适配器设置；训练在 train.adapter 填算法，推理在 adapter 填 weights 和 config。 |
| `encoder_weights` | string / null | `null` | 修复 encoder adapter 或 REPA 特征编码器路径，按所在表的任务选择。 |

## `generate` 独立命令

使用独立 TOML 和对应命令。图像生成 sampling 表使用上方 preview.samples 的图像采样字段。

| 字段 | 类型／可选值 | 默认值 | 填写方式 |
| --- | --- | --- | --- |
| `runtime` | object | `子表默认值` | 共享 CUDA/cuDNN 后端选项子表；各命令可独立设置。 |
| `model` | object | `必填` | 模型选择与本地权重路径子表；校准命令只填 family 与 vae。 |
| `sampling` | object | `子表默认值` | 推理采样子表，按图像、Wan 或 H3 选择对应字段。 |
| `adapter` | object / null | `null` | 适配器设置；训练在 train.adapter 填算法，推理在 adapter 填 weights 和 config。 |
| `output` | string | `必填` | 推理输出文件路径；图像通常为 PNG，视频按对应命令配置。 |

## `Wan generate-video` 独立命令

使用独立 TOML 和对应命令。图像生成 sampling 表使用上方 preview.samples 的图像采样字段。

| 字段 | 类型／可选值 | 默认值 | 填写方式 |
| --- | --- | --- | --- |
| `runtime` | object | `子表默认值` | 共享 CUDA/cuDNN 后端选项子表；各命令可独立设置。 |
| `model` | object | `必填` | 模型选择与本地权重路径子表；校准命令只填 family 与 vae。 |
| `sampling` | object | `子表默认值` | 推理采样子表，按图像、Wan 或 H3 选择对应字段。 |
| `adapter` | object / null | `null` | 适配器设置；训练在 train.adapter 填算法，推理在 adapter 填 weights 和 config。 |
| `high_noise` | object / null | `null` | Wan2.2 高噪声模型及适配器的专属设置；按视频生成示例填写。 |
| `output` | string | `必填` | 推理输出文件路径；图像通常为 PNG，视频按对应命令配置。 |

## `H3 generate-video` 独立命令

使用独立 TOML 和对应命令。图像生成 sampling 表使用上方 preview.samples 的图像采样字段。

| 字段 | 类型／可选值 | 默认值 | 填写方式 |
| --- | --- | --- | --- |
| `inputs` | object | `必填` | H3 提示词、首末帧、控制和参考输入子表。 |
| `sampling` | object | `子表默认值` | 推理采样子表，按图像、Wan 或 H3 选择对应字段。 |
| `runtime` | object | `子表默认值` | 共享 CUDA/cuDNN 后端选项子表；各命令可独立设置。 |
| `model` | object | `必填` | 模型选择与本地权重路径子表；校准命令只填 family 与 vae。 |
| `adapter` | object / null | `null` | 适配器设置；训练在 train.adapter 填算法，推理在 adapter 填 weights 和 config。 |
| `output` | string | `必填` | 推理输出文件路径；图像通常为 PNG，视频按对应命令配置。 |

## `mid-timestep` 独立命令

使用独立 TOML 和对应命令。图像生成 sampling 表使用上方 preview.samples 的图像采样字段。

| 字段 | 类型／可选值 | 默认值 | 填写方式 |
| --- | --- | --- | --- |
| `runtime` | object | `子表默认值` | 共享 CUDA/cuDNN 后端选项子表；各命令可独立设置。 |
| `model` | object | `必填` | 模型选择与本地权重路径子表；校准命令只填 family 与 vae。 |
| `dataset` | object | `必填` | 命名数据组及可选 defaults；每组声明自身 source。 |
| `seed` | integer | `42` | 随机种子；固定后便于比较配方，续训由状态文件恢复 RNG。 |
| `posterior` | "mean", "sample" / null | `null` | 训练 VAE 后验 mean/sample；省略遵循各家族策略，推理/预览使用均值。 |
| `calibration_pairs` | integer | `32` | 自动校准使用的配对数量；非默认值要求 calibrate=true。 |

## `wan-generate.sampling`

| 字段 | 类型／可选值 | 默认值 | 填写方式 |
| --- | --- | --- | --- |
| `prompt` | string | `""` | 数据组中覆盖所有同名 txt；空字符串也会覆盖。采样配置中为生成提示词。 |
| `negative_prompt` | string | `""` | 采样负提示词；是否支持以及 CFG 语义由模型变体决定。 |
| `width` | integer | `832` | 输出宽度，像素；一般 16 对齐，Qwen2.1/H3 使用 32 对齐。 |
| `height` | integer | `480` | 输出高度，像素；与宽度一起决定采样显存需求。 |
| `frames` | integer | `81` | 生成帧数；遵守家族帧数约束并与 fps 一起决定时长。 |
| `fps` | integer | `16` | 生成帧率；H3 使用原生 24 FPS。 |
| `steps` | integer | `40` | 采样迭代数；省略使用变体默认，Turbo 与非蒸馏模型不要套用同一数值。 |
| `solver` | "euler", "dpm++", "unipc" | `"unipc"` | Wan 采样求解器名称；使用该家族支持的取值。 |
| `shift` | number | `5.0` | 采样时间表移位；含义因模型而异，省略用原生默认。 |
| `guidance` | number | `5.0` | 该模型原生采样 guidance/CFG；省略使用变体默认。 |
| `seed` | integer | `42` | 采样噪声种子；比较训练版本时保持不变。 |
| `attention_backend` | "torch", "kitchen-int8" | `"torch"` | torch 使用原生注意力；kitchen-int8 需要相应运行环境，仅用于支持的推理。 |
| `conditioning_image` | string / null | `null` | 单张条件图路径；如 FLUX Kontext/Fill 或 Wan I2V，遵循模型约束。 |
| `control_video` | string / null | `null` | Wan 控制视频路径；仅在对应 I2V 控制输入需要时填写。 |

## `wan-generate.high_noise`

| 字段 | 类型／可选值 | 默认值 | 填写方式 |
| --- | --- | --- | --- |
| `weights` | string | `必填` | 本地权重文件、分片索引或支持的模型目录；替换为实际路径。 |
| `boundary` | number | `必填` | Wan2.2 高低噪声专家切换的归一化边界，0..1。 |
| `guidance` | number / null | `null` | 采样中为模型 guidance/CFG；method.guidance 为受支持模型的训练 embedding。 |
| `adapter` | object / null | `null` | 适配器设置；训练在 train.adapter 填算法，推理在 adapter 填 weights 和 config。 |

## `h3-generate.inputs`

| 字段 | 类型／可选值 | 默认值 | 填写方式 |
| --- | --- | --- | --- |
| `prompt` | string | `必填` | 本次推理或预览使用的提示词。 |
| `first_frame` | string / null | `null` | H3 首帧条件路径。 |
| `last_frame` | string / null | `null` | H3 末帧条件路径。 |
| `controls` | [object, …] | `[]` | H3 带时间索引的控制图记录列表。 |
| `target_index` | integer / null | `null` | 目标帧索引；应与数据记录和条件索引对应。 |
| `references` | [table, …] | `[]` | H3 有序图像/视频/音频参考记录列表。 |

## `h3-generate.sampling`

| 字段 | 类型／可选值 | 默认值 | 填写方式 |
| --- | --- | --- | --- |
| `width` | integer | `768` | 输出宽度，像素；一般 16 对齐，Qwen2.1/H3 使用 32 对齐。 |
| `height` | integer | `1344` | 输出高度，像素；与宽度一起决定采样显存需求。 |
| `frames` | integer | `124` | 生成帧数；遵守家族帧数约束并与 fps 一起决定时长。 |
| `fps` | integer | `24` | 生成帧率；H3 使用原生 24 FPS。 |
| `temporal_fine_bands` | integer | `0` | H3 原生时间频带设置；常规推理保留模型默认。 |
| `allow_experimental_duration` | boolean | `false` | 允许 H3 实验性时长；常规训练保持 false。 |
| `steps` | integer | `30` | 采样迭代数；省略使用变体默认，Turbo 与非蒸馏模型不要套用同一数值。 |
| `seed` | integer | `42` | 采样噪声种子；比较训练版本时保持不变。 |
| `attention_backend` | "torch", "kitchen-int8" | `"torch"` | torch 使用原生注意力；kitchen-int8 需要相应运行环境，仅用于支持的推理。 |
| `video_shift` | number | `12.0` | H3 视频时间表移位，默认 12.0；训练和采样使用匹配配方。 |
| `audio_shift` | number | `3.0` | H3 音频时间表移位，默认 3.0。 |
| `visual_condition_clean` | number | `0.999` | H3 视觉条件清洁度；默认 0.999。 |
| `audio_condition_clean` | number | `1.0` | H3 音频条件清洁度；默认 1.0。 |

## `mid-timestep.dataset.defaults`

| 字段 | 类型／可选值 | 默认值 | 填写方式 |
| --- | --- | --- | --- |
| `resolution` | [integer, integer] | `[1024, 1024]` | [宽, 高] 像素；分桶开启时作为目标面积基准，实际尺寸随宽高比变化。 |
| `repeats` | integer | `1` | 该组每轮样本重复次数；提高小数据组的相对出现频率，不改变 max_updates。 |
| `horizontal_flip_probability` | number | `0.5` | 修复训练成对水平翻转概率 0..1；方向敏感任务填 0。 |

## `mid-timestep.dataset.groups.NAME`

| 字段 | 类型／可选值 | 默认值 | 填写方式 |
| --- | --- | --- | --- |
| `resolution` | [integer, integer] | `[1024, 1024]` | [宽, 高] 像素；分桶开启时作为目标面积基准，实际尺寸随宽高比变化。 |
| `repeats` | integer | `1` | 该组每轮样本重复次数；提高小数据组的相对出现频率，不改变 max_updates。 |
| `horizontal_flip_probability` | number | `0.5` | 修复训练成对水平翻转概率 0..1；方向敏感任务填 0。 |
| `source` | string | `必填` | 数据源目录或该模型要求的记录文件；改动内容后重建缓存。 |
| `low_quality_source` | string / null | `null` | 修复输入 LQ 目录，与 source 的 HQ 按文件主体配对；不同时开启合成退化。 |
| `degradation` | object / null | `null` | 合成 LQ 的两阶段退化子表；与同组 low_quality_source 互斥。 |

---

[文档首页](../README.md) · [参数参考](parameters.md) · [故障排查](../troubleshooting.md)
