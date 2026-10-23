# 故障排查

[English](en/troubleshooting.md) · [文档首页](README.md)

先确定失败阶段：配置校验 → 编码缓存 → 训练 → 预览 → 导出/推理。保存报错前后的日志、运行命令和实际配置，有助于判断是资产、数据还是执行选项的问题。

<details>
<summary>本页导航</summary>

- [配置与资产](#配置与资产)
- [缓存与数据](#缓存与数据)
- [显存不足](#显存不足)
- [数值与训练效果](#数值与训练效果)
- [续训与输出](#续训与输出)
- [ComfyUI 与独立推理结果不同](#comfyui-与独立推理结果不同)
- [提供可定位的信息](#提供可定位的信息)

</details>

## 配置与资产

| 现象 | 检查与处理 |
| --- | --- |
| unknown / extra field | 确认字段所属表，使用[参数参考](training/parameters.md)和当前命令 `--help`；不要把训练配置直接当作生成配置 |
| 路径存在但找不到文件 | 相对路径以命令工作目录为准；检查分片文件是否完整、是否误填 tokenizer 路径为文本权重 |
| 权重 keys / shape / topology 不匹配 | 核对家族、variant、version、资产组件和格式；仅修改名称不会转换模型架构 |
| 分词器无法下载或离线缺文件 | 显式配置本地 `model.text.tokenizer.source`；必要时设置 subfolder，检查 processor/CLIP 的独立来源 |
| optional dependency missing | 按[安装指南](installation.md)安装对应 extra，并确认命令使用同一环境 |
| CUDA unavailable | 检查驱动、Torch CUDA 构建与可见设备；CPU 流程使用 FP32 并删除 CUDA 专属选项 |

## 缓存与数据

### 图片或 caption 没有生效

源图放在目录直属层级，caption 使用同名 UTF-8 文件。数据组或 defaults 中的 `prompt` 会覆盖侧文件；明确的空 `prompt = ""` 也会覆盖。修改源数据、条件图、VAE、文本设置或裁剪规则后重新执行 `cache`。

### 缓存摘要、布局或身份不匹配

不要手动编辑 safetensors 或 manifest。确认训练选择了对应模型和数据组的缓存，再重建该目录。每个组使用独立缓存目录；参考图、掩码和 HQ/LQ 必须遵守[配对规则](training/datasets.md)。

`cache` 每次完整重建，成功后替换目录。重建时旧缓存与新缓存会同时占用磁盘；不要让训练同时读取该目录。若日志报告回滚失败，先检查日志中的备份位置，保留它再排查文件权限与磁盘问题。

## 显存不足

| 出错阶段 | 优先检查 |
| --- | --- |
| 加载底模 | 适配器训练考虑 `frozen-blocks` / VBAR，或支持的量化底模；检查其他 GPU 进程 |
| 前向或反向 | 减小 batch/尺寸，开启 `recompute = "block"`；注意力保存占用大时考虑激活卸载 |
| 更大 bucket 首次出现 | 自动缓存容量不能保证所有未知尺寸都适用；降低最大面积或调整数据尺寸 |
| 预览或保存 | 预览需要额外 VAE/文本资产；减少预览尺寸和样本数，确认主机内存也充足 |

`residency = "resident"` 不会在 OOM 后自动切到卸载模式。H2D 在所选 `frozen-blocks` 模式内自动调节缓存；VBAR 在所选模式内管理驻留页。梯度累积增加有效批量，不减少单个 microbatch 的激活需求。详见[执行策略](training/execution.md)。

## 数值与训练效果

### loss 或梯度出现 NaN / Inf

检查输入缓存、学习率、损失权重和所选精度。FP16 可降低 `execution.fp16_loss_scale`，或在支持设备上使用 BF16。日志会在非有限梯度更新前报错，不会静默跳过该批次。修改影响数值的策略后不要强行绕过续训身份检查。

### loss 下降，但预览不符合预期

固定预览 prompt、seed、尺寸与采样设置，再检查 caption、触发词、训练层选择、数据比例和训练长度。比较分损失，确认没有被某一个较大的损失项主导。Base、Turbo 和蒸馏模型的 steps/guidance 不能通用。

### 训练能跑，预览失败

训练可从 latent 缓存读取，预览还需 VAE、文本权重、分词器及相应条件图。检查这些资产，确认预览配置与模型变体一致。关闭预览并不能解决实际生成资产不匹配。

## 续训与输出

- 用 `resume/` 中的 `.state-…safetensors`，不要用 `weights/` 中的 LoRA 续训。
- `max_updates` 表示目标总数；恢复已达到总数的状态不会再更新。
- `frozen-reference` 状态要求完全相同的冻结底模；替换量化文件也可能改变身份。
- 省略 `output.directory` 会创建新运行目录；显式目录的新训练会清理同名训练产物。继续原训练必须传 `--resume`。
- 默认保留一份 resume，正式权重不限；找不到更早状态时检查保留设置。

## ComfyUI 与独立推理结果不同

按以下顺序核对：

1. 使用 `weights/` 导出的文件，匹配底模家族、variant/version 和适配器算法。
2. 确认加载器支持导出的键布局；FLUX 系列等模型可能涉及融合投影拆分，不要仅重命名键。
3. 对齐 VAE、文本编码器、prompt、负面提示词、图像条件和参考图顺序。
4. 对齐采样器、时间/shift 约定、steps、guidance、尺寸和适配器强度。
5. 再检查量化、计算精度与注意力后端。跨运行时相同 seed 不一定产生相同噪声。

训练预览使用原生评估路径；训练启用 compile 不表示外部推理必须复制训练编译策略。格式规则见[权重转换](weight-conversion.md)，使用入口见[推理](inference.md)。

## 提供可定位的信息

报告问题时附运行命令、模型家族与变体、脱敏配置、完整异常、Torch/CUDA 版本，以及问题发生在缓存/训练/预览/导出的哪一阶段。日志位置与读取方法见[日志指南](logging.md)。

---

[文档首页](README.md) · [参数参考](training/parameters.md) · [故障排查](troubleshooting.md)
