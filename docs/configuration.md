# TOML 与命令行配置

[English](en/configuration.md) · [文档首页](README.md)

本页说明如何组织配置、覆盖参数和选择命令。字段类型、默认值与模型限制见[参数参考](training/parameters.md)。

<details>
<summary>本页导航</summary>

- [配置优先级](#配置优先级)
- [表的职责](#表的职责)
- [TOML 写法](#toml-写法)
- [CLI 字段与复杂值](#cli-字段与复杂值)
- [命令速查](#命令速查)
- [修改后需要做什么](#修改后需要做什么)

</details>

## 配置优先级

**配置默认值 → TOML → CLI 覆盖**。省略 TOML 时也可以完全通过 CLI 提供必填字段。

```bash
uv run --no-dev unitunner train train.toml --device cuda --train.learning-rate 5e-5
uv run --no-dev unitunner train --config train.toml --device cuda --train.learning-rate 5e-5
```

两种文件写法选择同一个配置。相对路径始终以命令的工作目录为基准，不是 TOML 所在目录。需要从不同目录启动时，模型与数据路径使用绝对路径。

## 表的职责

| 表 | 内容 | 适用命令 |
| --- | --- | --- |
| `model` | 家族、变体、权重、编码资产 | 训练、缓存、生成、修复 |
| `model.text` | 文本精度、量化、分词器/处理器 | 编码与文本条件推理 |
| `dataset.defaults` | 多组数据的公共默认设置 | 训练、缓存 |
| `dataset.groups.NAME` | 独立来源、缓存、尺寸和抽样设置 | 训练、缓存 |
| `train` / `train.adapter` | 更新、优化器、可训练范围 | 训练 |
| `method` / `losses` | 目标、辅助目标、修复损失 | 训练 |
| `execution` | 计算与显存策略 | 训练 |
| `preview` / `output` | 预览、权重、续训和日志 | 训练 |
| `sampling` / `adapter` | 采样参数与待加载适配器 | 生成 |
| `runtime` | TF32、cuDNN 等运行选项 | 支持该表的训练、缓存和推理命令 |

只添加当前任务需要的表。图像生成、视频生成、修复推理和时间校准使用各自配置结构，不将训练文件直接当作所有命令的输入。

## TOML 写法

```toml
[train]
learning_rate = 5e-5
accumulation_steps = 4
include_layers = ["layers.*"]

[dataset.defaults]
batch_size = 1
resolution = [1024, 1024]

[dataset.groups.portraits]
kind = "image"
source = "/data/portraits"
cache_directory = "/cache/portraits"
repeats = 2
```

这是配置片段，应合并到已有文件。同一表不能重复声明。字符串加引号，布尔值写 `true` / `false`；TOML 没有 `null`，参数表中的 `null` 表示省略字段。空字符串、空数组与省略的语义可能不同，例如 `prompt = ""` 会覆盖 caption 为明确的空文本。

## CLI 字段与复杂值

字段层级使用点号，字段名可使用连字符形式；动态数据组使用 `--set`：

```bash
uv run --no-dev unitunner validate train.toml \
  --train.learning-rate 5e-5 \
  --set 'dataset.groups.main.batch_size=1' \
  --set 'train.include_layers=["layers.*"]'
```

包含空格、数组、JSON 对象或 shell 特殊字符时对整个赋值加引号。常用字段也可以查看 `unitunner train --help`。切换全参数训练使用 `--parameter-mode full`，该快捷选项会同时清除适配器配置。

## 命令速查

| 命令 | 输入与结果 |
| --- | --- |
| `validate` | 校验配置；默认按训练配置检查，不加载模型 |
| `cache` | 编码数据；成功后替换对应缓存目录 |
| `train` | 读取数据并更新参数，发布权重与续训状态 |
| `generate` | 使用图像模型和可选适配器生成图像 |
| `generate-video` | Wan 视频或 H3 图像/音视频生成 |
| `restore` | 一步图像修复 |
| `mid-timestep` | 使用 VAE 与配对图像估计修复时间 |
| `weights` | 权重导入、转换等操作，使用 `--help` 查看子命令 |
| `caches import` | 按显式规格导入 MiniMaxH3 外部缓存 |

单独检查不同任务：

```bash
uv run --no-dev unitunner validate image-cache.toml --for cache
uv run --no-dev unitunner validate generate.toml --for generate
uv run --no-dev unitunner validate restore.toml --for restore
uv run --no-dev unitunner validate video.toml --for generate-video
uv run --no-dev unitunner validate calibration.toml --for mid-timestep
```

配置检查通过表示字段及组合合法，不表示资产文件已存在或与模型拓扑匹配。实际加载还会检查文件格式、键名和形状。

## 修改后需要做什么

| 改动 | 后续操作 |
| --- | --- |
| 图片、caption、参考图、VAE、文本编码/预处理 | 重建对应缓存 |
| 保存周期、输出位置、预览频率 | 检查输出目录与预览资产 |
| 模型、适配器算法、训练目标或数值策略 | 建立新训练，或确认现有续训状态通过身份检查 |
| 只修改推理 prompt、seed、steps | 使用生成配置；无需重新训练 |

继续阅读：[数据集](training/datasets.md) · [参数参考](training/parameters.md) · [故障排查](troubleshooting.md)。

---

[文档首页](README.md) · [参数参考](training/parameters.md) · [故障排查](troubleshooting.md)
