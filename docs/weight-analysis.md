# 分析训练权重

[English](en/weight-analysis.md) · [文档首页](README.md)

使用权重分析查看适配器更新是否集中、是否存在异常值，以及训练结果相对底模的变化。建议结合固定提示词预览或验证集判断效果。

## 生成第一份报告

```bash
uv run unituner analyze --list
uv run unituner analyze runs/my-run --base /models/base.safetensors
```

将 `runs/my-run` 换成训练输出目录，`--base` 指向训练时使用的原始底模。也可以直接分析一个权重文件：

```bash
uv run unituner analyze runs/my-run/weights/model.safetensors --base /models/base.safetensors
```

不提供 `--base` 时仍可查看权重统计，但不会得到完整的底模相对变化比较。比较编码器适配器时，使用对应的 VAE 底模。

默认选择 `weights/` 中不带步数后缀的导出文件，并单独分析编码器适配器；不会自动选择定期编号权重或 `resume/` 状态。定期编号权重和续训状态不能作为分析输入。无步数后缀的文件也可能是训练中的最近一次导出，请查看报告记录的更新数。

## 查看与保存报告

结果默认保存在训练目录的 `analysis/<时间戳>/`，也可以指定目录：

```bash
uv run unituner analyze runs/my-run --base /models/base.safetensors \
  --output reports/my-run --show never
```

| 文件 | 用途 |
| --- | --- |
| `summary.md` | 分析摘要和主要结果 |
| `figures.pdf`、PNG 图表 | 直接查看、分享或保留对照 |
| `layers.csv` | 逐层权重统计 |
| `diagnostics.csv`、`allocation.csv` | 更新大小、异常值和分布 |
| `labels.csv` | 图中简写与完整层名的对应关系 |
| `heads.csv`、`subspaces.csv` | 启用对应分析时的 head 和子空间结果 |

`--show never` 关闭自动显示，仍会保存图表和表格。生成报告时避免同时覆盖正在分析的权重文件。

## 图表如何阅读

| 图表 | 主要查看什么 |
| --- | --- |
| 层能量排名 | 更新主要集中在哪些层 |
| 更新幅度分布 | 同类模块是否存在明显偏大、偏小或持续为零的更新 |
| 谱曲线与有效秩 | 适配器容量的使用情况，可用于选择减小 rank 的对照实验 |
| 相对底模变化 | 更新量相对原始权重的大小 |
| 训练日志曲线 | loss、学习率、梯度和显存随训练的变化 |

LoRA 报告分析实际权重增量；全参数权重在提供匹配底模时分析两者差值。没有底模时，请按报告中的“存储权重”标记解读。

NaN／Inf 应优先排查。较大的更新量、较低的有效秩或更新集中，单独都不能证明训练失败或过拟合；还需要比较生成结果。

## 选择层和比较范围

```bash
uv run unituner analyze runs/my-run --base /models/base.safetensors \
  --layer '*attn*' --top-k 64 --top-layers 6
```

| 选项 | 用途 |
| --- | --- |
| `--layer` | 按层名筛选，可重复使用 |
| `--top-layers` | 增加或减少详细分析的层数，默认最多 3 层 |
| `--top-k` | 控制子空间比较的方向数 |
| `--compare` | 添加另一份完整权重进行比较 |
| `--key-map` | 提供明确的层名映射 |
| `--heads` | 在已知输出行分组时查看 head 统计 |

head 图描述权重大小和方向，不是图像生成时的注意力图。模型使用融合投影或特殊 head 布局时，请先确认分组。量化底模无法直接做完整逻辑权重比较时，提供对应的浮点文件。

## Notebook 中使用

```python
from unituner.analysis import analyze_run

report = analyze_run("runs/my-run", show="auto")
```

直接在 Python 单元格调用可显示交互图，支持缩放和悬停查看层名。通过 `!uv run` 启动命令时，查看保存的报告文件。

## 图表导出失败

PNG／PDF 导出需要 Chrome 或 Chromium。首次运行可自动下载浏览器；离线环境使用 `BROWSER_PATH` 指向已有浏览器。也可以提前安装：

```bash
uv run plotly_get_chrome -y
```

Linux 还需要浏览器的系统运行库。导出失败时先查看错误日志；已生成的 CSV 和 `summary.md` 可继续使用。

---

[文档首页](README.md) · [推理与模型使用](inference.md) · [故障排查](troubleshooting.md)
