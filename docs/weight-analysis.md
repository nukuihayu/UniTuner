# 最终权重统计分析

```bash
uv run unitunner analyze runs/oil-skin-lora
uv run unitunner analyze runs/oil-skin-lora --base /models/base.safetensors
uv run unitunner analyze --list
```

默认读取项目 `weights/` 下不带步数后缀的主权重和 `.encoder.safetensors`，分别分析；也可直接指定权重文件。排除 `*-00000001.safetensors`、对应的 encoder 定期导出和 `resume/`，没有最终导出时明确报错，不自动回退。也可直接选择最终 safetensors 文件或分片索引。

注意：当前训练器可能在保存边界更新无步数后缀的 latest 文件。因此这里“最终权重”指可消费的无步数导出文件，文件名本身不证明训练已完成；报告记录文件内的实际 update，不猜测完成状态。不要在训练器正在替换文件时同时分析。

<details>
<summary>本页导航</summary>

- [Notebook 与终端](#notebook-与终端)
- [图表如何阅读](#图表如何阅读)
- [指标与边界](#指标与边界)
- [比较和计算预算](#比较和计算预算)
- [LoRA ΔW 与合并权重 W₀+ΔW](#lora-δw-与合并权重-w₀δw)

</details>

## Notebook 与终端

```python
from unitunner.analysis import analyze_run
report = analyze_run("runs/oil-skin-lora", show="auto")
```

绘图统一使用 Plotly，不依赖 Matplotlib 或 Seaborn。Notebook 直接调用会显示可缩放、悬停查看完整参数路径的 Plotly 交互图；`!uv run ...` 是独立进程，不等同于 notebook 内联显示。终端输出数值摘要，支持的终端内联显示图像；桌面终端可打开查看器，无图形界面时保存文件。`--show never` 禁止显示。

PNG/PDF 导出由 Kaleido 完成。优先复用已有 Chrome/Chromium；未找到时自动下载到 Kaleido 用户缓存，后续复用，无需管理员权限。首次导出需要联网，自动下载最多等待 180 秒；离线环境可通过 `BROWSER_PATH` 指定已有浏览器。Linux 仍需浏览器的系统运行库。下载、浏览器启动或渲染失败时会给出明确错误，并保留 CSV 和 `summary.md`，不会挂起等待后台线程或回退其他绘图库。`--show never` 只禁止显示，仍会导出 PNG/PDF。

每次自动保存到 `项目/analysis/时间戳/`，包含 PNG、多页 `figures.pdf`、CSV 和 `summary.md`，没有 HTML 或中间数据文件。`--output` 可指定结果目录。

## 图表如何阅读

采用 Plotly 生成论文风格图：白底、统一无衬线字体、细坐标轴、低饱和分类配色与点形／线型双重编码，避免大数字卡片、树图和大小气泡。每个问题独立成图。完整方法、数据覆盖和文件身份移入 `figure-captions.md`，图面只保留必要的标题、坐标、图例与统计量。

- **更新能量排名**：前 20 层的平方范数占比，分母始终包含该角色全部符合条件的层，不把前 20 层重新归一化。图注记录 90% 能量所需层数、有效层数和覆盖范围。
- **层间关系散点图**：稳定秩／LoRA 容量与相对基模变化的关系，使用固定大小点，颜色区分模块。少于 3 个有效样本时不生成缺乏意义的关系图。无法匹配基模时显式使用可用指标，零或未定义的对数轴值不伪造。
- **模块箱线图**：RMS 的中位数、四分位数、须和观测点；最多 6 个模块一页。每组不超过 100 个样本时展示全部点，否则显示离群点。
- **谱曲线**：最多 6 个可计算谱的大范数层。Notebook 中可切换累计能量和归一化谱；论文 PNG/PDF 不显示交互按钮，导出默认累计能量视图。
- **逐层数值表**：每页最多 24 层，列出 RMS、相对变化、稳定秩、90% 能量秩和同类尺度偏差。不可用值显示破折号；小样本页自动缩短画布。
- **二维矩阵**：模块、奇异向量对齐与 head 热图统一使用参考图风格的蓝—浅色—红色色带。对齐图的颜色表示绝对余弦的对数大小；head 方向图则使用以 0 为中点、按实际范围对称缩放的有符号余弦（原始值范围 [-1, 1]）。分页使用相同色标。
- **训练日志曲线**：不同量纲分别成图，Notebook 支持联合悬停和范围滑块。

标签中的 L 是文件内自然排序编号，B 是序列化名称中的 block/layer 索引；`labels.csv` 保留完整路径。图中数值是描述性统计，不能单独证明过拟合或图像质量。

PNG 与多页 PDF 自动保存；PDF 中曲线、文字和坐标轴保留矢量，热图按渲染器支持保存。具体投稿仍需按目标期刊的栏宽、字号与格式要求调整。

## 指标与边界

`layers.csv` 包含 FP64 有限值统计、NaN/Inf、零值率、确定性采样分位数、基模匹配和谱指标。LoRA 分析 `ΔW=(alpha/r)BA`，而非把 A/B 因子的尺度直接当作模型变化；encoder LoRA 使用其真实单位缩放，LoHa/LoKr 按各自乘积计算。

`diagnostics.csv` 和 `allocation.csv` 进一步提供：

| 指标 | 定义与用途 | 不能据此断言 |
|---|---|---|
| 层能量占比 | 有效更新的平方 Frobenius 范数占比；全参使用匹配基模的差值，无基模时明确标为存储权重能量 | 大层能量多就是异常 |
| 有效层数 | `1 / sum(p_layer²)`；90% 能量所需层数描述更新集中程度 | 集中一定导致过拟合 |
| 稳定秩 | `sum(s²) / max(s²)` | 模型可直接压缩到此整数秩 |
| 90% 能量秩 | 解释 90% 平方奇异值和所需的方向数 | 能保持验证集质量 |
| LoRA 容量使用 | 上述秩除以 `min(r, in, out)`，仅用于 LoRA | 可把相同分母套给 LoHa/LoKr |
| 同类层尺度偏差 | 同角色、算法、形状、末级模块名内，log10(RMS) 的 median/MAD 稳健分数 | 分数超过 3.5 就是训练故障 |
| 子空间重叠 | `||UᵀV||F² / k`；同时报告主角度和谱能量覆盖 | 高重叠等于高图像质量 |

同类尺度偏差需要至少 5 个正有限样本、非零 MAD；不足时不制造分数。基模相对变化遇到零分母保持未定义。能量统计按角色隔离，并报告参与层数。FP64 非有限结果标记为空值。

重点排查 NaN/Inf、持续为零的有效更新、相对基模过大且影响固定样本质量的层。秩利用率低适合提出减秩对照实验；更新高度集中适合检查目标层和数据偏差；任何优化结论仍需验证集和固定 seed 预览支持。

## 比较和计算预算

```bash
uv run unitunner analyze runs/oil-skin-lora --base /models/base.safetensors \
  --layer '*attn*' --top-k 64 --top-layers 6
```

默认最多选择 3 层做子空间比较；`--layer` 可重复并显式选择层。`--compare` 可提供另一个完整模型权重，生成三方比较。基模与权重按明确名称、形状、已知官方／ComfyUI 投影映射或 `--key-map mapping.json` 匹配，不猜测下划线与点号关系。encoder 对照需要对应 VAE 基模和明确映射，不能拿 denoiser 基模冒充。

底模支持单文件、分片索引及官方仓库的本地 transformer 目录。已知布局差异使用只读投影视图处理：Qwen Image 2.1 的 `[gate_layer; proj]`、Z-Image 的 Q/K/V（含不等宽 GQA）、FLUX 的 QKV 和单流融合输入、FLUX3 QKV，以及 H3 的 QKV 和 SwiGLU 顺序。其他受支持模型复用训练加载器的明确官方层名映射。只在所分析的一层需要时拼接对应参考矩阵，不加载整模或改写底模。对普通 LoRA，同时识别 ComfyUI down/up/alpha 和 Diffusers A/B；后者读取官方 PEFT 的 alpha、逐层 pattern 和 RSLoRA 缩放元数据，无显式 alpha 时按 vanilla PEFT 的单位缩放处理。DoRA 不会被误算为普通 LoRA。

这修复的是存储布局匹配，不证明底模版本相同。量化打包权重仍标记为不可直接比较，需先提供浮点底模。Qwen 2.1 的 ComfyUI LoRA 本身保留拆分层名，见 [ComfyUI LoRA 映射](https://github.com/Comfy-Org/ComfyUI/blob/master/comfy/lora.py)；A/B 单位缩放依据 [Diffusers PEFT 加载逻辑](https://github.com/huggingface/diffusers/blob/main/src/diffusers/utils/peft_utils.py)。

精确谱分解受 `--svd-max-elements`、`--factor-budget` 限制；子空间矩阵受 `--matrix-budget` 限制，大矩阵使用固定 seed 的随机化 SVD，并报告捕获能量。卷积展开为 `[out, in×kernel]`；默认不解码 packed quantization，需浮点导出来分析逻辑权重。

**只分析最终导出无法恢复参数随时间的轨迹。** 日志里的 loss、梯度和学习率曲线是独立观测，不把多个最终文件伪装成训练时间序列。

参考：[LoRA](https://arxiv.org/abs/2106.09685)、[LyCORIS](https://arxiv.org/abs/2309.14859)、[随机化矩阵分解](https://arxiv.org/abs/0909.4061)。

图形采用独立右侧图例区域，图例不会占用坐标标题的位置。轴刻度与图例字号为 24 px，基础文字为 26 px，标题为 30 px；导出时随画布等比缩放。能量图每行预留 44 px，箱线图每页最多六组并统一使用深蓝描边。谱图的交互按钮仅用于 notebook，静态 PNG/PDF 隐藏按钮，保留右侧图例。

## LoRA ΔW 与合并权重 W₀+ΔW

提供 `--base` 时，LoRA 子空间和 head 图比较 **有效更新 ΔW** 与 **合并后的权重 W₀+ΔW**，不是因子 A/B，也不是把 W₀+ΔW 误标为原始基模 W₀。原有逐层基模相对变化等统计仍使用 W₀，两个比较口径分别标明。

- 奇异向量对齐矩阵：默认右奇异向量的 `|V_mergedᵀ V_delta|`，色标为 log10，上限固定为 1、下限按本图全部对比矩阵共同确定，10⁻⁴ 以下只在着色时截断，悬停保留实际数值。`--side left` 可选左奇异向量。每对矩阵取相同的非零方向数 k；LoRA 的有效秩可能小于 `--top-k`。报告主角度、子空间重叠和这个 k 对应的能量覆盖率，不能把重根中的基向量旋转当成结构变化。
- 层×head 幅度图：每个声明的输出行分组计算 `‖ΔW_h‖F / ‖(W₀+ΔW)_h‖F`，以 log10 着色。它衡量更新量相对合并权重的大小；分母接近零可能放大比例，不能直接等同于训练异常。
- 层×head 方向图：计算展平后 `cos(ΔW_h, (W₀+ΔW)_h)`，蓝色为反向、浅色为正交、红色为同向；任一范数为零时留空。这里不是激活注意力图，也不代表注意力概率或 head 重要性。

`heads.csv` 保存幅度比例、带符号余弦和比较对象，`subspaces.csv` 保存聚合指标，`principal-angles.csv` 保存逐方向主角度。head 热图每页最多 24 层×32 个 head，完整路径在 `labels.csv`；图注注明选择范围。需要显式 `--heads` 并确认输出行连续分组符合真实模型布局；融合 QKV、GQA、转置投影等不可仅凭维度推断。默认子空间/head 分析选择最多 3 层；用 `--top-layers` 增加层数，或重复 `--layer` 明确选择需要分析的层。全参权重继续使用训练权重与原始基模的比较，不伪装成 LoRA。

色条标题放在独立注释区，与最高刻度间保留留白；对齐图使用左右对称外边距。head 图按实际行列数调整画布，避免少量 head 被横向拉伸。对数对齐图与有符号 head 图使用不同的色标范围，不跨图按颜色直接比较数值。

所有旧图同步使用无衬线字体、蓝红分类色与 coolwarm 热图色带，色条标题独立排版。分析重点是训练产物：LoRA 使用有效 ΔW；全参产物匹配基模后，更新幅度箱线图和同类 RMS 比较使用 W训练−W₀，未匹配或非有限差值不混入。没有基模时明确标注存储权重。`diagnostics.csv` 的 `rms_kind` 标明来源；`layers.csv` 保留原始存储统计。全参谱图明确标为训练权重谱，不冒充差值谱。

---

[文档首页](README.md) · [参数参考](training/parameters.md) · [故障排查](troubleshooting.md)
