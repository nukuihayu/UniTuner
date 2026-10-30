# 特征目标与 Token 路由

简体中文 · [English](../en/training/objectives.md)

[文档目录](../README.md) · [训练指南](README.md) · [示例配置](../examples/README.md)

LayerSync、REPA 和 TREAD 可组合用于当前十个家族的流匹配训练，不接入图像修复。
LoRA／LoHa／LoKr 和 full 使用相同方法配置，层窗口须符合所选原生模型拓扑。
重算、编译和驻留是独立的[执行策略](execution.md)，不改变这里定义的训练目标。

<details>
<summary>本页导航</summary>

- [选择方法](#选择方法)
- [它们分别改变什么](#它们分别改变什么)
- [配置](#配置)
- [层与路由约束](#层与路由约束)
- [特征与梯度](#特征与梯度)
- [REPA 缓存](#repa-缓存)
- [Wavelet loss](#wavelet-loss)

</details>

## 选择方法

| 方法 | 职责 | 额外资源 |
| --- | --- | --- |
| LayerSync | 前层学生与停止梯度的后层教师做负余弦对齐 | 无需外部教师权重 |
| REPA | 原生目标特征经可训练投影后对齐固定编码器特征 | 匹配的 DINOv3 ViT 权重、摘要与目标特征缓存 |
| TREAD | 在指定层窗口路由一部分目标 token | 无需外部编码器；每次前向使用路由 RNG |

## 它们分别改变什么

LayerSync 与 REPA 在基础流匹配损失上增加监督：

```text
L_total = L_flow + w_sync · L_sync + w_repa · L_repa
```

LayerSync 用同一模型的后层特征作教师，不需要外部网络。REPA 引入独立预训练视觉表征，因此需要编码器身份与严格对应目标图的特征缓存。TREAD 改变指定层窗口中实际参与计算的目标 token，不额外增加一项损失。丢弃一半目标 token 不等于整个训练耗时减半，文本、参考条件和其他计算仍然存在。

先建立可复现的基础流匹配实验，每次启用一种方法，比较保留集输出、显存和吞吐，再决定是否组合。关闭可选方法时删除对应配置段；这些方法的权重要求大于 0，不能用 `weight=0` 充当关闭开关。

| 参数 | 默认值或要求 | 含义 |
| --- | --- | --- |
| LayerSync `weight` | `1.0`，大于 0 | 附加损失权重 |
| `student_layer`、`teacher_layer` | `0`、`1` | 从 0 开始的层号，教师层必须在学生之后 |
| REPA `weight`、`layer` | `1.0`、`0` | 投影对齐损失权重与特征提取层 |
| `encoder_dim` | 必填正整数 | 与真实编码器特征宽度一致 |
| `encoder_sha256` | 必填，64 位小写十六进制 | 固定编码器文件内容身份 |
| TREAD `drop_fraction` | `0.5`，严格介于 0 与 1 | 在指定窗口旁路的目标 token 比例 |
| `start_layer`、`end_layer` | `0`、`1` | 非空的左闭右开层窗口 |

## 配置

以下为训练 TOML 的附加片段；完整模板见[flow-features.toml](../examples/flow-features.toml)。
可以删除不使用的方法配置，不能将占位摘要用于实际缓存。

```toml
[method.tread]
drop_fraction = 0.5
start_layer = 0
end_layer = 2

[method.layersync]
weight = 0.2
student_layer = 0
teacher_layer = 1

[method.repa]
weight = 0.3
layer = 0
encoder_dim = 768
encoder_weights = "/absolute/path/dinov3-vit.safetensors"
# Replace with the actual lowercase SHA256 of encoder_weights.
encoder_sha256 = "0000000000000000000000000000000000000000000000000000000000000000"
encoder_device = "cpu"
cache_directory = "/absolute/path/repa-cache"
```

## 层与路由约束

层号从 0 开始，只计算模型主视觉/联合块；TREAD 的结束层不包含在窗口中。窗口不得超出该模型实际层数，LayerSync 的教师必须在学生之后。TREAD 每次前向采样一次有序目标子集，未选择 token 沿可微残差旁路传递；文本、独立参考、音频和 padding 保留原有顺序。推理预览不启用路由、不消耗路由 RNG。Wan 独立帧模式的所有原生监督槽都参与流匹配，REPA 特征也按照同一坐标顺序排列；没有 RCM 蒙版。

## 特征与梯度

LayerSync 使用停止教师梯度的负余弦损失。REPA 使用训练专属 FP32 LayerNorm/Linear 投影，目标特征停止梯度，时空网格按原公式三线性插值。特征捕获只覆盖本次前向，在反向之前解除，重算不重复收集；不改变模型权重键。两种目标和 TREAD 可以同时使用；图像修复训练暂未接入这些目标，配置会拒绝。

## REPA 缓存

运行 `unituner cache train.toml --device cuda` 时，REPA 编码器收到与原生 VAE 完全相同的目标裁剪和帧顺序，不从 latent 解码，也不另行中心裁剪。支持原生或官方 MLP DINOv3 ViT-S/B/L safetensors，保留原有 bicubic/antialias、ImageNet 归一化和逐帧处理，排除 CLS/register token。Alpha 保留在原生缓存与监督中，DINO 输入使用 RGB。编码器文件内容必须匹配配置摘要与输出宽度。训练只需要生成的特征缓存，不加载 DINO；手动提供预计算特征可调用 `save_repa_target` 和 `save_repa_manifest`，调用者负责确保裁剪和帧对应正确。

REPA 清单绑定每个原生缓存记录的完整摘要，训练前检查覆盖与内容，恢复身份包括特征清单摘要；投影器随 checkpoint 恢复，不混入推理适配器产物。

## Wavelet loss

八个图像家族的流匹配和图像修复可以附加 DWT 频带损失，适用于 adapter 和 full。
流匹配作用于模型原生速度 prediction/target；图像修复作用于解码图像 generated/HQ，
不会把速度目标误当成干净图像。Wan、MiniMaxH3 暂不接入此图像目标。

```toml
[method.wavelet]
weight = 0.5
wavelet = "db4"
levels = 3
ll = 0.1
lh = 0.01
hl = 0.01
hh = 0.05
ll_level_threshold = -1
normalize_bands = false
timestep_cutoff = 0.7
timestep_transition_width = 0.4
```

未配置时不增加计算。此实现提取自 [rockerBOO/wavelet-loss](https://github.com/rockerBOO/wavelet-loss)
的 DWT 分支，内置 Haar/db4 滤波器，不需要额外损失库。使用 zero 边界、完整卷积后
`1::2` 下采样，递归分解 LL，按每个频带的元素均值计算 MSE，再按权重求和。
LH 是高通 H／低通 W，HL 是低通 H／高通 W。负 LL 阈值按 `levels + threshold`
解析，默认只在最后一级计入 LL。开启归一化时，prediction 和 target 共用 target
的均值与标准差，不单独消除 prediction 的幅度误差。单系数频带的方差取0，避免无偏标准差未定义导致 NaN。

流匹配每个样本乘 `sigmoid((cutoff-sigma)*4/transition_width)`，sigma 是 `[0,1]`
的噪声比例，绝不传模型原生 timestep。多层图像的各层分别变换后取均值。
图像修复监督干净输出，不应用噪声时间衰减。所有频带运算在模型 autocast 外以 FP32
执行；独立调用和流匹配可对 prediction、target 求导；图像恢复会 detach HQ 目标，与其他图像监督损失一致。
SWT、实验性的 QWT、SNR-aware Huber 与逐频带逐层覆盖权重未在此接口开放。
附加目标会增加训练计算，不承诺收敛或图像质量必然提升。

---

[文档首页](../README.md) · [参数参考](parameters.md) · [故障排查](../troubleshooting.md)
