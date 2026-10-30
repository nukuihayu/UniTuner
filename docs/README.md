<p align="center"><img src="assets/unituner-logo.png" alt="UniTuner" width="560"></p>

# 使用文档

[English](en/README.md) · [项目首页](../README.zh-CN.md)

## 从这里开始

| 页面 | 内容 |
| --- | --- |
| [安装与环境](installation.md) | 依赖安装、CUDA、可选组件与离线资产 |
| [第一次训练 LoRA](getting-started.md) | 从图片、缓存到训练、预览与生成 |
| [TOML 与命令行](configuration.md) | 配置结构、覆盖优先级与命令速查 |
| [训练基础概念](concepts.md) | latent、流匹配、适配器与更新计数 |

## 训练工作流

| 页面 | 内容 |
| --- | --- |
| [训练指南](training/README.md) | 更新范围、优化器、预览、保存与续训 |
| [数据集](training/datasets.md) | 数据组、提示词、分桶、参考图与配对 |
| [图像修复教程](restoration-tutorial.md) | 从配对 LQ/HQ 到一步修复推理 |
| [视频与音视频](training/video.md) | Wan 与 MiniMaxH3 的资产、数据和命令 |
| [配置示例](examples/README.md) | 按任务与模型选择可复制的 TOML |

## 配置与方法参考

| 页面 | 内容 |
| --- | --- |
| [完整参数手册](training/parameters.md) | 字段类型、默认值、范围与填写位置 |
| [模型与资产](training/models.md) | 家族、变体、条件输入与资产组合 |
| [显存与执行策略](training/execution.md) | 常驻、H2D、VBAR、重算与编译 |
| [精度](training/precision.md) | 存储、计算、AMP 与导出精度 |
| [量化基模](training/quantized-base.md) | 冻结量化底模的适配器训练 |
| [文本编码](training/text-encoding.md) | 编码器、分词器、计算精度与量化 |
| [辅助训练目标](training/objectives.md) | LayerSync、REPA、TREAD 与小波损失 |
| [修复方法参考](training/restoration.md) | 时间校准、损失函数与编码器适配 |

## 使用模型与排查问题

| 页面 | 内容 |
| --- | --- |
| [推理与 ComfyUI](inference.md) | 使用导出的适配器和全参数模型 |
| [权重转换](weight-conversion.md) | 权重格式、消费者布局与精度转换 |
| [权重分析](weight-analysis.md) | 查看训练权重与适配器变化 |
| [日志与 TensorBoard](logging.md) | 终端、JSONL 与事件指标 |
| [故障排查](troubleshooting.md) | 资产、缓存、显存不足、续训与效果差异 |

## 项目维护参考

| 页面 | 内容 |
| --- | --- |
| [项目架构](architecture.md) | 模块边界与资源所有权 |
| [执行机制](training/execution-internals.md) | 权重生命周期与训练执行边界 |
