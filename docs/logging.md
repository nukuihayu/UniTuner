# 日志与训练监控

[English](en/logging.md) · [训练教程](training/README.md)

<details>
<summary>本页导航</summary>

- [终端显示](#终端显示)
- [Jupyter / Notebook](#jupyter--notebook)
- [JSONL 完整指标](#jsonl-完整指标)
- [TensorBoard](#tensorboard)
- [排查错误](#排查错误)

</details>

## 终端显示

交互终端原位更新进度、update/总数、loss、单次更新耗时、学习率、可选 D loss 和显存，不逐步刷 INFO。保存、预览切换和错误立即显示。

后台或重定向运行禁用动态进度条，默认每 60 秒输出摘要，首步和末步必报。通常省略 log_every_seconds，需要调整时才填写：

```toml
[output]
log_every_seconds = 120.0
```

非交互 JSON stdout 跟随摘要频率；需要逐步数据时读取 JSONL 或 TensorBoard。

## Jupyter / Notebook

Notebook 单元格会在支持的环境中自动显示进度。没有进度显示时，在训练命令前设置：

```bash
export UNITUNER_PROGRESS=always
```

默认值为 `auto`；保存终端输出时可用 `never` 关闭进度条。无论是否显示进度，JSONL 与 TensorBoard 都会记录每步完整指标。

## JSONL 完整指标

训练自动在输出目录的 logs/<run-name>.jsonl 记录每步分项指标，不受终端摘要频率影响。记录还包含时间、级别、阶段/上下文、源码位置和完整异常链。

省略 output.directory 时使用名称和时间戳；显式目录使用固定 run 名。从头训练覆盖本次日志，--resume 追加。启动时终端会显示实际路径。

## TensorBoard

```toml
[output]
tensorboard = "runs/tensorboard"
```

只填写事件根目录，run 名跟随训练输出名称，续训从恢复步接着记录。

```bash
tensorboard --logdir runs/tensorboard --host 0.0.0.0 --port 6006
```

| 指标 | 如何读 |
| --- | --- |
| loss/main 或 loss/generator | 完成一次有效更新后的按样本加权损失 |
| loss/generator/lrr、l1、dv3d、gan、histogram | 修复各分项，按各自原始尺度显示 |
| loss/discriminator | 独立判别器目标 |
| train/*/grad_norm、lr/* | 梯度范数和各参数组学习率 |
| train/update_samples、train/update_microbatches | 梯度累积实际消耗量 |
| performance/* | 更新耗时与样本吞吐 |
| memory/* | CUDA 当前分配、保留和峰值分配 |

## 排查错误

```bash
uv run --no-dev unituner train train.toml --device cuda --log-level DEBUG
```

终端级别支持 DEBUG/INFO/WARNING/ERROR，JSONL 独立保留 DEBUG 详情。错误会尽可能标明配置字段、设备、源文件、更新数和阶段。先看第一处异常，再查看启动时给出的 JSONL 路径。

配置错误发生在加载权重前，检查表路径、拼写和类型；数值及布尔值不要加引号。缓存身份不符时用实际模型/数据重建。OOM 先看 peak，再减 batch/尺寸、启用重算或合适的卸载策略，不能仅根据更新后的常驻显存判断需求。

---

[文档首页](README.md) · [参数参考](training/parameters.md) · [故障排查](troubleshooting.md)
