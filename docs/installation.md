# 安装与环境

[English](en/installation.md) · [文档首页](README.md)

本页完成环境准备。安装后跟随[第一次训练](getting-started.md)，或选择[模型对应的配置](examples/README.md)。

<details>
<summary>本页导航</summary>

- [环境要求](#环境要求)
- [基础安装](#基础安装)
- [默认完整依赖](#默认完整依赖)
- [检查 CUDA](#检查-cuda)
- [分词器与离线运行](#分词器与离线运行)
- [镜像与固定环境](#镜像与固定环境)
- [开发环境](#开发环境)

</details>

## 环境要求

| 项目 | 要求与用途 |
| --- | --- |
| Python | 3.12 或更高；可使用 `uv python install 3.13` |
| 包管理 | 使用 uv 和仓库 `uv.lock` 安装一致的依赖 |
| PyTorch | 项目依赖 Torch 2.13 系列；以锁文件解析的构建为准 |
| CUDA | NVIDIA GPU 与匹配驱动；BF16、FP16、训练编译、冻结块卸载需要 CUDA |
| CPU | 可运行 FP32 支持路径；不支持 CUDA 专属执行选项 |
| 模型文件 | 自行准备对应家族的 denoiser、VAE、文本编码器及必要辅助资产 |
| 磁盘与主机内存 | 容纳权重、编码缓存、输出与续训状态；卸载额外占用 RAM，重建缓存期间保留旧缓存 |

显存需求随模型、尺寸、batch、训练模式和损失变化。先使用适合目标模型的小批量配方，再按[执行策略](training/execution.md)配置。

## 基础安装

```bash
git clone https://github.com/nukuihayu/UniTuner.git
cd UniTuner
uv python install 3.13
uv sync --locked --no-dev
uv run --no-dev unituner --help
```

所有示例命令默认从仓库根目录执行。模型不会随着依赖安装自动下载。

## 默认完整依赖

普通安装和 `uv sync --locked --no-dev` 均包含全部运行依赖，不再提供 `media`、`text`、`kitchen`、`vbar` extras：

| 依赖 | 用途 |
| --- | --- |
| transformers、torchvision、sentencepiece、ftfy | 文本编码、分词器及处理器 |
| av、opencv-python-headless、pillow | 图像、视频和音频读写 |
| comfy-kitchen | 量化、权重转换及对应推理算子 |
| comfy-aimdo | VBAR 显存页管理 |

安装依赖不会自动启用量化或 VBAR，也不会下载模型。CUDA 专属功能仍须满足设备、驱动和配置要求；已有缓存且关闭预览的训练仍按需加载文本编码器。

## 检查 CUDA

```bash
uv run --no-sync --no-dev python -c "import torch; print('Torch:', torch.__version__); print('CUDA build:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available())"
```

`CUDA available: False` 时先检查驱动、当前 Python 环境和 Torch 构建。`--device cuda` 选择执行设备，不会安装驱动或自动换装 Torch。CPU 流程显式使用 `--device cpu --execution.precision fp32`；`auto` 根据权重精度判断，不保证适用于 CPU。

VBAR 从 `unituner train` 入口在 Torch 导入前初始化。在已导入 Torch 的 Notebook 进程中切换 VBAR，应启动新的 CLI 进程。

## 分词器与离线运行

文本权重和分词器是不同资产。默认分词器来源可下载分词文件，不下载编码器权重。离线运行时指定本地目录：

```toml
[model.text.tokenizer]
source = "/models/tokenizer"
offline = true
```

若目录中还有一层 `tokenizer/`，填写 `subfolder = "tokenizer"`。FLUX CLIP 和 Qwen 图像处理器有独立子表，见[文本编码](training/text-encoding.md)。

## 镜像与固定环境

需要 PyPI 镜像时：

```bash
UV_DEFAULT_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple uv sync --locked --no-dev
```

镜像只改变 Python 包来源，不是模型或 Python 解释器下载源。依赖准备完毕后可以使用 `uv run --no-sync --no-dev unituner …`，避免启动时再次同步。

## 开发环境

```bash
uv sync --locked --group dev
make check
make test
```

`make check` 执行 Ruff、Ty 和 Pyright；`make test` 执行项目测试。CUDA 和本地资产相关测试有各自运行条件。模型接入和资源所有权见[架构](architecture.md)。

---

[文档首页](README.md) · [参数参考](training/parameters.md) · [故障排查](troubleshooting.md)
