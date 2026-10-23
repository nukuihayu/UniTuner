# 安装与环境

[English](en/installation.md) · [文档首页](README.md)

本页完成环境准备。安装后跟随[第一次训练](getting-started.md)，或选择[模型对应的配置](examples/README.md)。

<details>
<summary>本页导航</summary>

- [环境要求](#环境要求)
- [基础安装](#基础安装)
- [可选依赖](#可选依赖)
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
git clone https://github.com/nukuihayu/UniTunner.git
cd UniTunner
uv python install 3.13
uv sync --locked --no-dev --extra media --extra text
uv run --no-dev unitunner --help
```

所有示例命令默认从仓库根目录执行。模型不会随着依赖安装自动下载。

## 可选依赖

| extra | 用途 | 何时需要 |
| --- | --- | --- |
| `media` | 图像、视频、音频读写 | 缓存生产、预览和生成 |
| `text` | 文本模型、分词器和处理器 | 编码缓存、文本条件预览和推理 |
| `kitchen` | comfy-kitchen 量化与可选算子 | 对应权重转换、量化或推理注意力路径 |
| `vbar` | comfy-aimdo 显存页管理 | `execution.residency = "vbar"` |

需要量化和 VBAR 时，在基础 extras 上一并安装：

```bash
uv sync --locked --no-dev --extra media --extra text --extra kitchen --extra vbar
```

再次同步时继续列出需要的 extras。已有缓存、关闭预览的训练不必加载文本编码器；修复训练仍需要 VAE 和图像读取。

## 检查 CUDA

```bash
uv run --no-sync --no-dev python -c "import torch; print('Torch:', torch.__version__); print('CUDA build:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available())"
```

`CUDA available: False` 时先检查驱动、当前 Python 环境和 Torch 构建。`--device cuda` 选择执行设备，不会安装驱动或自动换装 Torch。CPU 流程显式使用 `--device cpu --execution.precision fp32`；`auto` 根据权重精度判断，不保证适用于 CPU。

VBAR 从 `unitunner train` 入口在 Torch 导入前初始化。在已导入 Torch 的 Notebook 进程中切换 VBAR，应启动新的 CLI 进程。

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
UV_DEFAULT_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple uv sync --locked --no-dev --extra media --extra text
```

镜像只改变 Python 包来源，不是模型或 Python 解释器下载源。依赖准备完毕后可以使用 `uv run --no-sync --no-dev unitunner …`，避免启动时再次同步。

## 开发环境

```bash
uv sync --locked --group dev --extra media --extra text
make check
make test
```

`make check` 执行 Ruff、Ty 和 Pyright；`make test` 执行项目测试。CUDA、可选依赖和本地资产相关测试有各自运行条件。模型接入和资源所有权见[架构](architecture.md)。

---

[文档首页](README.md) · [参数参考](training/parameters.md) · [故障排查](troubleshooting.md)
