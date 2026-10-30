# 项目架构

简体中文 · [English](en/architecture.md)

UniTunner 将原生模型计算、训练方法、执行策略和外部产物分开维护。
不同模型共用训练编排，保留各自的条件布局、时间单位、预测方向与权重语义。
用户配置及模型能力见[文档目录](README.md)。本页描述当前职责。

<details>
<summary>本页导航</summary>

- [目录与职责](#目录与职责)
- [Adapter 参数与计算](#adapter-参数与计算)
- [模型家族的所有权](#模型家族的所有权)
- [训练与推理的边界](#训练与推理的边界)
- [配置、缓存与产物](#配置缓存与产物)
- [权重量化的所有权](#权重量化的所有权)
- [资源生命周期](#资源生命周期)
- [验证边界](#验证边界)

</details>

## 目录与职责

路径相对项目根目录：

```text
src/unitunner/
├── cli/             # 命令解析、配置转换与延迟分派
├── config/          # 公共严格配置与组合校验
├── models/          # 按模型家族组织的原生计算与专属组件
├── data/            # 媒体读取、缓存契约、数据集与退化
├── adapters/        # LoRA、LoHa、LoKr 数学与安装生命周期
├── training/        # 训练目标、参数计划、更新流程和预览
├── execution/       # 编译、重算、注意力与权重驻留
├── checkpoints/     # 更新边界训练状态的保存与恢复
├── artifacts/       # 权重格式转换、校验和原子发布
└── inference/       # 图像生成与一步修复的公共编排

tests/
├── config/          # 配置边界与实际组合约束
├── models/          # 原生模型公式、布局与梯度
├── comparison/      # 可选分配器测试入口
└── support/         # 正式测试共用的资产与断言

docs/
├── README.md        # 中文文档导航
├── getting-started.md # 安装与首次训练
├── architecture.md  # 当前职责与依赖
├── training/        # 中文训练、模型和执行专题
├── en/              # 对应的英文文档与示例索引
└── examples/        # 共用配置示例与中文任务索引
```

测试另按 adapters、artifacts、checkpoints、data、execution、training 等实际职责分组。

## Adapter 参数与计算

`adapters/lora.py`、`loha.py`、`lokr.py` 定义可训练参数、缩放和权重导出；
`construction.py` 负责构建，`attachment.py` 负责挂载与卸载，`loading.py` 和 `encoder.py` 负责产物读写。
优化计算集中在 `adapters/ops/`，按算法使用一致的命名：

| 文件 | 职责 |
| --- | --- |
| `ops/lora.py` / `ops/lora_cuda.py` | LoRA 线性计算与残差缩放、加法融合 |
| `ops/loha.py` / `ops/loha_cuda.py` | LoHa Hadamard 乘积与缩放融合 |
| `ops/lokr.py` / `ops/lokr_cuda.py` | LoKr 稠密 Kronecker 权重构造与反向乘积融合 |

每组普通模块负责适用条件、autograd 和原生 PyTorch 回退；`*_cuda.py` 只负责 Triton 内核，按需加载。
计算模块不依赖 adapter 参数类或挂载流程。优化保留原有矩阵乘法、低精度舍入及梯度归约顺序；
不满足条件的布局、编译或自动微分场景继续使用原生表达式。
对应数值测试位于 `tests/adapters/ops/test_{lora,loha,lokr}.py`，参数公式与加载测试保留在 `tests/adapters/`。

## 模型家族的所有权

`models/<family>/` 拥有原生 transformer、VAE、层与位置几何、文本编码、权重加载，
以及该家族专属的配置、缓存或采样组件。共享组件可以被显式复用，不为目录对称创建空包装。
模型原生代码不导入训练 workflow，也不调用外部参考训练器兜底。

`models/components/transformer_artifacts.py` 统一解析显式配置、内嵌元数据和外部配置的优先级，
并检查拓扑冲突；各家族加载入口负责模型构造、权重布局转换与精度策略。

公共字段放在 `config/`：模型身份、参数训练方式、数据来源、执行策略与输出。
只有具有独立原生语义的字段留在 `models/<family>/config/`；
多参考图、mask、控制来源和文本预算先使用公共字段，再按家族能力验证。
共用字段不意味着每个模型都支持每个取值，不能通过忽略字段实现表面兼容。

## 训练与推理的边界

| 所有者 | 负责 | 不负责 |
| --- | --- | --- |
| `training/integrations/` | 模型原生条件、时间、预测目标和 adapter 投影清单的连接 | 复制整个训练循环 |
| `training/methods/` | 流匹配、修复和特征目标的公式与状态 | 猜测模型原生时间或隐式转换输入 |
| `training/workflows/` | 图像／Wan／H3 更新流程、optimizer、累积、保存与预览编排 | 定义原生网络层 |
| `execution/` | 在已声明的模型区域安装编译、重算和驻留策略 | 改写训练目标或静默降级配置 |
| `inference/` 与模型专属推理组件 | 原生采样、VAE 解码和推理产物加载 | 依赖训练循环运行生成 |

LoRA／LoHa／LoKr 与 full 决定可训练参数；flow-matching／restoration 决定训练目标。
`training/objectives.py` 统一拥有 LayerSync、REPA 和 TREAD 的安装与释放；图像、Wan、H3 工作流复用同一入口。执行策略由 `execution/` 独立管理。
支持的组合在输入边界校验，不以散落的模型名称判断或后期补丁替代契约。

## 配置、缓存与产物

CLI 先加载不可变 Pydantic 配置，拒绝未知字段；配置加载不创建模型或探测 GPU。
媒体与文本的可选依赖在应用边界导入。所有相对配置路径以命令工作目录为基准。

`config/train.py`、`config/output.py` 分别拥有训练与输出字段；`config/training.py` 组合它们，
`training_validation.py` 分开校验全局策略、逐数据组约束和预览。全局规则每次配置加载仅执行一次。
`artifacts/formats/precision.py` 提供不依赖 Torch 的精度类型，`precision.py` 统一运行时 dtype 与计算上下文。
图像训练的家族组装和发布分别位于 `training/integrations/image_models.py` 与 `image_publication.py`；
公共预览仅在流匹配需要时加载所选家族的发布器及文本依赖。

`training/output.py` 统一运行目录的名称清理、时间戳与重名分配；
`training/telemetry.py` 使用同一规则或已确定的运行名称，负责指标记录与事件写入器的生命周期。

`data/cache/` 定义生产与消费共用的内容身份、manifest 和张量记录。
缓存身份覆盖对应模型与数值输入；改变源图、提示词或条件后重新构建。
`data/cache/storage.py` 统一记录的碰撞检查、原子写入与文件摘要；各家族存储入口负责原生张量和元数据契约。
媒体裁剪、后验和条件编码保持原生语义，不能为统一接口而改变监督数据。

`artifacts/` 管理外部权重的显式键转换、严格张量契约和原子文件发布。
适配器算法与消费者前缀转换在此边界处理，不改变训练内部参数或权重值。
`checkpoints/` 管理训练状态：模型／方法模块、optimizer、RNG 与数据游标；
推理发布文件与可恢复训练状态是不同产物。

## 权重量化的所有权

`artifacts/formats/precision_policy.py` 定义基模和文本编码器的层保护预设；独立转换与训练前量化共用它。`artifacts/quantization.py` 调用上游布局完成打包和序列化，`quantized_loading.py` 校验逻辑形状、附属张量与原生模型键。`models/components/weight_quantization.py` 负责逐层准备，`quantized.py` 负责统一的量化线性层计算。

训练按“CPU 加载浮点基模 → 每层在所选 GPU 量化并放回 CPU → 按驻留策略放置 → 安装适配器、优化器与执行策略”的顺序初始化。CPU 训练则在 CPU 量化。这样替换冻结投影时还没有活跃的 optimizer Parameter，也不需要完整浮点基模先驻留 GPU。

所有无需校准布局只量化权重，前向临时反量化后做浮点乘法，反向从保存的量化存储重建权重并计算输入梯度。缩放和校正跟随权重驻留，保留各自 dtype；不保存完整浮点基模，不启用 INT8 动态激活量化。量化配置属于续训身份。配置与限制见[量化基模训练](training/quantized-base.md)。

## 资源生命周期

训练 session 显式持有 adapter、执行安装、方法模块和 optimizer。
资源创建成功后登记，结束或异常时逆序释放；关闭前完成相关 backward 和设备工作。

`models/components/forward_override.py` 统一管理实例 `forward` 替换的所有权检查与恢复。
适配器和执行安装器组合使用该对象；设备传输、权重与计算图资源由各自安装器释放。

重算、编译与驻留通过具名原生区域安装，不能改变 live optimizer 的 Parameter 身份。
冻结权重即使不求自身梯度，也可能被输入梯度计算读取；驻留和 VBAR lease 必须覆盖这段生命周期。
VBAR 的进程级初始化必须发生在 Torch 导入前，其他限制见[执行策略](training/execution.md)。

特征捕获仅覆盖本次前向，不能因重算重复收集或跨更新保留计算图。
预览隔离随机状态，不推进训练数据游标。Checkpoint 在没有未完成计算的更新边界保存，
恢复先验证模型、配置与缓存身份，再恢复活动状态。

## 验证

测试覆盖模型公式、梯度、权重格式、训练更新和资源释放。


---

[文档首页](README.md) · [参数参考](training/parameters.md) · [故障排查](troubleshooting.md)
