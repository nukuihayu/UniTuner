# 原生训练审计

在训练配置 `[output]` 中设置 `audit = true`，或使用 `--output.audit true`。默认关闭，保持现有训练及外部启动方式兼容。审计直接接入原生训练作用域、参数安装、采样网格和发布器，不替换采样、优化器或 CLI；`--help` 和参数错误仍由原生入口处理。

每次运行在输出目录的 `training-audit/` 保存：

- `resolved-config.json`：解析并完成精度选择后的完整配置。
- `runtime-info.json`：resume 路径、日志／输出设置、命令参数。
- `manifest.json`、`targets.json`：审计版本、可用时的源码 commit、原生数值配置身份、实际模块、算法、rank／alpha、可训练参数量及 MLP／refiner 占比。
- `sigma-grid.json`：实际构建的原生 flow 网格；连续均匀采样记录其分布。Restoration 在 manifest 中标记 flow 控制不适用，单独记录 midtimestep、sigma、calibrate 和 posterior。
- `published-keys.json`、`exports.json`：原生发布器实际生成的键及最终文件的头部核对结果。直接复用 LoRA／LoHa／LoKr、ComfyUI／Diffusers 和融合投影的原生键映射，不从 LoRA 后缀猜测其他算法。
- `resume-files.json`：`resume/` 中的续训文件，与 `weights/` 中的推理产物分开登记。

审计按原生 `weights_path` 和最终别名识别阶段、最终、encoder 及 high-noise 文件。只有阶段文件不算完成；双专家训练要求两个最终专家产物。标准 adapter 产物核对原生发布器记录的全部键；未通过 adapter 发布器的完整模型／encoder 文件仅记录头部，状态明确为 `header_recorded`。

续训比较复用 `build_resume_identity` 的配置摘要，排除输出路径、日志、发布频率等运行信息；选层、rank、有效采样、优化器和数据配置变化仍会被发现。允许换输出目录，沿指定 checkpoint 的父目录寻找原审计。旧 checkpoint／旧 notebook 包装器的审计格式不强行套用新指纹，而由原生 restore 校验真正的数据和模型身份，并记录 legacy 状态。

审计只记录用户提供的模型引用，不新增模型文件存在性检查、清单或哈希扫描。训练器原有 checkpoint 恢复验证保持原样。审计不改变模型下载、默认选层、专家路由、噪声分布或随机数消费顺序。
