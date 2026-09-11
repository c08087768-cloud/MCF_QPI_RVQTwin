# 双域相位先验细节修正网络设计

## 目标

在不改变现有 `DualDomainRVQTwin`、既有 checkpoint 或历史实验含义的前提下，新增一个可公平比较的反演模型。新模型保留空间域和 Fourier 幅度域信息、RVQ phase prior 与像素级不确定度，同时引入多尺度解码器恢复量化瓶颈中容易丢失的高分辨率细节。

本轮首先提供代码、配置、单元测试和 official split 的 seed42 开发实验入口；不在代码中宣称其优于现有模型，也不自动启动三随机种子或 test 集评估。

## 非目标

- 不修改 `DualDomainRVQTwin` 的前向含义或它已经训练出的权重。
- 不改变 HDF5、数据增强、strict split 或 forward twin 的协议。
- 不把 clean-cycle 重新纳入主模型；新模型默认 cycle-off。
- 不在本轮实现真实相机计数噪声或跨域迁移训练。

## 架构

新增模型类 `DualDomainPriorRefiner`，配置名为 `dual_domain_prior_refiner`。

输入散斑同时进入两个逐尺度编码器：空间域编码器得到 `S1...S4`，Fourier 对数幅度编码器得到 `F1...F4`。每一尺度经门控融合得到 `G1...G4`。其中 `G4` 的空间尺寸和通道数经 `1x1` 投影后与 phase prior latent 对齐，形成 `z_speckle`。

`z_speckle` 经 phase prior 的 RVQ 量化器得到 `z_q`，再由 phase prior decoder 生成低频且结构受限的 `phase_prior`。与现有模型相同，训练时真实相位可以经冻结的 phase prior encoder 产生 token 和 latent 监督；推理时不访问真实相位。

融合特征 `G1...G4` 走新的多尺度解码器。解码器由最深层 `G4` 开始逐级上采样，并使用 `G3`、`G2`、`G1` 作为 skip connection，生成单通道细节图 `delta_phase`。最终输出是：

```text
phase = phase_prior + alpha * delta_phase
```

`alpha` 是正的可学习标量，采用 sigmoid 参数化并乘以配置上限 `detail_scale_max`。默认上限为 `0.25`。这样细节支路能补偿量化导致的边缘和纹理丢失，但不能无约束地绕开 phase prior。

不确定度头从与最终相位相同的多尺度解码器特征生成 `log_scale`。它仍表达每个像素的 Laplace scale，沿用项目中修复后的 MC-Laplace 不确定度、温度校准和指标流程。

## 兼容输出契约

新模型输出至少包含：

- `phase`：最终相位预测；
- `phase_prior`：RVQ phase prior 解码得到的基础相位；
- `detail_phase`：未乘 alpha 的细节修正；
- `detail_scale`：当前 alpha 值，便于诊断其是否贴近上限；
- `log_scale`：像素级不确定度头输出；
- `z_e`、`z_q`、`continuous_residual`、`vq_loss`、`perplexity`；
- 训练时的 `target_z_e`、`target_z_q`、`target_indices` 和 `token_logits`。

这样 `CompositeInverseLoss`、训练引擎、评估脚本、RVQ 证据脚本以及 checkpoint 契约无需为新模型复制一套流程。训练日志额外记录每项损失的原始值和加权值，以及 `detail_scale`、`mean_abs_detail_phase` 与 `mean_abs_phase_prior`。

## 对照实验

所有开发实验使用同一份 official HDF5、相同数字输入扰动、batch=32、80 epochs、AdamW、cosine scheduler、`val_phase_l1` 选模规则与 seed42。开发阶段仅访问 validation，不能因 validation 结果反复修改后访问 test。

| 配置 | 空间域 | 频域 | phase prior decoder | RVQ | 多尺度细节支路 |
| --- | --- | --- | --- | --- | --- |
| `controlled_resunet32` | 是 | 否 | 否 | 否 | ResUNet 主支路 |
| `dual_continuous` | 是 | 是 | 否 | 否 | 否 |
| `dual_prior_continuous` | 是 | 是 | 是 | 否 | 是 |
| `dual_prior_rvq1_refiner` | 是 | 是 | 是 | 一级 | 是 |
| `dual_prior_rvq2_refiner` | 是 | 是 | 是 | 两级 | 是 |

`dual_prior_continuous` 与 RVQ 配置应共享编码器、融合器、细节解码器、损失权重和参数量级；唯一因果变量是是否经过量化器。一级和两级 RVQ 同样只改变 `num_quantizers`。一级 refiner 必须先训练匹配的 `phase_rvqvae_rvq1_seed42`，并只加载该一级 phase prior；不能加载现有两级 phase prior。若关键模型相对受控 ResUNet 或其连续对照在 seed42 的 validation MAE 改善达到或超过 2%，才补跑 seed123 与 seed2026；否则报告 seed42 诊断结果并停止扩大训练。

## 错误处理与门禁

- `use_spatial=false` 且 `use_frequency=false` 必须在构造时抛出错误。
- `detail_scale_max` 必须是有限正数；无效配置在构造时抛出错误。
- `phase_prior_checkpoint` 仍是新训练模型的必需项。
- 评估必须从自包含 inference checkpoint 构建模型，不能隐式读取 phase prior 外部 checkpoint。
- forward twin 保持关闭；只有现有 clean-cycle 门槛全部通过后，才允许另行设计 cycle 实验。

## 测试

- 模型在 64x64 合成输入上输出正确的最终相位、基础相位、细节图、尺度和不确定度形状。
- 最终相位严格等于 `phase_prior + detail_scale * detail_phase`。
- `detail_scale` 始终大于零且不超过 `detail_scale_max`。
- 双域、连续、一级 RVQ 与两级 RVQ 均可前向；训练目标字段只在提供真实相位时出现。
- factory 可从新配置构建模型、加载 phase prior，并能以纯 inference checkpoint 独立恢复。
- 开发配置满足既有研究协议：cycle-off、official HDF5、统一训练超参数与显式 RVQ 级数。

## 验收标准

代码合入前必须通过全量 Python 测试。服务器运行前必须完成一次 GPU/AMP smoke，确认新模型能从 official HDF5 读取、训练一个有限 batch、保存自包含 checkpoint，并从该 checkpoint 完成 validation 评估。
