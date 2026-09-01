# MCF-QPI RVQ-Twin

面向公开 **MCF-QPI 真实散斑—相位配对数据** 的可复现实验工程。项目包含：

- 官方数据断点续传、安全解压、结构扫描、自动配对和人工抽查；
- 官方划分复现、严格哈希分组划分、少样本和跨域实验；
- 原论文风格 ResUNet 基线；
- 相位 Residual-VQ-VAE 先验；
- 空间域/Fourier 域双分支的 `MCF-RVQTwin` 主模型；
- 真实配对数据学习的经验 phase→speckle forward twin；
- 条件 DDPM/DDIM 生成式基线；
- MAE、RMSE、二维相关系数、SSIM、PSNR、梯度误差、不确定度、鲁棒性和延迟评估；
- 中文实验协议、数据泄漏检查、三随机种子和消融矩阵；
- mean-phase sanity baseline、相位先验码本评估、原论文公开策略风格复现；
- 空间域/频域/连续潜空间/经验闭环四项可执行消融。

> 该数据集已经包含真实实验配对，主实验不需要额外光学仿真。项目中的程序化数据只用于 smoke test，不能作为论文结果。

## 最快开始

```bash
bash scripts/setup_env.sh
source .venv/bin/activate
bash scripts/run_smoke_test.sh
bash scripts/run_data_preparation.sh
SEED=42 bash scripts/run_research_pipeline.sh
SEED=42 bash scripts/run_ablations.sh
```

详细说明从 [`START_HERE_中文.md`](START_HERE_中文.md) 开始。

## 研究模型

```text
真实远场散斑
 ├─ 空间域编码器 ─┐
 └─ FFT 对数幅度编码器 ─┤→ 门控融合 → Residual VQ 码本 → 连续细节残差
                         │                              │
                         └──────────────────────────────┴→ 相位解码器
                                                        └→ 像素不确定度
预测相位 → 可选经验 forward twin → 重建散斑一致性
```

经验 forward twin 是由公开真实配对学习的统计代理，不是解析多芯光纤传播模型，也不是实测 transmission matrix。

## 数据放置说明（重要）

本仓库**不包含**大型训练数据集（GitHub 单文件上限 100MB，而完整 HDF5 约 1.9GB）。正式训练前，请先准备好数据，二选一：

### 方式 A：直接放入已处理好的 HDF5（推荐，最快）

向数据集提供方/项目作者索取已处理文件，放到如下位置：

```text
data/processed/mcf_qpi_128.h5        # 约 1.9GB，训练直接读取
data/processed/mcf_qpi_manifest.csv  # 约 27MB，样本配对清单
```

### 方式 B：从公开源自动下载并处理

项目内置下载与处理脚本，会自动完成：下载 → 解压 → 配对 → 审计 → 转 HDF5。

```bash
bash scripts/run_data_preparation.sh
```

若原始数据 zip 不在默认位置，可用环境变量指定：

```bash
MCF_QPI_DATASET_ZIP=/path/to/MCF-QPI_数据集.zip bash scripts/run_data_preparation.sh
```

> 数据来源与下载链接见 `docs/10_公开来源与下载链接.md`。原始数据集约 740MB（内含两个 tar 包），处理耗时较长，建议在 GPU 服务器上执行。

`data/smoke/` 下已内置一份小型冒烟数据（约 5MB），无需任何准备即可跑通 `run_smoke_test.sh`。

## 数据许可与代码许可

MCF-QPI 数据由原作者通过 Optica/Figshare 发布，页面标注为 **CC BY 4.0**；引用数据时请同时引用原论文和 Figshare DOI。工程代码采用 MIT 许可。数据文件不会被重新分发到本项目压缩包中。


## 版本 0.2.0 的关键入口

```text
configs/research/paper_style_resunet.yaml
scripts/evaluate_phase_prior.py
scripts/evaluate_mean_phase.py
scripts/run_ablations.sh
docs/11_文献对照与本项目创新判定.md
docs/12_代码框架与文件职责.md
docs/13_完整训练评估决策树.md
```

打包前自检结果见 [`SMOKE_TEST_STATUS.md`](SMOKE_TEST_STATUS.md)。
