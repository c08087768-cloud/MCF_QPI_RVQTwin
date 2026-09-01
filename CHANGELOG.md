# Changelog

## 0.2.0

- 完成 MCF-QPI-only 数据下载、安全解压、结构扫描、自动配对、哈希审计与人工抽查流程；
- 实现原论文公开训练策略风格 ResUNet、更强监督 ResUNet、Phase-RVQ-VAE、DualDomain-RVQTwin、经验 forward twin；
- 加入条件 DDPM/DDIM 基线，并明确其不是 SpecDiffusion 官方代码逐层复现；
- 新增 mean-phase sanity baseline 和相位先验重建/码本占用评估；
- 新增空间域、频域、连续潜空间和无 forward twin 四项可执行消融；
- 支持 step-based 学习率调度，以对应原文“每 20,000 iterations 减半”的公开设置；
- 相机平移增强改为零填充而非周期回卷；
- SSIM、MAE、相关性等指标支持 valid mask，并同时输出归一化与弧度单位；
- 完成少样本、跨域、鲁棒性、延迟、不确定度与 risk–coverage 评估；
- 新增中文文献对照、创新边界、代码职责和完整决策树；
- 修正物理模型驱动 MCF 相位成像 DOI 为 `10.1364/OE.551221`；
- 加入数据集原文数量不一致的显式审计，避免静默修正；
- 17 项单元测试、全流程 smoke、Python 编译和 Shell 语法检查通过。
