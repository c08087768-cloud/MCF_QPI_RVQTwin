# 从这里开始：只基于 MCF-QPI 的完整研究路线

## 1. 先明确任务

公开 MCF-QPI 数据的机器学习任务是：

```text
输入：多芯光纤检测端相机记录的单幅远场散斑强度图
标签：测量端 SLM 显示并投影到多芯光纤端面的二维相位图
输出：与标签同尺寸的二维相位图
```

标签由 MNIST 手写数字和 Fashion-MNIST 灰度图映射到 `[0, π]`。这不是“恢复光纤内部三维折射率场”，也不是“预测动态大气畸变”。

## 2. 是否需要仿真数据

**主训练和主测试不需要仿真数据。** 公开数据已经是同一真实光学系统采集的成对实验数据。没有纤芯位置、每根纤芯固有相位、偏振、波长响应、传播距离和实测传输矩阵时，随意模拟一套多芯光纤散斑往往会引入比帮助更大的域差异。

本工程只有 `scripts/make_smoke_dataset.py` 会生成程序化小数据，作用是检查：

- PyTorch/CUDA 是否可用；
- 数据读取和 HDF5 是否正确；
- 各网络能否前向、反向和保存 checkpoint；
- 评估脚本能否输出 CSV/JSON/图片。

任何 smoke 指标都不能写进论文。

## 3. 推荐论文主线

仅把 U-Net 换成 VQ-VAE 不够，因为已有工作已经使用 U-Net/ResNet、条件扩散和物理模型驱动网络处理类似 MCF 相位恢复问题。建议主线为：

> **面向真实多芯光纤散斑相位恢复的数据高效双域残差矢量量化网络，结合经验测量闭环与校准不确定度。**

三个主要方法点：

1. `Spatial + Fourier` 双域散斑编码，显式利用散斑的局部结构和全局频谱；
2. 预训练 `Residual VQ-VAE` 相位先验，把逆网络限制在训练相位分布的离散状态空间，同时保留小连续残差；
3. 使用真实成对数据学习的经验 `phase → speckle` forward twin 做可选闭环，并输出异方差和 MC Dropout 不确定度。

工程上还必须把“严格数据审计、少样本、跨域、传感器扰动、拒识曲线”作为完整贡献的一部分。

## 4. 一次完整执行

```bash
# 创建环境
bash scripts/setup_env.sh
source .venv/bin/activate

# 先跑小型连通性测试
bash scripts/run_smoke_test.sh

# 下载、解压、配对、审计并缓存真实数据
bash scripts/run_data_preparation.sh

# 打开配对抽查图，确认无错配
# outputs/data_inspection/random_pairs.png

# 训练和评估一个随机种子
SEED=42 bash scripts/run_research_pipeline.sh

# 运行空间域、频域、无量化和无经验闭环四项核心消融
SEED=42 bash scripts/run_ablations.sh

# 单独训练较慢的扩散基线
SEED=42 bash scripts/run_diffusion_baseline.sh

# 正式论文至少三个随机种子
bash scripts/run_three_seeds.sh
```

## 5. 首次运行最容易犯的错误

1. **没有人工检查散斑—相位配对。** 自动脚本只根据目录和索引配对，不能替代人工抽查。
2. **将训练、验证、测试图像重新随机打乱。** 优先保留作者目录中的官方 split；另做严格哈希分组实验，而不是覆盖官方实验。
3. **对配对图同时做旋转或翻转。** 固定多芯光纤系统的散斑映射不是普通图像的平移/旋转等变关系，随意几何增强会制造不存在的物理样本。
4. **用测试集调参数。** 测试集只在最终模型冻结后运行。
5. **把二维相关系数写成复光场 fidelity。** 原论文所谓 fidelity 是二维图像相关系数，本工程明确输出 `fidelity_2d_correlation`。
6. **把经验 forward twin 称为 PINN 或传输矩阵。** 它只是数据驱动的前向代理；验证不好就关闭 cycle loss。
7. **从 MNIST/Fashion 结果推断临床组织性能。** 公开数据不含组织相位真值，相关结论需要新增组织实验数据。

## 6. 文档索引

- `docs/01_研究定位与相关工作.md`
- `docs/02_数据集标签与已知矛盾.md`
- `docs/03_数据准备与防泄漏.md`
- `docs/04_网络设计与损失函数.md`
- `docs/05_完整训练步骤.md`
- `docs/06_评估协议与统计.md`
- `docs/07_实验矩阵与消融.md`
- `docs/08_论文写作与创新边界.md`
- `docs/09_故障排查.md`
- `docs/10_公开来源与下载链接.md`
- `docs/11_文献对照与本项目创新判定.md`
- `docs/12_代码框架与文件职责.md`
- `docs/13_完整训练评估决策树.md`
- `DATA_STRUCTURE.md`
- `EXPERIMENT_STEPS.md`
- `SMOKE_TEST_STATUS.md`
