# 工程自检状态

生成日期：2026-08-04

## 已执行检查

```text
Python 全文件 compileall：通过
pytest：17/17 通过
Shell 脚本 bash -n：全部通过
端到端 smoke：数据生成、HDF5、Phase-RVQ-VAE、forward twin、ResUNet、
MCF-RVQTwin、conditional diffusion、评估与可视化流程均已跑通
```

## 当前执行环境

```text
Python：3.13.5
PyTorch：2.10.0+cpu
CUDA：当前打包环境不可用
```

因此，代码的 CPU 连通性已经验证；CUDA、AMP 和多 worker 路径已实现，但尚未在本打包环境的 NVIDIA GPU 上实际执行。请在您的 GPU 机器上先运行：

```bash
bash scripts/run_smoke_test.sh
```

## 尚未执行的科学实验

- 未在当前打包环境完整下载两个 MCF-QPI 官方 tar；
- 未使用全部 50,176 对真实实验图像训练正式模型；
- 未生成任何声称可作为论文结论的真实数据指标；
- 程序化 smoke 数据和对应指标只能证明软件流程连通，不能写入论文结果。

正式训练前必须人工检查下载后的配对、数据数量、split 和哈希审计报告。
