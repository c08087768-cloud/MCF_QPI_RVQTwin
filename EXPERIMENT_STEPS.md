# 完整实验步骤清单

## A. 一次性准备

```bash
cd MCF_QPI_RVQTwin
bash scripts/setup_env.sh
source .venv/bin/activate
bash scripts/run_smoke_test.sh
bash scripts/run_data_preparation.sh
```

人工签字确认：

- [ ] 两个官方 tar 下载完成并记录 SHA-256；
- [ ] 抽查至少 100 对散斑—相位无错配；
- [ ] manifest audit 通过；
- [ ] official split 数量差异已记录而非悄悄修改；
- [ ] HDF5 split/domain 数量正确；
- [ ] 测试集未参与任何训练和调参。

## B. 基线复现

```bash
python scripts/train_inverse.py --config configs/research/paper_style_resunet.yaml
python scripts/evaluate.py \
  --config configs/research/paper_style_resunet.yaml \
  --checkpoint outputs/research/paper_style_resunet_seed42/best.pt
```

记录：normalized MAE、fidelity 2D correlation、Fashion/Digits 分域和延迟。

## C. 主模型

先评估最弱基线和相位先验本身：

```bash
python scripts/evaluate_mean_phase.py \
  --config configs/research/baseline_resunet.yaml
python scripts/evaluate_phase_prior.py \
  --config configs/research/phase_rvqvae.yaml \
  --checkpoint outputs/research/phase_rvqvae_seed42/best.pt \
  --split test
```


```bash
python scripts/train_phase_rvqvae.py --config configs/research/phase_rvqvae.yaml
python scripts/train_forward_twin.py --config configs/research/forward_twin.yaml
python scripts/evaluate_forward_twin.py \
  --config configs/research/forward_twin.yaml \
  --checkpoint outputs/research/forward_twin_seed42/best.pt
python scripts/train_inverse.py --config configs/research/proposed_rvqtwin.yaml
python scripts/evaluate.py \
  --config configs/research/proposed_rvqtwin.yaml \
  --checkpoint outputs/research/proposed_rvqtwin_seed42/best.pt
```

## D. 核心消融

```bash
SEED=42 bash scripts/run_ablations.sh
```

它运行 spatial-only、frequency-only、dual-continuous 和 no-forward-twin。

## E. 外推和鲁棒性

```bash
python scripts/evaluate_cross_domain.py ...
python scripts/evaluate_robustness.py ...
python scripts/benchmark_latency.py ...
```

## F. 少样本

```bash
for SHOTS in 50 200 1000 5000; do
  python scripts/make_fewshot_manifest.py data/processed/mcf_qpi_manifest.csv \
    --shots "$SHOTS" --seed 42 \
    --output "data/processed/fewshot_${SHOTS}.csv"
  python scripts/cache_to_hdf5.py "data/processed/fewshot_${SHOTS}.csv" \
    --output "data/processed/fewshot_${SHOTS}.h5" --overwrite
  # 使用 --set data.hdf5=... 修改训练配置。
done
```

## G. 严格 split

```bash
python scripts/resplit_manifest.py data/processed/mcf_qpi_manifest.csv
python scripts/cache_to_hdf5.py data/processed/mcf_qpi_manifest_strict.csv \
  --output data/processed/mcf_qpi_strict_128.h5 --overwrite
```

重新训练全部基线和主模型，不能拿 official split 训练的权重直接评估 strict test 后称为公平比较。

## H. 三随机种子和表格

```bash
bash scripts/run_three_seeds.sh
python scripts/compare_experiments.py \
  outputs/research/eval_baseline_seed42/summary.json \
  outputs/research/eval_proposed_seed42/summary.json \
  --output outputs/research/main_table_seed42.csv
```

最终论文保留：

- 每个 seed 的配置和 checkpoint；
- 每个测试样本的指标；
- 数据清单和哈希；
- GPU/CUDA/PyTorch 信息；
- 失败实验和消融记录。
