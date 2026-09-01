#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"
SEED="${SEED:-42}"
DATA_H5="${DATA_H5:-data/processed/mcf_qpi_128.h5}"

PRIOR_DIR="outputs/research/phase_rvqvae_seed${SEED}"
TWIN_DIR="outputs/research/forward_twin_seed${SEED}"
PAPER_DIR="outputs/research/paper_style_resunet_seed${SEED}"
BASE_DIR="outputs/research/baseline_resunet_seed${SEED}"
PROP_DIR="outputs/research/proposed_rvqtwin_seed${SEED}"

# 0) 两个不依赖训练权重的 sanity 检查。
python scripts/evaluate_mean_phase.py --config configs/research/baseline_resunet.yaml \
  --set seed="$SEED" --set data.hdf5="$DATA_H5" \
  --output-dir "outputs/research/mean_phase_seed${SEED}"

# 1) 相位离散先验。先验只使用训练 split 的真实相位标签。
python scripts/train_phase_rvqvae.py --config configs/research/phase_rvqvae.yaml \
  --set seed="$SEED" --set data.hdf5="$DATA_H5" --set output_dir="$PRIOR_DIR"
python scripts/evaluate_phase_prior.py --config configs/research/phase_rvqvae.yaml \
  --checkpoint "$PRIOR_DIR/best.pt" --split test \
  --set seed="$SEED" --set data.hdf5="$DATA_H5" \
  --output-dir "outputs/research/phase_prior_eval_seed${SEED}"

# 2) 可选经验 phase→speckle 代理。先独立验证，再决定论文主模型是否保留 cycle loss。
python scripts/train_forward_twin.py --config configs/research/forward_twin.yaml \
  --set seed="$SEED" --set data.hdf5="$DATA_H5" --set output_dir="$TWIN_DIR"
python scripts/evaluate_forward_twin.py --config configs/research/forward_twin.yaml \
  --checkpoint "$TWIN_DIR/best.pt" --split val \
  --set seed="$SEED" --set data.hdf5="$DATA_H5" \
  --output-dir "outputs/research/forward_twin_eval_seed${SEED}"

# 3) 原论文公开训练策略风格复现 + 更强监督基线。
python scripts/train_inverse.py --config configs/research/paper_style_resunet.yaml \
  --set seed="$SEED" --set data.hdf5="$DATA_H5" --set output_dir="$PAPER_DIR"
python scripts/train_inverse.py --config configs/research/baseline_resunet.yaml \
  --set seed="$SEED" --set data.hdf5="$DATA_H5" --set output_dir="$BASE_DIR"

# 4) 完整主模型。若 forward twin 验证较差，请改跑 configs/ablations/no_forward_twin.yaml。
python scripts/train_inverse.py --config configs/research/proposed_rvqtwin.yaml \
  --set seed="$SEED" --set data.hdf5="$DATA_H5" \
  --set model.phase_prior_checkpoint="$PRIOR_DIR/best.pt" \
  --set forward_twin.checkpoint="$TWIN_DIR/best.pt" \
  --set output_dir="$PROP_DIR"

# 5) 完全冻结后评估保留测试集。
python scripts/evaluate.py --config configs/research/paper_style_resunet.yaml \
  --checkpoint "$PAPER_DIR/best.pt" \
  --set seed="$SEED" --set data.hdf5="$DATA_H5" \
  --set output_dir="outputs/research/eval_paper_style_seed${SEED}"
python scripts/evaluate.py --config configs/research/baseline_resunet.yaml \
  --checkpoint "$BASE_DIR/best.pt" \
  --set seed="$SEED" --set data.hdf5="$DATA_H5" \
  --set output_dir="outputs/research/eval_baseline_seed${SEED}"
python scripts/evaluate.py --config configs/research/proposed_rvqtwin.yaml \
  --checkpoint "$PROP_DIR/best.pt" \
  --set seed="$SEED" --set data.hdf5="$DATA_H5" \
  --set model.phase_prior_checkpoint="$PRIOR_DIR/best.pt" \
  --set output_dir="outputs/research/eval_proposed_seed${SEED}"

# 6) 分域、传感器扰动和纯网络延迟。
python scripts/evaluate_robustness.py --config configs/research/proposed_rvqtwin.yaml \
  --checkpoint "$PROP_DIR/best.pt" \
  --set seed="$SEED" --set data.hdf5="$DATA_H5" \
  --set model.phase_prior_checkpoint="$PRIOR_DIR/best.pt" \
  --output-dir "outputs/research/robustness_seed${SEED}"
python scripts/evaluate_cross_domain.py --config configs/research/proposed_rvqtwin.yaml \
  --checkpoint "$PROP_DIR/best.pt" \
  --set seed="$SEED" --set data.hdf5="$DATA_H5" \
  --set model.phase_prior_checkpoint="$PRIOR_DIR/best.pt" \
  --output-dir "outputs/research/cross_domain_seed${SEED}"
python scripts/benchmark_latency.py --config configs/research/proposed_rvqtwin.yaml \
  --checkpoint "$PROP_DIR/best.pt" \
  --set seed="$SEED" --set data.hdf5="$DATA_H5" \
  --set model.phase_prior_checkpoint="$PRIOR_DIR/best.pt" \
  --output "outputs/research/latency_seed${SEED}.json"

python scripts/compare_experiments.py \
  "outputs/research/mean_phase_seed${SEED}/summary.json" \
  "outputs/research/eval_paper_style_seed${SEED}/summary.json" \
  "outputs/research/eval_baseline_seed${SEED}/summary.json" \
  "outputs/research/eval_proposed_seed${SEED}/summary.json" \
  --output "outputs/research/main_table_seed${SEED}.csv"

echo "主流程完成。条件扩散基线耗时较长，请单独运行 scripts/run_diffusion_baseline.sh。"
