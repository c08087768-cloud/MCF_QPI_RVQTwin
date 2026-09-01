#!/usr/bin/env bash
# 只评估某个种子，复用已训练好的 best.pt，不重训。
# 用于三种子实验中途断点续跑：当 run_research_pipeline.sh 在评估阶段中断、
# 但 5 个模型权重已存在时，用本脚本从第 5 步（eval）接着跑。
#
# 用法（在 1 号卡上补 seed=123 的评估）：
#   MCFQPI_DEVICE=cuda:1 SEED=123 bash scripts/run_seed_eval_only.sh
#
# 前置条件：以下 best.pt 必须已存在（由训练阶段产出）：
#   outputs/research/phase_rvqvae_seed{SEED}/best.pt
#   outputs/research/paper_style_resunet_seed{SEED}/best.pt
#   outputs/research/baseline_resunet_seed{SEED}/best.pt
#   outputs/research/proposed_rvqtwin_seed{SEED}/best.pt
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"

SEED="${SEED:-123}"
DATA_H5="${DATA_H5:-data/processed/mcf_qpi_128.h5}"

PRIOR_DIR="outputs/research/phase_rvqvae_seed${SEED}"
PAPER_DIR="outputs/research/paper_style_resunet_seed${SEED}"
BASE_DIR="outputs/research/baseline_resunet_seed${SEED}"
PROP_DIR="outputs/research/proposed_rvqtwin_seed${SEED}"

echo "========== 只评估 SEED=$SEED（复用已有权重，不重训）=========="

# 0) 前置检查：5 个 best.pt 必须都在，否则报错退出，避免空跑。
checkpoints=(
  "$PRIOR_DIR/best.pt"
  "$PAPER_DIR/best.pt"
  "$BASE_DIR/best.pt"
  "$PROP_DIR/best.pt"
)
for ckpt in "${checkpoints[@]}"; do
  if [ ! -f "$ckpt" ]; then
    echo "[错误] 缺少权重：$ckpt"
    echo "该种子训练未完成，请先跑完整 run_research_pipeline.sh，而不是本脚本。"
    exit 1
  fi
done
echo "[检查] 4 个 best.pt 均存在，开始评估。"

# 1) 如果 mean_phase 尚未跑，补一个（很快，纯统计量）。
if [ ! -f "outputs/research/mean_phase_seed${SEED}/summary.json" ]; then
  echo "[补] mean_phase 未找到，执行..."
  python scripts/evaluate_mean_phase.py --config configs/research/baseline_resunet.yaml \
    --set seed="$SEED" --set data.hdf5="$DATA_H5" \
    --output-dir "outputs/research/mean_phase_seed${SEED}"
fi

# 2) 三个模型在 test 集评估。
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

# 3) 鲁棒性、跨域、延迟。
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

# 4) 汇总主表。
python scripts/compare_experiments.py \
  "outputs/research/mean_phase_seed${SEED}/summary.json" \
  "outputs/research/eval_paper_style_seed${SEED}/summary.json" \
  "outputs/research/eval_baseline_seed${SEED}/summary.json" \
  "outputs/research/eval_proposed_seed${SEED}/summary.json" \
  --output "outputs/research/main_table_seed${SEED}.csv"

echo "========== SEED=$SEED 评估完成 =========="
echo "主表：outputs/research/main_table_seed${SEED}.csv"
