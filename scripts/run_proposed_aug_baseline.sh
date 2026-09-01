#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"
export MCFQPI_DEVICE="${MCFQPI_DEVICE:-cuda:1}"

SEED="${SEED:-42}"
DATA_H5="${DATA_H5:-data/processed/mcf_qpi_128.h5}"
CONFIG="configs/tuning/proposed_aug_baseline.yaml"
TRAIN_DIR="outputs/tuning/proposed_aug_baseline_seed${SEED}"
EVAL_DIR="outputs/tuning/eval_proposed_aug_baseline_seed${SEED}"
PRIOR="outputs/research/phase_rvqvae_seed${SEED}/best.pt"
TWIN="outputs/research/forward_twin_seed${SEED}/best.pt"

echo "========== proposed augmentation=baseline, SEED=${SEED}, DEVICE=${MCFQPI_DEVICE} =========="
python scripts/train_inverse.py --config "$CONFIG" \
  --set seed="$SEED" \
  --set data.hdf5="$DATA_H5" \
  --set model.phase_prior_checkpoint="$PRIOR" \
  --set forward_twin.checkpoint="$TWIN" \
  --set output_dir="$TRAIN_DIR"

# 参数选择阶段只评估 clean validation，不读取 test。
python scripts/evaluate.py --config "$CONFIG" --checkpoint "$TRAIN_DIR/best.pt" \
  --set seed="$SEED" \
  --set data.hdf5="$DATA_H5" \
  --set model.phase_prior_checkpoint="$PRIOR" \
  --set forward_twin.checkpoint="$TWIN" \
  --set output_dir="$EVAL_DIR" \
  --set evaluation.split=val \
  --set evaluation.mc_samples=1 \
  --set evaluation.save_predictions=false

echo "增强强度单变量实验完成：$TRAIN_DIR"
