#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"
export MCFQPI_DEVICE="${MCFQPI_DEVICE:-cuda:1}"

SEED="${SEED:-42}"
DATA_H5="${DATA_H5:-data/processed/mcf_qpi_128.h5}"

run_experiment() {
  local name="$1"
  local config="$2"
  local train_dir="outputs/codebook_experiments/${name}_seed${SEED}"
  local eval_dir="outputs/codebook_experiments/eval_${name}_seed${SEED}"

  echo "========== ${name}, SEED=${SEED}, DEVICE=${MCFQPI_DEVICE} =========="
  python scripts/train_phase_rvqvae.py \
    --config "$config" \
    --set seed="$SEED" \
    --set data.hdf5="$DATA_H5" \
    --set output_dir="$train_dir"

  python scripts/evaluate_phase_prior.py \
    --config "$config" \
    --checkpoint "$train_dir/best.pt" \
    --split test \
    --set seed="$SEED" \
    --set data.hdf5="$DATA_H5" \
    --output-dir "$eval_dir"
}

run_experiment "codebook128" "configs/research/phase_rvqvae_codebook128.yaml"
run_experiment "vq025" "configs/research/phase_rvqvae_vq025.yaml"

echo "两项码本单变量实验完成。"

