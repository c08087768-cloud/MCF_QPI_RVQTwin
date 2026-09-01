#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"
SEED="${SEED:-42}"
DATA_H5="${DATA_H5:-data/processed/mcf_qpi_128.h5}"
PRIOR="outputs/research/phase_rvqvae_seed${SEED}/best.pt"

for NAME in spatial_only_rvq frequency_only_rvq dual_continuous no_forward_twin; do
  CONFIG="configs/ablations/${NAME}.yaml"
  OUT="outputs/ablations/${NAME}_seed${SEED}"
  python scripts/train_inverse.py --config "$CONFIG" \
    --set seed="$SEED" --set data.hdf5="$DATA_H5" \
    --set model.phase_prior_checkpoint="$PRIOR" --set output_dir="$OUT"
  python scripts/evaluate.py --config "$CONFIG" --checkpoint "$OUT/best.pt" \
    --set seed="$SEED" --set data.hdf5="$DATA_H5" \
    --set model.phase_prior_checkpoint="$PRIOR" \
    --set output_dir="outputs/ablations/eval_${NAME}_seed${SEED}"
done

echo "四项核心消融完成。"
