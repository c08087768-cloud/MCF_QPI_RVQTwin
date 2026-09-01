#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"
SEED="${SEED:-42}"
python scripts/train_diffusion.py --config configs/research/diffusion_baseline.yaml \
  --set seed="$SEED" --set output_dir="outputs/research/diffusion_seed${SEED}"
for STEPS in 25 50 100; do
  python scripts/evaluate_diffusion.py --config configs/research/diffusion_baseline.yaml \
    --checkpoint "outputs/research/diffusion_seed${SEED}/best.pt" \
    --set model.sample_steps="$STEPS" \
    --set output_dir="outputs/research/eval_diffusion_${STEPS}steps_seed${SEED}"
done
