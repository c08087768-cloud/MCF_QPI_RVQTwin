#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"
export MCFQPI_DEVICE="${MCFQPI_DEVICE:-cuda:1}"

python -c 'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(torch.cuda.current_device()))'
python scripts/make_smoke_dataset.py --overwrite
python scripts/train_phase_rvqvae.py --config configs/smoke/phase_rvqvae.yaml
python scripts/train_forward_twin.py --config configs/smoke/forward_twin.yaml
python scripts/train_inverse.py --config configs/smoke/proposed.yaml
python scripts/evaluate.py --config configs/smoke/proposed.yaml \
  --checkpoint outputs/smoke/proposed/best.inference.pt \
  --set output_dir=outputs/smoke/cuda_amp_eval
echo "CUDA/AMP smoke 通过：$MCFQPI_DEVICE"
