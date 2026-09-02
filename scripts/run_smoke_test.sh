#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"

python -m compileall -q src scripts tests
pytest -q
python scripts/make_smoke_dataset.py --overwrite
python scripts/train_phase_rvqvae.py --config configs/smoke/phase_rvqvae.yaml
python scripts/evaluate_phase_prior.py --config configs/smoke/phase_rvqvae.yaml \
  --checkpoint outputs/smoke/phase_rvqvae/best.pt --split test \
  --output-dir outputs/smoke/eval_phase_prior --max-batches 1
python scripts/evaluate_mean_phase.py --config configs/smoke/baseline.yaml \
  --output-dir outputs/smoke/eval_mean_phase
python scripts/train_forward_twin.py --config configs/smoke/forward_twin.yaml
python scripts/train_inverse.py --config configs/smoke/baseline.yaml
python scripts/train_inverse.py --config configs/smoke/proposed.yaml
python scripts/evaluate.py --config configs/smoke/proposed.yaml \
  --checkpoint outputs/smoke/proposed/best.inference.pt \
  --set output_dir=outputs/smoke/eval_proposed
python scripts/train_diffusion.py --config configs/smoke/diffusion.yaml
python scripts/evaluate_diffusion.py --config configs/smoke/diffusion.yaml \
  --checkpoint outputs/smoke/diffusion/best.pt \
  --set output_dir=outputs/smoke/eval_diffusion

echo "Smoke test 全部完成。注意：程序化 smoke 数据只验证代码，不可作为论文结果。"
