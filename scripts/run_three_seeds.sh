#!/usr/bin/env bash
set -euo pipefail
for SEED in 42 123 2026; do
  echo "========== SEED=$SEED =========="
  SEED="$SEED" bash scripts/run_research_pipeline.sh
done
