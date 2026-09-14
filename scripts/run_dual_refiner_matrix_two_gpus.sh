#!/usr/bin/env bash
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

DRY_RUN=0
RUN_TAG="${RUN_TAG:-dual_matrix_$(date -u +%Y%m%dT%H%M%SZ)}"
WORKERS_PER_GPU="${WORKERS_PER_GPU:-8}"
CPU_THREADS_PER_PROCESS="${CPU_THREADS_PER_PROCESS:-2}"
PYTHON_BIN="${MCFQPI_PYTHON:-/root/miniconda3/envs/mcf-qpi/bin/python}"
DATA_H5="${DATA_H5:-data/processed/official/mcf_qpi_128_v2.h5}"

usage() {
  cat <<'EOF'
Usage: bash scripts/run_dual_refiner_matrix_two_gpus.sh [options]

Options:
  --dry-run                  Print the resolved two-GPU plan without training.
  --run-tag TAG              Use TAG in new output and log directories.
  --workers-per-gpu N        DataLoader workers for each active GPU process (default: 8).
  --help                     Show this help.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --run-tag)
      [[ $# -ge 2 ]] || { echo "--run-tag requires a value" >&2; exit 2; }
      RUN_TAG="$2"
      shift 2
      ;;
    --workers-per-gpu)
      [[ $# -ge 2 ]] || { echo "--workers-per-gpu requires a value" >&2; exit 2; }
      WORKERS_PER_GPU="$2"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

[[ "$RUN_TAG" =~ ^[A-Za-z0-9._-]+$ ]] || {
  echo "RUN_TAG may contain only letters, digits, dot, underscore, and hyphen" >&2
  exit 2
}
[[ "$WORKERS_PER_GPU" =~ ^[1-9][0-9]*$ ]] || {
  echo "WORKERS_PER_GPU must be a positive integer" >&2
  exit 2
}
[[ "$CPU_THREADS_PER_PROCESS" =~ ^[1-9][0-9]*$ ]] || {
  echo "CPU_THREADS_PER_PROCESS must be a positive integer" >&2
  exit 2
}

LOG_ROOT="outputs/run_logs"
CONT_DIR="outputs/research_corrected/official/dual_prior_continuous_refiner_seed42"
CONT_EVAL="outputs/research_corrected/official/eval_continuous_seed42_val_${RUN_TAG}"
RVQ2_DIR="outputs/research_corrected/official/dual_prior_rvq2_refiner_seed42_${RUN_TAG}"
RVQ2_EVAL="outputs/research_corrected/official/eval_rvq2_seed42_val_${RUN_TAG}"
Q1_PRIOR_DIR="outputs/research_corrected/official/phase_rvqvae_rvq1_seed42_${RUN_TAG}"
Q1_PRIOR_EVAL="outputs/research_corrected/official/eval_phase_rvqvae_rvq1_seed42_val_${RUN_TAG}"
RVQ1_DIR="outputs/research_corrected/official/dual_prior_rvq1_refiner_seed42_${RUN_TAG}"
RVQ1_EVAL="outputs/research_corrected/official/eval_rvq1_seed42_val_${RUN_TAG}"
GPU0_LOG="${LOG_ROOT}/${RUN_TAG}_gpu0.nohup.log"
GPU1_LOG="${LOG_ROOT}/${RUN_TAG}_gpu1.nohup.log"

print_plan() {
  echo "RUN_TAG=${RUN_TAG}"
  echo "WORKERS_PER_GPU=${WORKERS_PER_GPU}"
  echo "MAX_CONCURRENT_DATALOADER_WORKERS=$((2 * WORKERS_PER_GPU))"
  echo "GPU0_STAGE_1=continuous_resume"
  echo "GPU0_STAGE_2=continuous_eval"
  echo "GPU0_STAGE_3=rvq2_train"
  echo "GPU0_STAGE_4=rvq2_eval"
  echo "GPU1_STAGE_1=rvq1_prior_train"
  echo "GPU1_STAGE_2=rvq1_prior_eval"
  echo "GPU1_STAGE_3=rvq1_refiner_train"
  echo "GPU1_STAGE_4=rvq1_refiner_eval"
  echo "GPU0_LOG=${GPU0_LOG}"
  echo "GPU1_LOG=${GPU1_LOG}"
}

print_plan
if [[ "$DRY_RUN" -eq 1 ]]; then
  exit 0
fi

[[ -x "$PYTHON_BIN" ]] || { echo "Python executable not found: $PYTHON_BIN" >&2; exit 1; }
[[ -f "$DATA_H5" ]] || { echo "HDF5 file not found: $DATA_H5" >&2; exit 1; }
[[ -f "$CONT_DIR/last.pt" ]] || { echo "Continuous resume checkpoint not found" >&2; exit 1; }
[[ -f "outputs/research_corrected/official/phase_rvqvae_seed42/best.pt" ]] || {
  echo "RVQ2 phase-prior checkpoint not found" >&2
  exit 1
}

for output in "$CONT_EVAL" "$RVQ2_DIR" "$RVQ2_EVAL" "$Q1_PRIOR_DIR" "$Q1_PRIOR_EVAL" "$RVQ1_DIR" "$RVQ1_EVAL"; do
  [[ ! -e "$output" ]] || { echo "Refusing to reuse existing output: $output" >&2; exit 1; }
done

mkdir -p "$LOG_ROOT"
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"

run_python() {
  local gpu="$1"
  shift
  env -u CUDA_VISIBLE_DEVICES \
    MCFQPI_DEVICE="cuda:${gpu}" \
    PYTHONPATH="$PYTHONPATH" \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS="$CPU_THREADS_PER_PROCESS" \
    MKL_NUM_THREADS="$CPU_THREADS_PER_PROCESS" \
    "$PYTHON_BIN" "$@"
}

run_gpu0() {
  local continuous_status="TRAIN_FAILED"
  local rvq2_status="TRAIN_FAILED"

  if time run_python 0 scripts/train_inverse.py \
    --config configs/research/dual_prior_continuous_refiner.yaml \
    --resume "$CONT_DIR/last.pt" \
    --set device=cuda:0 \
    --set seed=42 \
    --set reproducibility.mode=fast \
    --set deterministic=true \
    --set loader.num_workers="$WORKERS_PER_GPU" \
    --set loader.persistent_workers=false \
    --set output_dir="$CONT_DIR"
  then
    continuous_status="EVAL_FAILED"
    if time run_python 0 scripts/evaluate.py \
      --config configs/research/dual_prior_continuous_refiner.yaml \
      --checkpoint "$CONT_DIR/best.inference.pt" \
      --set device=cuda:0 \
      --set loader.num_workers="$WORKERS_PER_GPU" \
      --set loader.persistent_workers=false \
      --set evaluation.split=val \
      --set output_dir="$CONT_EVAL"
    then
      continuous_status="PASS"
    fi
  fi

  if time run_python 0 scripts/train_inverse.py \
    --config configs/research/dual_prior_rvq2_refiner.yaml \
    --set device=cuda:0 \
    --set seed=42 \
    --set reproducibility.mode=fast \
    --set deterministic=true \
    --set loader.num_workers="$WORKERS_PER_GPU" \
    --set loader.persistent_workers=false \
    --set output_dir="$RVQ2_DIR"
  then
    rvq2_status="EVAL_FAILED"
    if time run_python 0 scripts/evaluate.py \
      --config configs/research/dual_prior_rvq2_refiner.yaml \
      --checkpoint "$RVQ2_DIR/best.inference.pt" \
      --set device=cuda:0 \
      --set loader.num_workers="$WORKERS_PER_GPU" \
      --set loader.persistent_workers=false \
      --set evaluation.split=val \
      --set output_dir="$RVQ2_EVAL"
    then
      rvq2_status="PASS"
    fi
  fi

  echo "GPU0_STATUS continuous=${continuous_status} rvq2=${rvq2_status}"
  [[ "$continuous_status" == "PASS" && "$rvq2_status" == "PASS" ]]
}

run_gpu1() {
  local prior_status="TRAIN_FAILED"
  local rvq1_status="NOT_STARTED"

  if time run_python 1 scripts/train_phase_rvqvae.py \
    --config configs/research/phase_rvqvae_rvq1.yaml \
    --set device=cuda:1 \
    --set deterministic=true \
    --set loader.num_workers="$WORKERS_PER_GPU" \
    --set loader.persistent_workers=false \
    --set output_dir="$Q1_PRIOR_DIR"
  then
    prior_status="EVAL_FAILED"
    if time run_python 1 scripts/evaluate_phase_prior.py \
      --config configs/research/phase_rvqvae_rvq1.yaml \
      --checkpoint "$Q1_PRIOR_DIR/best.pt" \
      --split val \
      --output-dir "$Q1_PRIOR_EVAL" \
      --set device=cuda:1 \
      --set loader.num_workers="$WORKERS_PER_GPU" \
      --set loader.persistent_workers=false
    then
      prior_status="PASS"
      rvq1_status="TRAIN_FAILED"
      if time run_python 1 scripts/train_inverse.py \
        --config configs/research/dual_prior_rvq1_refiner.yaml \
        --set device=cuda:1 \
        --set seed=42 \
        --set reproducibility.mode=fast \
        --set deterministic=true \
        --set loader.num_workers="$WORKERS_PER_GPU" \
        --set loader.persistent_workers=false \
        --set model.phase_prior_checkpoint="$Q1_PRIOR_DIR/best.pt" \
        --set output_dir="$RVQ1_DIR"
      then
        rvq1_status="EVAL_FAILED"
        if time run_python 1 scripts/evaluate.py \
          --config configs/research/dual_prior_rvq1_refiner.yaml \
          --checkpoint "$RVQ1_DIR/best.inference.pt" \
          --set device=cuda:1 \
          --set loader.num_workers="$WORKERS_PER_GPU" \
          --set loader.persistent_workers=false \
          --set model.phase_prior_checkpoint="$Q1_PRIOR_DIR/best.pt" \
          --set evaluation.split=val \
          --set output_dir="$RVQ1_EVAL"
        then
          rvq1_status="PASS"
        fi
      fi
    fi
  fi

  echo "GPU1_STATUS phase_rvq1=${prior_status} rvq1=${rvq1_status}"
  [[ "$prior_status" == "PASS" && "$rvq1_status" == "PASS" ]]
}

run_gpu0 >"$GPU0_LOG" 2>&1 &
GPU0_PID=$!
run_gpu1 >"$GPU1_LOG" 2>&1 &
GPU1_PID=$!

echo "GPU0_PID=${GPU0_PID}"
echo "GPU1_PID=${GPU1_PID}"

GPU0_EXIT=0
GPU1_EXIT=0
wait "$GPU0_PID" || GPU0_EXIT=$?
wait "$GPU1_PID" || GPU1_EXIT=$?

echo "GPU0_EXIT=${GPU0_EXIT}"
echo "GPU1_EXIT=${GPU1_EXIT}"
if [[ "$GPU0_EXIT" -ne 0 || "$GPU1_EXIT" -ne 0 ]]; then
  exit 1
fi

