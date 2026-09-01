#!/usr/bin/env bash
# MCF-QPI 一键部署脚本 —— 在装有 NVIDIA GPU 的服务器上运行
# 用法：cd MCF_QPI_RVQTwin && bash deploy.sh
# 可选环境变量：
#   ENV_NAME   conda 环境名（默认 mcf-qpi）
#   PY_VER     Python 版本（默认 3.12）
#   TORCH_CU   手动指定 torch CUDA tag（如 cu121/cu124；默认从 nvidia-smi 自动推断）
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

ENV_NAME="${ENV_NAME:-mcf-qpi}"
PY_VER="${PY_VER:-3.12}"
TORCH_CU="${TORCH_CU:-}"

step() { echo ""; echo "======================================================"; echo ">> $1"; echo "======================================================"; }

step "0/6 检查 conda"
if ! command -v conda >/dev/null 2>&1; then
  echo "错误：未找到 conda。请先 source 你的 anaconda 初始化脚本后再重跑，例如：" >&2
  echo "  source ~/anaconda3/etc/profile.d/conda.sh" >&2
  exit 1
fi
conda --version

step "1/6 检查 GPU"
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "错误：未找到 nvidia-smi。请确认服务器有 NVIDIA GPU 且驱动已安装。" >&2
  exit 1
fi
nvidia-smi

step "2/6 确定 PyTorch CUDA 版本"
if [[ -z "$TORCH_CU" ]]; then
  DRV="$(nvidia-smi | grep -oiE 'cuda version:? *[0-9.]+' | head -1 | grep -oE '[0-9.]+' || true)"
  echo "驱动 CUDA 版本：${DRV:-未识别}"
  MAJOR="$(echo "$DRV" | cut -d. -f1)"
  MINOR="$(echo "$DRV" | cut -d. -f2)"
  if [[ "$MAJOR" == "12" ]]; then
    if [[ "${MINOR:-0}" -ge 4 ]]; then TORCH_CU="cu124"; else TORCH_CU="cu121"; fi
  elif [[ "$MAJOR" == "11" ]]; then
    TORCH_CU="cu118"
  else
    TORCH_CU="cu124"
  fi
fi
echo "使用 torch index：https://download.pytorch.org/whl/${TORCH_CU}"
echo "（如需手动指定：TORCH_CU=cu121 bash deploy.sh）"

step "3/6 创建/复用 conda 环境 ${ENV_NAME}（Python ${PY_VER}）"
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "环境 ${ENV_NAME} 已存在，跳过创建。"
else
  conda create -n "$ENV_NAME" python="$PY_VER" -y
fi

step "4/6 安装 PyTorch + 依赖（这一步下载较多，请耐心等待）"
conda run -n "$ENV_NAME" python -m pip install --upgrade pip -q
conda run -n "$ENV_NAME" python -m pip install torch --index-url "https://download.pytorch.org/whl/${TORCH_CU}"
conda run -n "$ENV_NAME" python -m pip install -r requirements.txt
conda run -n "$ENV_NAME" python -m pip install -e .

step "5/6 验证 GPU（必须看到 OK，否则不要继续）"
cat > /tmp/_mcf_check_gpu.py <<'PY'
import torch
assert torch.cuda.is_available(), "GPU 不可用！torch 可能装成了 CPU 版，或驱动与所选 CUDA tag 不匹配。"
print(f"OK GPU 可用：{torch.cuda.get_device_name(0)}  |  torch {torch.__version__}  |  CUDA {torch.version.cuda}")
PY
conda run -n "$ENV_NAME" python /tmp/_mcf_check_gpu.py
rm -f /tmp/_mcf_check_gpu.py

step "6/6 运行 smoke test 验证代码链路"
conda run -n "$ENV_NAME" bash scripts/run_smoke_test.sh

echo ""
echo "======================================================"
echo "部署完成 ✅ 环境 ${ENV_NAME} 已就绪，GPU 验证通过，smoke test 通过。"
echo "开始正式训练："
echo "  conda activate ${ENV_NAME}"
echo "  SEED=42 bash scripts/run_research_pipeline.sh"
echo "  SEED=42 bash scripts/run_ablations.sh"
echo "  SEED=42 bash scripts/run_diffusion_baseline.sh"
echo "  bash scripts/run_three_seeds.sh"
echo "======================================================"
