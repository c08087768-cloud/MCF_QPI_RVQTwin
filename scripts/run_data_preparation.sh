#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"

# 1) 获取两个真实公开 tar：优先用本地已有数据，其次从本地数据集 zip 解出，最后才在线下载。
DIGITS_TAR="data/raw/downloads/MCF_speckle_digits.tar"
FASHION_TAR="data/raw/downloads/fashion_MCF_QPI_dataset.tar"
mkdir -p data/raw/downloads

# 本地数据集 zip 可通过环境变量 MCF_QPI_DATASET_ZIP 指定；未指定时自动探测常见位置。
DATASET_ZIP="${MCF_QPI_DATASET_ZIP:-}"
if [ -z "$DATASET_ZIP" ]; then
  for candidate in \
    "$ROOT/../../MCF-QPI_数据集.zip" \
    "$ROOT/../../MCF_QPI_数据集.zip" \
    "$ROOT/MCF-QPI_数据集.zip"; do
    if [ -f "$candidate" ]; then DATASET_ZIP="$candidate"; break; fi
  done
fi

if [ -f "$DIGITS_TAR" ] && [ -f "$FASHION_TAR" ]; then
  echo "检测到本地 tar（data/raw/downloads/），跳过下载。"
elif [ -n "$DATASET_ZIP" ] && [ -f "$DATASET_ZIP" ]; then
  echo "从本地数据集 zip 解出 tar：$DATASET_ZIP"
  unzip -o "$DATASET_ZIP" "MCF_speckle_digits.tar" "fashion_MCF_QPI_dataset.tar" -d data/raw/downloads
else
  echo "未找到本地数据集，转为在线下载（约 740 MB）。"
  python scripts/download_mcf_qpi.py --datasets all --output-dir data/raw/downloads --hash
fi

# 2) 安全解压。不同数据条目分别进入独立子目录。
python scripts/extract_mcf_qpi.py \
  data/raw/downloads/fashion_MCF_QPI_dataset.tar \
  data/raw/downloads/MCF_speckle_digits.tar \
  --output-dir data/raw/extracted

# 3) 扫描实际结构。这里不预设作者压缩包内部目录名。
python scripts/inspect_raw_dataset.py data/raw/extracted

# 4) 生成带哈希的配对清单。若自动角色识别失败，先查看 inspection 报告；
#    只有在确认两个目录严格同序时，才人工添加 --allow-order-pairing。
python scripts/build_manifest.py data/raw/extracted \
  --output data/processed/mcf_qpi_manifest.csv \
  --compute-hashes \
  --assign-unknown-splits

# 5) 人工抽查、泄漏审计、论文数字对照。
python scripts/visualize_pairs.py data/processed/mcf_qpi_manifest.csv --count 20
python scripts/audit_manifest.py data/processed/mcf_qpi_manifest.csv --fail-on-error
python scripts/validate_public_dataset.py data/processed/mcf_qpi_manifest.csv

# 6) 生成训练友好的 HDF5。原论文已经把配对图统一到 128×128；若下载图也是方形，fit_pad 不会改变几何。
#    若压缩包中存在非方形图，则保持长宽比并用 valid_mask 排除填充区域。
python scripts/cache_to_hdf5.py data/processed/mcf_qpi_manifest.csv \
  --output data/processed/mcf_qpi_128.h5 \
  --size 128 --phase-encoding auto --resize-mode fit_pad \
  --normalization log_mean --overwrite

echo "数据准备完成。请先打开 outputs/data_inspection/random_pairs.png 人工确认配对。"
