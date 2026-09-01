# 数据结构说明

## 1. 原始下载目录

```text
data/
  raw/
    downloads/
      fashion_MCF_QPI_dataset.tar
      MCF_speckle_digits.tar
    extracted/
      ...作者压缩包的实际目录...
```

工程不假设压缩包内部的固定目录名。`inspect_raw_dataset.py` 扫描后，`build_manifest.py` 根据关键词和共同索引配对。

## 2. Manifest CSV

```text
sample_id
  工程唯一 ID

domain
  digits / fashion / unknown

split
  train / val / test / unknown

speckle_path
  真实散斑图绝对路径

phase_path
  真实相位标签绝对路径

pair_method
  normalized_key / natural_order_explicit

class_id
  能可靠推断时记录类别，否则为空

group_id
  严格分组 ID，优先为 phase SHA-256

speckle_sha256 / phase_sha256
  精确文件哈希

phase_phash
  感知哈希，只用于近似重复审计
```

## 3. HDF5

```text
speckle       [N, 1, 128, 128] float32
  每幅图按自身均值归一化并 log1p 压缩到约 [0,1]

phase         [N, 1, 128, 128] float32
  相位除以 π 后的 [0,1]

valid_mask    [N, 1, 128, 128] float32
  真正图像区域为 1，fit-pad 填充为 0

sample_id     [N] UTF-8

domain        [N] UTF-8
split         [N] UTF-8
class_id      [N] UTF-8
```

读取 Dataset 额外返回：

```text
phase_rad = phase × π
index
```

## 4. 输出目录

```text
outputs/<experiment>/
  resolved_config.yaml
  best.pt
  last.pt
  history.json
  training_summary.json
  tensorboard/

outputs/<evaluation>/
  per_sample_metrics.csv
  summary.json
  qualitative_examples.png
  risk_coverage.csv
  predictions.npz        # 可选
```

## 5. Checkpoint

```text
epoch
model
optimizer
scheduler
scaler
best_value
history
config
```

所有正式结果应保存 `resolved_config.yaml`，避免只留下权重而无法复现。
