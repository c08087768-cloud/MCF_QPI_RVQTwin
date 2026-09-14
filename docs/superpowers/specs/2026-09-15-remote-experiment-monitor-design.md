# 远程实验日志自动回传与 Codex 本地监控设计

## 1. 目标

在不允许 Codex 直接连接服务器的前提下，每 10 分钟自动把服务器上的轻量实验记录同步到本地，并由当前 Codex 任务检查实验进度、完成状态和异常。

实验训练与监控必须相互独立。同步或分析失败不能终止、暂停或修改服务器上的训练进程。

## 2. 已确认的约束

- 服务器：`root@192.168.0.3`，SSH 端口 `8706`。
- 服务器项目目录：`/home/user/MCF_QPI_RVQTwin_corrected`。
- 本地项目位于 Windows 的 `E:\laser and photonics\MCF-QPI_项目\MCF_QPI_RVQTwin`，由 WSL 通过 `/mnt/e/...` 访问。
- 免密 SSH 已由用户配置完成，任何密码、私钥或令牌都不能写入项目。
- Windows 在实验期间保持开机和联网，可以锁屏，但不能休眠。
- 同步和检查周期为 10 分钟。
- 自动发现服务器 `outputs/` 下的全部实验，不要求用户逐个填写实验名称。
- 连续 30 分钟没有产物更新时进入停滞检查。

## 3. 选定方案

采用“Windows 任务计划程序 + WSL 同步脚本 + 本地分析器 + Codex heartbeat”四层结构：

```text
服务器 outputs/ 与运行状态
            |
            | 免密 SSH / 增量 rsync，每 10 分钟
            v
Windows 任务计划程序启动 WSL 同步脚本
            |
            v
本地 outputs/remote_monitor/
            |
            | 只读本地状态和轻量结果
            v
当前 Codex 任务判断进度、失败、停滞和完成
```

不用 WSL `cron`，因为 WSL 退出后定时器可能不再运行。也不让 Codex 执行 SSH；Codex 只读取同步到工作区中的本地文件。

## 4. 组件和职责

### 4.1 WSL 同步脚本

文件：`scripts/monitor/sync_remote_experiments.sh`

职责：

1. 使用非阻塞文件锁，防止两个同步周期重叠。
2. 检查本地目录和免密 SSH 是否可用。
3. 从服务器采集 UTC 时间、实验 Python 进程、完整命令行和两张 GPU 的利用率、显存及计算进程。
4. 从服务器 `outputs/` 自动发现并增量同步轻量文件。
5. 调用本地分析器生成结构化报告。
6. 将本次同步成功或失败写入 `sync.log` 和状态文件。

同步包含：

- `*.log` 和 `*.nohup.log`；
- `history.json`；
- `training_summary.json`；
- `summary.json`；
- `resolved_config.yaml` 和 `reproducibility.json`；
- 校准、RVQ evidence、审计和检查结果等轻量 `*.json`、`*.csv`。

除持续增长的日志外，结构化文件默认设置 10 MiB 的单文件上限。超过上限的文件只进入服务器文件清单，不回传内容，从而排除大型逐样本 CSV。

默认排除：

- `*.pt`、`*.pth`、`*.ckpt`；
- `*.h5`、`*.hdf5`；
- 图片、TensorBoard event 文件和大型逐样本产物。

同步不使用 `--delete`，不会删除服务器或本地结果。正在增长的文件使用临时文件或 rsync 的安全更新方式，避免分析器读到半写入文件。

### 4.2 本地实验分析器

文件：`scripts/monitor/inspect_experiments.py`

分析器不连接服务器，只读取同步镜像和本次服务器状态快照。它自动发现实验，并尽量通过输出目录、日志名称、进程命令行和配置文件建立对应关系。无法可靠归属的日志仍作为独立记录报告，不进行猜测性合并。

每个实验生成以下状态之一：

- `running`：对应进程存在，且日志或结果仍在更新；
- `running_quiet`：30 分钟无新输出，但进程存在且 GPU 仍在计算；
- `suspected_stall`：超过 30 分钟无更新，进程存在但 GPU 没有相应活动；
- `failed`：日志出现明确错误，例如 traceback、非有限 loss、CUDA OOM 或进程错误退出；
- `exited_incomplete`：进程消失，但缺少完成证据；
- `training_complete`：训练摘要有效、完成轮数达到配置要求，并且服务器文件清单确认 `best.inference.pt` 和 `last.pt` 存在；
- `evaluation_complete`：评估摘要可解析，包含样本数和主要指标，所有数值有限且日志没有错误；
- `unknown`：证据不足，不能安全归类。

错误匹配以明确 traceback 和错误结尾为主，不能仅因为历史日志中出现普通单词 `error` 就误报。完成判定依赖结构化文件和服务器文件清单，不能仅凭进程消失或日志最后一行判断。

### 4.3 Windows 定时任务

文件：`scripts/monitor/install_monitor_task.ps1`

安装脚本注册一个当前 Windows 用户级任务，每 10 分钟调用默认 WSL 发行版执行同步脚本。任务设置为：

- 同一任务正在运行时不启动第二份；
- 网络暂时不可用时记录失败，不修改上一次成功同步的镜像；
- Windows 开机后按计划恢复运行；
- 使用绝对路径，不依赖交互式 shell 的当前目录；
- 不在命令行或任务定义中存放密码和私钥。

本地连接参数写入不纳入 Git 的配置文件。仓库只提供不含秘密的示例配置。

### 4.4 Codex 自动检查

为当前任务创建每 10 分钟运行一次的 heartbeat。它只读取本地 `outputs/remote_monitor/reports/latest_report.json` 和报告中引用的少量日志片段，不执行 SSH，也不启动或停止实验。

通知策略：

- 每跨过 10 个 epoch 汇报一次；
- 实验成功完成时立即汇报；
- 出现 traceback、NaN、CUDA 错误、OOM 或异常退出时立即汇报；
- 30 分钟无更新时，根据进程和 GPU 快照说明“仍在计算”或“疑似停滞”；
- SSH 同步连续失败两次时提示本地回传链路异常；
- 其余健康且无实质变化的检查保持安静。

多项实验同时变化时合并成一条简短报告。报告包含实验名称、状态、当前 epoch、最近训练/验证指标、最佳验证指标和基于最近 epoch 耗时得到的粗略剩余时间。证据不足时明确写“无法判断”，不把估计写成完成事实。

## 5. 本地文件布局

```text
scripts/monitor/
├── sync_remote_experiments.sh
├── inspect_experiments.py
├── install_monitor_task.ps1
└── monitor.example.yaml

outputs/remote_monitor/
├── mirror/                  # 服务器 outputs 的轻量镜像
├── status/
│   ├── latest.json          # 最新服务器状态快照
│   └── previous.json        # 上一个成功快照
├── reports/
│   ├── latest_report.json   # Codex 的主要输入
│   └── previous_report.json
└── sync.log
```

`outputs/remote_monitor/` 是运行产物，不提交 Git。实际使用的 `monitor.local.yaml` 是本机配置，也不提交 Git。

## 6. 配置接口

本机 YAML 配置至少包含：

- `REMOTE_HOST=root@192.168.0.3`；
- `REMOTE_PORT=8706`；
- `REMOTE_PROJECT=/home/user/MCF_QPI_RVQTwin_corrected`；
- `LOCAL_PROJECT=/mnt/e/laser and photonics/MCF-QPI_项目/MCF_QPI_RVQTwin`；
- `SYNC_INTERVAL_MINUTES=10`；
- `STALL_THRESHOLD_MINUTES=30`。

允许以后调整地址、目录和阈值而不改代码。使用 YAML 而不是直接 `source` shell 环境文件，使带空格和中文的路径能够按一个完整字符串读取。

## 7. 异常处理

- 单次 SSH 或网络失败：保留上次成功镜像，记录失败，不把全部实验误报为退出。
- 连续两次同步失败：报告“回传链路异常”，但不推断服务器实验失败。
- JSON 正在写入或损坏：保留上一个有效报告，将该文件标记为暂不可解析。
- 本地磁盘写入失败：同步返回非零状态并记录明确路径。
- 某一实验日志异常：不妨碍其他实验继续被发现和报告。
- Windows 或 WSL 重启：下一个计划周期继续增量同步，不删除历史文件。

## 8. 验证方案

### 自动测试

- 用临时目录模拟成功训练、运行中、失败、异常退出和评估完成。
- 验证 30 分钟边界以及 GPU 活跃/空闲时的不同结论。
- 验证历史错误文本不会覆盖后续成功完成证据。
- 验证 NaN、无穷值、损坏 JSON 和缺失字段被拒绝。
- 验证自动发现多项实验并保持目录隔离。
- 验证同步包含和排除规则，特别是不会回传 checkpoint 和 HDF5。
- 验证单实例锁和网络失败时保留上次成功状态。

### 人工验收

1. 手动执行一次同步，确认能读取服务器全部当前实验日志。
2. 检查本地目录中不存在 checkpoint、HDF5 或 SSH 秘密。
3. 注册 Windows 定时任务并等待一个周期，确认 WSL 原先关闭时仍能完成同步。
4. 构造完成、报错和 30 分钟无更新三种状态，检查报告结论。
5. 创建 Codex heartbeat，确认异常和完成会通知，健康无变化时保持安静。

## 9. 完成标准

只有满足以下条件才认为监控功能完成：

- Windows 定时任务连续两个周期成功运行；
- 本地镜像能够自动发现全部轻量实验记录；
- 不同步大型模型和数据文件；
- 进度、失败、停滞、异常退出和完成判定测试通过；
- Codex 无需服务器访问权限即可基于本地报告给出判断；
- 健康且无变化时不产生重复通知；
- 所有连接秘密均未进入仓库、日志或 Codex 报告。

## 10. 非目标

本轮不实现：

- Codex 直接 SSH 到服务器；
- 从本地启动、停止或修改服务器实验；
- 自动下载模型权重、HDF5、图片或完整逐样本结果；
- 通过 Git 提交实验日志；
- 在服务器安装新的常驻监控服务。
