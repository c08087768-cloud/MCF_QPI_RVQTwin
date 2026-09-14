# Remote Experiment Monitor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 每 10 分钟由 Windows 启动 WSL，将服务器全部实验的轻量日志和状态增量同步到本地，再由 Codex 仅凭本地报告识别进度、失败、停滞和完成。

**Architecture:** Windows 任务计划程序只负责启动一个带锁的 WSL shell 脚本；shell 脚本通过现有免密 SSH/rsync 获取轻量文件并执行本地 Python 分析器。Codex heartbeat 不连接服务器，只读取分析器生成的 `latest_report.json`，并按变化和告警规则决定是否通知。

**Tech Stack:** Bash、OpenSSH、rsync、Python 3.10+、PyYAML、pytest、Windows PowerShell ScheduledTasks、Codex heartbeat automation

**Spec:** `docs/superpowers/specs/2026-09-15-remote-experiment-monitor-design.md`

## Global Constraints

- 服务器固定为 `root@192.168.0.3:8706`，项目目录为 `/home/user/MCF_QPI_RVQTwin_corrected`。
- 本地项目固定为 `E:\laser and photonics\MCF-QPI_项目\MCF_QPI_RVQTwin`，WSL 路径为 `/mnt/e/laser and photonics/MCF-QPI_项目/MCF_QPI_RVQTwin`。
- 同步和 Codex 检查周期均为 10 分钟；停滞检查阈值为 30 分钟。
- 自动发现服务器 `outputs/` 下的全部实验；不能要求逐个配置运行名称。
- Codex 不得连接服务器；它只能读取本地 `outputs/remote_monitor/`。
- 同步不得使用 `--delete`，不得修改服务器实验，不得同步 checkpoint、HDF5、图片、TensorBoard event 或大型逐样本文件。
- 日志允许持续增量同步；非日志 JSON/CSV/YAML 单文件上限为 10 MiB。
- 单次网络失败保留上次有效状态；连续两次失败才通知回传链路异常。
- 实现必须与当前未提交的双 GPU 运行脚本分开提交。

---

## File Structure

- Create `scripts/monitor/inspect_experiments.py`: 纯本地分析器；把镜像、服务器快照和上次报告转换为稳定 JSON 报告。
- Create `scripts/monitor/collect_remote_status.py`: 通过标准输入临时发送到服务器执行；只读取进程、GPU和文件元数据并输出 JSON。
- Create `scripts/monitor/sync_remote_experiments.sh`: WSL 入口；负责锁、SSH、两段 rsync、原子更新和调用分析器。
- Create `scripts/monitor/monitor.example.yaml`: 不含秘密的配置模板。
- Create `scripts/monitor/install_monitor_task.ps1`: 创建、查看和删除 Windows 当前用户的 10 分钟定时任务。
- Create `tests/test_experiment_monitor.py`: 分析器状态机和数值有效性测试。
- Create `tests/test_remote_status_collector.py`: 服务器状态采集器的解析和文件清单测试。
- Create `tests/test_monitor_sync_script.py`: 同步脚本的配置、过滤、锁和失败保留测试。
- Create `tests/test_windows_monitor_task_script.py`: Windows 任务脚本的参数和安全设置静态契约测试。
- Modify `.gitignore`: 忽略 `scripts/monitor/monitor.local.yaml`。
- Create `docs/21_远程实验日志自动回传与Codex监控手册.md`: 安装、运行、检查、停用和故障排查说明。

---

### Task 1: 本地实验状态分析器

**Files:**
- Create: `scripts/monitor/inspect_experiments.py`
- Create: `tests/test_experiment_monitor.py`

**Interfaces:**
- Consumes: `mirror: Path`、`snapshot: dict[str, Any]`、`sync_state: dict[str, Any]`、`previous_report: dict[str, Any] | None`、`now_epoch: float`、`stall_seconds: int`。
- Produces: `build_report(mirror, snapshot, sync_state, previous_report, now_epoch, stall_seconds) -> dict[str, Any]`，报告 schema 为 `mcfqpi-remote-monitor-1.0`；CLI 写入 `--output` 指定路径。

- [ ] **Step 1: 写入状态机失败测试**

在 `tests/test_experiment_monitor.py` 创建固定 UTC 时间和临时镜像，至少覆盖以下断言：

```python
from __future__ import annotations

import json
from pathlib import Path

from scripts.monitor.inspect_experiments import build_report


NOW = 2_000_000_000.0


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_training_complete_requires_summary_epochs_and_checkpoint_inventory(tmp_path: Path):
    run = tmp_path / "mirror/official/model_seed42"
    write_json(run / "training_summary.json", {"epochs_completed": 80})
    (run / "resolved_config.yaml").write_text("training:\n  epochs: 80\n", encoding="utf-8")
    snapshot = {
        "collected_at_epoch": NOW,
        "processes": [],
        "gpus": [],
        "files": [
            {"path": "outputs/official/model_seed42/best.inference.pt", "mtime_epoch": NOW},
            {"path": "outputs/official/model_seed42/last.pt", "mtime_epoch": NOW},
        ],
    }
    report = build_report(tmp_path / "mirror", snapshot, {"failure_count": 0}, None, NOW, 1800)
    assert report["experiments"][0]["status"] == "training_complete"


def test_process_and_busy_gpu_after_threshold_is_running_quiet(tmp_path: Path):
    log = tmp_path / "mirror/run_logs/model_seed42.nohup.log"
    log.parent.mkdir(parents=True)
    log.write_text("Epoch 010/80 | train=0.4 | val=0.3\n", encoding="utf-8")
    snapshot = {
        "collected_at_epoch": NOW,
        "processes": [{"pid": 123, "command": "train_inverse.py --set output_dir=outputs/model_seed42"}],
        "gpus": [{"index": 1, "utilization_percent": 96, "compute_pids": [123]}],
        "files": [{"path": "outputs/run_logs/model_seed42.nohup.log", "mtime_epoch": NOW - 1900}],
    }
    report = build_report(tmp_path / "mirror", snapshot, {"failure_count": 0}, None, NOW, 1800)
    assert report["experiments"][0]["status"] == "running_quiet"


def test_idle_process_after_threshold_is_suspected_stall(tmp_path: Path):
    log = tmp_path / "mirror/run_logs/model_seed42.nohup.log"
    log.parent.mkdir(parents=True)
    log.write_text("Epoch 010/80 | train=0.4 | val=0.3\n", encoding="utf-8")
    snapshot = {
        "collected_at_epoch": NOW,
        "processes": [{"pid": 123, "command": "train_inverse.py --set output_dir=outputs/model_seed42"}],
        "gpus": [{"index": 1, "utilization_percent": 0, "compute_pids": []}],
        "files": [{"path": "outputs/run_logs/model_seed42.nohup.log", "mtime_epoch": NOW - 1900}],
    }
    report = build_report(tmp_path / "mirror", snapshot, {"failure_count": 0}, None, NOW, 1800)
    assert report["experiments"][0]["status"] == "suspected_stall"


def test_traceback_is_failed_and_nan_metric_is_invalid(tmp_path: Path):
    log = tmp_path / "mirror/run_logs/broken.nohup.log"
    log.parent.mkdir(parents=True)
    log.write_text("Traceback (most recent call last):\nRuntimeError: CUDA out of memory\n", encoding="utf-8")
    summary = tmp_path / "mirror/official/eval_bad/summary.json"
    write_json(summary, {"samples": 6000, "overall": {"mae_rad": float("nan")}})
    snapshot = {
        "collected_at_epoch": NOW,
        "processes": [],
        "gpus": [],
        "files": [
            {"path": "outputs/run_logs/broken.nohup.log", "mtime_epoch": NOW},
            {"path": "outputs/official/eval_bad/summary.json", "mtime_epoch": NOW},
        ],
    }
    report = build_report(tmp_path / "mirror", snapshot, {"failure_count": 0}, None, NOW, 1800)
    statuses = {item["name"]: item["status"] for item in report["experiments"]}
    assert statuses["broken"] == "failed"
    assert statuses["eval_bad"] != "evaluation_complete"


def test_previous_report_generates_only_ten_epoch_milestones(tmp_path: Path):
    log = tmp_path / "mirror/run_logs/model_seed42.nohup.log"
    log.parent.mkdir(parents=True)
    log.write_text("Epoch 020/80 | train=0.4 | val=0.3 | best=0.3\n", encoding="utf-8")
    snapshot = {
        "collected_at_epoch": NOW,
        "processes": [{"pid": 123, "command": "train_inverse.py --set output_dir=outputs/model_seed42"}],
        "gpus": [{"index": 1, "utilization_percent": 90, "compute_pids": [123]}],
        "files": [{"path": "outputs/run_logs/model_seed42.nohup.log", "mtime_epoch": NOW}],
    }
    previous = {
        "generated_at_epoch": NOW - 600,
        "experiments": [{"name": "model_seed42", "epoch": 19, "status": "running"}],
    }
    report = build_report(
        tmp_path / "mirror", snapshot, {"failure_count": 0}, previous, NOW, 1800
    )
    assert [event["kind"] for event in report["events"]] == ["epoch_milestone"]
    assert report["experiments"][0]["eta_seconds"] == 36_000


def test_two_consecutive_sync_failures_create_one_event(tmp_path: Path):
    snapshot = {"collected_at_epoch": NOW - 1200, "processes": [], "gpus": [], "files": []}
    sync_state = {"failure_count": 2, "last_error": "ssh exited 255"}
    report = build_report(tmp_path / "mirror", snapshot, sync_state, None, NOW, 1800)
    assert [event["kind"] for event in report["events"]] == ["sync_failure"]
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_experiment_monitor.py -q`

Expected: collection fails with `ModuleNotFoundError: No module named 'scripts.monitor'`.

- [ ] **Step 3: 实现最小分析器**

实现四个公开接口：`build_report(mirror: Path, snapshot: dict[str, Any], sync_state: dict[str, Any], previous_report: dict[str, Any] | None, now_epoch: float, stall_seconds: int) -> dict[str, Any]`、`parse_epoch_line(text: str) -> dict[str, float | int] | None`、`finite_metrics(value: object) -> bool` 和 `main() -> None`。

CLI 参数固定为：

```text
--mirror PATH
--snapshot PATH
--sync-state PATH
--previous-report PATH
--output PATH
--stall-minutes 30
```

报告顶层至少包含 `schema_version`、`generated_at_epoch`、`sync`、`experiments` 和 `events`。实验记录至少包含 `name`、`kind`、`status`、`epoch`、`total_epochs`、`metrics`、`last_update_epoch`、`process`、`gpu`、`evidence`。评估完成要求 `summary.json` 的 `samples` 为正整数，且 `overall` 中的数值指标均为有限值。`events` 只记录新出现的十轮里程碑、完成、失败、异常退出、疑似停滞和同步链路异常。

状态优先级固定为：有效完成证据 > 当前明确 traceback/非有限结果 > 活跃进程 > 无进程且不完整 > unknown。历史中已经被后续成功运行覆盖的错误不能压过有效完成摘要。

- [ ] **Step 4: 运行分析器测试**

Run: `python -m pytest tests/test_experiment_monitor.py -q`

Expected: all tests pass.

- [ ] **Step 5: 提交分析器**

```bash
git add scripts/monitor/inspect_experiments.py tests/test_experiment_monitor.py
git commit -m "feat: classify synchronized experiment status"
```

---

### Task 2: 只读服务器状态采集器

**Files:**
- Create: `scripts/monitor/collect_remote_status.py`
- Create: `tests/test_remote_status_collector.py`

**Interfaces:**
- Consumes: 命令行第一个参数 `PROJECT_ROOT`；可注入 `run_command(args) -> CompletedProcess[str]` 便于测试。
- Produces: stdout 上的 `mcfqpi-remote-status-1.0` JSON，包含进程、GPU和 `outputs/` 文件清单，不写服务器文件。

- [ ] **Step 1: 写入采集器失败测试**

测试至少验证：

```python
def test_collect_status_filters_experiment_processes_and_parses_gpus(tmp_path):
    outputs = tmp_path / "outputs/run"
    outputs.mkdir(parents=True)
    (outputs / "training_summary.json").write_text("{}", encoding="utf-8")
    result = collect_status(
        tmp_path,
        now_epoch=1234.0,
        process_text="321\t90\ttrain_inverse.py --set output_dir=outputs/run\n",
        gpu_text="GPU-aaaa, 1, 91, 4096\n",
        compute_text="GPU-aaaa, 321\n",
    )
    assert result["processes"][0]["pid"] == 321
    assert result["gpus"][0]["compute_pids"] == [321]
    assert result["files"][0]["path"] == "outputs/run/training_summary.json"
```

另测不存在 `outputs/` 时返回空文件清单，以及文件路径始终相对项目根目录且不跟随项目外符号链接。

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_remote_status_collector.py -q`

Expected: collection fails because `collect_remote_status.py` does not exist.

- [ ] **Step 3: 实现只读采集器**

使用 `ps -eo pid=,etimes=,args=` 获取进程；只保留命令包含 `train_inverse.py`、`train_phase_rvqvae.py`、`train_forward_twin.py`、`train_diffusion.py`、`evaluate.py`、`evaluate_phase_prior.py`、`evaluate_rvq_evidence.py` 或 `calibrate_uncertainty.py` 的记录。

使用 `nvidia-smi --query-gpu=uuid,index,utilization.gpu,memory.used --format=csv,noheader,nounits` 和 `nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader,nounits`，再通过 GPU UUID 映射计算 PID。`nvidia-smi` 不可用时返回 `gpus: []` 和一条 warning，不能使整个快照失败。

文件清单递归记录 `outputs/` 下文件的相对路径、大小和 mtime；它可以列出被排除的大文件，以便本地验证 checkpoint 是否存在，但不得读取其内容。

- [ ] **Step 4: 运行采集器测试**

Run: `python -m pytest tests/test_remote_status_collector.py -q`

Expected: all tests pass.

- [ ] **Step 5: 提交采集器**

```bash
git add scripts/monitor/collect_remote_status.py tests/test_remote_status_collector.py
git commit -m "feat: collect read-only remote experiment status"
```

---

### Task 3: WSL 增量同步入口

**Files:**
- Create: `scripts/monitor/sync_remote_experiments.sh`
- Create: `scripts/monitor/monitor.example.yaml`
- Create: `tests/test_monitor_sync_script.py`
- Modify: `.gitignore`

**Interfaces:**
- Consumes: `--config scripts/monitor/monitor.local.yaml` 和可选 `--dry-run`。
- Produces: `outputs/remote_monitor/mirror/`、`status/latest.json`、`status/sync_state.json`、`reports/latest_report.json` 和 `sync.log`；成功返回 0，锁占用返回 0 并记录 skip，真实同步失败返回非零。

- [ ] **Step 1: 写入同步脚本失败测试**

测试从 Python 调用 `bash`；Windows 上自动使用 `wsl.exe bash`。测试创建带空格的临时路径和 fake `ssh`/`rsync` 可执行文件，并断言：

```python
def test_dry_run_preserves_spaced_paths_and_never_uses_delete(tmp_path):
    completed = run_sync(tmp_path, "--dry-run")
    assert completed.returncode == 0
    assert "--delete" not in completed.stdout
    assert "*.pt" in completed.stdout
    assert "*.h5" in completed.stdout


def test_sync_uses_two_passes_and_keeps_previous_snapshot_on_ssh_failure(tmp_path):
    local_project = to_wsl_path(tmp_path / "project with spaces")
    status_dir = tmp_path / "project with spaces/outputs/remote_monitor/status"
    status_dir.mkdir(parents=True)
    latest = status_dir / "latest.json"
    latest.write_text('{"marker":"last-good"}', encoding="utf-8")
    fake_bin = tmp_path / "fake-bin"
    fake_bin.mkdir()
    write_wsl_executable(fake_bin / "ssh", "#!/usr/bin/env bash\nexit 255\n")
    write_wsl_executable(fake_bin / "rsync", "#!/usr/bin/env bash\nexit 0\n")
    config = write_config(tmp_path, local_project)
    completed = run_sync(
        config,
        extra_env={"PATH": f"{to_wsl_path(fake_bin)}:/usr/bin:/bin"},
    )
    assert completed.returncode != 0
    assert json.loads(latest.read_text(encoding="utf-8")) == {"marker": "last-good"}
    state = json.loads((status_dir / "sync_state.json").read_text(encoding="utf-8"))
    assert state["failure_count"] == 1


def test_script_declares_two_sync_passes_and_atomic_staging():
    script = Path("scripts/monitor/sync_remote_experiments.sh").read_text(encoding="utf-8")
    assert "SYNC_PASS=logs" in script
    assert "SYNC_PASS=structured" in script
    assert "--max-size=" in script
    assert "MIRROR_STAGING" in script
    assert "--delete" not in script
```

同一测试文件实现 `to_wsl_path`、`write_wsl_executable`、`write_config` 和 `run_sync`：Linux 直接使用 Bash；Windows 使用 `wsl.exe wslpath` 转换路径后调用 WSL Bash。`write_wsl_executable` 写入 LF 结尾并在 WSL 中执行 `chmod +x`。`write_config` 写入测试专用 YAML，端口为 8706、停滞阈值为 30、结构化文件上限为 10 MiB。

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_monitor_sync_script.py -q`

Expected: tests fail because the shell script and config template are absent.

- [ ] **Step 3: 添加配置模板和忽略规则**

`monitor.example.yaml` 使用以下完整键：

```yaml
remote_host: root@192.168.0.3
remote_port: 8706
remote_project: /home/user/MCF_QPI_RVQTwin_corrected
local_project: /mnt/e/laser and photonics/MCF-QPI_项目/MCF_QPI_RVQTwin
python_executable: python3
sync_interval_minutes: 10
stall_threshold_minutes: 30
structured_file_max_mib: 10
```

向 `.gitignore` 增加：

```gitignore
scripts/monitor/monitor.local.yaml
```

- [ ] **Step 4: 实现同步脚本**

脚本必须包含 `set -euo pipefail`，使用 `flock -n` 锁定 `outputs/remote_monitor/.sync.lock`，并按以下顺序执行：

1. 通过本地 Python/PyYAML读取配置，禁止 `eval` 和直接 `source` 配置。
2. 检查 `ssh`、`rsync`、Python 和远端连通性。
3. 将 `collect_remote_status.py` 通过 stdin 发送给远端 `python3 - REMOTE_PROJECT`，先写临时快照。
4. 第一段 rsync 只包含目录和 `*.log`/`*.nohup.log`，不设置大小上限。
5. 第二段 rsync 包含 JSON、CSV、YAML，设置 `--max-size=10m`。
6. 两段都显式排除 checkpoint、HDF5、图片、TensorBoard events，并且不得出现 `--delete`。
7. 两段同步先写入 `mirror.staging.<pid>`。它由上一个成功镜像通过硬链接副本初始化；文件系统不支持硬链接时退回普通副本。rsync 保持默认临时文件加 rename 行为，不能使用 `--inplace`。只有两段都成功才把 staging 目录切换为 `mirror/`；失败时删除精确 staging 目录并保留旧镜像。
8. 状态快照成功后用同目录原子 rename 更新 `latest.json`，并把旧值保存为 `previous.json`。
9. 调用分析器，并以相同方式更新 `latest_report.json`。
10. SSH/rsync 失败时保留上次成功镜像，更新 `status/sync_state.json` 中的连续失败计数；随后用上一个有效快照重新运行分析器，使第二次连续失败能够出现在新报告中。同步成功时计数归零。

所有用户可见日志行带 UTC 时间。命令中所有路径必须双引号包裹。

- [ ] **Step 5: 运行同步测试和 shell 语法检查**

Run: `python -m pytest tests/test_monitor_sync_script.py -q`

Expected: all tests pass.

Run on WSL/Linux: `bash -n scripts/monitor/sync_remote_experiments.sh`

Expected: exit code 0 with no output.

- [ ] **Step 6: 提交同步入口**

```bash
git add .gitignore scripts/monitor/sync_remote_experiments.sh scripts/monitor/monitor.example.yaml tests/test_monitor_sync_script.py
git commit -m "feat: sync remote experiment logs into local mirror"
```

---

### Task 4: Windows 十分钟定时任务

**Files:**
- Create: `scripts/monitor/install_monitor_task.ps1`
- Create: `tests/test_windows_monitor_task_script.py`

**Interfaces:**
- Consumes: PowerShell 参数 `-ProjectWslPath`、`-IntervalMinutes 10`、`-TaskName MCF-QPI-Remote-Monitor`、`-DryRun`、`-Uninstall`。
- Produces: 当前用户级 Scheduled Task；`-DryRun` 只显示将创建的 action/trigger/settings，`-Uninstall` 只删除精确任务名。

- [ ] **Step 1: 写入 PowerShell 契约失败测试**

静态契约测试读取脚本文本并断言：

```python
def test_windows_task_is_non_overlapping_and_has_safe_controls():
    text = Path("scripts/monitor/install_monitor_task.ps1").read_text(encoding="utf-8")
    assert "MultipleInstances" in text
    assert "IgnoreNew" in text
    assert "-DryRun" in text or "[switch]$DryRun" in text
    assert "[switch]$Uninstall" in text
    assert "wsl.exe" in text
    assert "sync_remote_experiments.sh" in text
```

另断言脚本不包含服务器密码、私钥正文、`RunLevel Highest` 或硬编码 Windows 用户名。

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_windows_monitor_task_script.py -q`

Expected: file-not-found failure.

- [ ] **Step 3: 实现任务安装脚本**

使用 `New-ScheduledTaskAction` 调用 `wsl.exe`，参数把 WSL 项目路径作为一个整体传入 shell。使用 `New-ScheduledTaskTrigger -Once` 加 10 分钟 repetition interval；设置 `MultipleInstances = IgnoreNew`、`StartWhenAvailable = true`，以当前用户和 limited run level 注册。

`-DryRun` 必须输出任务名、周期和完整 WSL action，但不注册任务。`-Uninstall` 在任务不存在时也成功返回。脚本结束时通过 `Get-ScheduledTask -TaskName` 显示状态。

- [ ] **Step 4: 运行契约测试和 PowerShell 语法检查**

Run: `python -m pytest tests/test_windows_monitor_task_script.py -q`

Expected: all tests pass.

Run on Windows PowerShell:

```powershell
$errors = $null
[void][System.Management.Automation.Language.Parser]::ParseFile(
  (Resolve-Path 'scripts/monitor/install_monitor_task.ps1'),
  [ref]$null,
  [ref]$errors
)
if ($errors.Count -gt 0) { $errors | Format-List; exit 1 }
```

Expected: exit code 0.

- [ ] **Step 5: 提交 Windows 任务支持**

```bash
git add scripts/monitor/install_monitor_task.ps1 tests/test_windows_monitor_task_script.py
git commit -m "feat: schedule WSL experiment synchronization"
```

---

### Task 5: 本地安装、端到端验收和 Codex heartbeat

**Files:**
- Create: `docs/21_远程实验日志自动回传与Codex监控手册.md`
- Runtime only, ignored: `scripts/monitor/monitor.local.yaml`
- Runtime only, ignored: `outputs/remote_monitor/`

**Interfaces:**
- Consumes: Tasks 1–4 的 CLI 和已经配置好的免密 SSH。
- Produces: 可复查的本地同步报告、Windows 定时任务和当前 Codex 任务的静默型 10 分钟 heartbeat。

- [ ] **Step 1: 写运行手册**

手册必须给出以下操作的完整命令：复制示例配置、一次性 `--dry-run`、手动同步、查看 `latest_report.json`、安装任务、立即触发任务、查询任务、卸载任务，以及 SSH/rsync/PyYAML 缺失时的诊断。明确说明 Codex 无服务器权限、`Ctrl+C` 不影响服务器训练、同步不会下载权重。

- [ ] **Step 2: 运行全部自动测试**

Run: `python -m pytest -q`

Expected: all tests pass; existing environment-specific skip is allowed only if its reason is printed and unrelated to monitor tests.

Run: `git diff --check`

Expected: no whitespace errors.

- [ ] **Step 3: 创建本机配置并执行一次 dry run**

从 `monitor.example.yaml` 复制为 `monitor.local.yaml`，保持已确认的服务器、本地路径和 10/30 分钟参数。运行：

```bash
bash scripts/monitor/sync_remote_experiments.sh \
  --config scripts/monitor/monitor.local.yaml \
  --dry-run
```

Expected: 输出精确远端、本地路径、两段 rsync 过滤规则和分析命令；不创建远端文件、不出现 `--delete`。

- [ ] **Step 4: 手动同步并验证本地报告**

```bash
bash scripts/monitor/sync_remote_experiments.sh \
  --config scripts/monitor/monitor.local.yaml
python3 -m json.tool outputs/remote_monitor/reports/latest_report.json >/dev/null
find outputs/remote_monitor/mirror -type f \
  \( -name '*.pt' -o -name '*.h5' -o -name '*.hdf5' \) -print
```

Expected: 同步返回 0；JSON 可解析；最后一个 `find` 没有输出；报告至少列出服务器当前已存在的实验或明确返回空实验列表。

- [ ] **Step 5: 安装并验证 Windows 定时任务**

先 dry run，再注册任务并立即触发：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File `
  scripts/monitor/install_monitor_task.ps1 -DryRun
powershell.exe -NoProfile -ExecutionPolicy Bypass -File `
  scripts/monitor/install_monitor_task.ps1
Start-ScheduledTask -TaskName 'MCF-QPI-Remote-Monitor'
Get-ScheduledTaskInfo -TaskName 'MCF-QPI-Remote-Monitor'
```

Expected: `LastTaskResult` 为 0；等待 10 分钟后 `sync.log` 增加一条成功记录。再等待一个周期，确认连续两个周期成功且没有重叠进程。

- [ ] **Step 6: 创建 Codex heartbeat**

使用 Codex automation 工具为当前任务创建每 10 分钟 heartbeat，名称为“监控 MCF-QPI 服务器实验”。保存的提示词使用以下语义：

```text
只读取本地 E:\laser and photonics\MCF-QPI_项目\MCF_QPI_RVQTwin\outputs\remote_monitor\reports\latest_report.json，以及该报告明确引用的本地日志片段。不得使用 SSH，不得连接服务器，不得启动、停止或修改实验。与上次已报告状态比较：每跨过 10 个 epoch、实验完成、首次出现失败/异常退出、首次出现疑似停滞，或同步连续失败两次时，用中文简要通知并给出证据。30 分钟无日志更新但进程和 GPU 仍活跃时，只说明仍在计算，不宣称卡死。健康且没有实质变化时保持安静。多个实验同时变化时合并汇报。
```

Expected: automation 状态为 active，目标是当前任务，通知策略不要求每轮成功都提醒。

- [ ] **Step 7: 提交手册并记录验收证据**

在手册末尾记录本机验证日期、两次成功同步的 UTC 时间、Windows 任务名和报告 schema；不得记录秘密、完整进程环境或私钥路径。

```bash
git add docs/21_远程实验日志自动回传与Codex监控手册.md
git commit -m "docs: add remote experiment monitor runbook"
```

- [ ] **Step 8: 最终检查工作区边界**

Run: `git status --short`

Expected: 本任务创建的受控文件均已提交；`monitor.local.yaml` 和 `outputs/remote_monitor/` 不出现在状态中；原先未提交的 `scripts/run_dual_refiner_matrix_two_gpus.sh` 与 `tests/test_dual_gpu_matrix_script.py` 仍保持原状，未被混入任何监控提交。
