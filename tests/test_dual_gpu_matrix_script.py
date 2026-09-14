from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_dual_refiner_matrix_two_gpus.sh"


def _bash_command(*arguments: str) -> list[str]:
    if os.name != "nt":
        bash = shutil.which("bash")
        if bash is None:
            pytest.skip("bash is unavailable")
        return [bash, str(SCRIPT), *arguments]

    wsl = shutil.which("wsl.exe")
    if wsl is None:
        pytest.skip("WSL is unavailable")
    drive = SCRIPT.drive.rstrip(":").lower()
    converted = f"/mnt/{drive}/{SCRIPT.as_posix().split(':', 1)[1].lstrip('/')}"
    return [wsl, "bash", converted, *arguments]


def test_dry_run_reports_two_gpu_dependency_plan() -> None:
    completed = subprocess.run(
        _bash_command("--dry-run", "--run-tag", "test-run", "--workers-per-gpu", "8"),
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    assert completed.returncode == 0, completed.stderr
    assert "RUN_TAG=test-run" in completed.stdout
    assert "WORKERS_PER_GPU=8" in completed.stdout
    assert "MAX_CONCURRENT_DATALOADER_WORKERS=16" in completed.stdout
    assert "GPU0_STAGE_1=continuous_resume" in completed.stdout
    assert "GPU0_STAGE_3=rvq2_train" in completed.stdout
    assert "GPU1_STAGE_1=rvq1_prior_train" in completed.stdout
    assert "GPU1_STAGE_3=rvq1_refiner_train" in completed.stdout
    assert completed.stdout.index("GPU1_STAGE_1") < completed.stdout.index("GPU1_STAGE_3")
