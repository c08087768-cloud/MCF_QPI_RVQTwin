from pathlib import Path

import pytest

from mcfqpi.protocol import create_freeze_record, verify_freeze_record


def test_freeze_detects_checkpoint_or_config_change(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    checkpoint = tmp_path / "model.pt"
    data = tmp_path / "data.h5"
    config.write_text("evaluation:\n  split: val\n", encoding="utf-8")
    checkpoint.write_bytes(b"weights")
    data.write_bytes(b"dataset")
    record = create_freeze_record(config, checkpoint, data_paths=[data], git_sha="abc")
    verify_freeze_record(record, config, checkpoint, data_paths=[data])

    checkpoint.write_bytes(b"changed")
    with pytest.raises(ValueError, match="freeze"):
        verify_freeze_record(record, config, checkpoint, data_paths=[data])


def test_freeze_id_is_deterministic(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    checkpoint = tmp_path / "model.pt"
    config.write_text("seed: 42\n", encoding="utf-8")
    checkpoint.write_bytes(b"weights")
    left = create_freeze_record(config, checkpoint, data_paths=[], git_sha="abc")
    right = create_freeze_record(config, checkpoint, data_paths=[], git_sha="abc")
    assert left["freeze_id"] == right["freeze_id"]
