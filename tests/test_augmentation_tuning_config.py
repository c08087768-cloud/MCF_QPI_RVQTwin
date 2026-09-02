from __future__ import annotations

from pathlib import Path

from mcfqpi.config import load_yaml


ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = ROOT / "configs" / "research" / "proposed_rvqtwin.yaml"
TUNED_CONFIG = ROOT / "configs" / "tuning" / "proposed_aug_baseline.yaml"
RUN_SCRIPT = ROOT / "scripts" / "run_proposed_aug_baseline.sh"


def _changed_leaf_values(base: dict, candidate: dict, prefix: str = "") -> dict[str, tuple[object, object]]:
    changed: dict[str, tuple[object, object]] = {}
    for key in base.keys() | candidate.keys():
        path = f"{prefix}.{key}" if prefix else key
        left = base.get(key)
        right = candidate.get(key)
        if isinstance(left, dict) and isinstance(right, dict):
            changed.update(_changed_leaf_values(left, right, path))
        elif left != right:
            changed[path] = (left, right)
    return changed


def test_legacy_tuned_config_records_baseline_augmentation_change() -> None:
    base = load_yaml(BASE_CONFIG)
    candidate = load_yaml(TUNED_CONFIG)
    changed = _changed_leaf_values(base, candidate)
    assert {key: changed[key] for key in changed if key.startswith("data.augmentation.")} == {
        "data.augmentation.background": (0.03, 0.02),
        "data.augmentation.blur_probability": (0.15, 0.10),
        "data.augmentation.dead_pixel_probability": (0.0005, 0.0002),
        "data.augmentation.gain": (0.15, 0.10),
        "data.augmentation.gaussian_noise": (0.015, 0.01),
        "data.augmentation.saturation_probability": (0.08, 0.05),
    }
    assert candidate["output_dir"] == "outputs/tuning/proposed_aug_baseline_seed42"


def test_runner_trains_seed42_and_evaluates_validation_only() -> None:
    text = RUN_SCRIPT.read_text(encoding="utf-8")
    assert 'SEED="${SEED:-42}"' in text
    assert 'MCFQPI_DEVICE="${MCFQPI_DEVICE:-cuda:1}"' in text
    assert "train_inverse.py" in text
    assert "evaluate.py" in text
    assert "evaluation.split=val" in text
    assert "evaluation.mc_samples=1" in text
    assert "evaluation.save_predictions=false" in text
    assert "evaluation.split=test" not in text
    assert "outputs/tuning/proposed_aug_baseline_seed${SEED}" in text
    assert "outputs/tuning/eval_proposed_aug_baseline_seed${SEED}" in text
