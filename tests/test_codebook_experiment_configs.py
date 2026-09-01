from __future__ import annotations

from pathlib import Path

from mcfqpi.config import load_yaml


ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = ROOT / "configs" / "research" / "phase_rvqvae.yaml"
SMALL_CONFIG = ROOT / "configs" / "research" / "phase_rvqvae_codebook128.yaml"
LOW_VQ_CONFIG = ROOT / "configs" / "research" / "phase_rvqvae_vq025.yaml"
RUN_SCRIPT = ROOT / "scripts" / "run_phase_prior_codebook_experiments.sh"


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


def test_codebook128_config_changes_only_codebook_and_output_dir() -> None:
    base = load_yaml(BASE_CONFIG)
    candidate = load_yaml(SMALL_CONFIG)
    assert _changed_leaf_values(base, candidate) == {
        "model.codebook_size": (256, 128),
        "output_dir": (
            "outputs/research/phase_rvqvae_seed42",
            "outputs/codebook_experiments/codebook128_seed42",
        ),
    }


def test_vq025_config_changes_only_vq_weight_and_output_dir() -> None:
    base = load_yaml(BASE_CONFIG)
    candidate = load_yaml(LOW_VQ_CONFIG)
    assert _changed_leaf_values(base, candidate) == {
        "loss.vq": (0.5, 0.25),
        "output_dir": (
            "outputs/research/phase_rvqvae_seed42",
            "outputs/codebook_experiments/vq025_seed42",
        ),
    }


def test_runner_trains_and_evaluates_both_experiments() -> None:
    text = RUN_SCRIPT.read_text(encoding="utf-8")
    assert "phase_rvqvae_codebook128.yaml" in text
    assert "phase_rvqvae_vq025.yaml" in text
    assert text.count("train_phase_rvqvae.py") == 1
    assert text.count("evaluate_phase_prior.py") == 1
    assert 'run_experiment "codebook128"' in text
    assert 'run_experiment "vq025"' in text
    assert 'SEED="${SEED:-42}"' in text
