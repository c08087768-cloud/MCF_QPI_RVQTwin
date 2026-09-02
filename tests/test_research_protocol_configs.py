from __future__ import annotations

from pathlib import Path

from mcfqpi.config import load_yaml


ROOT = Path(__file__).resolve().parents[1]


def _changes(left: dict, right: dict, prefix: str = "") -> dict[str, tuple[object, object]]:
    result: dict[str, tuple[object, object]] = {}
    for key in left.keys() | right.keys():
        path = f"{prefix}.{key}" if prefix else key
        a, b = left.get(key), right.get(key)
        if isinstance(a, dict) and isinstance(b, dict):
            result.update(_changes(a, b, path))
        elif a != b:
            result[path] = (a, b)
    return result


def test_proposed_defaults_to_cycle_off() -> None:
    config = load_yaml(ROOT / "configs/research/proposed_rvqtwin.yaml")

    assert config["forward_twin"]["enabled"] is False
    assert config["loss"]["cycle_l1"] == 0.0
    assert config["loss"]["cycle_spectral"] == 0.0


def test_clean_cycle_config_changes_only_cycle_fields_and_output() -> None:
    base = load_yaml(ROOT / "configs/research/proposed_rvqtwin.yaml")
    cycle = load_yaml(ROOT / "configs/research/proposed_rvqtwin_clean_cycle.yaml")

    assert _changes(base, cycle) == {
        "forward_twin.enabled": (False, True),
        "loss.cycle_l1": (0.0, 0.05),
        "loss.cycle_spectral": (0.0, 0.02),
        "output_dir": (
            "outputs/research_corrected/proposed_cycle_off_seed42",
            "outputs/research_corrected/proposed_clean_cycle_seed42",
        ),
    }
