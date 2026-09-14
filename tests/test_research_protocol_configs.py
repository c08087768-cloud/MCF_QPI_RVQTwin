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


def test_prior_refiner_seed42_configs_are_fair_and_cycle_off() -> None:
    names = [
        "dual_prior_continuous_refiner.yaml",
        "dual_prior_rvq1_refiner.yaml",
        "dual_prior_rvq2_refiner.yaml",
    ]
    configs = [load_yaml(ROOT / "configs/research" / name) for name in names]
    controlled = load_yaml(ROOT / "configs/research/controlled_resunet32.yaml")
    for config in configs:
        assert config["seed"] == 42
        assert config["data"]["hdf5"] == "data/processed/official/mcf_qpi_128_v2.h5"
        assert config["forward_twin"]["enabled"] is False
        assert config["loss"]["cycle_l1"] == 0.0
        assert config["loss"]["cycle_spectral"] == 0.0
        assert config["training"]["epochs"] == 80
        assert config["training"]["checkpoint_monitor"] == "val_phase_l1"
        assert config["model"]["type"] == "dual_domain_prior_refiner"
        assert config["model"]["base_channels"] == 14
        assert config["data"]["augmentation"] == controlled["data"]["augmentation"]

    assert configs[0]["model"]["use_quantization"] is False
    assert configs[1]["model"]["use_quantization"] is True
    assert configs[2]["model"]["use_quantization"] is True
    assert configs[1]["model"]["phase_prior_model"]["num_quantizers"] == 1
    assert configs[2]["model"]["phase_prior_model"]["num_quantizers"] == 2
    q1_prior = load_yaml(ROOT / "configs/research/phase_rvqvae_rvq1.yaml")
    assert q1_prior["model"]["num_quantizers"] == 1
    assert configs[1]["model"]["phase_prior_checkpoint"] == q1_prior["output_dir"] + "/best.pt"


def test_dual_domain_resunet_config_is_official_and_prior_free() -> None:
    config = load_yaml(ROOT / "configs/research/dual_domain_resunet.yaml")
    assert config["seed"] == 42
    assert config["data"]["hdf5"] == "data/processed/official/mcf_qpi_128_v2.h5"
    assert config["model"]["type"] == "dual_domain_resunet"
    assert config["model"]["base_channels"] == 21
    assert "phase_prior_checkpoint" not in config["model"]
    assert config["forward_twin"]["enabled"] is False
    assert config["loss"]["cycle_l1"] == 0.0
    assert config["loss"]["cycle_spectral"] == 0.0
    assert config["training"]["epochs"] == 80
    assert config["training"]["checkpoint_monitor"] == "val_phase_l1"
    controlled = load_yaml(ROOT / "configs/research/controlled_resunet32.yaml")
    assert config["data"]["augmentation"] == controlled["data"]["augmentation"]


def test_execution_manual_mentions_refiner_seed42_dev_order() -> None:
    manual = (ROOT / "docs/19_审核修复后正式实验执行手册.md").read_text(encoding="utf-8")
    assert "dual_prior_continuous_refiner.yaml" in manual
    assert "dual_prior_rvq1_refiner.yaml" in manual
    assert "dual_prior_rvq2_refiner.yaml" in manual
    assert "phase_rvqvae_rvq1.yaml" in manual
    assert "CUDA_VISIBLE_DEVICES=1" in manual
