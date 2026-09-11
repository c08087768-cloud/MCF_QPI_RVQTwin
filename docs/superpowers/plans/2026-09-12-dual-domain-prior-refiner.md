# Dual-Domain Prior Refiner Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a dual-domain multi-scale detail-refinement inverse model and a fair official-split seed42 development matrix without altering the existing proposed model.

**Architecture:** `DualDomainPriorRefiner` creates four spatial and Fourier feature scales, fuses matching scales with learned gates, maps the deepest fusion feature through the existing RVQ phase prior, and decodes all fused scales into a bounded detail image. It returns the old inverse-model fields plus `phase_prior`, `detail_phase`, and `detail_scale`, so existing loss, evaluator, and inference-checkpoint mechanisms continue to work.

**Tech Stack:** Python 3.12, PyTorch, PyYAML, pytest, current MCF-QPI training engine and HDF5 dataset.

**Spec:** `docs/superpowers/specs/2026-09-12-dual-domain-prior-refiner-design.md`

## Global Constraints

- Do not alter `DualDomainRVQTwin`, its output semantics, or an existing configuration file.
- New development runs are official HDF5 v2, cycle-off, batch 32, 80 epochs, AdamW at `1.5e-4` with `1e-4` weight decay, cosine minimum LR `1e-6`, and `val_phase_l1` selection.
- `detail_scale_max` is finite and positive; output `detail_scale` remains in `(0, detail_scale_max)`.
- `phase_prior_checkpoint` remains compulsory for a training model; inference must load only a self-contained checkpoint.
- On this Windows host pytest must add `--basetemp .pytest-tmp` because the default temporary directory is unavailable to Conda.

---

### Task 1: Add a tested multi-scale prior-refiner model

**Files:**
- Create: `src/mcfqpi/models/dual_domain_prior_refiner.py`
- Modify: `src/mcfqpi/models/__init__.py`
- Modify: `tests/test_models.py`

**Interfaces:**
- Consumes `PhaseRVQVAE`, `FourierMagnitude`, `ConvNormAct`, `DownBlock`, `UpBlock`, `ResidualBlock`, `set_requires_grad`.
- Produces `DualDomainPriorRefiner(phase_prior, *, base_channels=32, dropout=0.10, residual_scale=0.15, detail_scale_max=0.25, detail_scale_init=0.05, freeze_prior_encoder=True, freeze_codebook=True, freeze_decoder=False, use_spatial=True, use_frequency=True, use_quantization=True)`.
- Its forward method returns legacy inverse fields plus `phase_prior`, `detail_phase`, and scalar `detail_scale`.

- [ ] **Step 1: Write failing output-contract tests**

In `tests/test_models.py`, import `pytest` and `DualDomainPriorRefiner`, then add:

```python
def test_prior_refiner_composes_prior_and_detail_exactly() -> None:
    model = DualDomainPriorRefiner(
        _prior(), base_channels=8, dropout=0.0, detail_scale_max=0.25, detail_scale_init=0.05
    )
    phase = torch.rand(2, 1, 64, 64)
    output = model(torch.rand(2, 1, 64, 64), target_phase=phase)
    assert output["phase"].shape == phase.shape
    assert output["phase_prior"].shape == phase.shape
    assert output["detail_phase"].shape == phase.shape
    assert output["log_scale"].shape == phase.shape
    assert torch.allclose(
        output["phase"], output["phase_prior"] + output["detail_scale"] * output["detail_phase"]
    )
    assert 0.0 < float(output["detail_scale"]) < 0.25
    assert output["target_indices"].shape == (2, 2, 8, 8)


def test_prior_refiner_continuous_mode_has_no_token_targets() -> None:
    model = DualDomainPriorRefiner(_prior(), base_channels=8, dropout=0.0, use_quantization=False)
    output = model(torch.rand(2, 1, 64, 64), target_phase=torch.rand(2, 1, 64, 64))
    assert "token_logits" not in output
    assert "target_indices" not in output
    assert float(output["vq_loss"]) == 0.0


def test_prior_refiner_rejects_nonpositive_detail_scale_max() -> None:
    with pytest.raises(ValueError, match="detail_scale_max"):
        DualDomainPriorRefiner(_prior(), base_channels=8, detail_scale_max=0.0)
```

- [ ] **Step 2: Verify RED**

Run `conda run --no-capture-output -n mcf-qpi python -m pytest tests/test_models.py -q --basetemp .pytest-tmp`.

Expected: test collection fails because `DualDomainPriorRefiner` does not yet exist.

- [ ] **Step 3: Implement the minimal model**

Create the model with these concrete components:

```python
class PyramidEncoder(nn.Module):
    # stem plus three DownBlock layers; returns [skip1, skip2, skip3, bottleneck]
    # with channels [2B, 4B, 8B, 8B] at [H, H/2, H/4, H/8].


class GatedScaleFusion(nn.Module):
    # gate = sigmoid(Conv1x1(cat(spatial, frequency)))
    # return gate * spatial + (1 - gate) * frequency + residual_fusion(cat(...))


class DualDomainPriorRefiner(nn.Module):
    # fuse four scales to G1..G4; project G4 to z_e.
    # reuse the old RVQ, target-token, continuous-residual, freeze, and intervention behavior.
    # phase_prior = phase_prior.decode(z_q + continuous_residual).
    # upsample G4 with G3, G2, G1 through UpBlock to obtain detail features.
    # detail_phase = phase_head(detail_features); log_scale = uncertainty_head(detail_features).clamp(-7, 1).
    # detail_scale = detail_scale_max * sigmoid(detail_scale_logit).
    # phase = phase_prior + detail_scale * detail_phase.
```

If a branch is disabled, return zero tensors for its corresponding latent/fusion field; reject configurations where both branches are disabled. Validate that `detail_scale_init` is finite and strictly between zero and the maximum. Add the class to `models/__init__.py`.

- [ ] **Step 4: Verify GREEN**

Run `conda run --no-capture-output -n mcf-qpi python -m pytest tests/test_models.py -q --basetemp .pytest-tmp`.

Expected: all model tests pass.

- [ ] **Step 5: Commit**

Run `git add src/mcfqpi/models/dual_domain_prior_refiner.py src/mcfqpi/models/__init__.py tests/test_models.py` followed by `git commit -m "feat: add dual-domain prior refiner model"`.

### Task 2: Register factory construction and self-contained evaluation

**Files:**
- Modify: `src/mcfqpi/factory.py`
- Modify: `tests/test_checkpoint_contract.py`

**Interfaces:**
- Consumes the Task 1 model and `build_phase_prior`.
- Produces `build_inverse_architecture({"type": "dual_domain_prior_refiner", ...})` and training initialization from `phase_prior_checkpoint`.

- [ ] **Step 1: Write a failing factory/checkpoint test**

Add to `tests/test_checkpoint_contract.py`:

```python
def _refiner_config() -> dict:
    return {
        "type": "dual_domain_prior_refiner", "base_channels": 8, "dropout": 0.0,
        "detail_scale_max": 0.25, "detail_scale_init": 0.05,
        "phase_prior_model": {
            "latent_channels": 16, "base_channels": 8, "downsample_stages": 3,
            "codebook_size": 16, "num_quantizers": 2, "dropout": 0.0,
        },
    }


def test_refiner_inference_checkpoint_is_self_contained(tmp_path: Path) -> None:
    config = _refiner_config()
    model = build_inverse_architecture(config)
    path = tmp_path / "refiner.inference.pt"
    save_inference_checkpoint(model, path, model_config=config, metadata={"schema_version": "1.0"})
    loaded, checkpoint = build_inverse_from_checkpoint(path)
    assert checkpoint["model_config"] == config
    assert loaded(torch.rand(1, 1, 32, 32))["phase"].shape == (1, 1, 32, 32)
```

- [ ] **Step 2: Verify RED**

Run `conda run --no-capture-output -n mcf-qpi python -m pytest tests/test_checkpoint_contract.py -q --basetemp .pytest-tmp`.

Expected: factory reports unknown type `dual_domain_prior_refiner`.

- [ ] **Step 3: Implement factory support**

Import `DualDomainPriorRefiner`. Make the training initialization guard accept either `DualDomainRVQTwin` or `DualDomainPriorRefiner`. Add aliases `dual_domain_prior_refiner` and `prior_refiner` in `build_inverse_architecture`; construct the phase prior with `build_phase_prior(config.get("phase_prior_model", {}))`, then pass all Task 1 options.

- [ ] **Step 4: Verify GREEN**

Run `conda run --no-capture-output -n mcf-qpi python -m pytest tests/test_models.py tests/test_checkpoint_contract.py -q --basetemp .pytest-tmp`.

Expected: all selected tests pass.

- [ ] **Step 5: Commit**

Run `git add src/mcfqpi/factory.py tests/test_checkpoint_contract.py` followed by `git commit -m "feat: register prior refiner architecture"`.

### Task 3: Log raw, weighted, and refiner-specific diagnostics

**Files:**
- Modify: `src/mcfqpi/training/losses.py`
- Modify: `scripts/train_inverse.py`
- Create: `tests/test_training_diagnostics.py`

**Interfaces:**
- Produces `InverseLoss.weighted_terms(raw_terms) -> dict[str, Tensor]` using names `weighted_<term>`.
- Produces `InverseLoss.model_diagnostics(outputs) -> dict[str, Tensor]` with `detail_scale`, `mean_abs_detail_phase`, and `mean_abs_phase_prior` only for a refiner output.

- [ ] **Step 1: Write failing diagnostic tests**

Create `tests/test_training_diagnostics.py`:

```python
import torch

from mcfqpi.training.losses import InverseLoss, LossWeights


def test_weighted_terms_preserve_raw_terms() -> None:
    criterion = InverseLoss(LossWeights(phase_l1=1.0, gradient=0.15, token=0.2))
    raw = {"phase_l1": torch.tensor(2.0), "gradient": torch.tensor(4.0), "token": torch.tensor(3.0)}
    weighted = criterion.weighted_terms(raw)
    assert raw["gradient"].item() == 4.0
    assert weighted["weighted_phase_l1"].item() == 2.0
    assert weighted["weighted_gradient"].item() == 0.6
    assert weighted["weighted_token"].item() == 0.6


def test_refiner_diagnostics_are_optional() -> None:
    criterion = InverseLoss(LossWeights())
    diagnostics = criterion.model_diagnostics({
        "detail_scale": torch.tensor(0.05),
        "detail_phase": torch.tensor([[[[-2.0, 1.0]]]]),
        "phase_prior": torch.tensor([[[[0.2, -0.4]]]]),
    })
    assert diagnostics["detail_scale"].item() == 0.05
    assert diagnostics["mean_abs_detail_phase"].item() == 1.5
    assert diagnostics["mean_abs_phase_prior"].item() == 0.3
    assert criterion.model_diagnostics({}) == {}
```

- [ ] **Step 2: Verify RED**

Run `conda run --no-capture-output -n mcf-qpi python -m pytest tests/test_training_diagnostics.py -q --basetemp .pytest-tmp`.

Expected: `InverseLoss` lacks `weighted_terms`.

- [ ] **Step 3: Implement the two diagnostic helpers and use them**

In `InverseLoss.weighted_terms`, multiply each supplied term by `getattr(self.weights, name)` and prefix it `weighted_`. In `model_diagnostics`, use detached scalar tensors; calculate absolute means for the two images and return `{}` when the fields are absent.

Change the `step` closure in `scripts/train_inverse.py` to:

```python
loss, raw_terms = criterion(outputs, batch, forward_twin=forward_twin)
terms = {**raw_terms, **criterion.weighted_terms(raw_terms), **criterion.model_diagnostics(outputs)}
return loss, terms, outputs
```

- [ ] **Step 4: Verify GREEN**

Run `conda run --no-capture-output -n mcf-qpi python -m pytest tests/test_training_diagnostics.py tests/test_p0_regressions.py -q --basetemp .pytest-tmp`.

Expected: all selected tests pass, including existing P0 regression coverage.

- [ ] **Step 5: Commit**

Run `git add src/mcfqpi/training/losses.py scripts/train_inverse.py tests/test_training_diagnostics.py` followed by `git commit -m "feat: log weighted inverse-loss diagnostics"`.

### Task 4: Add the three seed42 refiner configurations and fairness tests

**Files:**
- Create: `configs/research/dual_prior_continuous_refiner.yaml`
- Create: `configs/research/phase_rvqvae_rvq1.yaml`
- Create: `configs/research/dual_prior_rvq1_refiner.yaml`
- Create: `configs/research/dual_prior_rvq2_refiner.yaml`
- Modify: `tests/test_research_protocol_configs.py`

**Interfaces:**
- Produces an official-HDF5 seed42 one-level phase-prior configuration plus three refiner configurations. Only quantization and RVQ-level count may differ between refiner configurations.

- [ ] **Step 1: Write a failing protocol test**

Append to `tests/test_research_protocol_configs.py`:

```python
def test_prior_refiner_seed42_configs_are_fair_and_cycle_off() -> None:
    names = ["dual_prior_continuous_refiner.yaml", "dual_prior_rvq1_refiner.yaml", "dual_prior_rvq2_refiner.yaml"]
    configs = [load_yaml(ROOT / "configs/research" / name) for name in names]
    for config in configs:
        assert config["seed"] == 42
        assert config["data"]["hdf5"] == "data/processed/official/mcf_qpi_128_v2.h5"
        assert config["forward_twin"]["enabled"] is False
        assert config["loss"]["cycle_l1"] == 0.0
        assert config["loss"]["cycle_spectral"] == 0.0
        assert config["training"]["epochs"] == 80
        assert config["training"]["checkpoint_monitor"] == "val_phase_l1"
    assert configs[0]["model"]["use_quantization"] is False
    assert configs[1]["model"]["phase_prior_model"]["num_quantizers"] == 1
    assert configs[2]["model"]["phase_prior_model"]["num_quantizers"] == 2
    q1_prior = load_yaml(ROOT / "configs/research/phase_rvqvae_rvq1.yaml")
    assert q1_prior["model"]["num_quantizers"] == 1
    assert configs[1]["model"]["phase_prior_checkpoint"] == q1_prior["output_dir"] + "/best.pt"
```

- [ ] **Step 2: Verify RED**

Run `conda run --no-capture-output -n mcf-qpi python -m pytest tests/test_research_protocol_configs.py -q --basetemp .pytest-tmp`.

Expected: `FileNotFoundError` for the first new YAML file.

- [ ] **Step 3: Add immutable, fair configurations**

Create `phase_rvqvae_rvq1.yaml` from the corrected phase-prior protocol: official v2 HDF5, no augmentation, 60 epochs, `num_quantizers: 1`, and output directory `outputs/research_corrected/official/phase_rvqvae_rvq1_seed42`. Its `best.pt` is the only permitted `phase_prior_checkpoint` in the RVQ1 refiner YAML.

Copy the corrected proposed protocol values to every refiner YAML: official v2 HDF5, augmentation, loader, phase-prior dimensions, loss weights, training schedule, and evaluation fields. Set model type `dual_domain_prior_refiner`, `detail_scale_max: 0.25`, `detail_scale_init: 0.05`, and unique `outputs/research_corrected/official/..._seed42` output paths.

Set continuous to `use_quantization: false`, `loss.vq: 0.0`, and `loss.token: 0.0`. Set RVQ1/RVQ2 to quantization true and phase-prior `num_quantizers` one/two with existing VQ and token weights. All cycle fields remain zero.

- [ ] **Step 4: Verify GREEN**

Run `conda run --no-capture-output -n mcf-qpi python -m pytest tests/test_research_protocol_configs.py tests/test_checkpoint_contract.py -q --basetemp .pytest-tmp`.

Expected: all selected tests pass.

- [ ] **Step 5: Commit**

Run `git add configs/research/phase_rvqvae_rvq1.yaml configs/research/dual_prior_continuous_refiner.yaml configs/research/dual_prior_rvq1_refiner.yaml configs/research/dual_prior_rvq2_refiner.yaml tests/test_research_protocol_configs.py` followed by `git commit -m "feat: add fair prior-refiner development configs"`.

### Task 5: Document run order and prove end-to-end local safety

**Files:**
- Modify: `docs/19_审核修复后正式实验执行手册.md`
- Modify: `tests/test_research_protocol_configs.py`

**Interfaces:**
- Produces instructions for seed42 validation-only order: controlled ResUNet-32, dual continuous, dual-prior continuous, RVQ2 refiner, then one-level phase-prior training followed by RVQ1 refiner.

- [ ] **Step 1: Write a failing manual-presence test**

Add this test:

```python
def test_execution_manual_mentions_refiner_seed42_dev_order() -> None:
    manual = (ROOT / "docs/19_审核修复后正式实验执行手册.md").read_text(encoding="utf-8")
    assert "dual_prior_continuous_refiner.yaml" in manual
    assert "dual_prior_rvq1_refiner.yaml" in manual
    assert "dual_prior_rvq2_refiner.yaml" in manual
    assert "CUDA_VISIBLE_DEVICES=1" in manual
```

- [ ] **Step 2: Verify RED**

Run `conda run --no-capture-output -n mcf-qpi python -m pytest tests/test_research_protocol_configs.py::test_execution_manual_mentions_refiner_seed42_dev_order -q --basetemp .pytest-tmp`.

Expected: assertion failure because the manual does not yet name the new configurations.

- [ ] **Step 3: Add concrete development commands**

Append a manual section listing the five configuration order above. State that runs use validation only and that test, seed123, seed2026, and strict split require at least 2% seed42 validation improvement. Document physical GPU1 mapping exactly as `CUDA_VISIBLE_DEVICES=1 MCFQPI_DEVICE=cuda:0`; for throughput development additionally use `reproducibility.mode=fast`, `deterministic=true`, `loader.num_workers=8`, and `loader.persistent_workers=false`.

- [ ] **Step 4: Verify full suite and CPU smoke**

Run:

```powershell
conda run --no-capture-output -n mcf-qpi python -m pytest -q --basetemp .pytest-tmp
conda run --no-capture-output -n mcf-qpi python scripts/train_inverse.py --config configs/research/dual_prior_rvq2_refiner.yaml --set device=cpu --set data.hdf5=data/smoke/mcf_smoke_64.h5 --set output_dir=outputs/smoke/prior_refiner_cpu --set training.epochs=1 --set training.max_train_batches=2 --set training.max_val_batches=1 --set loader.num_workers=0 --set loader.persistent_workers=false
```

Expected: full tests pass and the smoke output directory contains `best.inference.pt` and `training_summary.json`.

- [ ] **Step 5: Commit**

Run `git add docs/19_审核修复后正式实验执行手册.md tests/test_research_protocol_configs.py` followed by `git commit -m "docs: add prior-refiner development protocol"`.

## Plan Self-Review

- Task 1 covers the new multi-scale gated architecture, bounded detail path, phase-prior path, output contract, and invalid-config handling.
- Task 2 covers factory initialization and independent inference checkpoint restoration.
- Task 3 covers raw/weighted loss visibility and refiner diagnostics in train/validation history.
- Task 4 creates controlled continuous, RVQ1, and RVQ2 comparisons under a fixed official protocol.
- Task 5 documents GPU mapping and prevents unsupported test/three-seed expansion before the registered seed42 result.
