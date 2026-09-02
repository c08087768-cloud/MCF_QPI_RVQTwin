from __future__ import annotations

import torch

from mcfqpi.models import (
    ConditionalDenoiser,
    DiffusionPhaseReconstructor,
    DualDomainRVQTwin,
    EmpiricalForwardTwin,
    GaussianDiffusion,
    PhaseRVQVAE,
    ResUNetPhase,
)


def _prior() -> PhaseRVQVAE:
    return PhaseRVQVAE(
        latent_channels=16,
        base_channels=8,
        downsample_stages=3,
        codebook_size=16,
        num_quantizers=2,
        dropout=0.0,
    )


def test_phase_rvqvae_shapes_and_gradients() -> None:
    model = _prior()
    x = torch.rand(2, 1, 64, 64)
    output = model(x)
    assert output["phase"].shape == x.shape
    assert output["indices"].shape == (2, 2, 8, 8)
    loss = (output["phase"] - x).abs().mean() + output["loss"]
    loss.backward()
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_resunet_shape() -> None:
    model = ResUNetPhase(base_channels=8, dropout=0.0)
    output = model(torch.rand(2, 1, 64, 64))
    assert output["phase"].shape == (2, 1, 64, 64)
    assert output["log_scale"].shape == (2, 1, 64, 64)


def test_forward_twin_shape() -> None:
    model = EmpiricalForwardTwin(base_channels=8, dropout=0.0)
    output = model(torch.rand(2, 1, 64, 64))
    assert output["speckle"].shape == (2, 1, 64, 64)


def test_dual_domain_model_shape_and_targets() -> None:
    model = DualDomainRVQTwin(_prior(), base_channels=8, dropout=0.0)
    speckle = torch.rand(2, 1, 64, 64)
    phase = torch.rand(2, 1, 64, 64)
    output = model(speckle, target_phase=phase)
    assert output["phase"].shape == phase.shape
    assert output["target_indices"].shape == (2, 2, 8, 8)
    assert len(output["token_logits"]) == 2


def test_diffusion_training_and_sampling() -> None:
    denoiser = ConditionalDenoiser(base_channels=8, time_dim=32, dropout=0.0)
    diffusion = GaussianDiffusion(denoiser, timesteps=10, loss_type="mse")
    phase = torch.rand(2, 1, 32, 32)
    speckle = torch.rand(2, 1, 32, 32)
    loss, terms = diffusion.training_loss(phase, speckle)
    assert torch.isfinite(loss)
    assert "noise_mse" in terms
    loss.backward()
    wrapper = DiffusionPhaseReconstructor(diffusion, sample_steps=2, eta=0.0)
    output = wrapper(speckle)
    assert output["phase"].shape == phase.shape
    assert output["phase"].min() >= 0 and output["phase"].max() <= 1


def test_dual_domain_branch_and_quantization_ablations() -> None:
    speckle = torch.rand(2, 1, 64, 64)
    phase = torch.rand(2, 1, 64, 64)
    spatial_only = DualDomainRVQTwin(
        _prior(), base_channels=8, dropout=0.0, use_spatial=True, use_frequency=False
    )
    output_spatial = spatial_only(speckle, target_phase=phase)
    assert output_spatial["phase"].shape == phase.shape
    assert torch.count_nonzero(output_spatial["frequency_latent"]) == 0

    frequency_only = DualDomainRVQTwin(
        _prior(), base_channels=8, dropout=0.0, use_spatial=False, use_frequency=True
    )
    output_frequency = frequency_only(speckle, target_phase=phase)
    assert output_frequency["phase"].shape == phase.shape
    assert torch.count_nonzero(output_frequency["spatial_latent"]) == 0

    continuous = DualDomainRVQTwin(
        _prior(), base_channels=8, dropout=0.0, use_quantization=False
    )
    output_continuous = continuous(speckle, target_phase=phase)
    assert output_continuous["phase"].shape == phase.shape
    assert "token_logits" not in output_continuous
    assert "target_indices" not in output_continuous


def test_rvq_token_interventions_change_discrete_latent() -> None:
    model = DualDomainRVQTwin(_prior(), base_channels=8, dropout=0.0)
    speckle = torch.rand(2, 1, 64, 64)
    plain = model(speckle)
    shuffled = model(speckle, token_intervention="shuffle")
    replaced = model(speckle, token_intervention="mean")
    assert not torch.equal(plain["z_q"], shuffled["z_q"])
    assert replaced["z_q"].std(dim=(-2, -1)).max() == 0
