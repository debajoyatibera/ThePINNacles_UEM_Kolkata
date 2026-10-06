"""Tests for the simple CRP/IL-6 PINN architecture and physics residual."""

import numpy as np
import pytest
import torch

from src.biotwin.models.biological_ode import BiologicalParameters
from src.biotwin.models.pinn import CRPIL6PINN, physics_loss, physics_residual


@pytest.fixture
def params() -> BiologicalParameters:
    return BiologicalParameters(
        alpha_il6=1.8,
        k_il6=0.45,
        alpha_crp=1.2,
        k_crp=0.18,
    )


def test_pinn_can_be_instantiated() -> None:
    model = CRPIL6PINN(hidden_dim=16)
    assert isinstance(model, CRPIL6PINN)


def test_forward_pass_works() -> None:
    model = CRPIL6PINN(hidden_dim=8)
    inputs = torch.tensor([[0.0, 0.2], [1.0, 0.5]], dtype=torch.float32)
    outputs = model(inputs)

    assert outputs.shape == (2, 2)
    assert torch.isfinite(outputs).all()


def test_output_shape_is_correct_for_batch() -> None:
    model = CRPIL6PINN(hidden_dim=12)
    batch = torch.randn(7, 2, dtype=torch.float32)
    outputs = model(batch)

    assert outputs.shape == (7, 2)


def test_forward_output_is_finite() -> None:
    model = CRPIL6PINN(hidden_dim=10)
    batch = torch.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 0.4]], dtype=torch.float32)
    outputs = model(batch)

    assert torch.isfinite(outputs).all()


def test_physics_residual_function_works(params: BiologicalParameters) -> None:
    model = CRPIL6PINN(hidden_dim=16)
    time = torch.linspace(0.0, 1.0, 6, dtype=torch.float32)
    stimulus = torch.linspace(0.0, 2.0, 6, dtype=torch.float32)

    residual_il6, residual_crp = physics_residual(model, time, stimulus, params)

    assert residual_il6.shape == time.shape
    assert residual_crp.shape == time.shape
    assert torch.isfinite(residual_il6).all()
    assert torch.isfinite(residual_crp).all()


def test_physics_loss_returns_scalar() -> None:
    model = CRPIL6PINN(hidden_dim=8)
    time = torch.linspace(0.0, 2.0, 5, dtype=torch.float32)
    stimulus = torch.linspace(0.0, 0.8, 5, dtype=torch.float32)
    params = BiologicalParameters(1.5, 0.5, 1.0, 0.2)

    loss = physics_loss(model, time, stimulus, params)

    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_autograd_is_active_in_physics_residual() -> None:
    model = CRPIL6PINN(hidden_dim=8)
    time = torch.linspace(0.0, 2.0, 5, dtype=torch.float32, requires_grad=True)
    stimulus = torch.linspace(0.0, 1.0, 5, dtype=torch.float32)
    params = BiologicalParameters(1.5, 0.5, 1.0, 0.2)

    residual_il6, residual_crp = physics_residual(model, time, stimulus, params)

    assert residual_il6.requires_grad
    assert residual_crp.requires_grad


def test_deterministic_in_evaluation_mode() -> None:
    model = CRPIL6PINN(hidden_dim=12)
    model.eval()
    batch = torch.tensor([[0.0, 0.5], [2.0, 1.0], [5.0, 0.2]], dtype=torch.float32)

    first = model(batch)
    second = model(batch)

    assert torch.equal(first, second)


def test_cpu_execution_works() -> None:
    model = CRPIL6PINN(hidden_dim=12)
    model.to(torch.device("cpu"))
    batch = torch.randn(4, 2, dtype=torch.float32, device=torch.device("cpu"))
    outputs = model(batch)

    assert outputs.device.type == "cpu"


def test_biological_parameter_validation_is_respected() -> None:
    with pytest.raises(ValueError, match="alpha_il6"):
        BiologicalParameters(0.0, 0.5, 1.0, 0.2)
    with pytest.raises(ValueError, match="k_il6"):
        BiologicalParameters(1.0, 0.0, 1.0, 0.2)
    with pytest.raises(ValueError, match="alpha_crp"):
        BiologicalParameters(1.0, 0.5, 0.0, 0.2)
    with pytest.raises(ValueError, match="k_crp"):
        BiologicalParameters(1.0, 0.5, 1.0, 0.0)
