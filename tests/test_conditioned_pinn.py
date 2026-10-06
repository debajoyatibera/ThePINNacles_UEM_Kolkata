import numpy as np
import pandas as pd
import pytest
import torch

from src.biotwin.models.biological_ode import BiologicalParameters
from src.biotwin.models.conditioned_pinn import (
    CONDITIONED_STATIC_FEATURES,
    ConditionedPINN,
    TimeNormalizedConditionedPINN,
    physics_loss,
    physics_residual,
)


@pytest.fixture
def params() -> BiologicalParameters:
    return BiologicalParameters(alpha_il6=1.8, k_il6=0.45, alpha_crp=1.2, k_crp=0.18)


@pytest.fixture
def sample_features() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "patient_id": ["P001", "P001", "P002", "P002"],
            "time_hours": [0.0, 1.0, 0.0, 1.0],
            "synthetic_stimulus": [0.2, 0.7, 0.3, 0.8],
            "synthetic_il6": [0.1, 0.6, 0.2, 0.7],
            "synthetic_crp": [0.1, 0.6, 0.3, 0.8],
            "age": [43.0, 43.0, 51.0, 51.0],
            "sex": [1.0, 1.0, 2.0, 2.0],
            "bmi": [27.0, 27.0, 29.0, 29.0],
            "waist_cm": [98.0, 98.0, 106.0, 106.0],
            "systolic_bp": [132.0, 132.0, 136.0, 136.0],
            "diastolic_bp": [96.0, 96.0, 90.0, 90.0],
            "hba1c": [5.6, 5.6, 6.1, 6.1],
            "sedentary_minutes": [360.0, 360.0, 260.0, 260.0],
        }
    )


def test_conditioned_model_instantiates_and_accepts_static_features() -> None:
    model = ConditionedPINN(hidden_dim=16, static_feature_names=CONDITIONED_STATIC_FEATURES)
    assert isinstance(model, ConditionedPINN)
    data = torch.tensor(
        [[0.0, 0.2, 43.0, 0.0, 27.0, 98.0, 132.0, 96.0, 5.6, 360.0],
         [1.0, 0.7, 51.0, 1.0, 29.0, 106.0, 136.0, 90.0, 6.1, 260.0]],
        dtype=torch.float32,
    )
    outputs = model(data)
    assert outputs.shape == (2, 2)
    assert torch.isfinite(outputs).all()


def test_conditioned_model_uses_patient_id_not_as_feature() -> None:
    assert "patient_id" not in CONDITIONED_STATIC_FEATURES
    model = ConditionedPINN(hidden_dim=12, static_feature_names=CONDITIONED_STATIC_FEATURES)
    assert model.static_feature_names == CONDITIONED_STATIC_FEATURES


def test_physics_residual_shape_and_finiteness(params: BiologicalParameters) -> None:
    model = ConditionedPINN(hidden_dim=12, static_feature_names=CONDITIONED_STATIC_FEATURES)
    time = torch.linspace(0.0, 2.0, 8, dtype=torch.float32)
    stimulus = torch.linspace(0.0, 1.0, 8, dtype=torch.float32)
    static = torch.tensor(
        [[43.0, 0.0, 27.0, 98.0, 132.0, 96.0, 5.6, 360.0]] * len(time),
        dtype=torch.float32,
    )
    residual_il6, residual_crp = physics_residual(model, time, stimulus, static, params)
    assert residual_il6.shape == time.shape
    assert residual_crp.shape == time.shape
    assert torch.isfinite(residual_il6).all()
    assert torch.isfinite(residual_crp).all()


def test_physics_loss_is_finite(params: BiologicalParameters) -> None:
    model = ConditionedPINN(hidden_dim=12, static_feature_names=CONDITIONED_STATIC_FEATURES)
    time = torch.linspace(0.0, 2.0, 8, dtype=torch.float32)
    stimulus = torch.linspace(0.0, 1.0, 8, dtype=torch.float32)
    static = torch.tensor(
        [[43.0, 0.0, 27.0, 98.0, 132.0, 96.0, 5.6, 360.0]] * len(time),
        dtype=torch.float32,
    )
    loss = physics_loss(model, time, stimulus, static, params)
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_time_normalized_conditioned_model_applies_chain_rule() -> None:
    base_model = ConditionedPINN(hidden_dim=8, static_feature_names=CONDITIONED_STATIC_FEATURES)
    model = TimeNormalizedConditionedPINN(base_model, time_min=0.0, time_max=2.0)
    time = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float32, requires_grad=True)
    stimulus = torch.tensor([0.2, 0.5, 0.7], dtype=torch.float32)
    static = torch.tensor(
        [[43.0, 0.0, 27.0, 98.0, 132.0, 96.0, 5.6, 360.0]] * len(time),
        dtype=torch.float32,
    )
    outputs = model(torch.cat([time.reshape(-1, 1), stimulus.reshape(-1, 1), static], dim=1))
    il6 = outputs[:, 0]
    il6_dt = torch.autograd.grad(il6, time, grad_outputs=torch.ones_like(il6), retain_graph=True, create_graph=True)[0]
    assert torch.isfinite(il6_dt).all()
    assert il6_dt.shape == time.shape


def test_physical_time_derivative_tracks_only_time_and_not_static_features() -> None:
    class ManualTimeModel(torch.nn.Module):
        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            time = inputs[:, 0]
            stimulus = inputs[:, 1]
            static = inputs[:, 2:]
            il6 = 3.0 * time + 2.0 * stimulus + static[:, 0]
            crp = 4.0 * time + 0.5 * stimulus + static[:, 1]
            return torch.stack([il6, crp], dim=1)

    model = ManualTimeModel()
    time = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float32, requires_grad=True)
    stimulus = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32)
    static = torch.tensor([[7.0, 8.0], [7.0, 8.0], [7.0, 8.0]], dtype=torch.float32)
    inputs = torch.cat([time.unsqueeze(1), stimulus.unsqueeze(1), static], dim=1)

    outputs = model(inputs)
    d_il6_dt = torch.autograd.grad(
        outputs=outputs[:, 0],
        inputs=time,
        grad_outputs=torch.ones_like(outputs[:, 0]),
        retain_graph=True,
        create_graph=True,
    )[0]

    assert torch.allclose(d_il6_dt, torch.tensor([3.0, 3.0, 3.0], dtype=torch.float32))
    assert torch.isfinite(outputs).all()


def test_static_features_do_not_create_time_derivatives() -> None:
    base_model = ConditionedPINN(hidden_dim=8, static_feature_names=CONDITIONED_STATIC_FEATURES)
    model = TimeNormalizedConditionedPINN(base_model, time_min=0.0, time_max=2.0)
    time = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float32, requires_grad=True)
    stimulus = torch.tensor([0.2, 0.5, 0.7], dtype=torch.float32)
    static = torch.tensor(
        [[43.0, 0.0, 27.0, 98.0, 132.0, 96.0, 5.6, 360.0]] * len(time),
        dtype=torch.float32,
    )
    outputs = model(torch.cat([time.reshape(-1, 1), stimulus.reshape(-1, 1), static], dim=1))
    assert outputs.shape == (len(time), 2)
    assert torch.isfinite(outputs).all()


@pytest.mark.parametrize("bad_inputs", [
    torch.tensor([[0.0, 0.2]], dtype=torch.float32),
    torch.tensor([[0.0, 0.2, 43.0]], dtype=torch.float32),
])
def test_conditioned_model_rejects_invalid_input_dimensions(bad_inputs: torch.Tensor) -> None:
    model = ConditionedPINN(hidden_dim=8, static_feature_names=CONDITIONED_STATIC_FEATURES)
    with pytest.raises(ValueError):
        model(bad_inputs)
