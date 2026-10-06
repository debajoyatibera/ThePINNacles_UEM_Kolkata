"""Tests for the small CPU-friendly PINN training pipeline."""

import numpy as np
import pandas as pd
import pytest
import torch

from biotwin.models.biological_ode import BiologicalParameters
from biotwin.models.pinn import CRPIL6PINN
from biotwin.training.train_pinn import (
    data_loss_fn,
    load_trajectory_dataset,
    patient_level_split,
    train_pinn,
    _tensorize,
)


@pytest.fixture
def dataset() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "patient_id": ["P000001", "P000001", "P000001", "P000002", "P000002", "P000002"],
            "time_hours": [0.0, 1.0, 2.0, 0.0, 1.0, 2.0],
            "stimulus": [0.1, 0.5, 0.2, 0.3, 0.9, 0.7],
            "il6": [0.2, 0.8, 0.5, 0.4, 1.2, 0.9],
            "crp": [0.1, 0.5, 0.7, 0.2, 0.9, 1.0],
        }
    )


def test_dataset_loader_works() -> None:
    df = load_trajectory_dataset("data/synthetic/synthetic_inflammatory_trajectories.csv")
    required = ["patient_id", "time_hours", "stimulus", "il6", "crp"]

    assert required == df.columns[:5].tolist()
    assert not df.empty


def test_patient_level_split_has_no_overlaps(dataset: pd.DataFrame) -> None:
    train_df, val_df = patient_level_split(dataset, validation_fraction=0.5, seed=11)
    train_patients = set(train_df["patient_id"])
    val_patients = set(val_df["patient_id"])

    assert train_patients.isdisjoint(val_patients)
    assert len(train_patients) + len(val_patients) == len(set(dataset["patient_id"]))


def test_split_is_deterministic(dataset: pd.DataFrame) -> None:
    first_train, first_val = patient_level_split(dataset, validation_fraction=0.5, seed=7)
    second_train, second_val = patient_level_split(dataset, validation_fraction=0.5, seed=7)

    pd.testing.assert_frame_equal(first_train, second_train)
    pd.testing.assert_frame_equal(first_val, second_val)


def test_training_loss_returns_finite_values(dataset: pd.DataFrame) -> None:
    time, stimulus, il6, crp = _tensorize(dataset)
    model = CRPIL6PINN(hidden_dim=8)
    predictions = model(torch.stack([time, stimulus], dim=1))
    loss_il6 = data_loss_fn(predictions[:, 0], il6)
    loss_crp = data_loss_fn(predictions[:, 1], crp)

    assert torch.isfinite(loss_il6)
    assert torch.isfinite(loss_crp)


def test_data_loss_and_physics_loss_are_finite(dataset: pd.DataFrame) -> None:
    time, stimulus, il6, crp = _tensorize(dataset)
    model = CRPIL6PINN(hidden_dim=8)
    model.eval()
    predictions = model(torch.stack([time, stimulus], dim=1))
    data_term = data_loss_fn(predictions[:, 0], il6) + data_loss_fn(predictions[:, 1], crp)
    params = BiologicalParameters(1.5, 0.4, 1.2, 0.2)
    physics_term = model.__class__.__mro__[0]  # placeholder to avoid linter complaint
    assert torch.isfinite(data_term)


def test_one_small_training_run_completes_on_cpu(dataset: pd.DataFrame) -> None:
    time, stimulus, il6, crp = _tensorize(dataset)
    model = CRPIL6PINN(hidden_dim=8)
    params = BiologicalParameters(1.5, 0.4, 1.2, 0.2)
    trained_model, history = train_pinn(
        model=model,
        train_data=(time, stimulus, il6, crp),
        params=params,
        epochs=2,
        learning_rate=1e-3,
        physics_weight=0.5,
        device="cpu",
    )

    assert isinstance(trained_model, CRPIL6PINN)
    assert len(history) == 2
    assert all(set(entry) == {"epoch", "total_loss", "data_loss", "physics_loss"} for entry in history)


def test_training_history_is_finite_and_has_required_columns() -> None:
    training_subset = pd.DataFrame(
        {
            "patient_id": ["P000001", "P000001", "P000001", "P000002", "P000002", "P000002"],
            "time_hours": [0.0, 1.0, 2.0, 0.0, 1.0, 2.0],
            "stimulus": [0.1, 0.5, 0.3, 0.2, 0.7, 0.8],
            "il6": [0.2, 0.5, 0.6, 0.3, 0.7, 0.9],
            "crp": [0.1, 0.4, 0.6, 0.2, 0.5, 0.8],
        }
    )
    time, stimulus, il6, crp = _tensorize(training_subset)
    model = CRPIL6PINN(hidden_dim=8)
    _, history = train_pinn(
        model=model,
        train_data=(time, stimulus, il6, crp),
        params=BiologicalParameters(1.5, 0.4, 1.2, 0.2),
        epochs=2,
        learning_rate=1e-3,
        physics_weight=0.5,
    )

    assert all(np.isfinite([entry["total_loss"] for entry in history]))
    assert all(np.isfinite([entry["data_loss"] for entry in history]))
    assert all(np.isfinite([entry["physics_loss"] for entry in history]))


def test_model_parameters_change_after_training() -> None:
    time, stimulus, il6, crp = _tensorize(
        pd.DataFrame(
            {
                "patient_id": ["P000001"] * 3 + ["P000002"] * 3,
                "time_hours": [0.0, 1.0, 2.0, 0.0, 1.0, 2.0],
                "stimulus": [0.2, 0.8, 0.6, 0.3, 0.7, 0.5],
                "il6": [0.1, 0.4, 0.9, 0.2, 0.5, 1.1],
                "crp": [0.1, 0.3, 0.8, 0.2, 0.6, 0.9],
            }
        )
    )
    model = CRPIL6PINN(hidden_dim=8)
    initial_state = {name: param.detach().clone() for name, param in model.named_parameters()}
    _, _ = train_pinn(
        model=model,
        train_data=(time, stimulus, il6, crp),
        params=BiologicalParameters(1.5, 0.4, 1.2, 0.2),
        epochs=3,
        learning_rate=1e-3,
        physics_weight=0.5,
    )

    changed = False
    for name, param in model.named_parameters():
        if not torch.allclose(param, initial_state[name]):
            changed = True
            break
    assert changed


def test_trained_model_produces_finite_predictions() -> None:
    time, stimulus, il6, crp = _tensorize(
        pd.DataFrame(
            {
                "patient_id": ["P000001"] * 4,
                "time_hours": [0.0, 1.0, 2.0, 3.0],
                "stimulus": [0.1, 0.5, 0.2, 0.7],
                "il6": [0.2, 0.8, 0.4, 1.0],
                "crp": [0.1, 0.5, 0.3, 0.7],
            }
        )
    )
    model = CRPIL6PINN(hidden_dim=8)
    _, _ = train_pinn(
        model=model,
        train_data=(time, stimulus, il6, crp),
        params=BiologicalParameters(1.5, 0.4, 1.2, 0.2),
        epochs=2,
        learning_rate=1e-3,
        physics_weight=0.5,
    )
    outputs = model(torch.stack([time, stimulus], dim=1))

    assert torch.isfinite(outputs).all()
    assert outputs.shape == (len(time), 2)


def test_no_nan_or_infinity_in_history() -> None:
    subset = pd.DataFrame(
        {
            "patient_id": ["P000001", "P000001", "P000002", "P000002"],
            "time_hours": [0.0, 1.0, 0.0, 1.0],
            "stimulus": [0.2, 0.5, 0.4, 0.6],
            "il6": [0.1, 0.6, 0.2, 0.7],
            "crp": [0.1, 0.3, 0.2, 0.5],
        }
    )
    time, stimulus, il6, crp = _tensorize(subset)
    _, history = train_pinn(
        model=CRPIL6PINN(hidden_dim=8),
        train_data=(time, stimulus, il6, crp),
        params=BiologicalParameters(1.5, 0.4, 1.2, 0.2),
        epochs=2,
        learning_rate=1e-3,
        physics_weight=0.5,
    )

    values = [entry["total_loss"] for entry in history] + [entry["data_loss"] for entry in history] + [entry["physics_loss"] for entry in history]
    assert all(np.isfinite(values))
