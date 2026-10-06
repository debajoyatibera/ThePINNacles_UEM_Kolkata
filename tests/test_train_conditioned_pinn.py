import numpy as np
import pandas as pd
import pytest
import torch

from biotwin.models.biological_ode import BiologicalParameters
from biotwin.models.conditioned_pinn import CONDITIONED_STATIC_FEATURES
from biotwin.training.train_conditioned_pinn import (
    load_conditioned_dataset,
    patient_level_split,
    train_conditioned_pinn,
    tensorize_conditioned_dataset,
)


@pytest.fixture
def conditioned_dataset() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "patient_id": ["P001", "P001", "P002", "P002", "P003", "P003"],
            "time_hours": [0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
            "synthetic_stimulus": [0.2, 0.7, 0.3, 0.8, 0.4, 0.9],
            "synthetic_il6": [0.1, 0.6, 0.2, 0.7, 0.3, 0.8],
            "synthetic_crp": [0.2, 0.7, 0.3, 0.8, 0.4, 0.9],
            "age": [43.0, 43.0, 51.0, 51.0, 58.0, 58.0],
            "sex": [1.0, 1.0, 2.0, 2.0, 1.0, 1.0],
            "bmi": [27.0, 27.0, 29.0, 29.0, 31.0, 31.0],
            "waist_cm": [98.0, 98.0, 106.0, 106.0, 110.0, 110.0],
            "systolic_bp": [132.0, 132.0, 136.0, 136.0, 140.0, 140.0],
            "diastolic_bp": [96.0, 96.0, 90.0, 90.0, 88.0, 88.0],
            "hba1c": [5.6, 5.6, 6.1, 6.1, 5.9, 5.9],
            "sedentary_minutes": [360.0, 360.0, 260.0, 260.0, 300.0, 300.0],
        }
    )


def test_dataset_loader_uses_patient_conditioned_columns() -> None:
    dataset = load_conditioned_dataset("data/processed/patient_conditioned_digital_twin.csv")
    required = {"patient_id", "time_hours", "synthetic_stimulus", "synthetic_il6", "synthetic_crp"}
    assert required.issubset(dataset.columns)
    assert "patient_id" in dataset.columns
    assert dataset.dropna(subset=["time_hours"]).empty is False


def test_patient_level_split_has_zero_overlap(conditioned_dataset: pd.DataFrame) -> None:
    train_df, val_df = patient_level_split(conditioned_dataset, validation_fraction=0.5, seed=7)
    train_patients = set(train_df["patient_id"])
    val_patients = set(val_df["patient_id"])
    assert train_patients.isdisjoint(val_patients)
    assert len(train_patients) + len(val_patients) == len(set(conditioned_dataset["patient_id"]))


def test_tensorization_has_no_nan_or_inf(conditioned_dataset: pd.DataFrame) -> None:
    tensors = tensorize_conditioned_dataset(conditioned_dataset)
    for tensor in tensors:
        assert torch.isfinite(tensor).all()
        assert not torch.isnan(tensor).any()


def test_training_run_completes_and_losses_are_finite(conditioned_dataset: pd.DataFrame) -> None:
    train_df, _ = patient_level_split(conditioned_dataset, validation_fraction=0.5, seed=7)
    train_tensors = tensorize_conditioned_dataset(train_df)
    model, history = train_conditioned_pinn(
        model=None,
        train_data=train_tensors,
        params=BiologicalParameters(alpha_il6=1.8, k_il6=0.45, alpha_crp=1.2, k_crp=0.18),
        epochs=2,
        learning_rate=1e-3,
        physics_weight=0.5,
        device="cpu",
    )
    assert model is not None
    assert len(history) == 2
    assert all(np.isfinite([entry["total_loss"] for entry in history]))
    assert all(np.isfinite([entry["data_loss"] for entry in history]))
    assert all(np.isfinite([entry["physics_loss"] for entry in history]))
    assert all(param.isfinite().all() for param in model.parameters())


def test_static_feature_names_are_expected() -> None:
    assert CONDITIONED_STATIC_FEATURES == [
        "age",
        "sex",
        "bmi",
        "waist_cm",
        "systolic_bp",
        "diastolic_bp",
        "hba1c",
        "sedentary_minutes",
    ]
