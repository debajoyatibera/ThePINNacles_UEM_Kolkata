"""Focused tests for time normalization and the controlled experiment path."""

import json

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from src.biotwin.models.biological_ode import BiologicalParameters
from src.biotwin.models.pinn import CRPIL6PINN, TimeNormalizedPINN, physics_residual
from src.biotwin.training import diagnose_pinn as diagnostic
from src.biotwin.training.run_pinn_experiment import PREDICTION_COLUMNS
from src.biotwin.training.train_pinn import patient_level_split


class EchoInputs(nn.Module):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs


def test_time_wrapper_scales_only_time_and_preserves_stimulus() -> None:
    model = TimeNormalizedPINN(EchoInputs(), time_min=10.0, time_max=20.0)
    inputs = torch.tensor([[10.0, 0.2], [15.0, 0.7], [20.0, 1.4]])

    outputs = model(inputs)

    assert torch.allclose(outputs[:, 0], torch.tensor([0.0, 0.5, 1.0]))
    assert torch.equal(outputs[:, 1], inputs[:, 1])


def test_physics_residual_derivative_is_per_physical_hour() -> None:
    model = TimeNormalizedPINN(EchoInputs(), time_min=0.0, time_max=10.0)
    time = torch.tensor([0.0, 5.0, 10.0])
    stimulus = torch.tensor([0.2, 0.3, 0.4])
    params = BiologicalParameters(1.0, 0.1, 0.2, 0.05)

    residual_il6, _ = physics_residual(model, time, stimulus, params)
    expected = 0.1 - (stimulus - 0.1 * time / 10.0)

    assert torch.allclose(residual_il6, expected)


def test_time_wrapper_rejects_invalid_bounds() -> None:
    with pytest.raises(ValueError, match="time_max"):
        TimeNormalizedPINN(EchoInputs(), time_min=4.0, time_max=4.0)


def test_normalized_experiment_keeps_patient_holdout_and_serializes(tmp_path, monkeypatch) -> None:
    rows = []
    for patient_index in range(5):
        for hour in range(5):
            rows.append(
                {
                    "patient_id": f"P{patient_index:06d}",
                    "time_hours": float(hour),
                    "stimulus": 0.2 + 0.1 * (hour % 2),
                    "il6": 0.3 + patient_index * 0.1 + hour * 0.05,
                    "crp": 0.5 + patient_index * 0.2 + hour * 0.1,
                }
            )
    dataset_path = tmp_path / "synthetic.csv"
    pd.DataFrame(rows).to_csv(dataset_path, index=False)
    monkeypatch.setattr(diagnostic, "NORMALIZED_REPORT_PATH", tmp_path / "evaluation.json")
    monkeypatch.setattr(diagnostic, "NORMALIZED_HISTORY_PATH", tmp_path / "history.csv")
    monkeypatch.setattr(diagnostic, "NORMALIZED_PREDICTIONS_PATH", tmp_path / "predictions.csv")
    monkeypatch.setattr(diagnostic, "NORMALIZED_MODEL_PATH", tmp_path / "model.pt")

    report = diagnostic.run_time_normalized_experiment(
        dataset_path=dataset_path,
        epochs=2,
        seed=42,
    )

    predictions = pd.read_csv(diagnostic.NORMALIZED_PREDICTIONS_PATH)
    history = pd.read_csv(diagnostic.NORMALIZED_HISTORY_PATH)
    _, expected_validation = patient_level_split(pd.read_csv(dataset_path), seed=42)
    stored_report = json.loads(diagnostic.NORMALIZED_REPORT_PATH.read_text(encoding="utf-8"))
    checkpoint = torch.load(diagnostic.NORMALIZED_MODEL_PATH, map_location="cpu", weights_only=True)
    loaded_model = TimeNormalizedPINN(
        CRPIL6PINN(hidden_dim=checkpoint["hidden_dim"]),
        time_min=checkpoint["time_min"],
        time_max=checkpoint["time_max"],
    )
    loaded_model.load_state_dict(checkpoint["model_state_dict"])
    inputs = torch.tensor(predictions[["time_hours", "stimulus"]].to_numpy(dtype=np.float32))
    with torch.no_grad():
        outputs = loaded_model(inputs)

    assert predictions.columns.tolist() == PREDICTION_COLUMNS
    assert set(predictions.patient_id) == set(expected_validation.patient_id)
    assert history.columns.tolist() == ["epoch", "total_loss", "data_loss", "physics_loss"]
    assert len(history) == 2
    assert np.isfinite(history.select_dtypes("number").to_numpy()).all()
    assert report == stored_report
    assert report["training_configuration"]["stimulus_transform"] == "none; physical stimulus scale retained"
    assert torch.isfinite(outputs).all()