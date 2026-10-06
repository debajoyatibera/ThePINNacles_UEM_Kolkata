"""Focused tests for PINN evaluation and artifact contracts."""

import json

import numpy as np
import pandas as pd
import torch

from src.biotwin.training import run_pinn_experiment as experiment


def _small_synthetic_dataset() -> pd.DataFrame:
    rows = []
    for patient_index in range(5):
        for time_hours in range(6):
            stimulus = 0.2 + 0.05 * patient_index + 0.1 * (time_hours % 3)
            rows.append(
                {
                    "patient_id": f"P{patient_index:06d}",
                    "time_hours": float(time_hours),
                    "stimulus": stimulus,
                    "il6": 0.3 + 0.2 * patient_index + 0.1 * time_hours,
                    "crp": 0.5 + 0.4 * patient_index + 0.15 * time_hours,
                }
            )
    return pd.DataFrame(rows)


def test_calculate_metrics_matches_expected_values() -> None:
    metrics = experiment.calculate_metrics(np.array([1.0, 2.0]), np.array([1.0, 3.0]))

    assert metrics["mae"] == 0.5
    assert metrics["rmse"] == 2**-0.5
    assert metrics["r2"] == -1.0


def test_calculate_metrics_rejects_invalid_values() -> None:
    try:
        experiment.calculate_metrics(np.array([1.0]), np.array([np.nan]))
    except ValueError as error:
        assert "finite" in str(error)
    else:
        raise AssertionError("non-finite predictions must be rejected")


def test_experiment_artifacts_use_held_out_patients_and_loadable_model(
    tmp_path, monkeypatch
) -> None:
    dataset_path = tmp_path / "synthetic.csv"
    _small_synthetic_dataset().to_csv(dataset_path, index=False)
    output_dir = tmp_path / "outputs"
    monkeypatch.setattr(experiment, "PREDICTIONS_DIR", output_dir / "predictions")
    monkeypatch.setattr(experiment, "MODEL_PATH", output_dir / "models" / "crp_il6_pinn.pt")
    monkeypatch.setattr(
        experiment, "HISTORY_PATH", output_dir / "predictions" / "pinn_training_history.csv"
    )
    monkeypatch.setattr(
        experiment,
        "PREDICTIONS_PATH",
        output_dir / "predictions" / "pinn_validation_predictions.csv",
    )
    monkeypatch.setattr(
        experiment, "EVALUATION_PATH", output_dir / "predictions" / "pinn_evaluation.json"
    )
    monkeypatch.setattr(
        experiment, "LOSS_FIGURE_PATH", output_dir / "figures" / "pinn_training_loss.png"
    )
    monkeypatch.setattr(
        experiment,
        "VALIDATION_FIGURE_PATH",
        output_dir / "figures" / "pinn_validation_example.png",
    )

    report = experiment.run_experiment(
        epochs=1,
        learning_rate=1e-3,
        physics_weight=1.0,
        seed=17,
        validation_fraction=0.2,
        dataset_path=dataset_path,
    )

    validation = pd.read_csv(experiment.PREDICTIONS_PATH)
    source = pd.read_csv(dataset_path)
    train, expected_validation = experiment.patient_level_split(source, seed=17)
    history = pd.read_csv(experiment.HISTORY_PATH)
    saved_report = json.loads(experiment.EVALUATION_PATH.read_text(encoding="utf-8"))
    loaded_model = experiment.load_checkpoint(experiment.MODEL_PATH)
    inputs = torch.tensor(
        validation[["time_hours", "stimulus"]].to_numpy(dtype=np.float32),
        dtype=torch.float32,
    )
    with torch.no_grad():
        outputs = loaded_model(inputs)

    assert validation.columns.tolist() == experiment.PREDICTION_COLUMNS
    assert set(validation.patient_id) == set(expected_validation.patient_id)
    assert set(validation.patient_id).isdisjoint(set(train.patient_id))
    assert len(validation) == len(expected_validation)
    assert history.columns.tolist() == ["epoch", "total_loss", "data_loss", "physics_loss"]
    assert set(
        [
            "training_configuration",
            "train_patient_count",
            "validation_patient_count",
            "epoch_count",
            "seed",
            "il6_mae",
            "il6_rmse",
            "il6_r2",
            "crp_mae",
            "crp_rmse",
            "crp_r2",
            "validation_physics_loss",
            "validation_physics_residual_metric",
            "scientific_limitation",
        ]
    ).issubset(saved_report)
    assert saved_report == report
    assert experiment.SCIENTIFIC_LIMITATION == saved_report["scientific_limitation"]
    assert torch.isfinite(outputs).all()
    assert outputs.shape == (len(validation), 2)