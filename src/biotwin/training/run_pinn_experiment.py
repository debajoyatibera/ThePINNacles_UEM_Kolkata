"""Train and evaluate the synthetic CRP/IL-6 PINN experiment.

All reported metrics measure agreement with synthetic ODE-generated data and
do not establish clinical validity.
"""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn

from src.biotwin.models.biological_ode import BiologicalParameters
from src.biotwin.models.pinn import CRPIL6PINN, physics_loss, physics_residual
from src.biotwin.training.train_pinn import (
    TRAINING_DATA_PATH,
    _tensorize,
    data_loss_fn,
    load_trajectory_dataset,
    patient_level_split,
    train_pinn,
)


PREDICTIONS_DIR = Path("outputs/predictions")
MODEL_PATH = Path("outputs/models/crp_il6_pinn.pt")
HISTORY_PATH = PREDICTIONS_DIR / "pinn_training_history.csv"
PREDICTIONS_PATH = PREDICTIONS_DIR / "pinn_validation_predictions.csv"
EVALUATION_PATH = PREDICTIONS_DIR / "pinn_evaluation.json"
LOSS_FIGURE_PATH = Path("outputs/figures/pinn_training_loss.png")
VALIDATION_FIGURE_PATH = Path("outputs/figures/pinn_validation_example.png")
PREDICTION_COLUMNS = [
    "patient_id",
    "time_hours",
    "stimulus",
    "il6_actual",
    "il6_predicted",
    "crp_actual",
    "crp_predicted",
]
SCIENTIFIC_LIMITATION = (
    "Performance metrics at this stage measure agreement with the synthetic "
    "biological simulator and do not establish clinical validity."
)


def calculate_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    """Calculate MAE, RMSE, and R2 for one target series."""
    actual = np.asarray(actual, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    if actual.shape != predicted.shape or actual.size == 0:
        raise ValueError("actual and predicted must have the same non-empty shape")
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("actual and predicted values must be finite")

    errors = predicted - actual
    squared_total = float(np.sum((actual - np.mean(actual)) ** 2))
    r2 = 1.0 - float(np.sum(errors**2)) / squared_total if squared_total > 0 else 0.0
    return {
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "r2": r2,
    }


def predict_frame(model: nn.Module, validation_data: pd.DataFrame) -> pd.DataFrame:
    """Build the compact output frame from only the supplied validation rows."""
    time, stimulus, _, _ = _tensorize(validation_data)
    model.eval()
    with torch.no_grad():
        outputs = model(torch.stack([time, stimulus], dim=1)).cpu().numpy()
    if not np.isfinite(outputs).all():
        raise FloatingPointError("model produced non-finite validation predictions")
    predictions = validation_data[["patient_id", "time_hours", "stimulus"]].copy()
    predictions["il6_actual"] = validation_data["il6"].to_numpy()
    predictions["il6_predicted"] = outputs[:, 0]
    predictions["crp_actual"] = validation_data["crp"].to_numpy()
    predictions["crp_predicted"] = outputs[:, 1]
    return predictions[PREDICTION_COLUMNS]


def evaluate_model(
    model: nn.Module,
    validation_data: pd.DataFrame,
    params: BiologicalParameters,
) -> dict[str, Any]:
    """Evaluate target metrics and ODE residuals on held-out patients."""
    predictions = predict_frame(model, validation_data)
    time, stimulus, _, _ = _tensorize(validation_data)
    model.eval()
    validation_physics_loss = float(physics_loss(model, time, stimulus, params).detach().cpu())
    residual_il6, residual_crp = physics_residual(model, time, stimulus, params)
    residual_magnitude = float(
        torch.cat([residual_il6.abs(), residual_crp.abs()]).mean().detach().cpu()
    )
    if not np.isfinite([validation_physics_loss, residual_magnitude]).all():
        raise FloatingPointError("model produced non-finite validation physics metrics")
    return {
        "il6": calculate_metrics(predictions["il6_actual"], predictions["il6_predicted"]),
        "crp": calculate_metrics(predictions["crp_actual"], predictions["crp_predicted"]),
        "validation_physics_loss": validation_physics_loss,
        "validation_physics_residual_metric": residual_magnitude,
    }


def _loss_components(
    model: nn.Module,
    data: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    params: BiologicalParameters,
) -> dict[str, float]:
    time, stimulus, target_il6, target_crp = data
    model.eval()
    with torch.no_grad():
        outputs = model(torch.stack([time, stimulus], dim=1))
        data_term = data_loss_fn(outputs[:, 0], target_il6) + data_loss_fn(outputs[:, 1], target_crp)
    physics_term = physics_loss(model, time, stimulus, params)
    data_value = float(data_term.detach().cpu())
    physics_value = float(physics_term.detach().cpu())
    return {
        "total_loss": data_value + physics_value,
        "data_loss": data_value,
        "physics_loss": physics_value,
    }


def save_checkpoint(model: CRPIL6PINN, path: str | Path, seed: int) -> None:
    """Save a CPU-loadable state dict with the model construction metadata."""
    checkpoint = {
        "model_state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "hidden_dim": model.network[0].out_features,
        "seed": seed,
    }
    torch.save(checkpoint, path)


def load_checkpoint(path: str | Path) -> CRPIL6PINN:
    """Recreate a PINN from a saved experiment checkpoint."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = CRPIL6PINN(hidden_dim=int(checkpoint["hidden_dim"]))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def _write_figures(history: list[dict[str, float]], predictions: pd.DataFrame) -> None:
    LOSS_FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    epochs = [entry["epoch"] for entry in history]
    fig, axis = plt.subplots(figsize=(8, 5))
    for column, label in (
        ("total_loss", "Total loss"),
        ("data_loss", "Data loss"),
        ("physics_loss", "Physics loss"),
    ):
        axis.plot(epochs, [entry[column] for entry in history], label=label)
    axis.set(xlabel="Epoch", ylabel="Loss", title="PINN training losses")
    axis.set_yscale("log")
    axis.legend()
    fig.tight_layout()
    fig.savefig(LOSS_FIGURE_PATH, dpi=140)
    plt.close(fig)

    example_patient = predictions["patient_id"].iloc[0]
    example = predictions[predictions["patient_id"] == example_patient]
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    for axis, target, actual_label, predicted_label in (
        (axes[0], "il6", "il6_actual", "il6_predicted"),
        (axes[1], "crp", "crp_actual", "crp_predicted"),
    ):
        axis.plot(example["time_hours"], example[actual_label], label="Actual")
        axis.plot(example["time_hours"], example[predicted_label], label="Predicted")
        axis.set_ylabel(target.upper())
        axis.legend()
    axes[-1].set_xlabel("Time (hours)")
    fig.suptitle(f"Held-out patient {example_patient}: actual vs predicted")
    fig.tight_layout()
    fig.savefig(VALIDATION_FIGURE_PATH, dpi=140)
    plt.close(fig)


def run_experiment(
    *,
    epochs: int = 100,
    learning_rate: float = 1e-3,
    physics_weight: float = 1.0,
    seed: int = 42,
    validation_fraction: float = 0.2,
    dataset_path: str | Path = TRAINING_DATA_PATH,
) -> dict[str, Any]:
    """Train, evaluate on held-out patients, and save all experiment artifacts."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(2)
    dataset = load_trajectory_dataset(dataset_path)
    train_df, validation_df = patient_level_split(dataset, validation_fraction, seed)
    if set(train_df["patient_id"]).intersection(validation_df["patient_id"]):
        raise RuntimeError("patient-level split leaked IDs between training and validation")

    params = BiologicalParameters(1.8, 0.45, 1.2, 0.18)
    train_tensors = _tensorize(train_df)
    model = CRPIL6PINN(hidden_dim=32)
    initial_model = deepcopy(model)
    initial_train_loss = _loss_components(model, train_tensors, params)
    initial_evaluation = evaluate_model(initial_model, validation_df, params)

    trained_model, history = train_pinn(
        model=model,
        train_data=train_tensors,
        params=params,
        epochs=epochs,
        learning_rate=learning_rate,
        physics_weight=physics_weight,
        device="cpu",
    )
    final_train_loss = _loss_components(trained_model, train_tensors, params)
    trained_evaluation = evaluate_model(trained_model, validation_df, params)
    prediction_frame = predict_frame(trained_model, validation_df)

    numeric_values = [
        *initial_train_loss.values(),
        *final_train_loss.values(),
        *trained_evaluation["il6"].values(),
        *trained_evaluation["crp"].values(),
        trained_evaluation["validation_physics_loss"],
        trained_evaluation["validation_physics_residual_metric"],
    ]
    if not np.isfinite(numeric_values).all():
        raise FloatingPointError("experiment metrics contain non-finite values; artifacts were not saved")

    report: dict[str, Any] = {
        "training_configuration": {
            "optimizer": "Adam",
            "device": "cpu",
            "learning_rate": learning_rate,
            "physics_weight": physics_weight,
            "hidden_dim": 32,
            "validation_fraction": validation_fraction,
            "ode_parameters": {
                "alpha_il6": params.alpha_il6,
                "k_il6": params.k_il6,
                "alpha_crp": params.alpha_crp,
                "k_crp": params.k_crp,
            },
        },
        "seed": seed,
        "train_patient_count": int(train_df["patient_id"].nunique()),
        "validation_patient_count": int(validation_df["patient_id"].nunique()),
        "epoch_count": epochs,
        "initial_train_loss": initial_train_loss,
        "final_train_loss": final_train_loss,
        "initial_validation": initial_evaluation,
        "trained_validation": trained_evaluation,
        "il6_mae": trained_evaluation["il6"]["mae"],
        "il6_rmse": trained_evaluation["il6"]["rmse"],
        "il6_r2": trained_evaluation["il6"]["r2"],
        "crp_mae": trained_evaluation["crp"]["mae"],
        "crp_rmse": trained_evaluation["crp"]["rmse"],
        "crp_r2": trained_evaluation["crp"]["r2"],
        "validation_physics_loss": trained_evaluation["validation_physics_loss"],
        "validation_physics_residual_metric": trained_evaluation["validation_physics_residual_metric"],
        "scientific_limitation": SCIENTIFIC_LIMITATION,
        "data_source": "synthetic ODE-generated trajectories only",
    }

    for output_path in (PREDICTIONS_DIR, MODEL_PATH.parent, LOSS_FIGURE_PATH.parent):
        output_path.mkdir(parents=True, exist_ok=True)
    prediction_frame.to_csv(PREDICTIONS_PATH, index=False)
    pd.DataFrame(history, columns=["epoch", "total_loss", "data_loss", "physics_loss"]).to_csv(
        HISTORY_PATH, index=False
    )
    EVALUATION_PATH.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    save_checkpoint(trained_model, MODEL_PATH, seed)
    _write_figures(history, prediction_frame)
    return report


def main() -> None:
    report = run_experiment()
    print(json.dumps(report, indent=2))
    print(f"Checkpoint: {MODEL_PATH}")
    print(f"Predictions: {PREDICTIONS_PATH}")
    print(f"Evaluation: {EVALUATION_PATH}")
    print(f"History: {HISTORY_PATH}")
    print(f"Figures: {LOSS_FIGURE_PATH}, {VALIDATION_FIGURE_PATH}")


if __name__ == "__main__":
    main()