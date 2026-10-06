"""Controlled full experiment comparing the existing time-normalized PINN to the
patient-conditioned PINN on the synthetic patient-conditioned Digital Twin data.

The experiment is intentionally narrow: same split, seed, optimizer, learning
rate, hidden dimension, physics weight, and device for both models. The only
architectural difference is patient conditioning.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from biotwin.models.biological_ode import BiologicalParameters
from biotwin.models.conditioned_pinn import (
    CONDITIONED_STATIC_FEATURES,
    ConditionedPINN,
    StaticFeatureScaler,
    TimeNormalizedConditionedPINN,
    save_scaler_artifact,
    physics_loss as conditioned_physics_loss,
    physics_residual as conditioned_physics_residual,
)
from biotwin.models.pinn import CRPIL6PINN, TimeNormalizedPINN, physics_loss, physics_residual
from biotwin.training.train_conditioned_pinn import (
    load_conditioned_dataset,
    patient_level_split,
    tensorize_conditioned_dataset,
    train_conditioned_pinn,
)
from biotwin.training.train_pinn import data_loss_fn, train_pinn

DATA_PATH = Path("data/processed/patient_conditioned_digital_twin.csv")
MODEL_DIR = Path("outputs/models")
PREDICTIONS_DIR = Path("outputs/predictions")
FIGURES_DIR = Path("outputs/figures")

SEED = 42
VALIDATION_FRACTION = 0.2
EPOCHS = 50
LEARNING_RATE = 1e-3
HIDDEN_DIM = 32
PHYSICS_WEIGHT = 1.0
DEVICE = "cpu"

BASELINE_MODEL_PATH = MODEL_DIR / "time_normalized_baseline_full.pt"
CONDITIONED_MODEL_PATH = MODEL_DIR / "conditioned_pinn_full.pt"
CONDITIONED_SCALER_PATH = MODEL_DIR / "conditioned_pinn_scaler.json"
BASELINE_HISTORY_PATH = PREDICTIONS_DIR / "baseline_time_normalized_training_history.csv"
CONDITIONED_HISTORY_PATH = PREDICTIONS_DIR / "conditioned_pinn_training_history.csv"
BASELINE_PREDICTIONS_PATH = PREDICTIONS_DIR / "baseline_time_normalized_validation_predictions.csv"
CONDITIONED_PREDICTIONS_PATH = PREDICTIONS_DIR / "conditioned_pinn_validation_predictions.csv"
BASELINE_EVAL_PATH = PREDICTIONS_DIR / "baseline_time_normalized_evaluation.json"
CONDITIONED_EVAL_PATH = PREDICTIONS_DIR / "conditioned_pinn_evaluation.json"
SPLIT_PATH = PREDICTIONS_DIR / "conditioned_experiment_split.json"
COMPARISON_PATH = PREDICTIONS_DIR / "pinn_conditioning_comparison.json"
LOSS_FIGURE_PATH = FIGURES_DIR / "conditioned_pinn_training_loss.png"
IL6_FIGURE_PATH = FIGURES_DIR / "conditioned_pinn_validation_il6_example.png"
CRP_FIGURE_PATH = FIGURES_DIR / "conditioned_pinn_validation_crp_example.png"

PARAMS = BiologicalParameters(alpha_il6=1.8, k_il6=0.45, alpha_crp=1.2, k_crp=0.18)
SCIENTIFIC_LIMITATION = (
    "The evaluation measures agreement with the competition's synthetic/model-simulated Digital Twin "
    "trajectories and does not establish clinical validity."
)


def calculate_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
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
        "r2": float(r2),
    }


def save_split_artifact(train_df: pd.DataFrame, validation_df: pd.DataFrame) -> dict[str, Any]:
    split_payload = {
        "seed": SEED,
        "validation_fraction": VALIDATION_FRACTION,
        "train_patient_count": int(train_df["patient_id"].nunique()),
        "validation_patient_count": int(validation_df["patient_id"].nunique()),
        "train_patient_ids": sorted(train_df["patient_id"].drop_duplicates().tolist()),
        "validation_patient_ids": sorted(validation_df["patient_id"].drop_duplicates().tolist()),
    }
    SPLIT_PATH.write_text(json.dumps(split_payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return split_payload


def _baseline_predict(model: torch.nn.Module, validation_df: pd.DataFrame) -> pd.DataFrame:
    time = torch.tensor(validation_df["time_hours"].to_numpy(dtype=np.float32), dtype=torch.float32)
    stimulus = torch.tensor(validation_df["synthetic_stimulus"].to_numpy(dtype=np.float32), dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        outputs = model(torch.stack([time, stimulus], dim=1)).cpu().numpy()
    if not np.isfinite(outputs).all():
        raise FloatingPointError("baseline model produced non-finite validation predictions")
    frame = validation_df[["patient_id", "time_hours", "synthetic_stimulus"]].copy()
    frame = frame.rename(columns={"synthetic_stimulus": "stimulus"})
    frame["il6_actual"] = validation_df["synthetic_il6"].to_numpy()
    frame["il6_predicted"] = outputs[:, 0]
    frame["crp_actual"] = validation_df["synthetic_crp"].to_numpy()
    frame["crp_predicted"] = outputs[:, 1]
    return frame


def _conditioned_predict(model: torch.nn.Module, validation_df: pd.DataFrame, scaler: StaticFeatureScaler) -> pd.DataFrame:
    time = torch.tensor(validation_df["time_hours"].to_numpy(dtype=np.float32), dtype=torch.float32)
    stimulus = torch.tensor(validation_df["synthetic_stimulus"].to_numpy(dtype=np.float32), dtype=torch.float32)
    static_array = scaler.transform(validation_df)
    static_tensor = torch.tensor(static_array, dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        model_input = torch.cat([time.unsqueeze(1), stimulus.unsqueeze(1), static_tensor], dim=1)
        outputs = model(model_input).cpu().numpy()
    if not np.isfinite(outputs).all():
        raise FloatingPointError("conditioned model produced non-finite validation predictions")
    frame = validation_df[["patient_id", "time_hours", "synthetic_stimulus"]].copy()
    frame = frame.rename(columns={"synthetic_stimulus": "stimulus"})
    frame["il6_actual"] = validation_df["synthetic_il6"].to_numpy()
    frame["il6_predicted"] = outputs[:, 0]
    frame["crp_actual"] = validation_df["synthetic_crp"].to_numpy()
    frame["crp_predicted"] = outputs[:, 1]
    return frame


def _evaluate_baseline(model: torch.nn.Module, validation_df: pd.DataFrame) -> dict[str, Any]:
    predictions = _baseline_predict(model, validation_df)
    time = torch.tensor(validation_df["time_hours"].to_numpy(dtype=np.float32), dtype=torch.float32)
    stimulus = torch.tensor(validation_df["synthetic_stimulus"].to_numpy(dtype=np.float32), dtype=torch.float32)
    target_il6 = torch.tensor(validation_df["synthetic_il6"].to_numpy(dtype=np.float32), dtype=torch.float32)
    target_crp = torch.tensor(validation_df["synthetic_crp"].to_numpy(dtype=np.float32), dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        pred_outputs = model(torch.stack([time, stimulus], dim=1))
        val_data_loss = data_loss_fn(pred_outputs[:, 0], target_il6) + data_loss_fn(pred_outputs[:, 1], target_crp)
    phys = physics_loss(model, time, stimulus, PARAMS)
    residual_il6, residual_crp = physics_residual(model, time, stimulus, PARAMS)
    metrics = {
        "il6": calculate_metrics(predictions["il6_actual"].to_numpy(), predictions["il6_predicted"].to_numpy()),
        "crp": calculate_metrics(predictions["crp_actual"].to_numpy(), predictions["crp_predicted"].to_numpy()),
        "validation_data_loss": float(val_data_loss.detach().cpu()),
        "validation_physics_loss": float(phys.detach().cpu()),
        "mean_absolute_il6_residual": float(residual_il6.abs().mean().detach().cpu()),
        "mean_absolute_crp_residual": float(residual_crp.abs().mean().detach().cpu()),
    }
    numeric_values = [
        float(metrics["il6"]["mae"]),
        float(metrics["il6"]["rmse"]),
        float(metrics["il6"]["r2"]),
        float(metrics["crp"]["mae"]),
        float(metrics["crp"]["rmse"]),
        float(metrics["crp"]["r2"]),
        float(metrics["validation_data_loss"]),
        float(metrics["validation_physics_loss"]),
        float(metrics["mean_absolute_il6_residual"]),
        float(metrics["mean_absolute_crp_residual"]),
    ]
    if not np.isfinite(numeric_values).all():
        raise FloatingPointError("baseline evaluation metrics contain non-finite values")
    return {"metrics": metrics, "predictions": predictions}


def _evaluate_conditioned(model: torch.nn.Module, validation_df: pd.DataFrame, scaler: StaticFeatureScaler) -> dict[str, Any]:
    predictions = _conditioned_predict(model, validation_df, scaler)
    time = torch.tensor(validation_df["time_hours"].to_numpy(dtype=np.float32), dtype=torch.float32)
    stimulus = torch.tensor(validation_df["synthetic_stimulus"].to_numpy(dtype=np.float32), dtype=torch.float32)
    static_tensor = torch.tensor(scaler.transform(validation_df), dtype=torch.float32)
    il6_target = torch.tensor(validation_df["synthetic_il6"].to_numpy(dtype=np.float32), dtype=torch.float32)
    crp_target = torch.tensor(validation_df["synthetic_crp"].to_numpy(dtype=np.float32), dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        model_input = torch.cat([time.unsqueeze(1), stimulus.unsqueeze(1), static_tensor], dim=1)
        outputs = model(model_input)
        val_data_loss = data_loss_fn(outputs[:, 0], il6_target) + data_loss_fn(outputs[:, 1], crp_target)
    phys = conditioned_physics_loss(model, time, stimulus, static_tensor, PARAMS)
    residual_il6, residual_crp = conditioned_physics_residual(model, time, stimulus, static_tensor, PARAMS)
    metrics = {
        "il6": calculate_metrics(predictions["il6_actual"].to_numpy(), predictions["il6_predicted"].to_numpy()),
        "crp": calculate_metrics(predictions["crp_actual"].to_numpy(), predictions["crp_predicted"].to_numpy()),
        "validation_data_loss": float(val_data_loss.detach().cpu()),
        "validation_physics_loss": float(phys.detach().cpu()),
        "mean_absolute_il6_residual": float(residual_il6.abs().mean().detach().cpu()),
        "mean_absolute_crp_residual": float(residual_crp.abs().mean().detach().cpu()),
    }
    numeric_values = [
        float(metrics["il6"]["mae"]),
        float(metrics["il6"]["rmse"]),
        float(metrics["il6"]["r2"]),
        float(metrics["crp"]["mae"]),
        float(metrics["crp"]["rmse"]),
        float(metrics["crp"]["r2"]),
        float(metrics["validation_data_loss"]),
        float(metrics["validation_physics_loss"]),
        float(metrics["mean_absolute_il6_residual"]),
        float(metrics["mean_absolute_crp_residual"]),
    ]
    if not np.isfinite(numeric_values).all():
        raise FloatingPointError("conditioned evaluation metrics contain non-finite values")
    return {"metrics": metrics, "predictions": predictions}


def _write_training_loss_figure(baseline_history: list[dict[str, float]], conditioned_history: list[dict[str, float]]) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    epochs_baseline = [entry["epoch"] for entry in baseline_history]
    epochs_conditioned = [entry["epoch"] for entry in conditioned_history]
    fig, axis = plt.subplots(figsize=(8, 5))
    axis.plot(epochs_baseline, [entry["total_loss"] for entry in baseline_history], label="Baseline total loss")
    axis.plot(epochs_conditioned, [entry["total_loss"] for entry in conditioned_history], label="Conditioned total loss")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Total loss")
    axis.set_title("Training loss comparison")
    axis.legend()
    fig.tight_layout()
    fig.savefig(LOSS_FIGURE_PATH, dpi=140)
    plt.close(fig)


def _write_example_figure(baseline_predictions: pd.DataFrame, conditioned_predictions: pd.DataFrame, target: str, path: Path) -> None:
    patient_id = baseline_predictions["patient_id"].iloc[0]
    baseline_example = baseline_predictions[baseline_predictions["patient_id"] == patient_id].copy()
    conditioned_example = conditioned_predictions[conditioned_predictions["patient_id"] == patient_id].copy()
    fig, axis = plt.subplots(figsize=(8, 4))
    axis.plot(baseline_example["time_hours"], baseline_example[f"{target}_actual"], label="Actual")
    axis.plot(baseline_example["time_hours"], baseline_example[f"{target}_predicted"], label="Baseline prediction")
    axis.plot(conditioned_example["time_hours"], conditioned_example[f"{target}_predicted"], label="Conditioned prediction")
    axis.set_xlabel("Time (hours)")
    axis.set_ylabel(f"{target.upper()} value")
    axis.set_title(f"Validation example: {target.upper()} - patient {patient_id}")
    axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _save_model_checkpoint(model: torch.nn.Module, path: Path, metadata: dict[str, Any]) -> None:
    checkpoint = {
        "model_state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "metadata": metadata,
    }
    torch.save(checkpoint, path)


def _build_delta_summary(baseline_metrics: dict[str, Any], conditioned_metrics: dict[str, Any]) -> dict[str, float]:
    deltas: dict[str, float] = {}
    for name in ("mae", "rmse"):
        deltas[f"il6_{name}"] = float(conditioned_metrics["il6"][name] - baseline_metrics["il6"][name])
        deltas[f"crp_{name}"] = float(conditioned_metrics["crp"][name] - baseline_metrics["crp"][name])
    deltas["il6_r2"] = float(conditioned_metrics["il6"]["r2"] - baseline_metrics["il6"]["r2"])
    deltas["crp_r2"] = float(conditioned_metrics["crp"]["r2"] - baseline_metrics["crp"]["r2"])
    deltas["validation_data_loss"] = float(conditioned_metrics["validation_data_loss"] - baseline_metrics["validation_data_loss"])
    deltas["validation_physics_loss"] = float(conditioned_metrics["validation_physics_loss"] - baseline_metrics["validation_physics_loss"])
    deltas["mean_absolute_il6_residual"] = float(conditioned_metrics["mean_absolute_il6_residual"] - baseline_metrics["mean_absolute_il6_residual"])
    deltas["mean_absolute_crp_residual"] = float(conditioned_metrics["mean_absolute_crp_residual"] - baseline_metrics["mean_absolute_crp_residual"])
    return deltas


def run_controlled_experiment() -> dict[str, Any]:
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(2)

    dataset = load_conditioned_dataset(DATA_PATH)
    train_df, validation_df = patient_level_split(dataset, validation_fraction=VALIDATION_FRACTION, seed=SEED)
    if set(train_df["patient_id"]).intersection(validation_df["patient_id"]):
        raise RuntimeError("patient-level split leaked IDs between training and validation")

    split_info = save_split_artifact(train_df, validation_df)

    time_min = float(train_df["time_hours"].min())
    time_max = float(train_df["time_hours"].max())

    baseline_time = torch.tensor(train_df["time_hours"].to_numpy(dtype=np.float32), dtype=torch.float32)
    baseline_stimulus = torch.tensor(train_df["synthetic_stimulus"].to_numpy(dtype=np.float32), dtype=torch.float32)
    baseline_il6 = torch.tensor(train_df["synthetic_il6"].to_numpy(dtype=np.float32), dtype=torch.float32)
    baseline_crp = torch.tensor(train_df["synthetic_crp"].to_numpy(dtype=np.float32), dtype=torch.float32)
    torch.manual_seed(SEED)
    baseline_model = TimeNormalizedPINN(CRPIL6PINN(hidden_dim=HIDDEN_DIM), time_min=time_min, time_max=time_max)
    baseline_start = time.perf_counter()
    baseline_model, baseline_history = train_pinn(
        model=baseline_model,
        train_data=(baseline_time, baseline_stimulus, baseline_il6, baseline_crp),
        params=PARAMS,
        epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        physics_weight=PHYSICS_WEIGHT,
        device=DEVICE,
    )
    baseline_duration = time.perf_counter() - baseline_start
    baseline_eval = _evaluate_baseline(baseline_model, validation_df)
    baseline_predictions = baseline_eval["predictions"]
    baseline_predictions.to_csv(BASELINE_PREDICTIONS_PATH, index=False)
    pd.DataFrame(baseline_history, columns=["epoch", "total_loss", "data_loss", "physics_loss"]).to_csv(
        BASELINE_HISTORY_PATH, index=False
    )
    baseline_report = {
        "model_type": "TimeNormalizedPINN",
        "train_patient_count": int(train_df["patient_id"].nunique()),
        "validation_patient_count": int(validation_df["patient_id"].nunique()),
        "training_duration_seconds": float(baseline_duration),
        "initial_total_loss": float(baseline_history[0]["total_loss"]),
        "final_total_loss": float(baseline_history[-1]["total_loss"]),
        "initial_data_loss": float(baseline_history[0]["data_loss"]),
        "final_data_loss": float(baseline_history[-1]["data_loss"]),
        "initial_physics_loss": float(baseline_history[0]["physics_loss"]),
        "final_physics_loss": float(baseline_history[-1]["physics_loss"]),
        "metrics": baseline_eval["metrics"],
        "history": baseline_history,
    }
    BASELINE_EVAL_PATH.write_text(json.dumps(baseline_report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    _save_model_checkpoint(
        baseline_model,
        BASELINE_MODEL_PATH,
        {
            "model_type": "TimeNormalizedPINN",
            "seed": SEED,
            "epochs": EPOCHS,
            "learning_rate": LEARNING_RATE,
            "hidden_dim": HIDDEN_DIM,
            "physics_weight": PHYSICS_WEIGHT,
            "validation_fraction": VALIDATION_FRACTION,
            "time_min": time_min,
            "time_max": time_max,
        },
    )

    scaler = StaticFeatureScaler(feature_names=CONDITIONED_STATIC_FEATURES).fit(train_df)
    conditioned_time, conditioned_stimulus, conditioned_static, conditioned_il6, conditioned_crp = tensorize_conditioned_dataset(
        train_df, scaler=scaler
    )
    torch.manual_seed(SEED)
    conditioned_base = ConditionedPINN(hidden_dim=HIDDEN_DIM, static_feature_names=CONDITIONED_STATIC_FEATURES)
    conditioned_base.set_feature_scaler(scaler)
    conditioned_model = TimeNormalizedConditionedPINN(conditioned_base, time_min=time_min, time_max=time_max)
    conditioned_start = time.perf_counter()
    conditioned_model, conditioned_history = train_conditioned_pinn(
        model=conditioned_model,
        train_data=(conditioned_time, conditioned_stimulus, conditioned_static, conditioned_il6, conditioned_crp),
        params=PARAMS,
        epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        physics_weight=PHYSICS_WEIGHT,
        device=DEVICE,
    )
    conditioned_duration = time.perf_counter() - conditioned_start
    conditioned_eval = _evaluate_conditioned(conditioned_model, validation_df, scaler)
    conditioned_predictions = conditioned_eval["predictions"]
    conditioned_predictions.to_csv(CONDITIONED_PREDICTIONS_PATH, index=False)
    pd.DataFrame(conditioned_history, columns=["epoch", "total_loss", "data_loss", "physics_loss"]).to_csv(
        CONDITIONED_HISTORY_PATH, index=False
    )
    conditioned_report = {
        "model_type": "TimeNormalizedConditionedPINN",
        "train_patient_count": int(train_df["patient_id"].nunique()),
        "validation_patient_count": int(validation_df["patient_id"].nunique()),
        "training_duration_seconds": float(conditioned_duration),
        "initial_total_loss": float(conditioned_history[0]["total_loss"]),
        "final_total_loss": float(conditioned_history[-1]["total_loss"]),
        "initial_data_loss": float(conditioned_history[0]["data_loss"]),
        "final_data_loss": float(conditioned_history[-1]["data_loss"]),
        "initial_physics_loss": float(conditioned_history[0]["physics_loss"]),
        "final_physics_loss": float(conditioned_history[-1]["physics_loss"]),
        "metrics": conditioned_eval["metrics"],
        "history": conditioned_history,
    }
    CONDITIONED_EVAL_PATH.write_text(json.dumps(conditioned_report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    _save_model_checkpoint(
        conditioned_model,
        CONDITIONED_MODEL_PATH,
        {
            "model_type": "TimeNormalizedConditionedPINN",
            "seed": SEED,
            "epochs": EPOCHS,
            "learning_rate": LEARNING_RATE,
            "hidden_dim": HIDDEN_DIM,
            "physics_weight": PHYSICS_WEIGHT,
            "validation_fraction": VALIDATION_FRACTION,
            "time_min": time_min,
            "time_max": time_max,
            "static_feature_names": CONDITIONED_STATIC_FEATURES,
        },
    )
    save_scaler_artifact(
        CONDITIONED_SCALER_PATH,
        scaler,
        train_patient_ids=sorted(train_df["patient_id"].drop_duplicates().tolist()),
        time_min=time_min,
        time_max=time_max,
    )

    comparison_deltas = _build_delta_summary(baseline_eval["metrics"], conditioned_eval["metrics"])
    _write_training_loss_figure(baseline_history, conditioned_history)
    _write_example_figure(
        baseline_predictions[["patient_id", "time_hours", "il6_actual", "il6_predicted"]],
        conditioned_predictions[["patient_id", "time_hours", "il6_actual", "il6_predicted"]],
        "il6",
        IL6_FIGURE_PATH,
    )
    _write_example_figure(
        baseline_predictions[["patient_id", "time_hours", "crp_actual", "crp_predicted"]],
        conditioned_predictions[["patient_id", "time_hours", "crp_actual", "crp_predicted"]],
        "crp",
        CRP_FIGURE_PATH,
    )

    comparison_report = {
        "experiment_configuration": {
            "seed": SEED,
            "epochs": EPOCHS,
            "learning_rate": LEARNING_RATE,
            "hidden_dim": HIDDEN_DIM,
            "physics_weight": PHYSICS_WEIGHT,
            "device": DEVICE,
            "validation_fraction": VALIDATION_FRACTION,
            "optimizer": "Adam",
            "time_min_hours": time_min,
            "time_max_hours": time_max,
            "data_path": DATA_PATH.as_posix(),
        },
        "dataset": {
            "name": "patient_conditioned_digital_twin",
            "dynamic_inputs": ["time_hours", "synthetic_stimulus"],
            "static_features": CONDITIONED_STATIC_FEATURES,
            "targets": ["synthetic_il6", "synthetic_crp"],
            "patient_id_usage": "grouping and splitting only; not a model feature",
            "data_provenance": "NHANES baseline fields + synthetic model-simulated trajectories",
        },
        "split": split_info,
        "baseline": baseline_report,
        "conditioned": conditioned_report,
        "deltas": comparison_deltas,
        "interpretation": {
            "error_deltas_negative_improve": True,
            "r2_deltas_positive_improve": True,
            "scientific_limitation": SCIENTIFIC_LIMITATION,
        },
        "limitations": [
            "This evaluates agreement with synthetic/model-simulated trajectories only.",
            "It does not establish clinical validity or real-world prediction performance.",
        ],
    }
    COMPARISON_PATH.write_text(json.dumps(comparison_report, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    return comparison_report


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    report = run_controlled_experiment()
    print(json.dumps({
        "configuration": report["experiment_configuration"],
        "split": report["split"],
        "baseline_final_total_loss": report["baseline"]["final_total_loss"],
        "conditioned_final_total_loss": report["conditioned"]["final_total_loss"],
        "baseline_il6_r2": report["baseline"]["metrics"]["il6"]["r2"],
        "conditioned_il6_r2": report["conditioned"]["metrics"]["il6"]["r2"],
        "baseline_crp_r2": report["baseline"]["metrics"]["crp"]["r2"],
        "conditioned_crp_r2": report["conditioned"]["metrics"]["crp"]["r2"],
    }, indent=2))
    print(f"Baseline checkpoint: {BASELINE_MODEL_PATH}")
    print(f"Conditioned checkpoint: {CONDITIONED_MODEL_PATH}")
    print(f"Comparison JSON: {COMPARISON_PATH}")


if __name__ == "__main__":
    main()
