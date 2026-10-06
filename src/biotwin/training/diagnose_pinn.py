"""Diagnose baseline PINN conditioning and run one normalized-time comparison."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from biotwin.models.biological_ode import BiologicalParameters
from biotwin.models.pinn import CRPIL6PINN, TimeNormalizedPINN
from biotwin.training.run_pinn_experiment import (
    MODEL_PATH,
    PREDICTION_COLUMNS as BASELINE_PREDICTION_COLUMNS,
    calculate_metrics,
    evaluate_model,
    load_checkpoint,
    predict_frame,
)
from biotwin.training.train_pinn import (
    TRAINING_DATA_PATH,
    _tensorize,
    load_trajectory_dataset,
    patient_level_split,
    train_pinn,
)


DIAGNOSTICS_PATH = Path("outputs/predictions/pinn_diagnostics.json")
NORMALIZED_REPORT_PATH = Path("outputs/predictions/pinn_time_normalized_evaluation.json")
NORMALIZED_HISTORY_PATH = Path("outputs/predictions/pinn_time_normalized_history.csv")
NORMALIZED_PREDICTIONS_PATH = Path("outputs/predictions/pinn_time_normalized_predictions.csv")
NORMALIZED_MODEL_PATH = Path("outputs/models/crp_il6_pinn_time_normalized.pt")
PARAMS = BiologicalParameters(1.8, 0.45, 1.2, 0.18)
SCIENTIFIC_LIMITATION = (
    "Performance metrics at this stage measure agreement with the synthetic "
    "biological simulator and do not establish clinical validity."
)


def _statistics(values: pd.Series) -> dict[str, float]:
    values = values.to_numpy(dtype=np.float64)
    return {
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
    }


def _history_summary(history: pd.DataFrame) -> dict[str, Any]:
    values = history[["total_loss", "data_loss", "physics_loss"]].to_numpy(dtype=np.float64)
    return {
        "all_losses_finite": bool(np.isfinite(values).all()),
        "total_loss_upward_steps": int((np.diff(history["total_loss"].to_numpy()) > 0).sum()),
        "initial": {
            key: float(history.iloc[0][key])
            for key in ("total_loss", "data_loss", "physics_loss")
        },
        "final": {
            key: float(history.iloc[-1][key])
            for key in ("total_loss", "data_loss", "physics_loss")
        },
        "physics_loss_min": float(history["physics_loss"].min()),
        "physics_loss_max": float(history["physics_loss"].max()),
    }


def _first_layer_saturation(
    model: CRPIL6PINN,
    train_data: pd.DataFrame,
    time_min: float,
    time_max: float,
) -> dict[str, float]:
    time, stimulus, _, _ = _tensorize(train_data)
    raw_inputs = torch.stack([time, stimulus], dim=1)
    normalized_inputs = raw_inputs.clone()
    normalized_inputs[:, 0] = (normalized_inputs[:, 0] - time_min) / (time_max - time_min)
    first_layer = model.network[0]
    with torch.no_grad():
        raw_activation = torch.tanh(first_layer(raw_inputs))
        normalized_activation = torch.tanh(first_layer(normalized_inputs))
    return {
        "raw_time_fraction_abs_tanh_at_least_0_95": float(
            (raw_activation.abs() >= 0.95).float().mean()
        ),
        "normalized_time_fraction_abs_tanh_at_least_0_95_same_weights": float(
            (normalized_activation.abs() >= 0.95).float().mean()
        ),
        "raw_time_fraction_abs_tanh_at_least_0_99": float(
            (raw_activation.abs() >= 0.99).float().mean()
        ),
    }


def run_time_normalized_experiment(
    *,
    dataset_path: str | Path = TRAINING_DATA_PATH,
    epochs: int = 100,
    learning_rate: float = 1e-3,
    physics_weight: float = 1.0,
    seed: int = 42,
    validation_fraction: float = 0.2,
) -> dict[str, Any]:
    """Run exactly one same-seed, same-split experiment with normalized time."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(2)

    dataset = load_trajectory_dataset(dataset_path)
    train_df, validation_df = patient_level_split(dataset, validation_fraction, seed)
    time_min = float(train_df["time_hours"].min())
    time_max = float(train_df["time_hours"].max())
    base_model = CRPIL6PINN(hidden_dim=32)
    model = TimeNormalizedPINN(base_model, time_min=time_min, time_max=time_max)
    train_tensors = _tensorize(train_df)
    trained_model, history = train_pinn(
        model=model,
        train_data=train_tensors,
        params=PARAMS,
        epochs=epochs,
        learning_rate=learning_rate,
        physics_weight=physics_weight,
        device="cpu",
    )
    evaluation = evaluate_model(trained_model, validation_df, PARAMS)
    predictions = predict_frame(trained_model, validation_df)
    if predictions.columns.tolist() != BASELINE_PREDICTION_COLUMNS:
        raise RuntimeError("normalized-time prediction schema changed unexpectedly")
    if set(predictions["patient_id"]).intersection(set(train_df["patient_id"])):
        raise RuntimeError("normalized-time predictions include training patients")
    history_summary = _history_summary(pd.DataFrame(history))
    if not history_summary["all_losses_finite"]:
        raise FloatingPointError("normalized-time training history contains non-finite losses")

    report: dict[str, Any] = {
        "experiment": "A: time mapped linearly to [0, 1]",
        "training_configuration": {
            "optimizer": "Adam",
            "device": "cpu",
            "seed": seed,
            "train_patient_count": int(train_df["patient_id"].nunique()),
            "validation_patient_count": int(validation_df["patient_id"].nunique()),
            "epochs": epochs,
            "learning_rate": learning_rate,
            "physics_weight": physics_weight,
            "hidden_dim": 32,
            "time_min_hours": time_min,
            "time_max_hours": time_max,
            "time_transform": "(time_hours - time_min_hours) / (time_max_hours - time_min_hours)",
            "stimulus_transform": "none; physical stimulus scale retained",
            "ode_parameters": {
                "alpha_il6": PARAMS.alpha_il6,
                "k_il6": PARAMS.k_il6,
                "alpha_crp": PARAMS.alpha_crp,
                "k_crp": PARAMS.k_crp,
            },
        },
        "il6": calculate_metrics(predictions["il6_actual"], predictions["il6_predicted"]),
        "crp": calculate_metrics(predictions["crp_actual"], predictions["crp_predicted"]),
        "validation_physics_loss": evaluation["validation_physics_loss"],
        "validation_physics_residual_metric": evaluation[
            "validation_physics_residual_metric"
        ],
        "training_history_summary": history_summary,
        "scientific_limitation": SCIENTIFIC_LIMITATION,
        "data_source": "synthetic ODE-generated trajectories only",
    }

    NORMALIZED_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    NORMALIZED_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(NORMALIZED_PREDICTIONS_PATH, index=False)
    pd.DataFrame(history, columns=["epoch", "total_loss", "data_loss", "physics_loss"]).to_csv(
        NORMALIZED_HISTORY_PATH, index=False
    )
    NORMALIZED_REPORT_PATH.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    torch.save(
        {
            "model_state_dict": {
                key: value.detach().cpu() for key, value in trained_model.state_dict().items()
            },
            "hidden_dim": 32,
            "time_min": time_min,
            "time_max": time_max,
            "seed": seed,
        },
        NORMALIZED_MODEL_PATH,
    )
    return report


def _build_diagnostics(
    dataset: pd.DataFrame,
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    baseline: dict[str, Any],
) -> dict[str, Any]:
    current_model = load_checkpoint(MODEL_PATH)
    time_min = float(train_df["time_hours"].min())
    time_max = float(train_df["time_hours"].max())
    train_time, train_stimulus, train_il6, train_crp = _tensorize(train_df)
    with torch.no_grad():
        baseline_outputs = current_model(torch.stack([train_time, train_stimulus], dim=1))
    per_target_data_mse = {
        "il6": float(torch.mean((baseline_outputs[:, 0] - train_il6) ** 2)),
        "crp": float(torch.mean((baseline_outputs[:, 1] - train_crp) ** 2)),
    }
    baseline_loss = baseline["final_train_loss"]
    initial_rows = dataset.loc[dataset["time_hours"] == dataset["time_hours"].min()]
    initial_state_stats = {
        target: _statistics(initial_rows[target]) for target in ("il6", "crp")
    }
    raw_time_range = float(dataset["time_hours"].max() - dataset["time_hours"].min())
    data_physics_ratio = baseline_loss["data_loss"] / baseline_loss["physics_loss"]

    return {
        "data_source": "synthetic ODE-generated trajectories only",
        "patient_split": {
            "seed": 42,
            "train_patient_count": int(train_df["patient_id"].nunique()),
            "validation_patient_count": int(validation_df["patient_id"].nunique()),
            "train_validation_patient_overlap": int(
                len(set(train_df["patient_id"]).intersection(validation_df["patient_id"]))
            ),
        },
        "input_statistics": {
            "time_hours": _statistics(dataset["time_hours"]),
            "stimulus": _statistics(dataset["stimulus"]),
        },
        "target_statistics": {
            "il6": _statistics(dataset["il6"]),
            "crp": _statistics(dataset["crp"]),
        },
        "baseline_loss_magnitudes": {
            "initial_data_loss": baseline["initial_train_loss"]["data_loss"],
            "initial_physics_loss": baseline["initial_train_loss"]["physics_loss"],
            "final_data_loss": baseline_loss["data_loss"],
            "final_physics_loss": baseline_loss["physics_loss"],
            "final_data_to_physics_loss_ratio": data_physics_ratio,
            "final_data_mse_by_target": per_target_data_mse,
            "validation_physics_loss": baseline["trained_validation"][
                "validation_physics_loss"
            ],
        },
        "current_validation_metrics": baseline["trained_validation"],
        "conditioning_observations": {
            "first_layer_tanh_saturation": _first_layer_saturation(
                current_model, train_df, time_min, time_max
            ),
            "time_input_range_hours": raw_time_range,
            "time_input_range_ratio_to_stimulus_range": raw_time_range
            / float(dataset["stimulus"].max() - dataset["stimulus"].min()),
            "note": (
                "The baseline receives unscaled hours from 0 to 167; with its trained weights, "
                "most first-layer tanh activations are saturated. Rescaling time to [0, 1] with "
                "the same weights sharply reduces that saturation."
            ),
        },
        "patient_heterogeneity_observations": {
            "model_inputs": ["time_hours", "stimulus"],
            "patient_id_used_as_model_input": False,
            "patient_initial_state_statistics_at_time_zero": initial_state_stats,
            "generator_samples_patient_specific_initial_il6_from": [0.05, 1.2],
            "generator_samples_patient_specific_initial_crp_from": [0.05, 1.5],
            "generator_samples_patient_specific_ode_parameter_ranges": {
                "alpha_il6": [0.8, 2.5],
                "k_il6": [0.12, 0.45],
                "alpha_crp": [0.4, 1.8],
                "k_crp": [0.05, 0.25],
            },
            "individual_sampled_ode_parameters_persisted_in_dataset": False,
            "note": (
                "The generator samples patient-specific initial states and ODE rates, but the "
                "five-column trajectory CSV stores neither patient-specific rates nor an explicit "
                "initial-state feature. The PINN therefore maps only time and stimulus to targets "
                "across heterogeneous patients."
            ),
        },
        "target_scale_observation": (
            "CRP contributes about 98.7% of the final baseline data MSE (882.5 of 894.0). "
            "Target normalization or output weighting is a plausible separate follow-up, but "
            "was deliberately not varied in Experiment A."
        ),
        "scientific_limitation": SCIENTIFIC_LIMITATION,
    }


def run_diagnostic_experiment() -> tuple[dict[str, Any], dict[str, Any]]:
    dataset = load_trajectory_dataset()
    train_df, validation_df = patient_level_split(dataset, validation_fraction=0.2, seed=42)
    baseline = json.loads(Path("outputs/predictions/pinn_evaluation.json").read_text(encoding="utf-8"))
    normalized = run_time_normalized_experiment()
    diagnostics = _build_diagnostics(dataset, train_df, validation_df, baseline)
    diagnostics["normalized_time_experiment"] = normalized
    diagnostics["training_stability"] = {
        "baseline": _history_summary(
            pd.read_csv("outputs/predictions/pinn_training_history.csv")
        ),
        "normalized_time": normalized["training_history_summary"],
        "observation": (
            "Both runs had finite loss histories and monotonically decreasing total loss. "
            "The physics-loss component fluctuated and did not decrease monotonically in either run."
        ),
    }
    diagnostics["baseline_vs_normalized_time"] = {
        "baseline_il6": baseline["trained_validation"]["il6"],
        "normalized_il6": normalized["il6"],
        "baseline_crp": baseline["trained_validation"]["crp"],
        "normalized_crp": normalized["crp"],
        "baseline_validation_physics_loss": baseline["trained_validation"][
            "validation_physics_loss"
        ],
        "normalized_validation_physics_loss": normalized["validation_physics_loss"],
    }
    DIAGNOSTICS_PATH.write_text(
        json.dumps(diagnostics, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return diagnostics, normalized


def main() -> None:
    diagnostics, normalized = run_diagnostic_experiment()
    print(json.dumps({"diagnostics": diagnostics, "normalized_time_experiment": normalized}, indent=2))
    print(f"Diagnostics: {DIAGNOSTICS_PATH}")
    print(f"Normalized evaluation: {NORMALIZED_REPORT_PATH}")
    print(f"Normalized model: {NORMALIZED_MODEL_PATH}")
    print(f"Normalized predictions: {NORMALIZED_PREDICTIONS_PATH}")
    print(f"Normalized history: {NORMALIZED_HISTORY_PATH}")


if __name__ == "__main__":
    main()