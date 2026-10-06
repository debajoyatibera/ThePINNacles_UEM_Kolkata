"""Train the patient-conditioned CRP/IL-6 PINN on the synthetic Digital Twin data."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from src.biotwin.models.biological_ode import BiologicalParameters
from src.biotwin.models.conditioned_pinn import (
    CONDITIONED_STATIC_FEATURES,
    ConditionedPINN,
    StaticFeatureScaler,
    physics_loss,
)

TRAINING_DATA_PATH = Path("data/processed/patient_conditioned_digital_twin.csv")


def load_conditioned_dataset(csv_path: str | Path = TRAINING_DATA_PATH) -> pd.DataFrame:
    """Load the patient-conditioned digital twin dataset and validate core schema.

    Baseline covariates may legitimately contain missing values in the raw bridge
    dataset because the synthetic conditioning step imputed them before training.
    The training tensorizer enforces the stricter no-NaN/no-inf contract on
    model-ready inputs.
    """
    dataset = pd.read_csv(csv_path)
    required = [
        "patient_id",
        "time_hours",
        "synthetic_stimulus",
        "synthetic_il6",
        "synthetic_crp",
        *CONDITIONED_STATIC_FEATURES,
    ]
    missing = [column for column in required if column not in dataset.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    if dataset.empty:
        raise ValueError("Conditioned patient dataset cannot be empty")
    if dataset["patient_id"].isna().any():
        raise ValueError("patient_id values must not be missing")
    for column in ["time_hours", "synthetic_stimulus", "synthetic_il6", "synthetic_crp"]:
        numeric = pd.to_numeric(dataset[column], errors="coerce")
        if numeric.isna().any() or not np.isfinite(numeric.to_numpy(dtype=np.float32)).all():
            raise ValueError(f"Column '{column}' contains NaN or non-finite values")
    return dataset


def patient_level_split(
    dataset: pd.DataFrame,
    validation_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split by unique patient ID so each entire trajectory stays in one split."""
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")
    rng = np.random.default_rng(seed)
    patient_ids = dataset["patient_id"].drop_duplicates().to_numpy().copy()
    rng.shuffle(patient_ids)
    validation_count = max(1, int(np.floor(len(patient_ids) * validation_fraction)))
    validation_patients = set(patient_ids[:validation_count].tolist())
    train_mask = ~dataset["patient_id"].isin(validation_patients)
    validation_mask = dataset["patient_id"].isin(validation_patients)
    return dataset.loc[train_mask].reset_index(drop=True), dataset.loc[validation_mask].reset_index(drop=True)


def tensorize_conditioned_dataset(
    data: pd.DataFrame,
    scaler: StaticFeatureScaler | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert patient-conditioned rows into tensors for model training.

    Missing baseline covariates are imputed through the train-only scaler before
    the tensors are validated. This ensures the model sees finite inputs but keeps
    the raw patient-conditioned dataset untouched.
    """
    required = ["time_hours", "synthetic_stimulus", "synthetic_il6", "synthetic_crp", *CONDITIONED_STATIC_FEATURES]
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns for tensorization: {missing}")
    time = torch.tensor(data["time_hours"].to_numpy(dtype=np.float32), dtype=torch.float32)
    stimulus = torch.tensor(data["synthetic_stimulus"].to_numpy(dtype=np.float32), dtype=torch.float32)
    il6_target = torch.tensor(data["synthetic_il6"].to_numpy(dtype=np.float32), dtype=torch.float32)
    crp_target = torch.tensor(data["synthetic_crp"].to_numpy(dtype=np.float32), dtype=torch.float32)

    static_df = data[CONDITIONED_STATIC_FEATURES].copy()
    if scaler is None:
        scaler = StaticFeatureScaler(feature_names=CONDITIONED_STATIC_FEATURES).fit(static_df)
        static_array = scaler.transform(static_df)
    else:
        static_array = scaler.transform(static_df)

    static_tensor = torch.tensor(static_array, dtype=torch.float32)
    for tensor_name, tensor in {
        "time": time,
        "stimulus": stimulus,
        "static_features": static_tensor,
        "synthetic_il6": il6_target,
        "synthetic_crp": crp_target,
    }.items():
        if not torch.isfinite(tensor).all():
            raise ValueError(f"{tensor_name} contains NaN or infinite values")
    return time, stimulus, static_tensor, il6_target, crp_target


def data_loss_fn(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Mean squared error at the per-output level."""
    return torch.mean((predictions - targets) ** 2)


def train_conditioned_pinn(
    model: nn.Module | None,
    train_data: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    params: BiologicalParameters,
    epochs: int = 5,
    learning_rate: float = 1e-3,
    physics_weight: float = 1.0,
    device: str = "cpu",
) -> tuple[nn.Module, list[dict[str, float]]]:
    """Train the conditioned PINN with data and ODE residual losses."""
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if not isinstance(params, BiologicalParameters):
        raise ValueError("params must be a BiologicalParameters instance")

    time, stimulus, static_features, targets_il6, targets_crp = train_data
    if time.dim() != 1 or stimulus.dim() != 1 or static_features.dim() != 2:
        raise ValueError("train_data must contain 1D time/stimulus and 2D static features")

    if model is None:
        model = ConditionedPINN(hidden_dim=16, static_feature_names=CONDITIONED_STATIC_FEATURES)
    model = model.to(device=device)
    time = time.to(device=device)
    stimulus = stimulus.to(device=device)
    static_features = static_features.to(device=device)
    targets_il6 = targets_il6.to(device=device)
    targets_crp = targets_crp.to(device=device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    history: list[dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        model_input = torch.cat([time.unsqueeze(1), stimulus.unsqueeze(1), static_features], dim=1)
        predictions = model(model_input)
        data_term = data_loss_fn(predictions[:, 0], targets_il6) + data_loss_fn(predictions[:, 1], targets_crp)
        physics_term = physics_loss(model, time, stimulus, static_features, params)
        total_loss = data_term + physics_weight * physics_term
        total_loss.backward()
        optimizer.step()
        history.append(
            {
                "epoch": float(epoch),
                "total_loss": float(total_loss.detach().cpu()),
                "data_loss": float(data_term.detach().cpu()),
                "physics_loss": float(physics_term.detach().cpu()),
            }
        )
    return model, history


def main() -> None:
    """Run a small CPU smoke-training loop on a handful of patient trajectories."""
    dataset = load_conditioned_dataset(TRAINING_DATA_PATH)
    train_df, validation_df = patient_level_split(dataset, validation_fraction=0.2, seed=42)
    patient_subset = train_df["patient_id"].drop_duplicates().head(5).tolist()
    subset = train_df[train_df["patient_id"].isin(patient_subset)].reset_index(drop=True)
    scaler = StaticFeatureScaler(feature_names=CONDITIONED_STATIC_FEATURES).fit(subset)
    time, stimulus, static_features, target_il6, target_crp = tensorize_conditioned_dataset(subset, scaler=scaler)
    model = ConditionedPINN(hidden_dim=16, static_feature_names=CONDITIONED_STATIC_FEATURES)
    model.set_feature_scaler(scaler)
    params = BiologicalParameters(alpha_il6=1.8, k_il6=0.45, alpha_crp=1.2, k_crp=0.18)
    trained_model, history = train_conditioned_pinn(
        model=model,
        train_data=(time, stimulus, static_features, target_il6, target_crp),
        params=params,
        epochs=3,
        learning_rate=1e-3,
        physics_weight=1.0,
        device="cpu",
    )
    output = {
        "seed": 42,
        "train_patients": int(subset["patient_id"].nunique()),
        "validation_patients": int(validation_df["patient_id"].nunique()),
        "epochs": 3,
        "learning_rate": 1e-3,
        "physics_weight": 1.0,
        "history": history,
        "final_total_loss": history[-1]["total_loss"],
        "final_data_loss": history[-1]["data_loss"],
        "final_physics_loss": history[-1]["physics_loss"],
        "feature_names": CONDITIONED_STATIC_FEATURES,
        "feature_scaler": {
            "means": scaler.means,
            "stds": scaler.stds,
            "medians": scaler.medians,
        },
    }
    output_path = Path("outputs/predictions/conditioned_pinn_smoke.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(f"Saved smoke experiment artifact to {output_path}")
    print(f"Train patients: {subset['patient_id'].nunique()}")
    print(f"Validation patients: {validation_df['patient_id'].nunique()}")
    print(f"Final total loss: {history[-1]['total_loss']:.6f}")
    print(f"Parameters finite: {all(param.isfinite().all() for param in trained_model.parameters())}")


if __name__ == "__main__":
    main()
