"""Minimal patient-conditioned PINN extension for the CRP/IL-6 system.

The model accepts physical time, synthetic stimulus, and static patient
baseline features as inputs, returning IL-6 and CRP predictions. Static features
are kept constant across all time points for a patient and are normalized using
training-only statistics to avoid data leakage.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src.biotwin.models.biological_ode import BiologicalParameters

CONDITIONED_STATIC_FEATURES = [
    "age",
    "sex",
    "bmi",
    "waist_cm",
    "systolic_bp",
    "diastolic_bp",
    "hba1c",
    "sedentary_minutes",
]


def encode_sex(values: Sequence[float] | pd.Series | np.ndarray) -> np.ndarray:
    """Encode the binary NHANES sex indicator deterministically as 0/1.

    The processed patient-conditioned dataset uses sex codes 1 and 2. For the
    model we map 1 -> 0 and 2 -> 1 to keep the feature binary and stable.
    """
    series = pd.Series(values, copy=False)
    numeric = pd.to_numeric(series, errors="coerce")
    encoded = np.zeros(len(numeric), dtype=np.float32)
    valid_mask = numeric.notna().to_numpy()
    if valid_mask.any():
        encoded[valid_mask] = np.where(numeric.loc[valid_mask].to_numpy() == 2.0, 1.0, 0.0).astype(np.float32)
    return encoded


@dataclass
class StaticFeatureScaler:
    """Normalize static baseline covariates with train-only statistics."""

    feature_names: list[str] = field(default_factory=lambda: list(CONDITIONED_STATIC_FEATURES))
    means: dict[str, float] = field(default_factory=dict)
    stds: dict[str, float] = field(default_factory=dict)
    medians: dict[str, float] = field(default_factory=dict)

    def fit(self, dataset: pd.DataFrame) -> "StaticFeatureScaler":
        missing = [feature for feature in self.feature_names if feature not in dataset.columns]
        if missing:
            raise ValueError(f"Dataset is missing static feature columns: {missing}")
        means: dict[str, float] = {}
        stds: dict[str, float] = {}
        medians: dict[str, float] = {}
        for feature in self.feature_names:
            values = pd.to_numeric(dataset[feature], errors="coerce")
            if feature == "sex":
                encoded = pd.Series(encode_sex(values), index=dataset.index)
                values = encoded
                fill_value = float(values.median()) if values.notna().any() else 0.0
            else:
                fill_value = float(values.median()) if values.notna().any() else 0.0
                values = values.fillna(fill_value)
            medians[feature] = float(fill_value)
            means[feature] = float(values.mean())
            std = float(values.std(ddof=0))
            if not np.isfinite(std) or std == 0.0:
                std = 1.0
            stds[feature] = float(std)
        self.means = means
        self.stds = stds
        self.medians = medians
        return self

    def transform(self, dataset: pd.DataFrame) -> np.ndarray:
        if not self.means:
            raise ValueError("StaticFeatureScaler must be fit before use")
        matrix = np.empty((len(dataset), len(self.feature_names)), dtype=np.float32)
        for index, feature in enumerate(self.feature_names):
            values = pd.to_numeric(dataset[feature], errors="coerce")
            if feature == "sex":
                values = pd.Series(encode_sex(values), index=dataset.index)
                fill_value = self.medians.get(feature, float(values.median()))
                values = values.fillna(fill_value)
            else:
                fill_value = self.medians.get(feature, float(values.median()))
                values = values.fillna(fill_value)
            matrix[:, index] = ((values.to_numpy(dtype=np.float32) - self.means[feature]) / self.stds[feature]).astype(np.float32)
        return matrix

    def to_dict(
        self,
        train_patient_ids: Sequence[int | float] | None = None,
        time_min: float | None = None,
        time_max: float | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "feature_names": list(self.feature_names),
            "means": {feature: float(self.means.get(feature, 0.0)) for feature in self.feature_names},
            "stds": {feature: float(self.stds.get(feature, 1.0)) for feature in self.feature_names},
            "medians": {feature: float(self.medians.get(feature, 0.0)) for feature in self.feature_names},
            "zero_variance_policy": "set_to_one",
            "normalization": "standardize_static_features",
        }
        if train_patient_ids is not None:
            payload["train_patient_ids"] = [float(patient_id) for patient_id in train_patient_ids]
        if time_min is not None:
            payload["time_min"] = float(time_min)
        if time_max is not None:
            payload["time_max"] = float(time_max)
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "StaticFeatureScaler":
        feature_names = list(payload.get("feature_names") or CONDITIONED_STATIC_FEATURES)
        scaler = cls(feature_names=feature_names)
        scaler.means = {feature: float(payload.get("means", {}).get(feature, 0.0)) for feature in feature_names}
        scaler.stds = {feature: float(payload.get("stds", {}).get(feature, 1.0)) for feature in feature_names}
        scaler.medians = {feature: float(payload.get("medians", {}).get(feature, 0.0)) for feature in feature_names}
        return scaler

    def transform_tensor(self, dataset: pd.DataFrame) -> torch.Tensor:
        return torch.tensor(self.transform(dataset), dtype=torch.float32)


def save_scaler_artifact(
    path: str | Path,
    scaler: StaticFeatureScaler,
    train_patient_ids: Sequence[int | float] | None = None,
    time_min: float | None = None,
    time_max: float | None = None,
) -> Path:
    """Persist the exact train-only static feature normalization state used by the conditioned PINN."""
    artifact_path = Path(path)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    payload = scaler.to_dict(train_patient_ids=train_patient_ids, time_min=time_min, time_max=time_max)
    artifact_path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return artifact_path


def load_scaler_artifact(path: str | Path) -> StaticFeatureScaler:
    """Load a persisted static-feature scaler artifact and validate its compatibility."""
    artifact_path = Path(path)
    if not artifact_path.exists():
        raise FileNotFoundError(
            f"Persisted static-feature scaler artifact is missing at {artifact_path}. "
            "Restore the reproducibility artifact before dashboard inference."
        )
    with artifact_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    scaler = StaticFeatureScaler.from_dict(payload)
    if list(scaler.feature_names) != list(CONDITIONED_STATIC_FEATURES):
        raise ValueError(
            "Persisted scaler feature names do not match the conditioned PINN schema: "
            f"expected {CONDITIONED_STATIC_FEATURES}, found {scaler.feature_names}"
        )
    all_values = list(scaler.means.values()) + list(scaler.stds.values()) + list(scaler.medians.values())
    if not all(np.isfinite(value) for value in all_values):
        raise ValueError("Persisted scaler contains non-finite normalization statistics.")
    return scaler


class ConditionedPINN(nn.Module):
    """Feed-forward PINN mapping [time, stimulus, static_features] -> [IL6, CRP]."""

    def __init__(self, hidden_dim: int = 32, static_feature_names: list[str] | None = None) -> None:
        super().__init__()
        self.static_feature_names = list(static_feature_names or CONDITIONED_STATIC_FEATURES)
        self.input_dim = 2 + len(self.static_feature_names)
        self.network = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 2),
        )
        self.static_feature_scaler = StaticFeatureScaler(feature_names=self.static_feature_names)
        self.static_feature_scaler.means = {feature: 0.0 for feature in self.static_feature_names}
        self.static_feature_scaler.stds = {feature: 1.0 for feature in self.static_feature_names}
        self.static_feature_scaler.medians = {feature: 0.0 for feature in self.static_feature_names}

    def set_feature_scaler(self, scaler: StaticFeatureScaler) -> None:
        self.static_feature_scaler = scaler

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return predictions shaped [N, 2] for IL6 and CRP."""
        if not torch.is_tensor(inputs):
            raise TypeError("inputs must be a torch.Tensor")
        if inputs.dim() != 2 or inputs.shape[1] != self.input_dim:
            raise ValueError(
                f"inputs must have shape [N, {self.input_dim}] with columns "
                "[time, stimulus, static patient features]"
            )
        if not torch.isfinite(inputs).all():
            raise ValueError("model inputs must be finite")
        return self.network(inputs)


class TimeNormalizedConditionedPINN(nn.Module):
    """Wrap a conditioned PINN with affine time scaling in [0, 1]."""

    def __init__(self, model: nn.Module, time_min: float, time_max: float) -> None:
        super().__init__()
        if not torch.isfinite(torch.tensor([time_min, time_max])).all() or time_max <= time_min:
            raise ValueError("time_max must be finite and greater than finite time_min")
        self.model = model
        self.static_feature_names = list(getattr(model, "static_feature_names", CONDITIONED_STATIC_FEATURES))
        self.register_buffer("time_min", torch.tensor(float(time_min), dtype=torch.float32))
        self.register_buffer("time_range", torch.tensor(float(time_max - time_min), dtype=torch.float32))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(inputs):
            raise TypeError("inputs must be a torch.Tensor")
        expected_dim = 2 + len(self.static_feature_names)
        if inputs.dim() != 2 or inputs.shape[1] != expected_dim:
            raise ValueError(
                f"inputs must have shape [N, {expected_dim}] with columns "
                "[time, stimulus, static patient features]"
            )
        normalized_time = (inputs[:, 0] - self.time_min) / self.time_range
        normalized_inputs = torch.cat(
            [normalized_time.unsqueeze(1), inputs[:, 1:2], inputs[:, 2:]],
            dim=1,
        )
        return self.model(normalized_inputs)


def physics_residual(
    model: nn.Module,
    time: torch.Tensor,
    stimulus: torch.Tensor,
    static_features: torch.Tensor,
    params: BiologicalParameters,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute ODE residuals with derivatives with respect to physical time."""
    if not isinstance(params, BiologicalParameters):
        raise ValueError("params must be a BiologicalParameters instance")
    if not torch.is_tensor(time) or not torch.is_tensor(stimulus):
        raise TypeError("time and stimulus must be torch tensors")
    if not torch.is_tensor(static_features):
        raise TypeError("static_features must be a torch.Tensor")
    if time.shape != stimulus.shape:
        raise ValueError("time and stimulus must have the same shape")
    if time.dim() != 1:
        raise ValueError("time and stimulus must be one-dimensional tensors")
    if static_features.dim() != 2 or static_features.shape[0] != time.shape[0]:
        raise ValueError("static_features must have shape [N, F] and match the batch size")

    time = time.to(dtype=torch.float32).clone().detach().requires_grad_(True)
    stimulus = stimulus.to(dtype=torch.float32)
    static_features = static_features.to(dtype=torch.float32)
    model_input = torch.cat([time.unsqueeze(1), stimulus.unsqueeze(1), static_features], dim=1)

    model.eval()
    with torch.set_grad_enabled(True):
        predictions = model(model_input)
        if predictions.shape != (time.shape[0], 2):
            raise ValueError("model output must have shape [N, 2]")

        il6_pred = predictions[:, 0]
        crp_pred = predictions[:, 1]

        il6_dt = torch.autograd.grad(
            outputs=il6_pred,
            inputs=time,
            grad_outputs=torch.ones_like(il6_pred),
            retain_graph=True,
            create_graph=True,
        )[0]
        crp_dt = torch.autograd.grad(
            outputs=crp_pred,
            inputs=time,
            grad_outputs=torch.ones_like(crp_pred),
            retain_graph=True,
            create_graph=True,
        )[0]

    residual_il6 = il6_dt - (params.alpha_il6 * stimulus - params.k_il6 * il6_pred)
    residual_crp = crp_dt - (params.alpha_crp * il6_pred - params.k_crp * crp_pred)
    return residual_il6, residual_crp


def physics_loss(
    model: nn.Module,
    time: torch.Tensor,
    stimulus: torch.Tensor,
    static_features: torch.Tensor,
    params: BiologicalParameters,
) -> torch.Tensor:
    """Return the mean squared residual loss used by the conditioned PINN."""
    residual_il6, residual_crp = physics_residual(model, time, stimulus, static_features, params)
    return torch.mean(residual_il6 ** 2) + torch.mean(residual_crp ** 2)
