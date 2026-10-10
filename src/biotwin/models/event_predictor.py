"""Interpretable synthetic adverse-event prediction model.

This module provides a Logistic Regression baseline for predicting the
synthetically generated cardiometabolic deterioration event within the next
2 hours.

The model is a competition/research PoC and is not clinically validated.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


EVENT_TARGET = "event_in_next_2h"

DEFAULT_FEATURES = [
    "age",
    "bmi",
    "systolic_bp",
    "diastolic_bp",
    "fasting_glucose",
    "hba1c",
    "hrv_rmssd_mean_6h",
    "hrv_rmssd_std_6h",
    "resting_heart_rate_mean_6h",
    "resting_heart_rate_std_6h",
    "sleep_hours_mean_6h",
    "sleep_hours_std_6h",
    "stress_score_mean_6h",
    "stress_score_std_6h",
    "physical_activity_mean_6h",
    "physical_activity_std_6h",
    "hrv_rmssd_slope_6h",
    "resting_heart_rate_slope_6h",
    "stress_score_slope_6h",
    "physical_activity_slope_6h",
]


@dataclass
class EventPredictor:
    """Trainable Logistic Regression event predictor."""

    feature_names: list[str]
    classifier: Pipeline | None = None

    def __post_init__(self) -> None:
        if not self.feature_names:
            raise ValueError("feature_names must not be empty")

        if len(set(self.feature_names)) != len(self.feature_names):
            raise ValueError("feature_names must be unique")

    def _validate_dataset(self, dataset: pd.DataFrame) -> None:
        """Validate the training/inference dataframe."""
        if not isinstance(dataset, pd.DataFrame):
            raise ValueError("dataset must be a pandas DataFrame")

        required = set(self.feature_names) | {EVENT_TARGET}
        missing = required - set(dataset.columns)

        if missing:
            raise ValueError(
                f"dataset is missing required columns: {sorted(missing)}"
            )

        if dataset.empty:
            raise ValueError("dataset must not be empty")

        features = dataset[self.feature_names]

        if not features.apply(
            lambda column: pd.api.types.is_numeric_dtype(column)
        ).all():
            raise ValueError("all predictor features must be numeric")

        if not np.isfinite(features.to_numpy(dtype=float)).all():
            raise ValueError("predictor features must contain only finite values")

        target = pd.to_numeric(dataset[EVENT_TARGET], errors="coerce")

        if target.isna().any():
            raise ValueError("event target must not contain missing values")

        if not target.isin([0, 1]).all():
            raise ValueError("event target must contain only 0 and 1")

    def fit(
        self,
        dataset: pd.DataFrame,
        class_weight: str | None = "balanced",
        random_state: int = 42,
    ) -> "EventPredictor":
        """Fit a standardized Logistic Regression classifier."""
        self._validate_dataset(dataset)

        X = dataset[self.feature_names].to_numpy(dtype=float)
        y = dataset[EVENT_TARGET].to_numpy(dtype=int)

        if np.unique(y).size < 2:
            raise ValueError("event target must contain both classes")

        self.classifier = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=1000,
                        class_weight=class_weight,
                        random_state=random_state,
                    ),
                ),
            ]
        )

        self.classifier.fit(X, y)
        return self

    def _require_fitted(self) -> Pipeline:
        """Return the fitted pipeline or raise a clear error."""
        if self.classifier is None:
            raise RuntimeError("EventPredictor must be fitted before prediction")
        return self.classifier

    def predict_probability(
        self,
        dataset: pd.DataFrame,
    ) -> np.ndarray:
        """Return probability of the simulated event within the next 2 hours."""
        model = self._require_fitted()

        if not isinstance(dataset, pd.DataFrame):
            raise ValueError("dataset must be a pandas DataFrame")

        missing = set(self.feature_names) - set(dataset.columns)
        if missing:
            raise ValueError(
                f"dataset is missing required predictor columns: {sorted(missing)}"
            )

        X = dataset[self.feature_names].to_numpy(dtype=float)

        if not np.isfinite(X).all():
            raise ValueError("predictor features must contain only finite values")

        probabilities = model.predict_proba(X)[:, 1]

        if not np.isfinite(probabilities).all():
            raise FloatingPointError(
                "event predictor produced non-finite probabilities"
            )

        return probabilities.astype(float)

    def predict(
        self,
        dataset: pd.DataFrame,
        threshold: float = 0.5,
    ) -> np.ndarray:
        """Return binary event predictions using the supplied probability threshold."""
        if not np.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be between 0 and 1")

        probabilities = self.predict_probability(dataset)
        return (probabilities >= threshold).astype(np.int8)

    def save(self, path: str | Path) -> Path:
        """Persist the fitted predictor."""
        model = self._require_fitted()

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "feature_names": list(self.feature_names),
            "classifier": model,
            "target": EVENT_TARGET,
            "model_type": "standardized_logistic_regression",
            "prediction_horizon_hours": 2,
            "synthetic_target": True,
        }

        joblib.dump(payload, output_path)
        return output_path

    @classmethod
    def load(cls, path: str | Path) -> "EventPredictor":
        """Load a previously saved predictor."""
        input_path = Path(path)

        if not input_path.exists():
            raise FileNotFoundError(f"Model artifact not found: {input_path}")

        payload = joblib.load(input_path)

        if not isinstance(payload, dict):
            raise ValueError("invalid event predictor artifact")

        feature_names = payload.get("feature_names")
        classifier = payload.get("classifier")

        if not isinstance(feature_names, list) or not feature_names:
            raise ValueError("event predictor artifact has invalid feature names")

        if classifier is None:
            raise ValueError("event predictor artifact has no classifier")

        predictor = cls(
            feature_names=[str(feature) for feature in feature_names],
            classifier=classifier,
        )

        return predictor


def create_default_event_predictor() -> EventPredictor:
    """Return the project's default Logistic Regression predictor."""
    return EventPredictor(feature_names=list(DEFAULT_FEATURES))
