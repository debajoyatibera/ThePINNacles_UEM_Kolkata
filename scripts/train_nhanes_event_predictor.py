"""Train the NHANES-linked synthetic 2-hour adverse-event predictor.

REAL:
- NHANES participant IDs and baseline observations.

SYNTHETIC:
- wearable telemetry
- adverse-event target

The target is a synthetic competition PoC outcome and is not clinically
validated or suitable for medical decision-making.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler


DATA_PATH = Path(
    "data/synthetic/nhanes_adverse_event_dataset.csv"
)

MODEL_PATH = Path(
    "outputs/models/nhanes_adverse_event_predictor.joblib"
)

PREDICTION_PATH = Path(
    "outputs/predictions/nhanes_adverse_event_predictions.csv"
)

METRICS_PATH = Path(
    "outputs/predictions/nhanes_adverse_event_metrics.txt"
)

TARGET = "event_in_next_2h"

FEATURES = [
    "age",
    "bmi",
    "systolic_bp",
    "diastolic_bp",
    "hba1c",
    "sedentary_minutes",
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


def main() -> None:
    df = pd.read_csv(DATA_PATH)

    required = set(FEATURES) | {
        "patient_id",
        "timestamp",
        "forecast_end_hours",
        TARGET,
        "event_probability_simulated",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Dataset is missing required columns: {sorted(missing)}"
        )

    if df[FEATURES].isna().any().any():
        raise ValueError("Predictor features contain missing values")

    if not np.isfinite(
        df[FEATURES].to_numpy(dtype=float)
    ).all():
        raise ValueError(
            "Predictor features contain non-finite values"
        )

    if TARGET in FEATURES:
        raise ValueError("Target leakage detected")

    forbidden = {
        "event_probability_simulated",
        "forecast_end_hours",
    }

    leakage = forbidden.intersection(FEATURES)

    if leakage:
        raise ValueError(
            f"Forbidden leakage columns in features: {sorted(leakage)}"
        )

    patients = (
        df["patient_id"]
        .drop_duplicates()
        .sort_values()
        .to_numpy()
        .copy()
    )

    rng = np.random.default_rng(42)
    rng.shuffle(patients)

    split_index = int(len(patients) * 0.8)

    train_patients = set(
        patients[:split_index]
    )

    test_patients = set(
        patients[split_index:]
    )

    train = df[
        df["patient_id"].isin(train_patients)
    ].copy()

    test = df[
        df["patient_id"].isin(test_patients)
    ].copy()

    X_train = train[FEATURES].to_numpy(dtype=float)
    y_train = train[TARGET].to_numpy(dtype=int)

    X_test = test[FEATURES].to_numpy(dtype=float)
    y_test = test[TARGET].to_numpy(dtype=int)

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42,
    )

    model.fit(
        X_train_scaled,
        y_train,
    )

    probabilities = model.predict_proba(
        X_test_scaled
    )[:, 1]

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    roc_auc = roc_auc_score(
        y_test,
        probabilities,
    )

    pr_auc = average_precision_score(
        y_test,
        probabilities,
    )

    baseline_probability = float(
        y_train.mean()
    )

    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    PREDICTION_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    artifact = {
        "scaler": scaler,
        "model": model,
        "features": FEATURES,
        "target": TARGET,
        "horizon_hours": 2,
        "synthetic_target": True,
        "data_source": "NHANES + synthetic wearable",
    }

    joblib.dump(
        artifact,
        MODEL_PATH,
    )

    prediction_output = test[
        [
            "patient_id",
            "timestamp",
            "forecast_end_hours",
            TARGET,
        ]
    ].copy()

    prediction_output[
        "predicted_probability"
    ] = probabilities

    prediction_output[
        "predicted_event"
    ] = predictions

    prediction_output.to_csv(
        PREDICTION_PATH,
        index=False,
    )

    metrics = [
        "NHANES-linked synthetic adverse-event predictor",
        "",
        f"Dataset rows: {len(df)}",
        f"Patients: {df['patient_id'].nunique()}",
        f"Train rows: {len(train)}",
        f"Test rows: {len(test)}",
        f"Train patients: {len(train_patients)}",
        f"Test patients: {len(test_patients)}",
        f"Training event rate: {y_train.mean():.6f}",
        f"Test event rate: {y_test.mean():.6f}",
        f"Constant baseline PR-AUC: {baseline_probability:.6f}",
        f"Model ROC-AUC: {roc_auc:.6f}",
        f"Model PR-AUC: {pr_auc:.6f}",
        "",
        "IMPORTANT:",
        "The adverse-event target is synthetic.",
        "Metrics are a competition/research PoC result.",
        "This model is not clinically validated.",
    ]

    METRICS_PATH.write_text(
        "\n".join(metrics) + "\n"
    )

    print("\nNHANES-linked event predictor trained.")
    print(f"Dataset rows: {len(df)}")
    print(f"Patients: {df['patient_id'].nunique()}")
    print(f"Train rows: {len(train)}")
    print(f"Test rows: {len(test)}")
    print(f"Train patients: {len(train_patients)}")
    print(f"Test patients: {len(test_patients)}")
    print(f"Training event rate: {y_train.mean():.4f}")
    print(f"Test event rate: {y_test.mean():.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"PR-AUC: {pr_auc:.4f}")
    print(f"Model: {MODEL_PATH}")
    print(f"Predictions: {PREDICTION_PATH}")
    print(f"Metrics: {METRICS_PATH}")


if __name__ == "__main__":
    main()
