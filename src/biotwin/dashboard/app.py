"""Streamlit demo for the CRP Digital Twin research PoC.

This module keeps UI code separate from reusable model/data logic so the demo can
be exercised by tests without launching a browser.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from biotwin.models.conditioned_pinn import (
    CONDITIONED_STATIC_FEATURES,
    ConditionedPINN,
    load_scaler_artifact,
    StaticFeatureScaler,
    TimeNormalizedConditionedPINN,
)
from biotwin.models.event_predictor import (
    DEFAULT_FEATURES,
    EventPredictor,
)

try:
    import streamlit as st
except ImportError:  # pragma: no cover - handled by runtime if Streamlit is absent.
    st = None

REPO_ROOT = Path(__file__).resolve().parents[3]
BASELINE_DATA_PATH = REPO_ROOT / "data/processed/nhanes_modeling_cohort.csv"
DYNAMIC_DATA_PATH = REPO_ROOT / "data/processed/patient_conditioned_digital_twin.csv"
CHECKPOINT_PATH = REPO_ROOT / "outputs/models/conditioned_pinn_full.pt"
SCALER_PATH = REPO_ROOT / "outputs/models/conditioned_pinn_scaler.json"
SPLIT_PATH = REPO_ROOT / "outputs/predictions/conditioned_experiment_split.json"
EVENT_MODEL_PATH = REPO_ROOT / "outputs/models/nhanes_adverse_event_predictor_compatible.joblib"
EVENT_DATA_PATH = REPO_ROOT / "data/synthetic/nhanes_adverse_event_latest.csv"


def load_baseline_data() -> pd.DataFrame:
    """Load the real NHANES processed cohort used for participant selection."""
    data = pd.read_csv(BASELINE_DATA_PATH)
    required = ["patient_id", "age", "sex", "bmi", "waist_cm", "systolic_bp", "diastolic_bp", "hba1c", "sedentary_minutes"]
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError(f"NHANES cohort is missing required baseline columns: {missing}")
    return data


def load_dynamic_data() -> pd.DataFrame:
    """Load the patient-conditioned synthetic Digital Twin trajectories."""
    data = pd.read_csv(DYNAMIC_DATA_PATH)
    required = ["patient_id", "time_hours", "synthetic_stimulus", "synthetic_il6", "synthetic_crp"]
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError(f"Patient-conditioned dataset is missing required columns: {missing}")
    return data.sort_values(["patient_id", "time_hours"]).reset_index(drop=True)


def load_persisted_scaler() -> StaticFeatureScaler:
    """Load the exact reproducibility artifact that was fitted on the training split."""
    if not SCALER_PATH.exists():
        raise FileNotFoundError(
            f"Persisted static-feature scaler artifact is missing at {SCALER_PATH}. "
            "Restore outputs/models/conditioned_pinn_scaler.json before dashboard inference."
        )
    scaler = load_scaler_artifact(SCALER_PATH)
    if list(scaler.feature_names) != list(CONDITIONED_STATIC_FEATURES):
        raise ValueError(
            "Persisted scaler is incompatible with the conditioned PINN: "
            f"expected {CONDITIONED_STATIC_FEATURES}, found {scaler.feature_names}"
        )
    return scaler


def load_event_predictor() -> EventPredictor:
    """Load the trained synthetic 2-hour adverse-event predictor."""
    if not EVENT_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Adverse-event model is missing at {EVENT_MODEL_PATH}. "
            "Train the event predictor before dashboard inference."
        )

    predictor = EventPredictor.load(EVENT_MODEL_PATH)

    if not predictor.feature_names:
        raise ValueError("Persisted adverse-event predictor has no features")

    return predictor


def load_model() -> tuple[torch.nn.Module, StaticFeatureScaler, dict[str, Any]]:
    """Load the existing trained conditioned PINN checkpoint without retraining."""
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")
    metadata = checkpoint["metadata"]
    static_feature_names = list(metadata.get("static_feature_names", CONDITIONED_STATIC_FEATURES))
    model_base = ConditionedPINN(hidden_dim=int(metadata["hidden_dim"]), static_feature_names=static_feature_names)
    scaler = load_persisted_scaler()
    model_base.set_feature_scaler(scaler)
    model = TimeNormalizedConditionedPINN(
        model_base,
        time_min=float(metadata["time_min"]),
        time_max=float(metadata["time_max"]),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, scaler, metadata


def prepare_patient_features(patient_id: float | int) -> dict[str, Any]:
    """Prepare the real baseline and synthetic dynamic inputs for one patient."""
    patient_key = float(patient_id)
    cohort = load_baseline_data()
    dynamic_df = load_dynamic_data()
    baseline_row = cohort[cohort["patient_id"] == patient_key].copy()
    if baseline_row.empty:
        raise ValueError(f"No NHANES row found for patient_id={patient_id}")
    baseline_row = baseline_row.iloc[0]
    dynamic_patient = dynamic_df[dynamic_df["patient_id"] == patient_key].sort_values("time_hours").reset_index(drop=True)
    if dynamic_patient.empty:
        raise ValueError(f"No synthetic dynamic trajectory found for patient_id={patient_id}")
    scaler = load_persisted_scaler()
    static_df = pd.DataFrame([baseline_row])[CONDITIONED_STATIC_FEATURES].copy()
    static_features = scaler.transform(static_df).astype(np.float32)
    return {
        "patient_id": patient_key,
        "baseline_row": baseline_row.to_dict(),
        "dynamic_df": dynamic_patient,
        "static_features": static_features,
    }


def run_patient_simulation(patient_id: float | int) -> dict[str, Any]:
    """Run deterministic inference for the selected patient using the trained checkpoint."""
    model, scaler, metadata = load_model()
    patient_features = prepare_patient_features(patient_id)
    dynamic_df = patient_features["dynamic_df"]
    baseline_features = pd.DataFrame([patient_features["baseline_row"]])[CONDITIONED_STATIC_FEATURES].copy()
    static_matrix = np.repeat(scaler.transform(baseline_features), len(dynamic_df), axis=0).astype(np.float32)
    model_input = np.column_stack(
        [
            dynamic_df["time_hours"].to_numpy(dtype=np.float32),
            dynamic_df["synthetic_stimulus"].to_numpy(dtype=np.float32),
            static_matrix,
        ]
    )
    model.eval()
    with torch.no_grad():
        outputs = model(torch.tensor(model_input, dtype=torch.float32))
    il6_pred = outputs[:, 0].cpu().numpy().astype(np.float32)
    crp_pred = outputs[:, 1].cpu().numpy().astype(np.float32)
    if not np.isfinite(il6_pred).all() or not np.isfinite(crp_pred).all():
        raise FloatingPointError("Model-simulated IL-6/CRP trajectories contain non-finite values")
    return {
        "model": model,
        "metadata": metadata,
        "patient_id": float(patient_id),
        "baseline_row": patient_features["baseline_row"],
        "dynamic_df": dynamic_df,
        "static_features": static_matrix,
        "il6_pred": il6_pred,
        "crp_pred": crp_pred,
    }


def run_event_prediction(patient_id: float | int) -> dict[str, Any]:
    """Predict the synthetic adverse-event probability within the next 2 hours."""
    predictor = load_event_predictor()

    event_df = pd.read_csv(EVENT_DATA_PATH)

    # The event dataset uses synthetic IDs such as P000001, while the
    # NHANES dashboard participant selector uses numeric IDs.
    numeric_id = int(float(patient_id))
    patient_key = numeric_id

    patient_data = (
        event_df[event_df["patient_id"] == patient_key]
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    if patient_data.empty:
        raise ValueError(
            f"No adverse-event feature data found for patient_id={patient_id} "
            f"(mapped to {patient_key})"
        )

    latest_row = patient_data.iloc[[-1]].copy()

    missing = [
        feature
        for feature in predictor.feature_names
        if feature not in latest_row.columns
    ]
    if missing:
        raise ValueError(
            f"Adverse-event dataset is missing predictor features: {missing}"
        )

    probability = float(predictor.predict_probability(latest_row)[0])
    predicted_event = int(predictor.predict(latest_row)[0])

    if not 0.0 <= probability <= 1.0:
        raise ValueError(
            f"Event probability is outside [0, 1]: {probability}"
        )

    return {
        "patient_id": patient_id,
        "event_patient_id": patient_key,
        "timestamp": latest_row["timestamp"].iloc[0],
        "forecast_end_hours": float(latest_row["forecast_end_hours"].iloc[0]),
        "event_probability": probability,
        "predicted_event": predicted_event,
        "feature_row": latest_row.iloc[0].to_dict(),
    }



def calculate_simulated_trend(values: np.ndarray | pd.Series) -> str:
    """Classify the simulated inflammation trajectory using the trajectory direction."""
    series = np.asarray(values, dtype=np.float32).reshape(-1)
    if series.size == 0:
        raise ValueError("Trend input cannot be empty")
    if series.size == 1:
        return "relatively stable"
    delta = float(series[-1] - series[0])
    if np.isclose(delta, 0.0, atol=1e-6):
        return "relatively stable"
    return "increasing" if delta > 0 else "decreasing"


def _format_value(value: Any) -> str:
    if pd.isna(value):
        return "Unavailable"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.2f}"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return str(value)


def _decode_sex(value: Any) -> str:
    """Display the NHANES sex code in a human-readable form without modifying the data."""
    try:
        sex_code = int(float(value))
    except (TypeError, ValueError):
        return "Unknown"
    if sex_code == 1:
        return "Male"
    if sex_code == 2:
        return "Female"
    return "Unknown"


def _render_ui() -> None:
    if st is None:  # pragma: no cover - guarded at runtime
        raise RuntimeError("Streamlit is not installed in this environment.")

    st.title("CRP Digital Twin — Inflammation Simulation PoC")
    st.caption("Physics-Informed Digital Twin using NHANES baseline + synthetic dynamic inputs")
    st.caption(
        "Digital Twin concept: a participant's real NHANES baseline profile conditions a physics-informed model, "
        "while a synthetic dynamic stimulus drives a simulated IL-6/CRP trajectory."
    )

    cohort = load_baseline_data()
    patient_ids = sorted(pd.to_numeric(cohort["patient_id"], errors="coerce").dropna().astype(int).unique().tolist())
    selected_patient_id = st.selectbox("Select NHANES participant", patient_ids, index=0)

    baseline_row = cohort[cohort["patient_id"] == float(selected_patient_id)].iloc[0].to_dict()
    dynamic_df = load_dynamic_data()
    patient_dynamic = dynamic_df[dynamic_df["patient_id"] == float(selected_patient_id)].sort_values("time_hours").reset_index(drop=True)
    result = run_patient_simulation(selected_patient_id)
    event_result = run_event_prediction(selected_patient_id)

    st.markdown("### Patient Selection")
    st.write(f"Selected participant: {selected_patient_id}")

    st.markdown("### REAL NHANES BASELINE")
    st.caption("Baseline clinical values are from the NHANES 2021–2023 cross-sectional dataset.")
    columns = st.columns(4)
    baseline_fields = [
        ("Age", baseline_row.get("age")),
        ("Sex", _decode_sex(baseline_row.get("sex"))),
        ("BMI", baseline_row.get("bmi")),
        ("Waist (cm)", baseline_row.get("waist_cm")),
        ("Systolic BP", baseline_row.get("systolic_bp")),
        ("Diastolic BP", baseline_row.get("diastolic_bp")),
        ("HbA1c", baseline_row.get("hba1c")),
        ("Sedentary minutes", baseline_row.get("sedentary_minutes")),
        ("Observed baseline CRP", baseline_row.get("crp")),
    ]
    for index, (label, value) in enumerate(baseline_fields):
        with columns[index % 4]:
            st.metric(label=label, value=_format_value(value))
            if label == "Observed baseline CRP":
                st.caption("Real NHANES baseline observation")

    st.markdown("### SYNTHETIC DYNAMIC INPUT")
    st.caption("The dynamic stimulus shown here is synthetic/model-generated for this competition PoC; it is not an observed wearable stream.")
    stimulus_chart = pd.DataFrame(
        {
            "Time (hours since simulation start)": patient_dynamic["time_hours"].to_numpy(dtype=np.float32),
            "Synthetic dynamic stimulus": patient_dynamic["synthetic_stimulus"].to_numpy(dtype=np.float32),
        }
    ).set_index("Time (hours since simulation start)")
    st.line_chart(stimulus_chart, x_label="Time (hours since simulation start)")
    st.dataframe(patient_dynamic[["time_hours", "synthetic_stimulus"]].head(10), use_container_width=True)

    st.markdown("### DIGITAL TWIN SIMULATION")
    il6_series = pd.Series(result["il6_pred"], index=patient_dynamic["time_hours"].to_numpy())
    crp_series = pd.Series(result["crp_pred"], index=patient_dynamic["time_hours"].to_numpy())
    il6_chart = pd.DataFrame(
        {
            "Time (hours since simulation start)": patient_dynamic["time_hours"].to_numpy(dtype=np.float32),
            "Model-simulated IL-6": il6_series.to_numpy(dtype=np.float32),
        }
    ).set_index("Time (hours since simulation start)")
    crp_chart = pd.DataFrame(
        {
            "Time (hours since simulation start)": patient_dynamic["time_hours"].to_numpy(dtype=np.float32),
            "Model-simulated CRP": crp_series.to_numpy(dtype=np.float32),
        }
    ).set_index("Time (hours since simulation start)")
    st.markdown("#### MODEL-SIMULATED IL-6")
    st.line_chart(il6_chart, x_label="Time (hours since simulation start)")
    st.markdown("#### MODEL-SIMULATED CRP")
    st.line_chart(crp_chart, x_label="Time (hours since simulation start)")
    st.markdown("### 2-HOUR SIMULATED ADVERSE-EVENT FORECAST")
    st.caption(
        "Real NHANES baseline + synthetic wearable features are used to estimate the probability "
        "of a simulated adverse event within the next 2 hours. This is a competition PoC, "
        "not a clinical prediction."
    )

    event_columns = st.columns(3)

    with event_columns[0]:
        st.metric(
            "Event probability",
            f"{event_result['event_probability'] * 100:.1f}%"
        )

    with event_columns[1]:
        alert_label = (
            "Simulated event alert"
            if event_result["predicted_event"] == 1
            else "No simulated event alert"
        )
        st.metric(
            "Forecast",
            alert_label
        )

    with event_columns[2]:
        st.metric(
            "Forecast horizon",
            "2 hours"
        )

    st.caption(
        f"Latest synthetic observation: {event_result['timestamp']} "
        f"· Event data ID: {event_result['event_patient_id']}"
    )

    if event_result["predicted_event"] == 1:
        st.warning(
            "Simulated adverse-event alert: the model predicts elevated probability "
            "of the synthetic target within the next 2 hours."
        )
    else:
        st.success(
            "No simulated adverse-event alert at the current model threshold."
        )

    st.info(
        "Important: this event target, wearable stream, and prediction are synthetic. "
        "The model has not been clinically validated and must not be used for medical decisions."
    )

    st.info(
        "Controlled synthetic evaluation: The controlled experiment did not demonstrate predictive "
        "improvement from patient conditioning."
    )

    st.markdown("### MODEL OUTPUT TREND")
    combined_signal = il6_series.to_numpy() + crp_series.to_numpy()
    trend = calculate_simulated_trend(combined_signal)
    trend_label = "Stable" if trend == "relatively stable" else trend.title()
    st.info(f"Model-simulated trajectory: {trend_label}")
    st.caption("Not a clinical indicator.")

    st.markdown("### DATA & MODEL PROVENANCE")
    st.write("REAL: NHANES baseline clinical observations.")
    st.write("SYNTHETIC: Wearable telemetry and dynamic stimulus.")
    st.write("MODEL-PREDICTED: 2-hour simulated adverse-event probability.")
    st.write("MODEL-SIMULATED: IL-6 and CRP trajectories.")
    st.write("NHANES does not provide longitudinal IL-6/CRP or continuous wearable telemetry in this project.")
    st.write("This is a competition/research PoC and is not clinically validated.")

    st.markdown("### LIMITATIONS")
    st.write("- synthetic dynamic data")
    st.write("- model-simulated IL-6/CRP")
    st.write("- no clinical validation")
    st.write("- cross-sectional NHANES baseline")


def main() -> None:
    if st is None:
        raise RuntimeError("Streamlit is not installed. Install the project environment before running the dashboard.")
    _render_ui()


if __name__ == "__main__":
    main()
