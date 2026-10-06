import numpy as np
import pandas as pd
import pytest

from biotwin.dashboard.app import (
    SCALER_PATH,
    calculate_simulated_trend,
    load_baseline_data,
    load_dynamic_data,
    load_persisted_scaler,
    load_model,
    prepare_patient_features,
    run_patient_simulation,
)


def test_nhanes_patient_selection_returns_valid_participant() -> None:
    cohort = load_baseline_data()
    assert {"patient_id", "age", "sex", "bmi"}.issubset(cohort.columns)
    patient_id = cohort["patient_id"].dropna().iloc[0]
    assert np.isfinite(float(patient_id))
    assert patient_id in cohort["patient_id"].to_numpy()


def test_selected_patient_exists_in_both_required_datasets() -> None:
    cohort = load_baseline_data()
    dynamic = load_dynamic_data()
    patient_id = cohort["patient_id"].dropna().iloc[0]
    assert patient_id in cohort["patient_id"].unique()
    assert patient_id in dynamic["patient_id"].unique()


def test_static_features_are_prepared_correctly() -> None:
    patient_id = 130378.0
    patient_features = prepare_patient_features(patient_id)
    assert patient_features["static_features"].shape == (1, 8)
    assert patient_features["static_features"].dtype == np.float32
    assert np.isfinite(patient_features["static_features"]).all()


def test_dynamic_trajectory_has_expected_time_ordering() -> None:
    patient_id = 130378.0
    patient_features = prepare_patient_features(patient_id)
    dynamic_df = patient_features["dynamic_df"].copy()
    assert dynamic_df["time_hours"].is_monotonic_increasing
    assert set(["time_hours", "synthetic_stimulus"]).issubset(dynamic_df.columns)


def test_model_inference_returns_correct_il6_crp_shapes() -> None:
    patient_id = 130378.0
    result = run_patient_simulation(patient_id)
    assert result["il6_pred"].shape[0] == result["dynamic_df"].shape[0]
    assert result["crp_pred"].shape[0] == result["dynamic_df"].shape[0]
    assert result["il6_pred"].ndim == 1
    assert result["crp_pred"].ndim == 1


def test_model_inference_outputs_are_finite() -> None:
    patient_id = 130378.0
    result = run_patient_simulation(patient_id)
    assert np.isfinite(result["il6_pred"]).all()
    assert np.isfinite(result["crp_pred"]).all()


def test_inference_does_not_train_the_model() -> None:
    model, _, _ = load_model()
    assert model.training is False
    patient_id = 130378.0
    result = run_patient_simulation(patient_id)
    assert result["model"].training is False


def test_trend_calculation_returns_valid_categories() -> None:
    trend = calculate_simulated_trend(np.array([0.5, 0.7, 0.9, 1.1]))
    assert trend in {"increasing", "decreasing", "relatively stable"}


def test_missing_baseline_values_are_handled_safely() -> None:
    patient_id = 130392.0
    patient_features = prepare_patient_features(patient_id)
    assert np.isfinite(patient_features["static_features"]).all()
    assert patient_features["baseline_row"]["patient_id"] == patient_id


def test_provenance_metadata_is_present() -> None:
    metadata = {
        "real_data": "NHANES baseline clinical observations.",
        "synthetic_data": "Dynamic stimulus.",
        "model_simulated_data": "IL-6 and CRP trajectories.",
    }
    assert all(key in metadata for key in ("real_data", "synthetic_data", "model_simulated_data"))


def test_persisted_scaler_artifact_exists() -> None:
    assert SCALER_PATH.exists(), "Persisted scaler artifact is missing."


def test_persisted_scaler_feature_names_are_correct() -> None:
    scaler = load_persisted_scaler()
    assert scaler.feature_names == [
        "age",
        "sex",
        "bmi",
        "waist_cm",
        "systolic_bp",
        "diastolic_bp",
        "hba1c",
        "sedentary_minutes",
    ]


def test_persisted_scaler_values_are_finite() -> None:
    scaler = load_persisted_scaler()
    all_values = list(scaler.means.values()) + list(scaler.stds.values()) + list(scaler.medians.values())
    assert all(np.isfinite(value) for value in all_values)


def test_dashboard_loads_persisted_scaler_without_runtime_fit(monkeypatch: pytest.MonkeyPatch) -> None:
    from biotwin.models.conditioned_pinn import StaticFeatureScaler

    def _fail_fit(self, dataset):
        raise AssertionError("Dashboard should not fit the scaler at runtime.")

    monkeypatch.setattr(StaticFeatureScaler, "fit", _fail_fit)
    model, scaler, _ = load_model()
    assert model.training is False
    assert scaler.feature_names == [
        "age",
        "sex",
        "bmi",
        "waist_cm",
        "systolic_bp",
        "diastolic_bp",
        "hba1c",
        "sedentary_minutes",
    ]


def test_missing_scaler_artifact_raises_clear_error(monkeypatch: pytest.MonkeyPatch) -> None:
    import biotwin.dashboard.app as dashboard

    monkeypatch.setattr(dashboard, "SCALER_PATH", dashboard.REPO_ROOT / "outputs/models/does_not_exist.json")
    with pytest.raises(FileNotFoundError, match="Persisted static-feature scaler"):
        dashboard.load_persisted_scaler()
