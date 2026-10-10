"""Tests for the synthetic adverse-event forecasting dataset builder."""

import numpy as np
import pandas as pd
import pytest

from biotwin.data.generate_adverse_events import build_adverse_event_dataset


@pytest.fixture
def baseline() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "patient_id": [f"P{number:06d}" for number in range(1, 11)],
            "age": [24, 74, 65, 50, 49, 35, 60, 42, 55, 31],
            "sex": [
                "male", "female", "female", "male", "female",
                "male", "female", "male", "female", "male",
            ],
            "bmi": [27.2, 29.3, 17.4, 30.5, 31.7, 23.0, 26.0, 22.0, 28.0, 25.0],
            "systolic_bp": [120, 145, 118, 135, 140, 125, 130, 115, 138, 122],
            "diastolic_bp": [78, 92, 72, 88, 90, 80, 84, 70, 86, 76],
            "fasting_glucose": [92, 125, 88, 110, 118, 95, 105, 85, 115, 90],
            "hba1c": [5.1, 6.4, 4.9, 5.9, 6.2, 5.2, 5.7, 4.8, 6.0, 5.0],
            "smoking_status": [
                "never", "former", "never", "never", "never",
                "current", "former", "never", "current", "never",
            ],
            "physical_activity_level": [
                "moderate", "moderate", "moderate", "low", "moderate",
                "high", "low", "high", "moderate", "high",
            ],
        }
    )


@pytest.fixture
def wearable(baseline: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for patient_number, patient_id in enumerate(baseline["patient_id"], start=1):
        for hour in range(24):
            rows.append(
                {
                    "patient_id": patient_id,
                    "timestamp": pd.Timestamp("2026-01-01") + pd.Timedelta(hours=hour),
                    "hours_since_start": float(hour),
                    "sleep_hours": 0.5 if hour < 6 else 0.05,
                    "hrv_rmssd": 50.0 - 0.1 * hour + patient_number * 0.2,
                    "resting_heart_rate": 70.0 + 0.2 * hour + patient_number * 0.1,
                    "stress_score": 30.0 + 0.5 * hour,
                    "physical_activity": 1.0 if 8 <= hour <= 18 else 0.2,
                }
            )

    return pd.DataFrame(rows)


@pytest.fixture
def dataset(
    baseline: pd.DataFrame,
    wearable: pd.DataFrame,
) -> pd.DataFrame:
    return build_adverse_event_dataset(
        baseline_df=baseline,
        wearable_df=wearable,
        observation_window_hours=6,
        forecast_horizon_hours=2,
        seed=42,
    )


def test_builds_expected_patient_and_row_counts(dataset: pd.DataFrame) -> None:
    assert dataset["patient_id"].nunique() == 10
    assert (dataset.groupby("patient_id").size() == 17).all()
    assert len(dataset) == 10 * 17


def test_uses_only_valid_history_and_forecast_horizon(
    dataset: pd.DataFrame,
) -> None:
    assert dataset["hours_since_start"].min() == 5.0
    assert dataset["hours_since_start"].max() == 21.0
    assert (dataset["forecast_end_hours"] == dataset["hours_since_start"] + 2).all()


def test_required_event_columns_are_present(dataset: pd.DataFrame) -> None:
    required = {
        "patient_id",
        "timestamp",
        "hours_since_start",
        "hrv_rmssd_mean_6h",
        "resting_heart_rate_mean_6h",
        "sleep_hours_mean_6h",
        "stress_score_mean_6h",
        "physical_activity_mean_6h",
        "hrv_rmssd_slope_6h",
        "resting_heart_rate_slope_6h",
        "stress_score_slope_6h",
        "physical_activity_slope_6h",
        "event_probability_simulated",
        "event_in_next_2h",
        "age",
        "sex",
        "bmi",
        "systolic_bp",
        "diastolic_bp",
        "fasting_glucose",
        "hba1c",
        "smoking_status",
        "physical_activity_level",
    }

    assert required.issubset(dataset.columns)


def test_no_missing_values(dataset: pd.DataFrame) -> None:
    assert not dataset.isna().any().any()


def test_event_target_is_binary(dataset: pd.DataFrame) -> None:
    assert set(dataset["event_in_next_2h"].unique()).issubset({0, 1})


def test_event_probability_is_valid(dataset: pd.DataFrame) -> None:
    probabilities = dataset["event_probability_simulated"]

    assert np.isfinite(probabilities).all()
    assert probabilities.between(0.0, 1.0).all()


def test_patient_and_timestamp_pairs_are_unique(
    dataset: pd.DataFrame,
) -> None:
    assert not dataset.duplicated(["patient_id", "timestamp"]).any()


def test_patients_have_no_overlap_with_each_other(dataset: pd.DataFrame) -> None:
    assert dataset["patient_id"].nunique() == 10
    assert dataset.groupby("patient_id").size().nunique() == 1


def test_same_seed_produces_identical_dataset(
    baseline: pd.DataFrame,
    wearable: pd.DataFrame,
) -> None:
    first = build_adverse_event_dataset(
        baseline_df=baseline,
        wearable_df=wearable,
        seed=42,
    )
    second = build_adverse_event_dataset(
        baseline_df=baseline,
        wearable_df=wearable,
        seed=42,
    )

    pd.testing.assert_frame_equal(first, second)


def test_different_seed_changes_event_target(
    baseline: pd.DataFrame,
    wearable: pd.DataFrame,
) -> None:
    first = build_adverse_event_dataset(
        baseline_df=baseline,
        wearable_df=wearable,
        seed=42,
    )
    second = build_adverse_event_dataset(
        baseline_df=baseline,
        wearable_df=wearable,
        seed=43,
    )

    assert not first["event_in_next_2h"].equals(
        second["event_in_next_2h"]
    )


def test_future_observations_do_not_change_current_features(
    baseline: pd.DataFrame,
    wearable: pd.DataFrame,
) -> None:
    original = build_adverse_event_dataset(
        baseline_df=baseline,
        wearable_df=wearable,
        seed=42,
    )

    modified_wearable = wearable.copy()
    future_mask = modified_wearable["hours_since_start"] >= 18

    modified_wearable.loc[future_mask, "hrv_rmssd"] += 1000.0
    modified_wearable.loc[future_mask, "resting_heart_rate"] += 1000.0
    modified_wearable.loc[future_mask, "stress_score"] = 0.0
    modified_wearable.loc[future_mask, "physical_activity"] += 1000.0

    modified = build_adverse_event_dataset(
        baseline_df=baseline,
        wearable_df=modified_wearable,
        seed=42,
    )

    current_mask = original["hours_since_start"] < 18

    feature_columns = [
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

    pd.testing.assert_frame_equal(
        original.loc[current_mask, feature_columns].reset_index(drop=True),
        modified.loc[current_mask, feature_columns].reset_index(drop=True),
    )


def test_rejects_unknown_wearable_patient_ids(
    baseline: pd.DataFrame,
    wearable: pd.DataFrame,
) -> None:
    invalid = wearable.copy()
    invalid.loc[0, "patient_id"] = "P999999"

    with pytest.raises(ValueError, match="absent from baseline"):
        build_adverse_event_dataset(baseline, invalid)


@pytest.mark.parametrize(
    ("observation_window", "forecast_horizon"),
    [(5, 2), (6, 1), (7, 3)],
)
def test_rejects_noncompetition_configuration(
    baseline: pd.DataFrame,
    wearable: pd.DataFrame,
    observation_window: int,
    forecast_horizon: int,
) -> None:
    with pytest.raises(ValueError, match="competition PoC configuration"):
        build_adverse_event_dataset(
            baseline_df=baseline,
            wearable_df=wearable,
            observation_window_hours=observation_window,
            forecast_horizon_hours=forecast_horizon,
        )
