"""Tests for the synthetic wearable telemetry generator."""

import numpy as np
import pandas as pd
import pytest

from src.biotwin.data.generate_wearable import generate_synthetic_wearable


@pytest.fixture
def baseline() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "patient_id": [f"P{number:06d}" for number in range(1, 11)],
            "age": [24, 74, 65, 50, 49, 35, 60, 42, 55, 31],
            "bmi": [27.2, 29.3, 17.4, 30.5, 31.7, 23.0, 26.0, 22.0, 28.0, 25.0],
            "physical_activity_level": [
                "moderate", "moderate", "moderate", "low", "moderate",
                "high", "low", "high", "moderate", "high",
            ],
            "smoking_status": [
                "never", "former", "never", "never", "never",
                "current", "former", "never", "current", "never",
            ],
        }
    )


@pytest.fixture
def telemetry(baseline: pd.DataFrame) -> pd.DataFrame:
    return generate_synthetic_wearable(baseline, days=2, seed=42)


def test_generates_expected_number_of_rows(telemetry: pd.DataFrame) -> None:
    assert len(telemetry) == 10 * 48


def test_all_baseline_patient_ids_are_used_without_extras(
    baseline: pd.DataFrame, telemetry: pd.DataFrame
) -> None:
    assert set(telemetry["patient_id"]) == set(baseline["patient_id"])


def test_each_patient_has_expected_observation_count(telemetry: pd.DataFrame) -> None:
    assert (telemetry.groupby("patient_id").size() == 48).all()


def test_hourly_frequency_has_24_observations_per_day(telemetry: pd.DataFrame) -> None:
    observations_per_day = telemetry.groupby(
        ["patient_id", telemetry["timestamp"].dt.date]
    ).size()

    assert (observations_per_day == 24).all()


def test_timestamps_and_elapsed_hours_are_ordered_per_patient(
    telemetry: pd.DataFrame,
) -> None:
    for _, patient_rows in telemetry.groupby("patient_id", sort=False):
        assert patient_rows["timestamp"].is_monotonic_increasing
        assert patient_rows["hours_since_start"].is_monotonic_increasing
        assert patient_rows["hours_since_start"].iloc[0] == 0
        assert patient_rows["hours_since_start"].iloc[-1] == 47
        assert patient_rows["timestamp"].iloc[0] == pd.Timestamp("2026-01-01 00:00:00")
        assert patient_rows["timestamp"].diff().dropna().eq(pd.Timedelta(hours=1)).all()


def test_required_values_are_present_and_within_bounds(telemetry: pd.DataFrame) -> None:
    required_columns = [
        "patient_id",
        "timestamp",
        "hours_since_start",
        "sleep_hours",
        "hrv_rmssd",
        "resting_heart_rate",
        "stress_score",
        "physical_activity",
    ]

    assert not telemetry[required_columns].isna().any().any()
    assert (telemetry["hrv_rmssd"] > 0).all()
    assert (telemetry["resting_heart_rate"] > 0).all()
    assert telemetry["stress_score"].between(0, 100).all()
    assert (telemetry["physical_activity"] >= 0).all()
    assert (telemetry["sleep_hours"] >= 0).all()


def test_same_seed_produces_identical_data(baseline: pd.DataFrame) -> None:
    first = generate_synthetic_wearable(baseline, days=2, seed=42)
    second = generate_synthetic_wearable(baseline, days=2, seed=42)

    pd.testing.assert_frame_equal(first, second)


def test_different_seeds_produce_different_data(baseline: pd.DataFrame) -> None:
    first = generate_synthetic_wearable(baseline, days=2, seed=42)
    second = generate_synthetic_wearable(baseline, days=2, seed=43)

    assert not first.equals(second)


def test_sleep_and_activity_show_day_night_pattern(telemetry: pd.DataFrame) -> None:
    by_hour = telemetry.assign(hour=telemetry["timestamp"].dt.hour).groupby("hour").mean(
        numeric_only=True
    )
    night_hours = [22, 23, 0, 1, 2, 3, 4, 5]
    day_hours = [10, 11, 12, 13, 14, 15, 16]

    assert by_hour.loc[night_hours, "sleep_hours"].mean() > 5 * by_hour.loc[
        day_hours, "sleep_hours"
    ].mean()
    assert by_hour.loc[day_hours, "physical_activity"].mean() > 5 * by_hour.loc[
        night_hours, "physical_activity"
    ].mean()


def test_signals_vary_over_time_and_between_patients(telemetry: pd.DataFrame) -> None:
    signals = [
        "sleep_hours",
        "hrv_rmssd",
        "resting_heart_rate",
        "stress_score",
        "physical_activity",
    ]

    assert (telemetry[signals].nunique() > 1).all()
    assert (telemetry.groupby("patient_id")[signals].std().min() > 0).all()


@pytest.mark.parametrize("days", [0, -1, 1.5, True])
def test_rejects_invalid_days(baseline: pd.DataFrame, days: object) -> None:
    with pytest.raises(ValueError, match="days must be a positive integer"):
        generate_synthetic_wearable(baseline, days=days, seed=42)  # type: ignore[arg-type]


def test_rejects_missing_or_duplicate_patient_ids() -> None:
    missing_id = pd.DataFrame({"patient_id": [None]})
    duplicate_id = pd.DataFrame({"patient_id": ["P000001", "P000001"]})

    with pytest.raises(ValueError, match="patient_id"):
        generate_synthetic_wearable(missing_id, days=1)
    with pytest.raises(ValueError, match="patient_id"):
        generate_synthetic_wearable(duplicate_id, days=1)


def test_rejects_frequency_that_does_not_divide_day(baseline: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="frequency"):
        generate_synthetic_wearable(baseline, days=1, frequency="7h")