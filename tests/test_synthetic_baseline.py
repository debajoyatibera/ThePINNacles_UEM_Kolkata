"""Tests for the synthetic clinical baseline generator."""

import numpy as np
import pandas as pd
import pytest

from src.biotwin.data.generate_nhanes import generate_synthetic_baseline


REQUIRED_COLUMNS = {
    "patient_id",
    "age",
    "sex",
    "height_cm",
    "weight_kg",
    "bmi",
    "systolic_bp",
    "diastolic_bp",
    "resting_heart_rate",
    "fasting_glucose",
    "hba1c",
    "triglycerides",
    "hdl",
    "ldl",
    "smoking_status",
    "alcohol_use",
    "physical_activity_level",
    "baseline_crp",
    "baseline_il6",
}


def test_generates_requested_number_of_patients_and_columns() -> None:
    baseline = generate_synthetic_baseline(n_patients=125, seed=7)

    assert len(baseline) == 125
    assert REQUIRED_COLUMNS.issubset(baseline.columns)


def test_patient_ids_are_unique_and_well_formed() -> None:
    baseline = generate_synthetic_baseline(n_patients=100, seed=7)

    assert baseline["patient_id"].is_unique
    assert baseline["patient_id"].str.fullmatch(r"P\d{6}").all()
    assert baseline["patient_id"].iloc[0] == "P000001"
    assert baseline["patient_id"].iloc[-1] == "P000100"


def test_bmi_is_calculated_from_height_and_weight() -> None:
    baseline = generate_synthetic_baseline(n_patients=100, seed=7)
    expected_bmi = baseline["weight_kg"] / (baseline["height_cm"] / 100.0) ** 2

    np.testing.assert_allclose(baseline["bmi"], expected_bmi, rtol=1e-12)


def test_numeric_baseline_features_vary_across_patients() -> None:
    baseline = generate_synthetic_baseline(n_patients=100, seed=7)

    assert baseline["ldl"].nunique() > 1


def test_required_values_are_present_and_positive_where_required() -> None:
    baseline = generate_synthetic_baseline(n_patients=100, seed=7)

    assert not baseline[list(REQUIRED_COLUMNS)].isna().any().any()
    assert (baseline[["height_cm", "weight_kg", "bmi", "baseline_crp", "baseline_il6"]] > 0).all().all()
    assert baseline["age"].between(18, 90).all()
    assert all(pd.api.types.is_numeric_dtype(baseline[column]) for column in REQUIRED_COLUMNS - {
        "patient_id",
        "sex",
        "smoking_status",
        "alcohol_use",
        "physical_activity_level",
    })


def test_same_seed_produces_identical_data() -> None:
    first = generate_synthetic_baseline(n_patients=100, seed=42)
    second = generate_synthetic_baseline(n_patients=100, seed=42)

    pd.testing.assert_frame_equal(first, second)


def test_different_seeds_produce_different_populations() -> None:
    first = generate_synthetic_baseline(n_patients=100, seed=42)
    second = generate_synthetic_baseline(n_patients=100, seed=43)

    assert not first.equals(second)


def test_categorical_values_use_documented_categories() -> None:
    baseline = generate_synthetic_baseline(n_patients=250, seed=7)

    assert set(baseline["sex"]).issubset({"female", "male"})
    assert set(baseline["smoking_status"]).issubset({"never", "former", "current"})
    assert set(baseline["alcohol_use"]).issubset({"none", "occasional", "regular"})
    assert set(baseline["physical_activity_level"]).issubset({"low", "moderate", "high"})


@pytest.mark.parametrize("n_patients", [0, -1, 1.5, True])
def test_rejects_invalid_patient_counts(n_patients: object) -> None:
    with pytest.raises(ValueError, match="n_patients must be a positive integer"):
        generate_synthetic_baseline(n_patients=n_patients)  # type: ignore[arg-type]