"""Tests for preparing the observed-CRP NHANES modeling cohort."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.biotwin.data.prepare_modeling_cohort import (
    IMPORTANT_VARIABLES,
    prepare_modeling_cohort,
)


@pytest.fixture
def source_baseline() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "patient_id": [101, 102, 103, 104],
            "age": [35, 42, np.nan, 58],
            "sex": [1, 2, 2, 1],
            "weight_kg": [72.0, np.nan, 81.0, 69.0],
            "height_cm": [175.0, 164.0, 181.0, 168.0],
            "bmi": [23.5, 27.0, 24.7, 24.4],
            "waist_cm": [79.0, 88.0, 86.0, 83.0],
            "systolic_bp": [118.0, 130.0, 126.0, 122.0],
            "diastolic_bp": [74.0, 82.0, 78.0, 76.0],
            "hba1c": [5.1, 5.7, np.nan, 5.4],
            "crp": [1.2, np.nan, 0.7, np.nan],
            "vigorous_activity_frequency": [2, 3, 1, 2],
            "vigorous_activity_frequency_unit": ["W"] * 4,
            "vigorous_activity_duration": [30, 45, 20, 30],
            "moderate_activity_frequency": [3, 2, 4, 1],
            "moderate_activity_frequency_unit": ["W"] * 4,
            "moderate_activity_duration": [20, 25, 30, 15],
            "sedentary_minutes": [300, 420, np.nan, 360],
        }
    )


@pytest.fixture
def prepared_files(
    tmp_path: Path, source_baseline: pd.DataFrame
) -> tuple[Path, Path, Path, pd.DataFrame, dict[str, object], bytes]:
    source_path = tmp_path / "nhanes_baseline.csv"
    cohort_path = tmp_path / "nhanes_modeling_cohort.csv"
    report_path = tmp_path / "nhanes_modeling_cohort_report.json"
    source_baseline.to_csv(source_path, index=False)
    original_source_bytes = source_path.read_bytes()

    cohort, report = prepare_modeling_cohort(source_path, cohort_path, report_path)
    return source_path, cohort_path, report_path, cohort, report, original_source_bytes


def test_missing_crp_rows_are_excluded_and_observed_rows_retained(
    prepared_files: tuple[Path, Path, Path, pd.DataFrame, dict[str, object], bytes]
) -> None:
    _, _, _, cohort, _, _ = prepared_files

    assert cohort["patient_id"].tolist() == [101, 103]
    assert cohort["crp"].tolist() == [1.2, 0.7]


def test_crp_is_not_imputed_or_replaced(
    source_baseline: pd.DataFrame,
    prepared_files: tuple[Path, Path, Path, pd.DataFrame, dict[str, object], bytes],
) -> None:
    _, _, _, cohort, _, _ = prepared_files

    assert cohort["crp"].tolist() == source_baseline.loc[source_baseline["crp"].notna(), "crp"].tolist()
    assert cohort["crp"].isna().sum() == 0


def test_patient_ids_are_unique_and_crp_observed_is_true(
    prepared_files: tuple[Path, Path, Path, pd.DataFrame, dict[str, object], bytes]
) -> None:
    _, _, _, cohort, _, _ = prepared_files

    assert cohort["patient_id"].is_unique
    assert cohort["patient_id"].notna().all()
    assert cohort["crp_observed"].eq(True).all()
    assert cohort["crp_observed"].dtype == bool


def test_baseline_source_csv_is_unchanged(
    prepared_files: tuple[Path, Path, Path, pd.DataFrame, dict[str, object], bytes]
) -> None:
    source_path, _, _, _, _, original_source_bytes = prepared_files

    assert source_path.read_bytes() == original_source_bytes


def test_json_report_contains_required_counts_and_missingness(
    prepared_files: tuple[Path, Path, Path, pd.DataFrame, dict[str, object], bytes]
) -> None:
    _, _, report_path, cohort, report, _ = prepared_files
    saved_report = json.loads(report_path.read_text(encoding="utf-8"))
    required_keys = {
        "source_row_count",
        "modeling_cohort_row_count",
        "excluded_row_count",
        "crp_observed_count",
        "crp_missing_count",
        "percentage_baseline_with_observed_crp",
        "unique_patient_id_count",
        "duplicate_patient_id_count",
        "missing_value_counts",
    }

    assert required_keys <= saved_report.keys()
    assert saved_report == report
    assert report["source_row_count"] == 4
    assert report["modeling_cohort_row_count"] == 2
    assert report["excluded_row_count"] == 2
    assert report["crp_observed_count"] == 2
    assert report["crp_missing_count"] == 2
    assert report["percentage_baseline_with_observed_crp"] == 50.0
    assert report["unique_patient_id_count"] == 2
    assert report["duplicate_patient_id_count"] == 0
    assert set(report["missing_value_counts"]) == set(IMPORTANT_VARIABLES)
    assert report["missing_value_counts"]["crp"] == 0
    assert report["missing_value_counts"]["age"] == 1
    assert len(cohort.columns) == len(pd.read_csv(prepared_files[1]).columns)


def test_output_keeps_all_baseline_columns_and_adds_status(
    source_baseline: pd.DataFrame,
    prepared_files: tuple[Path, Path, Path, pd.DataFrame, dict[str, object], bytes],
) -> None:
    _, cohort_path, _, _, _, _ = prepared_files
    saved_cohort = pd.read_csv(cohort_path)

    assert set(source_baseline.columns) <= set(saved_cohort.columns)
    assert "crp_observed" in saved_cohort.columns
    assert len(saved_cohort.columns) == len(source_baseline.columns) + 1


def test_source_with_duplicate_patient_ids_is_rejected(
    tmp_path: Path, source_baseline: pd.DataFrame
) -> None:
    source_baseline.loc[1, "patient_id"] = source_baseline.loc[0, "patient_id"]
    source_path = tmp_path / "duplicate_ids.csv"
    source_baseline.to_csv(source_path, index=False)

    with pytest.raises(ValueError, match="patient_id values must be unique"):
        prepare_modeling_cohort(
            source_path,
            tmp_path / "cohort.csv",
            tmp_path / "report.json",
        )