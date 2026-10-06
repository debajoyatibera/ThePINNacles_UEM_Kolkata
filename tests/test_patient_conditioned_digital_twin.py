"""Focused tests for the NHANES-conditioned synthetic trajectory bridge."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from biotwin.data.generate_patient_conditioned_digital_twin import (
    BASELINE_FEATURES,
    DEFAULT_SOURCE_PATH,
    generate_patient_conditioned_digital_twin,
)


@pytest.fixture
def nhanes_sample(tmp_path: Path) -> tuple[Path, pd.DataFrame]:
    source = pd.read_csv(DEFAULT_SOURCE_PATH).head(16).copy()
    source_path = tmp_path / "nhanes_sample.csv"
    source.to_csv(source_path, index=False)
    return source_path, source


def _generate(
    tmp_path: Path,
    source_path: Path,
    *,
    name: str,
    seed: int = 42,
    hours: int = 24,
) -> tuple[pd.DataFrame, dict[str, object]]:
    return generate_patient_conditioned_digital_twin(
        source_path=source_path,
        output_path=tmp_path / f"{name}.csv",
        report_path=tmp_path / f"{name}.json",
        seed=seed,
        time_points_per_patient=hours,
    )


def test_output_schema_ids_hourly_grid_and_provenance(
    tmp_path: Path, nhanes_sample: tuple[Path, pd.DataFrame]
) -> None:
    source_path, source = nhanes_sample
    output, report = _generate(tmp_path, source_path, name="bridge")
    expected_columns = {
        "patient_id",
        "time_hours",
        *BASELINE_FEATURES,
        "synthetic_stimulus",
        "synthetic_il6",
        "synthetic_crp",
        "imputed_baseline_fields",
        "data_source",
        "dynamic_data_source",
        "target_type",
    }

    assert set(output.columns) == expected_columns
    assert set(output["patient_id"]) == set(source["patient_id"])
    assert not output.duplicated(["patient_id", "time_hours"]).any()
    assert output.groupby("patient_id").size().eq(24).all()
    assert all(
        np.array_equal(group["time_hours"].to_numpy(), np.arange(24))
        for _, group in output.groupby("patient_id", sort=False)
    )
    assert report["number_of_unique_nhanes_participants"] == len(source)
    assert report["number_of_trajectory_rows"] == len(source) * 24
    assert report["time_points_per_participant"] == 24
    assert "crp" not in report["baseline_feature_list"]
    assert output["data_source"].eq("NHANES_2021_2023").all()
    assert output["dynamic_data_source"].eq("SYNTHETIC").all()
    assert output["target_type"].eq("MODEL_SIMULATED").all()
    assert output.loc[output["patient_id"] == source["patient_id"].iloc[0], "imputed_baseline_fields"].eq("NONE").all()
    assert report["no_observed_longitudinal_nhanes_data_created"] is True
    assert "stimulus is generated" in report["data_provenance_statement"]

    legacy_ids = set(
        pd.read_csv("data/synthetic/synthetic_inflammatory_trajectories.csv")["patient_id"]
    )
    assert set(output["patient_id"]).isdisjoint(legacy_ids)


def test_seed_controls_deterministic_synthetic_dynamics(
    tmp_path: Path, nhanes_sample: tuple[Path, pd.DataFrame]
) -> None:
    source_path, _ = nhanes_sample
    first, first_report = _generate(tmp_path, source_path, name="first", seed=42, hours=12)
    repeat, repeat_report = _generate(tmp_path, source_path, name="repeat", seed=42, hours=12)
    changed, changed_report = _generate(tmp_path, source_path, name="changed", seed=99, hours=12)

    pd.testing.assert_frame_equal(first, repeat)
    assert first_report == repeat_report
    assert not first[["synthetic_stimulus", "synthetic_il6", "synthetic_crp"]].equals(
        changed[["synthetic_stimulus", "synthetic_il6", "synthetic_crp"]]
    )
    assert first_report["random_seed"] == 42
    assert changed_report["random_seed"] == 99


def test_dynamic_values_are_finite_nonnegative_and_bounded(
    tmp_path: Path, nhanes_sample: tuple[Path, pd.DataFrame]
) -> None:
    source_path, _ = nhanes_sample
    output, _ = _generate(tmp_path, source_path, name="bounded", hours=24)
    dynamic = output[["synthetic_stimulus", "synthetic_il6", "synthetic_crp"]]

    assert np.isfinite(dynamic.to_numpy()).all()
    assert dynamic.ge(0).all().all()
    assert output["synthetic_stimulus"].le(5.0).all()


def test_real_baselines_are_preserved_and_input_is_not_modified(
    tmp_path: Path, nhanes_sample: tuple[Path, pd.DataFrame]
) -> None:
    source_path, source = nhanes_sample
    original_bytes = source_path.read_bytes()
    output, _ = _generate(tmp_path, source_path, name="preserved", hours=8)
    baseline_rows = output.drop_duplicates("patient_id").set_index("patient_id")[BASELINE_FEATURES]
    expected = source.set_index("patient_id")[BASELINE_FEATURES]

    pd.testing.assert_frame_equal(baseline_rows, expected)
    assert source_path.read_bytes() == original_bytes


def test_missing_baselines_are_imputed_only_for_conditioning_and_reported(
    tmp_path: Path, nhanes_sample: tuple[Path, pd.DataFrame]
) -> None:
    source_path, source = nhanes_sample
    imputed_features = ["bmi", "waist_cm", "systolic_bp", "diastolic_bp", "hba1c", "sedentary_minutes"]
    source.loc[0, imputed_features] = np.nan
    source.loc[0, "sex"] = np.nan
    source.to_csv(source_path, index=False)

    output, report = _generate(tmp_path, source_path, name="missing", hours=8)
    first_patient = output.loc[output["patient_id"] == source.loc[0, "patient_id"]]
    strategy = report["imputation_strategy"]

    assert first_patient[imputed_features + ["sex"]].isna().all().all()
    expected_imputed_fields = ";".join(
        feature for feature in BASELINE_FEATURES if feature == "sex" or feature in imputed_features
    )
    assert first_patient["imputed_baseline_fields"].eq(expected_imputed_fields).all()
    assert strategy["feature_counts"]["bmi"] == 1
    assert strategy["feature_counts"]["sex"] == 1
    assert strategy["missing_baseline_rows"] > 0
    assert strategy["row_indicator_column"] == "imputed_baseline_fields"


def test_duplicate_nhanes_ids_are_rejected(
    tmp_path: Path, nhanes_sample: tuple[Path, pd.DataFrame]
) -> None:
    source_path, source = nhanes_sample
    source.loc[1, "patient_id"] = source.loc[0, "patient_id"]
    source.to_csv(source_path, index=False)

    with pytest.raises(ValueError, match="patient_id values must be unique"):
        _generate(tmp_path, source_path, name="duplicate", hours=4)