"""Prepare a CRP-observed modeling cohort from the real NHANES baseline.

NHANES provides cross-sectional, participant-level baseline measurements here;
it does not provide continuous wearable data or longitudinal CRP measurements.
CRP is retained only when observed in the NHANES laboratory data. Missing CRP is
never imputed or replaced with synthetic values. Longitudinal CRP/IL-6
trajectories remain a separate synthetic or model-simulated component.
"""

import json
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_SOURCE_PATH = Path("data/processed/nhanes_baseline.csv")
DEFAULT_COHORT_PATH = Path("data/processed/nhanes_modeling_cohort.csv")
DEFAULT_REPORT_PATH = Path("data/processed/nhanes_modeling_cohort_report.json")

IMPORTANT_VARIABLES = [
    "age",
    "sex",
    "bmi",
    "weight_kg",
    "height_cm",
    "waist_cm",
    "systolic_bp",
    "diastolic_bp",
    "hba1c",
    "crp",
    "sedentary_minutes",
]


def _validate_patient_ids(data: pd.DataFrame, dataset_name: str) -> None:
    if "patient_id" not in data.columns:
        raise ValueError(f"{dataset_name} must contain patient_id")
    if data["patient_id"].isna().any():
        raise ValueError(f"{dataset_name} patient_id values must not be missing")
    if data["patient_id"].duplicated().any():
        raise ValueError(f"{dataset_name} patient_id values must be unique")


def _validate_source(source: pd.DataFrame) -> None:
    required_columns = {"patient_id", "crp", *IMPORTANT_VARIABLES}
    missing_columns = sorted(required_columns - set(source.columns))
    if missing_columns:
        raise ValueError(f"Source baseline is missing required columns: {missing_columns}")
    _validate_patient_ids(source, "Source baseline")


def _validate_cohort(cohort: pd.DataFrame, source_row_count: int) -> None:
    _validate_patient_ids(cohort, "Modeling cohort")
    if "crp" not in cohort.columns:
        raise ValueError("Modeling cohort must contain crp")
    if cohort["crp"].isna().any():
        raise ValueError("Modeling cohort crp values must not be missing")
    if (cohort["crp"] <= 0).any():
        raise ValueError("Modeling cohort crp values must be positive")
    if len(cohort) > source_row_count:
        raise ValueError("Modeling cohort cannot contain more rows than the source baseline")
    if "crp_observed" not in cohort.columns or not cohort["crp_observed"].eq(True).all():
        raise ValueError("crp_observed must be True for every modeling-cohort row")


def prepare_modeling_cohort(
    source_path: Path = DEFAULT_SOURCE_PATH,
    cohort_path: Path = DEFAULT_COHORT_PATH,
    report_path: Path = DEFAULT_REPORT_PATH,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Write and return the rows with observed CRP plus their quality report.

    Every baseline column is retained; only rows with missing ``crp`` are
    excluded. The source file is read but never overwritten.
    """
    source_path = Path(source_path)
    cohort_path = Path(cohort_path)
    report_path = Path(report_path)
    resolved_paths = [source_path.resolve(), cohort_path.resolve(), report_path.resolve()]
    if len(set(resolved_paths)) != len(resolved_paths):
        raise ValueError("Source, cohort, and report paths must be different")

    source = pd.read_csv(source_path)
    _validate_source(source)

    cohort = source.loc[source["crp"].notna()].copy()
    cohort["crp_observed"] = True
    _validate_cohort(cohort, len(source))

    source_row_count = int(len(source))
    cohort_row_count = int(len(cohort))
    observed_count = int(source["crp"].notna().sum())
    missing_count = int(source["crp"].isna().sum())
    report: dict[str, Any] = {
        "source_row_count": source_row_count,
        "modeling_cohort_row_count": cohort_row_count,
        "excluded_row_count": source_row_count - cohort_row_count,
        "crp_observed_count": observed_count,
        "crp_missing_count": missing_count,
        "percentage_baseline_with_observed_crp": (
            round(observed_count / source_row_count * 100, 2) if source_row_count else 0.0
        ),
        "unique_patient_id_count": int(cohort["patient_id"].nunique(dropna=True)),
        "duplicate_patient_id_count": int(source["patient_id"].duplicated().sum()),
        "missing_value_counts": {
            variable: int(cohort[variable].isna().sum()) for variable in IMPORTANT_VARIABLES
        },
    }

    cohort_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    cohort.to_csv(cohort_path, index=False)
    with report_path.open("w", encoding="utf-8") as report_file:
        json.dump(report, report_file, indent=2)
        report_file.write("\n")

    return cohort, report


def main() -> None:
    """Create the default CRP-observed cohort and print its size summary."""
    cohort, report = prepare_modeling_cohort()
    percentage = report["percentage_baseline_with_observed_crp"]
    print(
        f"NHANES baseline: {report['source_row_count']} participants; "
        f"CRP-observed modeling cohort: {len(cohort)} ({percentage:.2f}%); "
        f"excluded for missing CRP: {report['excluded_row_count']}"
    )


if __name__ == "__main__":
    main()