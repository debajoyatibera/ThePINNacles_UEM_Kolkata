"""Generate a patient-conditioned synthetic Digital Twin dataset from NHANES baselines.

This module bridges real NHANES cross-sectional baseline observations to a
synthetic dynamic Digital Twin that is clearly labelled as synthetic and
model-simulated. It does not create longitudinal observed NHANES measurements.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from biotwin.models.biological_ode import BiologicalParameters, simulate_trajectory

DEFAULT_SOURCE_PATH = Path("data/processed/nhanes_modeling_cohort.csv")
DEFAULT_OUTPUT_PATH = Path("data/processed/patient_conditioned_digital_twin.csv")
DEFAULT_REPORT_PATH = Path("data/processed/patient_conditioned_digital_twin_report.json")
DEFAULT_TIME_POINTS_PER_PARTICIPANT = 168

BASELINE_FEATURES = [
    "age",
    "sex",
    "bmi",
    "waist_cm",
    "systolic_bp",
    "diastolic_bp",
    "hba1c",
    "sedentary_minutes",
]
OUTPUT_COLUMNS = [
    "patient_id",
    "time_hours",
    "age",
    "sex",
    "bmi",
    "waist_cm",
    "systolic_bp",
    "diastolic_bp",
    "hba1c",
    "sedentary_minutes",
    "synthetic_stimulus",
    "synthetic_il6",
    "synthetic_crp",
    "imputed_baseline_fields",
    "data_source",
    "dynamic_data_source",
    "target_type",
]


def _validate_source(dataset: pd.DataFrame) -> None:
    required = {"patient_id", *BASELINE_FEATURES}
    missing = sorted(required - set(dataset.columns))
    if missing:
        raise ValueError(f"Source cohort is missing required baseline columns: {missing}")
    if dataset.empty:
        raise ValueError("Source cohort cannot be empty")
    if dataset["patient_id"].isna().any():
        raise ValueError("patient_id values must not be missing")
    if dataset["patient_id"].duplicated().any():
        raise ValueError("patient_id values must be unique")


def _prepare_conditioning_features(
    source: pd.DataFrame,
) -> tuple[pd.Series, dict[str, int], pd.Series]:
    conditioning = source[BASELINE_FEATURES].copy()
    missing = pd.DataFrame(False, index=source.index, columns=BASELINE_FEATURES)
    imputation_counts: dict[str, int] = {}

    for feature in BASELINE_FEATURES:
        values = pd.to_numeric(conditioning[feature], errors="coerce")
        missing[feature] = values.isna()
        imputation_counts[feature] = int(values.isna().sum())
        if values.isna().all():
            raise ValueError(f"Cannot impute baseline feature with no observed values: {feature}")
        fill_value = values.mode().iloc[0] if feature == "sex" else values.median()
        conditioning[feature] = values.fillna(fill_value)

    # Percentile ranks avoid embedding hand-picked clinical thresholds or weights.
    conditioning_score = conditioning.rank(method="average", pct=True).mean(axis=1)
    imputed_fields = missing.apply(
        lambda row: ";".join(row.index[row].tolist()), axis=1
    )
    return conditioning_score, imputation_counts, imputed_fields


def _synthetic_stimulus_for_patient(
    conditioning_score: float,
    time_hours: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    baseline = 0.05 + 0.30 * conditioning_score
    peak = 1.2 + 2.5 * conditioning_score
    pulse_center = rng.uniform(0.0, max(1.0, float(time_hours[-1])))
    pulse_width = rng.uniform(2.0, 12.0)
    pulse_scale = rng.uniform(0.9, 1.1)
    pulse = peak * pulse_scale * np.exp(
        -0.5 * ((time_hours - pulse_center) / pulse_width) ** 2
    )
    cyclical = 0.05 * np.sin(2.0 * np.pi * time_hours / 24.0)
    return np.clip(baseline + pulse + cyclical, 0.0, 5.0)


def generate_patient_conditioned_digital_twin(
    source_path: str | Path = DEFAULT_SOURCE_PATH,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    report_path: str | Path = DEFAULT_REPORT_PATH,
    seed: int = 42,
    time_points_per_patient: int = DEFAULT_TIME_POINTS_PER_PARTICIPANT,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Create a synthetic dynamic Digital Twin dataset anchored to NHANES baselines."""
    if isinstance(time_points_per_patient, bool) or not isinstance(
        time_points_per_patient, (int, np.integer)
    ):
        raise ValueError("time_points_per_patient must be a positive integer")
    if time_points_per_patient <= 0:
        raise ValueError("time_points_per_patient must be positive")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")

    source_file = Path(source_path)
    output_file = Path(output_path)
    report_file = Path(report_path)
    source = pd.read_csv(source_file)
    _validate_source(source)

    conditioning_score, feature_imputation_counts, imputed_fields = (
        _prepare_conditioning_features(source)
    )
    missing_baseline_rows = int(imputed_fields.ne("").sum())
    imputed_fields = imputed_fields.mask(imputed_fields.eq(""), "NONE")

    params = BiologicalParameters(alpha_il6=1.8, k_il6=0.45, alpha_crp=1.2, k_crp=0.18)
    time_grid = np.arange(time_points_per_patient, dtype=float)
    rng = np.random.default_rng(int(seed))
    trajectories: list[pd.DataFrame] = []

    for position, (_, patient_row) in enumerate(source.iterrows()):
        patient_id = patient_row["patient_id"]
        initial_il6 = float(rng.uniform(0.05, 1.2))
        initial_crp = float(rng.uniform(0.05, 1.5))
        synthetic_stimulus = _synthetic_stimulus_for_patient(
            float(conditioning_score.iloc[position]), time_grid, rng
        )
        trajectory = simulate_trajectory(
            initial_il6=initial_il6,
            initial_crp=initial_crp,
            stimulus=synthetic_stimulus,
            dt=1.0,
            params=params,
        )

        dynamic = trajectory.rename(
            columns={
                "time": "time_hours",
                "stimulus": "synthetic_stimulus",
                "il6": "synthetic_il6",
                "crp": "synthetic_crp",
            }
        )
        dynamic["patient_id"] = patient_id
        dynamic["imputed_baseline_fields"] = imputed_fields.iloc[position]
        for feature in BASELINE_FEATURES:
            dynamic[feature] = patient_row[feature]
        dynamic["data_source"] = "NHANES_2021_2023"
        dynamic["dynamic_data_source"] = "SYNTHETIC"
        dynamic["target_type"] = "MODEL_SIMULATED"

        trajectories.append(dynamic[OUTPUT_COLUMNS])

    output = pd.concat(trajectories, ignore_index=True)
    if output.empty:
        raise ValueError("Generated digital twin output cannot be empty")
    if output.duplicated(subset=["patient_id", "time_hours"]).any():
        raise ValueError("Generated output contains duplicate patient/time rows")
    if not np.isfinite(output[["synthetic_stimulus", "synthetic_il6", "synthetic_crp"]]).all().all():
        raise FloatingPointError("Generated synthetic dynamic columns contain non-finite values")
    if (output[["synthetic_stimulus", "synthetic_il6", "synthetic_crp"]] < 0).any().any():
        raise ValueError("Synthetic dynamic values must be non-negative")
    if (output["synthetic_stimulus"] > 5.0).any():
        raise ValueError("Synthetic stimulus exceeds the bounded operating range")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_file, index=False)

    report: dict[str, Any] = {
        "source_dataset": source_file.as_posix(),
        "number_of_unique_nhanes_participants": int(output["patient_id"].nunique()),
        "number_of_trajectory_rows": int(len(output)),
        "time_points_per_participant": int(time_points_per_patient),
        "random_seed": int(seed),
        "baseline_feature_list": BASELINE_FEATURES,
        "synthetic_feature_list": ["synthetic_stimulus", "synthetic_il6", "synthetic_crp"],
        "conditioning_method": (
            "Equal-weight mean of within-cohort percentile ranks for the listed baseline features; "
            "this is a synthetic construction and not a clinically validated relationship."
        ),
        "synthetic_stimulus_bounds": [0.0, 5.0],
        "time_step_hours": 1.0,
        "ode_parameters": {
            "alpha_il6": 1.8,
            "k_il6": 0.45,
            "alpha_crp": 1.2,
            "k_crp": 0.18,
        },
        "ode_parameter_source": (
            "Existing competition PoC configuration from run_pinn_experiment.py; "
            "not fitted to NHANES or clinical data."
        ),
        "imputation_strategy": {
            "method": (
                "Median imputation for numeric baseline covariates and mode imputation for sex, "
                "used only for synthetic conditioning. Original baseline "
                "values remain unchanged in the output."
            ),
            "feature_counts": feature_imputation_counts,
            "missing_baseline_rows": missing_baseline_rows,
            "row_indicator_column": "imputed_baseline_fields",
        },
        "data_provenance_statement": (
            "NHANES 2021-2023 baseline values are real observed participant characteristics. "
            "The bounded synthetic stimulus is generated from baseline-feature percentile ranks. "
            "Synthetic IL-6 and CRP trajectories are model-simulated with the existing biological "
            "ODE. None of the dynamic values are observed clinical measurements."
        ),
        "limitation_statement": (
            "This dataset is a competition/research PoC and does not represent observed longitudinal "
            "clinical measurements. NHANES is cross-sectional and does not provide longitudinal IL-6, "
            "longitudinal CRP, or continuous wearable telemetry."
        ),
        "no_observed_longitudinal_nhanes_data_created": True,
        "no_observed_nhanes_il6_created": True,
        "no_observed_nhanes_crp_longitudinal_data_created": True,
        "nhanes_crp_used_for_conditioning": False,
        "missing_baseline_value_counts": feature_imputation_counts,
        "imputation_summary": {
            "missing_baseline_rows": missing_baseline_rows,
            "feature_counts": feature_imputation_counts,
        },
    }

    report_file.parent.mkdir(parents=True, exist_ok=True)
    report_file.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return output, report


def main() -> None:
    """Generate the patient-conditioned digital twin dataset and its report."""
    data, report = generate_patient_conditioned_digital_twin(
        source_path=DEFAULT_SOURCE_PATH,
        output_path=DEFAULT_OUTPUT_PATH,
        report_path=DEFAULT_REPORT_PATH,
        seed=42,
        time_points_per_patient=DEFAULT_TIME_POINTS_PER_PARTICIPANT,
    )
    print(f"Unique NHANES participants: {report['number_of_unique_nhanes_participants']}")
    print(f"Trajectory rows: {report['number_of_trajectory_rows']}")
    print(f"Time points per participant: {report['time_points_per_participant']}")
    print(f"Output dataset: {DEFAULT_OUTPUT_PATH.as_posix()}")
    print(f"Report: {DEFAULT_REPORT_PATH.as_posix()}")
    print(f"Columns: {list(data.columns)}")


if __name__ == "__main__":
    main()
