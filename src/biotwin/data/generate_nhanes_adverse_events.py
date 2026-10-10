"""Build a synthetic 2-hour adverse-event dataset linked to real NHANES IDs.

REAL:
- NHANES participant identifiers
- NHANES baseline clinical/activity observations

SYNTHETIC:
- wearable telemetry
- temporal wearable features
- adverse-event target

This is a competition/research PoC. The target is synthetic and is not
clinically validated or suitable for medical decision-making.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from biotwin.data.generate_adverse_events import _build_temporal_features


STATIC_FEATURES = [
    "age",
    "bmi",
    "systolic_bp",
    "diastolic_bp",
    "hba1c",
    "sedentary_minutes",
]

SPECIAL_MISSING_CODES = {9999, 7777, 6666}


def prepare_nhanes_baseline(
    baseline_df: pd.DataFrame,
) -> pd.DataFrame:
    """Prepare the NHANES static fields used by the event PoC."""
    required = {"patient_id", *STATIC_FEATURES}

    missing = required - set(baseline_df.columns)
    if missing:
        raise ValueError(
            f"NHANES baseline is missing required columns: {sorted(missing)}"
        )

    data = baseline_df[list(required)].copy()

    if data["patient_id"].isna().any():
        raise ValueError("NHANES patient_id values must not be missing")

    if data["patient_id"].duplicated().any():
        raise ValueError("NHANES patient_id values must be unique")

    for column in STATIC_FEATURES:
        data[column] = pd.to_numeric(data[column], errors="coerce")
        data.loc[data[column].isin(SPECIAL_MISSING_CODES), column] = np.nan

    # Cohort-median imputation is used only to make the synthetic PoC complete.
    # These values are not presented as observed measurements.
    for column in STATIC_FEATURES:
        median = data[column].median()

        if not np.isfinite(median):
            raise ValueError(
                f"Cannot calculate a finite cohort median for {column}"
            )

        data[column] = data[column].fillna(float(median))

    return data


def _create_synthetic_event_target(
    features: pd.DataFrame,
    baseline_df: pd.DataFrame,
    forecast_horizon_hours: int,
    seed: int,
) -> pd.DataFrame:
    """Generate an independent synthetic deterioration-event target."""
    if forecast_horizon_hours != 2:
        raise ValueError("forecast_horizon_hours must be 2 for this PoC")

    rng = np.random.default_rng(seed)

    static = baseline_df[
        [
            "patient_id",
            "age",
            "bmi",
            "systolic_bp",
            "diastolic_bp",
            "hba1c",
            "sedentary_minutes",
        ]
    ].copy()

    merged = features.merge(
        static,
        on="patient_id",
        how="left",
        validate="many_to_one",
    )

    age_component = np.clip(
        (merged["age"] - 45.0) / 25.0,
        -1.0,
        2.0,
    )

    bmi_component = np.clip(
        (merged["bmi"] - 25.0) / 8.0,
        -1.0,
        2.0,
    )

    bp_component = np.clip(
        ((merged["systolic_bp"] - 120.0) / 25.0)
        + ((merged["diastolic_bp"] - 80.0) / 15.0),
        -2.0,
        3.0,
    )

    hba1c_component = np.clip(
        (merged["hba1c"] - 5.5) / 1.5,
        -1.0,
        3.0,
    )

    sedentary_component = np.clip(
        (merged["sedentary_minutes"] - 300.0) / 300.0,
        -1.0,
        3.0,
    )

    hrv_change = np.clip(
        -merged["hrv_rmssd_slope_6h"] / 2.5,
        -2.0,
        3.0,
    )

    hr_change = np.clip(
        merged["resting_heart_rate_slope_6h"] / 1.5,
        -2.0,
        3.0,
    )

    stress_change = np.clip(
        merged["stress_score_slope_6h"] / 4.0,
        -2.0,
        3.0,
    )

    sleep_disruption = np.clip(
        (7.0 - merged["sleep_hours_mean_6h"]) / 2.0,
        -1.0,
        3.0,
    )

    activity_disruption = np.clip(
        -merged["physical_activity_slope_6h"] / 0.25,
        -2.0,
        3.0,
    )

    max_time = merged.groupby("patient_id")["hours_since_start"].transform("max")

    merged["forecast_end_hours"] = (
        merged["hours_since_start"] + float(forecast_horizon_hours)
    )

    valid_history = merged["hrv_rmssd_mean_6h"].notna()
    valid_future_horizon = merged["forecast_end_hours"] <= max_time

    merged = merged.loc[
        valid_history & valid_future_horizon
    ].copy()

    latent_score = (
        -3.0
        + 0.22 * age_component.loc[merged.index]
        + 0.28 * bmi_component.loc[merged.index]
        + 0.25 * bp_component.loc[merged.index]
        + 0.20 * hba1c_component.loc[merged.index]
        + 0.15 * sedentary_component.loc[merged.index]
        + 0.55 * hrv_change.loc[merged.index]
        + 0.50 * hr_change.loc[merged.index]
        + 0.50 * stress_change.loc[merged.index]
        + 0.35 * sleep_disruption.loc[merged.index]
        + 0.25 * activity_disruption.loc[merged.index]
    )

    noise = rng.normal(
        0.0,
        0.55,
        size=len(merged),
    )

    latent_score = (
        latent_score.to_numpy(dtype=float)
        + noise
    )

    probability = 1.0 / (1.0 + np.exp(-latent_score))
    probability = np.clip(probability, 0.0, 1.0)

    event = rng.binomial(
        1,
        probability,
    ).astype(np.int8)

    merged["event_probability_simulated"] = probability
    merged["event_in_next_2h"] = event

    return merged


def build_nhanes_adverse_event_dataset(
    baseline_df: pd.DataFrame,
    wearable_df: pd.DataFrame,
    observation_window_hours: int = 6,
    forecast_horizon_hours: int = 2,
    seed: int = 42,
) -> pd.DataFrame:
    """Build the NHANES-linked synthetic event dataset."""
    baseline = prepare_nhanes_baseline(baseline_df)


    if observation_window_hours != 6:
        raise ValueError("This PoC requires a 6-hour observation window")

    if forecast_horizon_hours != 2:
        raise ValueError("This PoC requires a 2-hour forecast horizon")

    features = _build_temporal_features(
        wearable_df=wearable_df,
        observation_window_hours=observation_window_hours,
    )

    result = _create_synthetic_event_target(
        features=features,
        baseline_df=baseline,
        forecast_horizon_hours=forecast_horizon_hours,
        seed=seed,
    )

    static_columns = [
        "patient_id",
        "age",
        "bmi",
        "systolic_bp",
        "diastolic_bp",
        "hba1c",
        "sedentary_minutes",
    ]

    result = result.merge(
        baseline[static_columns],
        on="patient_id",
        how="left",
        validate="many_to_one",
        suffixes=("", "_baseline"),
    )

    result = result.drop(
        columns=[
            "age",
            "bmi",
            "systolic_bp",
            "diastolic_bp",
            "hba1c",
            "sedentary_minutes",
        ]
    )

    result = result.rename(
        columns={
            "age_baseline": "age",
            "bmi_baseline": "bmi",
            "systolic_bp_baseline": "systolic_bp",
            "diastolic_bp_baseline": "diastolic_bp",
            "hba1c_baseline": "hba1c",
            "sedentary_minutes_baseline": "sedentary_minutes",
        }
    )

    return result.sort_values(
        ["patient_id", "timestamp"]
    ).reset_index(drop=True)


def main() -> None:
    """Generate the default NHANES-linked event dataset."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--baseline",
        type=Path,
        default=Path(
            "data/processed/nhanes_modeling_cohort.csv"
        ),
    )

    parser.add_argument(
        "--wearable",
        type=Path,
        default=Path(
            "data/synthetic/nhanes_synthetic_wearable.csv"
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "data/synthetic/nhanes_adverse_event_dataset.csv"
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    baseline = pd.read_csv(args.baseline)
    wearable = pd.read_csv(args.wearable)

    dataset = build_nhanes_adverse_event_dataset(
        baseline_df=baseline,
        wearable_df=wearable,
        seed=args.seed,
    )

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataset.to_csv(
        args.output,
        index=False,
    )

    print("NHANES-linked synthetic adverse-event dataset generated.")
    print(f"Patients: {dataset['patient_id'].nunique()}")
    print(f"Rows: {len(dataset)}")
    print(
        f"Event rate: "
        f"{dataset['event_in_next_2h'].mean():.4f}"
    )
    print(f"Output: {args.output.as_posix()}")


if __name__ == "__main__":
    main()
