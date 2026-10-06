"""Generate synthetic wearable telemetry for the competition PoC.

The output is simulated, not real wearable data, and is intended only for
competition PoC experimentation. Physiological relationships are simulation
assumptions, are not clinically validated, and must not be interpreted as
clinical findings or patient measurements.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


_START_TIMESTAMP = pd.Timestamp("2026-01-01 00:00:00")
_REQUIRED_BASELINE_COLUMNS = {"patient_id"}


def _ar1_noise(
    rng: np.random.Generator,
    n_patients: int,
    n_steps: int,
    persistence: float,
    innovation_scale: float,
) -> np.ndarray:
    """Create patient-wise temporally correlated noise."""
    noise = np.empty((n_patients, n_steps), dtype=float)
    noise[:, 0] = rng.normal(0.0, innovation_scale, size=n_patients)
    innovation_sd = innovation_scale * np.sqrt(1.0 - persistence**2)
    for step in range(1, n_steps):
        noise[:, step] = (
            persistence * noise[:, step - 1]
            + rng.normal(0.0, innovation_sd, size=n_patients)
        )
    return noise


def generate_synthetic_wearable(
    baseline_df: pd.DataFrame,
    days: int = 14,
    frequency: str = "1h",
    seed: int = 42,
) -> pd.DataFrame:
    """Generate reproducible synthetic telemetry linked to baseline patient IDs.

    The frequency must be a positive fixed interval that divides 24 hours.
    ``sleep_hours`` is the amount of sleep represented during each interval.
    """
    if not isinstance(baseline_df, pd.DataFrame):
        raise ValueError("baseline_df must be a pandas DataFrame")
    if not _REQUIRED_BASELINE_COLUMNS.issubset(baseline_df.columns):
        raise ValueError("baseline_df must contain a patient_id column")
    if baseline_df.empty:
        raise ValueError("baseline_df must contain at least one patient")
    if baseline_df["patient_id"].isna().any() or baseline_df["patient_id"].duplicated().any():
        raise ValueError("baseline_df patient_id values must be non-null and unique")
    if isinstance(days, bool) or not isinstance(days, (int, np.integer)) or days <= 0:
        raise ValueError("days must be a positive integer")

    try:
        interval = pd.Timedelta(frequency)
    except (TypeError, ValueError) as error:
        raise ValueError("frequency must be a positive fixed interval dividing 24 hours") from error
    day_ns = pd.Timedelta(days=1).value
    if interval <= pd.Timedelta(0) or day_ns % interval.value != 0:
        raise ValueError("frequency must be a positive fixed interval dividing 24 hours")

    rng = np.random.default_rng(seed)
    patient_count = len(baseline_df)
    steps_per_day = day_ns // interval.value
    step_count = int(days * steps_per_day)
    interval_hours = interval / pd.Timedelta(hours=1)
    elapsed_hours = np.arange(step_count, dtype=float) * interval_hours
    hours_of_day = np.mod(elapsed_hours, 24.0)

    def baseline_values(column: str, default: float) -> np.ndarray:
        if column not in baseline_df:
            return np.full(patient_count, default, dtype=float)
        values = pd.to_numeric(baseline_df[column], errors="coerce").to_numpy(dtype=float)
        return np.where(np.isfinite(values), values, default)

    age = baseline_values("age", 45.0)
    bmi = baseline_values("bmi", 25.0)
    activity_category = (
        baseline_df["physical_activity_level"].astype("string").fillna("moderate").to_numpy()
        if "physical_activity_level" in baseline_df
        else np.full(patient_count, "moderate")
    )
    activity_factor = np.select(
        [activity_category == "low", activity_category == "high"], [0.78, 1.25], default=1.0
    )
    smoking = (
        baseline_df["smoking_status"].astype("string").fillna("never").to_numpy()
        if "smoking_status" in baseline_df
        else np.full(patient_count, "never")
    )
    smoking_effect = np.where(smoking == "current", 3.0, np.where(smoking == "former", 1.0, 0.0))

    typical_sleep = np.clip(rng.normal(7.35, 0.65, size=patient_count), 5.5, 9.2)
    bedtime = rng.normal(22.4, 0.8, size=patient_count)
    sleep_midpoint = np.mod(bedtime + typical_sleep / 2.0, 24.0)
    distance_from_sleep_midpoint = (hours_of_day[None, :] - sleep_midpoint[:, None] + 12.0) % 24.0 - 12.0
    sleep_profile = np.exp(-0.5 * (distance_from_sleep_midpoint / 2.65) ** 2)
    sleep_profile /= sleep_profile.sum(axis=1, keepdims=True)
    daily_sleep_variation = rng.normal(1.0, 0.055, size=(patient_count, int(days)))
    sleep_day_factor = np.repeat(daily_sleep_variation, int(steps_per_day), axis=1)[:, :step_count]
    sleep_noise = _ar1_noise(rng, patient_count, step_count, 0.72, 0.06)
    sleep_hours = (
        sleep_profile
        * typical_sleep[:, None]
        * sleep_day_factor
        * np.maximum(0.5, 1.0 + sleep_noise)
    )
    sleep_hours *= (
        typical_sleep[:, None] * sleep_day_factor
    ) / sleep_hours.reshape(patient_count, int(days), int(steps_per_day)).sum(axis=2).repeat(
        int(steps_per_day), axis=1
    )[:, :step_count]

    waking_profile = np.clip(1.0 - sleep_profile * int(steps_per_day), 0.0, 1.0)
    daylight_activity = (
        0.30
        + 0.70 * np.exp(-0.5 * ((hours_of_day - 13.0) / 4.0) ** 2)
        + 0.18 * np.exp(-0.5 * ((hours_of_day - 8.5) / 1.5) ** 2)
    )
    day_activity_variation = rng.lognormal(mean=0.0, sigma=0.12, size=(patient_count, int(days)))
    activity_day_factor = np.repeat(day_activity_variation, int(steps_per_day), axis=1)[:, :step_count]
    activity_noise = _ar1_noise(rng, patient_count, step_count, 0.78, 0.22)
    physical_activity = (
        (0.12 + 1.45 * daylight_activity[None, :])
        * waking_profile
        * activity_factor[:, None]
        * activity_day_factor
        * np.exp(activity_noise)
    )
    physical_activity = np.maximum(0.0, physical_activity)

    circadian = np.sin(2.0 * np.pi * (hours_of_day - 8.0) / 24.0)
    stress_tendency = rng.normal(38.0, 10.0, size=patient_count)
    stress_noise = _ar1_noise(rng, patient_count, step_count, 0.88, 8.0)
    stress_score = (
        stress_tendency[:, None]
        + 8.0 * np.sin(2.0 * np.pi * (hours_of_day[None, :] - 10.0) / 24.0)
        + 0.45 * stress_noise
        + 3.0 * (1.0 - waking_profile)
    )
    stress_score = np.clip(stress_score, 0.0, 100.0)

    patient_resting_hr = (
        rng.normal(64.0, 5.5, size=patient_count)
        + 0.10 * (age - 45.0)
        + 0.12 * (bmi - 25.0)
        + smoking_effect
    )
    hr_noise = _ar1_noise(rng, patient_count, step_count, 0.82, 2.6)
    resting_heart_rate = (
        patient_resting_hr[:, None]
        + 2.2 * circadian[None, :]
        + 0.45 * stress_score
        + 3.0 * physical_activity
        - 2.0 * sleep_profile * int(steps_per_day)
        + hr_noise
    )
    resting_heart_rate = np.maximum(25.0, resting_heart_rate)

    patient_hrv = (
        rng.normal(52.0, 9.0, size=patient_count)
        - 0.18 * (age - 45.0)
        - 0.32 * (bmi - 25.0)
        - 1.0 * smoking_effect
    )
    hrv_noise = _ar1_noise(rng, patient_count, step_count, 0.84, 3.0)
    hrv_rmssd = (
        patient_hrv[:, None]
        - 0.10 * stress_score
        + 2.0 * sleep_profile * int(steps_per_day)
        - 0.55 * physical_activity
        - 1.4 * circadian[None, :]
        + hrv_noise
    )
    hrv_rmssd = np.maximum(5.0, hrv_rmssd)

    timestamps = pd.date_range(
        start=_START_TIMESTAMP, periods=step_count, freq=interval
    )
    output = pd.DataFrame(
        {
            "patient_id": np.repeat(baseline_df["patient_id"].to_numpy(), step_count),
            "timestamp": np.tile(timestamps.to_numpy(), patient_count),
            "hours_since_start": np.tile(elapsed_hours, patient_count),
            "sleep_hours": sleep_hours.reshape(-1),
            "hrv_rmssd": hrv_rmssd.reshape(-1),
            "resting_heart_rate": resting_heart_rate.reshape(-1),
            "stress_score": stress_score.reshape(-1),
            "physical_activity": physical_activity.reshape(-1),
        }
    )
    return output


def main() -> None:
    """Load the baseline population and export default hourly telemetry."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline",
        type=Path,
        default=Path("data/synthetic/synthetic_baseline.csv"),
        help="Path to the synthetic baseline CSV",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/synthetic/synthetic_wearable.csv"),
        help="Path for the synthetic wearable CSV",
    )
    args = parser.parse_args()

    baseline = pd.read_csv(args.baseline)
    wearable = generate_synthetic_wearable(baseline, days=14, frequency="1h", seed=42)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    wearable.to_csv(args.output, index=False)
    print("Synthetic wearable telemetry generated successfully.")
    print(f"Patients: {baseline['patient_id'].nunique()}")
    print("Days: 14")
    print("Frequency: 1h")
    print(f"Rows: {len(wearable)}")
    print(f"Output: {args.output.as_posix()}")


if __name__ == "__main__":
    main()