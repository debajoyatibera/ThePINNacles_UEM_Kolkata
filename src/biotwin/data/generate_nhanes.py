"""Generate a synthetic, NHANES-inspired clinical baseline population.

This generator creates fictional data for competition and research PoC
experimentation because real hospital data are unavailable. It is inspired by
the types of variables commonly found in NHANES, but is not actual NHANES data
and does not reproduce its population statistics. Its values are not clinical
patient data and must not be interpreted for diagnosis or clinical decisions.
Relationships between variables are simulation assumptions, not causal medical
claims.
"""

from pathlib import Path

import numpy as np
import pandas as pd


def generate_synthetic_baseline(
    n_patients: int = 1000,
    seed: int = 42,
) -> pd.DataFrame:
    """Return one reproducible synthetic baseline row per adult patient.

    Categories are ``female``/``male`` for sex; ``never``/``former``/``current``
    for smoking_status; ``none``/``occasional``/``regular`` for alcohol_use;
    and ``low``/``moderate``/``high`` for physical_activity_level.
    """
    if isinstance(n_patients, bool) or not isinstance(n_patients, (int, np.integer)):
        raise ValueError("n_patients must be a positive integer")
    if n_patients <= 0:
        raise ValueError("n_patients must be a positive integer")

    rng = np.random.default_rng(seed)
    count = int(n_patients)

    age = rng.integers(18, 91, size=count)
    sex = rng.choice(["female", "male"], size=count, p=[0.51, 0.49])

    height_cm = np.where(
        sex == "female",
        rng.triangular(145.0, 163.0, 182.0, size=count),
        rng.triangular(155.0, 176.0, 198.0, size=count),
    )
    latent_bmi = rng.triangular(17.0, 27.0, 42.0, size=count)
    weight_kg = latent_bmi * (height_cm / 100.0) ** 2
    bmi = weight_kg / (height_cm / 100.0) ** 2

    diastolic_bp = rng.triangular(55.0, 78.0, 100.0, size=count)
    pulse_pressure = rng.triangular(25.0, 42.0, 70.0, size=count)
    systolic_bp = diastolic_bp + pulse_pressure
    resting_heart_rate = rng.triangular(48.0, 72.0, 110.0, size=count)

    smoking_status = rng.choice(
        ["never", "former", "current"], size=count, p=[0.60, 0.24, 0.16]
    )
    alcohol_use = rng.choice(
        ["none", "occasional", "regular"], size=count, p=[0.30, 0.48, 0.22]
    )
    physical_activity_level = rng.choice(
        ["low", "moderate", "high"], size=count, p=[0.30, 0.48, 0.22]
    )

    glucose_mode = np.clip(88.0 + 0.65 * (bmi - 25.0) + 0.12 * (age - 45.0), 78.0, 135.0)
    fasting_glucose = rng.triangular(65.0, glucose_mode, 165.0)
    hba1c_mode = np.clip(4.8 + 0.025 * (fasting_glucose - 90.0), 4.4, 7.0)
    hba1c = rng.triangular(4.2, hba1c_mode, 9.5)
    triglycerides = rng.triangular(40.0, 115.0 + 2.0 * (bmi - 25.0), 350.0)
    hdl = rng.triangular(25.0, 55.0 - 0.55 * (bmi - 25.0), 100.0)
    ldl = rng.triangular(40.0, 115.0, 220.0, size=count)

    smoking_effect = np.select(
        [smoking_status == "current", smoking_status == "former"], [0.30, 0.12], default=0.0
    )
    metabolic_effect = 0.012 * (fasting_glucose - 100.0)
    crp_log_mean = (
        np.log(1.5)
        + 0.045 * (bmi - 25.0)
        + 0.009 * (age - 45.0)
        + smoking_effect
        + metabolic_effect
    )
    il6_log_mean = (
        np.log(2.0)
        + 0.025 * (bmi - 25.0)
        + 0.006 * (age - 45.0)
        + 0.65 * smoking_effect
        + 0.006 * (fasting_glucose - 100.0)
    )
    baseline_crp = rng.lognormal(mean=crp_log_mean, sigma=0.48)
    baseline_il6 = rng.lognormal(mean=il6_log_mean, sigma=0.35)

    return pd.DataFrame(
        {
            "patient_id": [f"P{patient_number:06d}" for patient_number in range(1, count + 1)],
            "age": age,
            "sex": sex,
            "height_cm": height_cm,
            "weight_kg": weight_kg,
            "bmi": bmi,
            "systolic_bp": systolic_bp,
            "diastolic_bp": diastolic_bp,
            "resting_heart_rate": resting_heart_rate,
            "fasting_glucose": fasting_glucose,
            "hba1c": hba1c,
            "triglycerides": triglycerides,
            "hdl": hdl,
            "ldl": ldl,
            "smoking_status": smoking_status,
            "alcohol_use": alcohol_use,
            "physical_activity_level": physical_activity_level,
            "baseline_crp": baseline_crp,
            "baseline_il6": baseline_il6,
        }
    )


if __name__ == "__main__":
    output_path = Path("data/synthetic/synthetic_baseline.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    baseline = generate_synthetic_baseline(n_patients=1000, seed=42)
    baseline.to_csv(output_path, index=False)
    print("Synthetic baseline generated successfully.")
    print(f"Patients: {len(baseline)}")
    print(f"Output: {output_path.as_posix()}")
    print(f"DataFrame shape: {baseline.shape}")