"""Build a real NHANES 2021-2023 cross-sectional baseline dataset.

This pipeline preserves observed missing values and does not create longitudinal
records, impute clinical measurements, or add variables absent from NHANES.
"""

from pathlib import Path

import pandas as pd


NHANES_FILES = {
    "DEMO_L.xpt": ["SEQN", "RIDAGEYR", "RIAGENDR"],
    "BMX_L.xpt": ["SEQN", "BMXWT", "BMXHT", "BMXBMI", "BMXWAIST"],
    "BPXO_L.xpt": [
        "SEQN",
        "BPXOSY1",
        "BPXOSY2",
        "BPXOSY3",
        "BPXODI1",
        "BPXODI2",
        "BPXODI3",
    ],
    "GHB_L.xpt": ["SEQN", "LBXGH"],
    "HSCRP_L.xpt": ["SEQN", "LBXHSCRP"],
    "PAQ_L.xpt": [
        "SEQN",
        "PAD790Q",
        "PAD790U",
        "PAD800",
        "PAD810Q",
        "PAD810U",
        "PAD820",
        "PAD680",
    ],
}

OUTPUT_COLUMNS = [
    "patient_id",
    "age",
    "sex",
    "weight_kg",
    "height_cm",
    "bmi",
    "waist_cm",
    "systolic_bp",
    "diastolic_bp",
    "hba1c",
    "crp",
    "vigorous_activity_frequency",
    "vigorous_activity_frequency_unit",
    "vigorous_activity_duration",
    "moderate_activity_frequency",
    "moderate_activity_frequency_unit",
    "moderate_activity_duration",
    "sedentary_minutes",
]

RENAMED_COLUMNS = {
    "SEQN": "patient_id",
    "RIDAGEYR": "age",
    "RIAGENDR": "sex",
    "BMXWT": "weight_kg",
    "BMXHT": "height_cm",
    "BMXBMI": "bmi",
    "BMXWAIST": "waist_cm",
    "LBXGH": "hba1c",
    "LBXHSCRP": "crp",
    "PAD790Q": "vigorous_activity_frequency",
    "PAD790U": "vigorous_activity_frequency_unit",
    "PAD800": "vigorous_activity_duration",
    "PAD810Q": "moderate_activity_frequency",
    "PAD810U": "moderate_activity_frequency_unit",
    "PAD820": "moderate_activity_duration",
    "PAD680": "sedentary_minutes",
}


def validate_nhanes_baseline(data: pd.DataFrame) -> None:
    """Raise ``ValueError`` when baseline identifiers or clinical values fail checks."""
    if "patient_id" not in data.columns:
        raise ValueError("patient_id column is required")
    if data["patient_id"].isna().any():
        raise ValueError("patient_id must not be missing")
    if data["patient_id"].duplicated().any():
        raise ValueError("patient_id values must be unique")

    lower_bounds = {
        "age": 0,
        "weight_kg": 0,
        "height_cm": 0,
        "bmi": 0,
        "waist_cm": 0,
        "hba1c": 0,
        "crp": 0,
        "systolic_bp": 0,
        "diastolic_bp": 0,
    }
    for column, lower_bound in lower_bounds.items():
        if column not in data.columns:
            raise ValueError(f"{column} column is required")
        values = data[column].dropna()
        invalid = (values < lower_bound).any() if column == "age" else (values <= lower_bound).any()
        if invalid:
            comparison = "non-negative" if column == "age" else "positive"
            raise ValueError(f"{column} must be {comparison} when present")


def _read_nhanes_file(data_dir: Path, filename: str, columns: list[str]) -> pd.DataFrame:
    path = data_dir / filename
    data = pd.read_sas(path, format="xport")
    missing_columns = sorted(set(columns) - set(data.columns))
    if missing_columns:
        raise ValueError(f"{filename} is missing required columns: {missing_columns}")
    if data["SEQN"].isna().any():
        raise ValueError(f"{filename} contains missing SEQN values")
    if data["SEQN"].duplicated().any():
        raise ValueError(f"{filename} contains duplicate SEQN values")
    return data.loc[:, columns].copy()


def build_nhanes_baseline(data_dir: Path) -> pd.DataFrame:
    """Load, merge, and validate the requested real NHANES baseline variables.

    Outer joins retain every participant identifier present in any of the six
    source files. Missing measurements remain NaN.
    """
    data_dir = Path(data_dir)
    tables = [
        _read_nhanes_file(data_dir, filename, columns)
        for filename, columns in NHANES_FILES.items()
    ]

    merged = tables[0]
    for table in tables[1:]:
        merged = merged.merge(table, on="SEQN", how="outer", validate="one_to_one")

    merged["systolic_bp"] = merged[["BPXOSY1", "BPXOSY2", "BPXOSY3"]].mean(axis=1)
    merged["diastolic_bp"] = merged[["BPXODI1", "BPXODI2", "BPXODI3"]].mean(axis=1)
    merged = merged.drop(
        columns=[
            "BPXOSY1",
            "BPXOSY2",
            "BPXOSY3",
            "BPXODI1",
            "BPXODI2",
            "BPXODI3",
        ]
    )
    merged = merged.rename(columns=RENAMED_COLUMNS).loc[:, OUTPUT_COLUMNS]
    validate_nhanes_baseline(merged)
    return merged


def main() -> None:
    """Write the processed real NHANES baseline to its designated location."""
    baseline = build_nhanes_baseline(Path("data/raw/nhanes"))
    output_path = Path("data/processed/nhanes_baseline.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    baseline.to_csv(output_path, index=False)
    print(f"Wrote {len(baseline)} participants and {len(baseline.columns)} columns to {output_path}")


if __name__ == "__main__":
    main()