"""Tests for processing the real NHANES baseline files."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from biotwin.data.process_nhanes import (
    NHANES_FILES,
    OUTPUT_COLUMNS,
    build_nhanes_baseline,
    validate_nhanes_baseline,
)


@pytest.fixture
def nhanes_frames() -> dict[str, pd.DataFrame]:
    return {
        "DEMO_L.xpt": pd.DataFrame(
            {"SEQN": [1, 2, 3], "RIDAGEYR": [30, 40, 50], "RIAGENDR": [1, 2, 1]}
        ),
        "BMX_L.xpt": pd.DataFrame(
            {
                "SEQN": [1, 2, 3],
                "BMXWT": [70.0, np.nan, 80.0],
                "BMXHT": [170.0, 165.0, 180.0],
                "BMXBMI": [24.2, 25.0, 24.7],
                "BMXWAIST": [80.0, 82.0, 84.0],
            }
        ),
        "BPXO_L.xpt": pd.DataFrame(
            {
                "SEQN": [1, 2, 3],
                "BPXOSY1": [120.0, 130.0, np.nan],
                "BPXOSY2": [126.0, np.nan, np.nan],
                "BPXOSY3": [132.0, 140.0, np.nan],
                "BPXODI1": [70.0, 80.0, np.nan],
                "BPXODI2": [74.0, np.nan, np.nan],
                "BPXODI3": [78.0, 90.0, np.nan],
            }
        ),
        "GHB_L.xpt": pd.DataFrame({"SEQN": [1, 2, 3], "LBXGH": [5.2, np.nan, 6.1]}),
        "HSCRP_L.xpt": pd.DataFrame(
            {"SEQN": [1, 2, 3], "LBXHSCRP": [1.1, np.nan, 2.2]}
        ),
        "PAQ_L.xpt": pd.DataFrame(
            {
                "SEQN": [1, 2, 3],
                "PAD790Q": [2.0, 3.0, np.nan],
                "PAD790U": ["W", "W", None],
                "PAD800": [30.0, 45.0, np.nan],
                "PAD810Q": [1.0, 2.0, np.nan],
                "PAD810U": ["W", "W", None],
                "PAD820": [20.0, 25.0, np.nan],
                "PAD680": [300.0, 420.0, np.nan],
            }
        ),
    }


@pytest.fixture
def baseline(nhanes_frames: dict[str, pd.DataFrame], monkeypatch: pytest.MonkeyPatch) -> pd.DataFrame:
    read_calls: list[tuple[Path, str]] = []

    def fake_read_sas(path: Path, *, format: str) -> pd.DataFrame:
        read_calls.append((Path(path), format))
        return nhanes_frames[Path(path).name].copy()

    monkeypatch.setattr(pd, "read_sas", fake_read_sas)
    result = build_nhanes_baseline(Path("temporary-nhanes"))

    assert len(read_calls) == len(NHANES_FILES)
    assert all(file_format == "xport" for _, file_format in read_calls)
    return result


def test_builds_expected_baseline_and_averages_all_bp_readings(baseline: pd.DataFrame) -> None:
    assert baseline.columns.tolist() == OUTPUT_COLUMNS
    assert baseline["patient_id"].tolist() == [1, 2, 3]
    assert baseline.loc[0, "systolic_bp"] == pytest.approx(126.0)
    assert baseline.loc[0, "diastolic_bp"] == pytest.approx(74.0)
    assert baseline.loc[1, "systolic_bp"] == pytest.approx(135.0)
    assert baseline.loc[1, "diastolic_bp"] == pytest.approx(85.0)


def test_all_missing_bp_readings_remain_nan(baseline: pd.DataFrame) -> None:
    assert pd.isna(baseline.loc[2, "systolic_bp"])
    assert pd.isna(baseline.loc[2, "diastolic_bp"])


def test_missing_clinical_measurements_are_preserved(baseline: pd.DataFrame) -> None:
    assert len(baseline) == 3
    assert pd.isna(baseline.loc[1, "weight_kg"])
    assert pd.isna(baseline.loc[1, "hba1c"])
    assert pd.isna(baseline.loc[1, "crp"])


def test_rejects_duplicate_patient_ids() -> None:
    data = pd.DataFrame(
        {
            "patient_id": [1, 1],
            "age": [30, 40],
            "weight_kg": [70.0, 80.0],
            "height_cm": [170.0, 180.0],
            "bmi": [24.2, 24.7],
            "waist_cm": [80.0, 84.0],
            "hba1c": [5.2, 6.1],
            "crp": [1.1, 2.2],
            "systolic_bp": [120.0, 130.0],
            "diastolic_bp": [70.0, 80.0],
        }
    )

    with pytest.raises(ValueError, match="patient_id.*unique"):
        validate_nhanes_baseline(data)


@pytest.mark.parametrize(
    ("column", "invalid_value"),
    [
        ("age", -1),
        ("weight_kg", 0),
        ("height_cm", 0),
        ("bmi", 0),
        ("waist_cm", 0),
        ("hba1c", 0),
        ("crp", 0),
        ("systolic_bp", 0),
        ("diastolic_bp", 0),
    ],
)
def test_rejects_invalid_clinical_values(column: str, invalid_value: float) -> None:
    data = pd.DataFrame(
        {
            "patient_id": [1],
            "age": [30],
            "weight_kg": [70.0],
            "height_cm": [170.0],
            "bmi": [24.2],
            "waist_cm": [80.0],
            "hba1c": [5.2],
            "crp": [1.1],
            "systolic_bp": [120.0],
            "diastolic_bp": [70.0],
        }
    )
    data.loc[0, column] = invalid_value

    with pytest.raises(ValueError, match=column):
        validate_nhanes_baseline(data)


def test_rejects_missing_patient_id() -> None:
    data = pd.DataFrame(
        {
            "patient_id": [np.nan],
            "age": [30],
            "weight_kg": [70.0],
            "height_cm": [170.0],
            "bmi": [24.2],
            "waist_cm": [80.0],
            "hba1c": [5.2],
            "crp": [1.1],
            "systolic_bp": [120.0],
            "diastolic_bp": [70.0],
        }
    )

    with pytest.raises(ValueError, match="patient_id must not be missing"):
        validate_nhanes_baseline(data)