"""Tests for the synthetic inflammatory trajectory generator."""

import numpy as np
import pandas as pd
import pytest

from src.biotwin.data.generate_inflammatory_trajectories import generate_inflammatory_trajectories


REQUIRED_COLUMNS = ["patient_id", "time_hours", "stimulus", "il6", "crp"]


def test_correct_row_count_and_patient_count() -> None:
    df = generate_inflammatory_trajectories(n_patients=10, n_hours=12, seed=7)

    assert len(df) == 10 * 12
    assert df["patient_id"].nunique() == 10


def test_required_columns_are_present() -> None:
    df = generate_inflammatory_trajectories(n_patients=5, n_hours=10, seed=9)

    assert df.columns.tolist() == REQUIRED_COLUMNS


def test_observations_per_patient_are_correct() -> None:
    df = generate_inflammatory_trajectories(n_patients=20, n_hours=24, seed=11)
    counts = df.groupby("patient_id").size()

    assert (counts == 24).all()


def test_time_starts_at_zero_and_increases_consistently() -> None:
    df = generate_inflammatory_trajectories(n_patients=4, n_hours=8, seed=13)

    for _, patient_df in df.groupby("patient_id", sort=False):
        assert patient_df["time_hours"].iloc[0] == 0.0
        assert (patient_df["time_hours"].diff().dropna() > 0).all()


def test_stimulus_and_concentrations_are_finite_and_nonnegative() -> None:
    df = generate_inflammatory_trajectories(n_patients=15, n_hours=20, seed=17)
    numeric = df[["time_hours", "stimulus", "il6", "crp"]].to_numpy()

    assert np.isfinite(numeric).all()
    assert (df["stimulus"] >= 0).all()
    assert (df["il6"] >= 0).all()
    assert (df["crp"] >= 0).all()


def test_same_seed_produces_identical_data() -> None:
    first = generate_inflammatory_trajectories(n_patients=25, n_hours=30, seed=42)
    second = generate_inflammatory_trajectories(n_patients=25, n_hours=30, seed=42)

    pd.testing.assert_frame_equal(first, second)


def test_different_seeds_produce_different_data() -> None:
    first = generate_inflammatory_trajectories(n_patients=25, n_hours=30, seed=42)
    second = generate_inflammatory_trajectories(n_patients=25, n_hours=30, seed=43)

    assert not first.equals(second)


def test_patients_have_non_identical_trajectories() -> None:
    df = generate_inflammatory_trajectories(n_patients=25, n_hours=20, seed=19)
    patient_summary = df.groupby("patient_id")[["il6", "crp"]].mean()

    assert patient_summary["il6"].nunique() > 1
    assert patient_summary["crp"].nunique() > 1


def test_no_duplicate_patient_and_time_combinations() -> None:
    df = generate_inflammatory_trajectories(n_patients=12, n_hours=16, seed=23)

    assert not df.duplicated(subset=["patient_id", "time_hours"]).any()


def test_csv_save_and_reload_round_trip(tmp_path: pytest.TempPathFactory) -> None:
    df = generate_inflammatory_trajectories(n_patients=8, n_hours=9, seed=29)
    output_path = tmp_path / "synthetic_inflammatory_trajectories.csv"
    df.to_csv(output_path, index=False)
    reloaded = pd.read_csv(output_path)

    pd.testing.assert_frame_equal(df, reloaded)


def test_rejects_invalid_n_patients() -> None:
    with pytest.raises(ValueError, match="n_patients"):
        generate_inflammatory_trajectories(n_patients=0, seed=1)
    with pytest.raises(ValueError, match="n_patients"):
        generate_inflammatory_trajectories(n_patients=-1, seed=1)


def test_rejects_invalid_n_hours() -> None:
    with pytest.raises(ValueError, match="n_hours"):
        generate_inflammatory_trajectories(n_patients=3, n_hours=0, seed=1)
    with pytest.raises(ValueError, match="n_hours"):
        generate_inflammatory_trajectories(n_patients=3, n_hours=-1, seed=1)
