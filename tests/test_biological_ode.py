"""Tests for the simplified synthetic inflammatory ODE model."""

import numpy as np
import pandas as pd
import pytest

from src.biotwin.models.biological_ode import (
    BiologicalParameters,
    inflammatory_ode,
    simulate_trajectory,
)


@pytest.fixture
def params() -> BiologicalParameters:
    return BiologicalParameters(
        alpha_il6=2.0,
        k_il6=0.5,
        alpha_crp=1.5,
        k_crp=0.2,
    )


def test_inflammatory_ode_derivatives(params: BiologicalParameters) -> None:
    derivatives = inflammatory_ode(
        t=0.0,
        state=np.array([4.0, 3.0]),
        stimulus=2.0,
        params=params,
    )

    assert derivatives == [2.0, 5.4]


def test_positive_stimulus_produces_il6(params: BiologicalParameters) -> None:
    d_il6_dt, _ = inflammatory_ode(0.0, np.array([0.0, 0.0]), 1.0, params)

    assert d_il6_dt > 0


def test_positive_il6_produces_crp(params: BiologicalParameters) -> None:
    _, d_crp_dt = inflammatory_ode(0.0, np.array([2.0, 0.0]), 0.0, params)

    assert d_crp_dt > 0


def test_zero_stimulus_causes_decay(params: BiologicalParameters) -> None:
    trajectory = simulate_trajectory(
        initial_il6=10.0,
        initial_crp=8.0,
        stimulus=np.zeros(301),
        dt=0.1,
        params=params,
    )

    assert trajectory["il6"].iloc[-1] < trajectory["il6"].iloc[0]
    assert trajectory["crp"].iloc[-1] < trajectory["crp"].iloc[0]


def test_coarse_time_step_preserves_finite_nonnegative_decay() -> None:
    fast_decay = BiologicalParameters(1.0, 0.8, 1.0, 0.6)
    trajectory = simulate_trajectory(10.0, 5.0, np.zeros(4), 10.0, fast_decay)

    assert np.isfinite(trajectory[["il6", "crp"]].to_numpy()).all()
    assert (trajectory[["il6", "crp"]] >= 0).all().all()
    assert trajectory["il6"].iloc[-1] < trajectory["il6"].iloc[0]
    assert trajectory["crp"].iloc[-1] < trajectory["crp"].iloc[0]


def test_repeated_simulation_is_deterministic(params: BiologicalParameters) -> None:
    stimulus = np.linspace(0.0, 2.0, 50)
    first = simulate_trajectory(0.5, 0.25, stimulus, 0.1, params)
    second = simulate_trajectory(0.5, 0.25, stimulus, 0.1, params)

    pd.testing.assert_frame_equal(first, second)


def test_output_columns_and_length(params: BiologicalParameters) -> None:
    stimulus = np.array([0.0, 1.0, 0.5, 0.0])
    trajectory = simulate_trajectory(0.0, 0.0, stimulus, 0.25, params)

    assert trajectory.columns.tolist() == ["time", "stimulus", "il6", "crp"]
    assert len(trajectory) == len(stimulus)
    assert trajectory["time"].tolist() == [0.0, 0.25, 0.5, 0.75]


def test_trajectory_is_finite_and_nonnegative(params: BiologicalParameters) -> None:
    trajectory = simulate_trajectory(
        0.0,
        0.0,
        np.array([0.0, 1.0, 2.0, 1.0, 0.0]),
        0.1,
        params,
    )

    assert np.isfinite(trajectory[["time", "stimulus", "il6", "crp"]].to_numpy()).all()
    assert (trajectory[["il6", "crp"]] >= 0).all().all()


def test_larger_stimulus_produces_stronger_response(params: BiologicalParameters) -> None:
    low = simulate_trajectory(0.0, 0.0, np.ones(40), 0.1, params)
    high = simulate_trajectory(0.0, 0.0, np.full(40, 3.0), 0.1, params)

    assert high["il6"].iloc[-1] > low["il6"].iloc[-1]
    assert high["crp"].iloc[-1] > low["crp"].iloc[-1]


@pytest.mark.parametrize("invalid_value", [0.0, -1.0, np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("parameter_name", ["alpha_il6", "k_il6", "alpha_crp", "k_crp"])
def test_biological_parameters_must_be_finite_and_positive(
    invalid_value: float, parameter_name: str
) -> None:
    parameter_values = {
        "alpha_il6": 2.0,
        "k_il6": 0.5,
        "alpha_crp": 1.5,
        "k_crp": 0.2,
    }
    parameter_values[parameter_name] = invalid_value

    with pytest.raises(ValueError, match=parameter_name):
        BiologicalParameters(**parameter_values)


@pytest.mark.parametrize(
    ("initial_il6", "initial_crp"),
    [(-1.0, 0.0), (0.0, -1.0), (np.nan, 0.0), (0.0, np.inf)],
)
def test_invalid_initial_states_are_rejected(
    params: BiologicalParameters,
    initial_il6: float,
    initial_crp: float,
) -> None:
    with pytest.raises(ValueError, match="initial"):
        simulate_trajectory(initial_il6, initial_crp, np.zeros(2), 0.1, params)


@pytest.mark.parametrize("dt", [0.0, -0.1, np.nan, np.inf])
def test_nonpositive_or_nonfinite_dt_is_rejected(
    params: BiologicalParameters, dt: float
) -> None:
    with pytest.raises(ValueError, match="dt"):
        simulate_trajectory(0.0, 0.0, np.zeros(2), dt, params)


@pytest.mark.parametrize(
    "stimulus",
    [np.array([0.0, np.nan]), np.array([0.0, np.inf]), np.array([0.0, -1.0])],
)
def test_invalid_stimulus_is_rejected(
    params: BiologicalParameters, stimulus: np.ndarray
) -> None:
    with pytest.raises(ValueError, match="stimulus"):
        simulate_trajectory(0.0, 0.0, stimulus, 0.1, params)


def test_empty_or_multidimensional_stimulus_is_rejected(
    params: BiologicalParameters,
) -> None:
    for stimulus in (np.array([]), np.zeros((2, 2))):
        with pytest.raises(ValueError, match="one-dimensional"):
            simulate_trajectory(0.0, 0.0, stimulus, 0.1, params)