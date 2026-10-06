"""Independent qualitative behavior checks for the biological ODE API."""

import numpy as np
import pandas as pd
import pytest

from src.biotwin.models.biological_ode import (
    BiologicalParameters,
    simulate_trajectory,
)


@pytest.fixture
def params() -> BiologicalParameters:
    return BiologicalParameters(
        alpha_il6=1.8,
        k_il6=0.45,
        alpha_crp=1.2,
        k_crp=0.18,
    )


def test_zero_stimulus_eventually_decays_without_invalid_values(
    params: BiologicalParameters,
) -> None:
    trajectory = simulate_trajectory(8.0, 3.0, np.zeros(201), 0.1, params)
    concentrations = trajectory[["il6", "crp"]].to_numpy()

    assert trajectory["il6"].iloc[-1] < trajectory["il6"].iloc[0]
    assert trajectory["crp"].iloc[-1] < trajectory["crp"].iloc[0]
    assert np.isfinite(concentrations).all()
    assert (concentrations >= 0).all()


def test_constant_positive_stimulus_increases_both_markers(
    params: BiologicalParameters,
) -> None:
    trajectory = simulate_trajectory(0.0, 0.0, np.ones(121), 0.1, params)
    concentrations = trajectory[["il6", "crp"]].to_numpy()

    assert trajectory["il6"].iloc[-1] > trajectory["il6"].iloc[0]
    assert trajectory["crp"].iloc[-1] > trajectory["crp"].iloc[0]
    assert np.isfinite(concentrations).all()
    assert (concentrations >= 0).all()


def test_higher_constant_stimulus_produces_greater_response(
    params: BiologicalParameters,
) -> None:
    lower = simulate_trajectory(0.0, 0.0, np.full(101, 0.5), 0.1, params)
    higher = simulate_trajectory(0.0, 0.0, np.full(101, 2.0), 0.1, params)

    assert higher["il6"].iloc[-1] > lower["il6"].iloc[-1]
    assert higher["crp"].iloc[-1] > lower["crp"].iloc[-1]


def test_temporary_pulse_produces_delayed_response_then_decline(
    params: BiologicalParameters,
) -> None:
    stimulus = np.zeros(151)
    stimulus[20:50] = 2.0
    trajectory = simulate_trajectory(0.0, 0.0, stimulus, 0.1, params)
    il6 = trajectory["il6"].to_numpy()
    crp = trajectory["crp"].to_numpy()
    il6_peak = int(np.argmax(il6))
    crp_peak = int(np.argmax(crp))

    assert il6[30:55].max() > il6[19]
    assert crp[30:60].max() > crp[19]
    assert crp_peak > il6_peak
    assert il6[-1] < il6[il6_peak]
    assert crp[-1] < crp[crp_peak]


def test_stronger_il6_production_parameter_increases_il6_response(
    params: BiologicalParameters,
) -> None:
    stronger_production = BiologicalParameters(
        alpha_il6=3.6,
        k_il6=params.k_il6,
        alpha_crp=params.alpha_crp,
        k_crp=params.k_crp,
    )
    stimulus = np.ones(101)
    reference = simulate_trajectory(0.0, 0.0, stimulus, 0.1, params)
    stronger = simulate_trajectory(0.0, 0.0, stimulus, 0.1, stronger_production)

    assert stronger["il6"].iloc[-1] > reference["il6"].iloc[-1]


def test_identical_simulations_are_numerically_identical(
    params: BiologicalParameters,
) -> None:
    stimulus = np.concatenate((np.zeros(10), np.ones(20), np.zeros(30)))
    first = simulate_trajectory(0.2, 0.1, stimulus, 0.2, params)
    second = simulate_trajectory(0.2, 0.1, stimulus, 0.2, params)

    pd.testing.assert_frame_equal(first, second, check_exact=True)


def test_time_grid_matches_stimulus_length_and_spacing(
    params: BiologicalParameters,
) -> None:
    stimulus = np.linspace(0.0, 1.0, 17)
    dt = 0.25
    trajectory = simulate_trajectory(0.0, 0.0, stimulus, dt, params)

    assert len(trajectory) == len(stimulus)
    assert trajectory["time"].iloc[0] == 0.0
    assert (trajectory["time"].diff().dropna() > 0).all()
    np.testing.assert_allclose(trajectory["time"].diff().dropna(), dt)


@pytest.mark.parametrize(
    "stimulus",
    [np.zeros(41), np.ones(41), np.concatenate((np.zeros(10), np.ones(10), np.zeros(21)))],
)
def test_simulated_concentrations_remain_finite_and_nonnegative(
    params: BiologicalParameters,
    stimulus: np.ndarray,
) -> None:
    trajectory = simulate_trajectory(1.0, 0.5, stimulus, 0.2, params)
    concentrations = trajectory[["il6", "crp"]].to_numpy()

    assert np.isfinite(concentrations).all()
    assert (concentrations >= 0).all()