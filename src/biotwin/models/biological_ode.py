"""Simplified synthetic inflammatory dynamics for a competition/research PoC.

This biological model is synthetic and is not clinically validated. It must
not be used for diagnosis, treatment, or medical decision-making.
"""

from dataclasses import dataclass, fields

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BiologicalParameters:
    """Positive rate and production parameters for the two-state model."""

    alpha_il6: float
    k_il6: float
    alpha_crp: float
    k_crp: float

    def __post_init__(self) -> None:
        for parameter in fields(self):
            value = getattr(self, parameter.name)
            try:
                valid = bool(np.isfinite(value) and value > 0)
            except (TypeError, ValueError):
                valid = False
            if not valid:
                raise ValueError(f"{parameter.name} must be finite and strictly positive")


def inflammatory_ode(
    t: float,
    state: np.ndarray,
    stimulus: float,
    params: BiologicalParameters,
) -> list[float]:
    """Return ``[dIL6/dt, dCRP/dt]`` for one state and stimulus value."""
    if not np.isfinite(t):
        raise ValueError("t must be finite")
    if not isinstance(params, BiologicalParameters):
        raise ValueError("params must be a BiologicalParameters instance")

    state_values = np.asarray(state, dtype=float)
    if state_values.shape != (2,) or not np.isfinite(state_values).all():
        raise ValueError("state must contain two finite values: IL6 and CRP")
    if (state_values < 0).any():
        raise ValueError("IL6 and CRP state values must be non-negative")
    if not np.isfinite(stimulus) or stimulus < 0:
        raise ValueError("stimulus must be finite and non-negative")

    il6, crp = state_values
    d_il6_dt = params.alpha_il6 * stimulus - params.k_il6 * il6
    d_crp_dt = params.alpha_crp * il6 - params.k_crp * crp
    return [float(d_il6_dt), float(d_crp_dt)]


def simulate_trajectory(
    initial_il6: float,
    initial_crp: float,
    stimulus: np.ndarray,
    dt: float,
    params: BiologicalParameters,
) -> pd.DataFrame:
    """Integrate the IL6/CRP system with RK4 over a stimulus time series.

    Each stimulus value corresponds to one output time, beginning at time zero.
    Stimulus is linearly interpolated between adjacent time points during the
    RK4 step. ``dt`` is the spacing between observations.
    """
    if not isinstance(params, BiologicalParameters):
        raise ValueError("params must be a BiologicalParameters instance")
    try:
        valid_dt = bool(np.isfinite(dt) and dt > 0)
    except (TypeError, ValueError):
        valid_dt = False
    if not valid_dt:
        raise ValueError("dt must be finite and strictly positive")

    initial_state = np.asarray([initial_il6, initial_crp], dtype=float)
    if not np.isfinite(initial_state).all():
        raise ValueError("initial IL6 and CRP must be finite")
    if (initial_state < 0).any():
        raise ValueError("initial IL6 and CRP must be non-negative")

    stimulus_values = np.asarray(stimulus, dtype=float)
    if stimulus_values.ndim != 1 or stimulus_values.size == 0:
        raise ValueError("stimulus must be a non-empty one-dimensional time series")
    if not np.isfinite(stimulus_values).all() or (stimulus_values < 0).any():
        raise ValueError("stimulus values must be finite and non-negative")

    state = initial_state.copy()
    trajectory = np.empty((stimulus_values.size, 2), dtype=float)
    trajectory[0] = state

    for index in range(stimulus_values.size - 1):
        first_stimulus = float(stimulus_values[index])
        last_stimulus = float(stimulus_values[index + 1])
        decay_rate = max(params.k_il6, params.k_crp)
        substep_count = max(1, int(np.ceil(dt * decay_rate / 0.5)))
        substep = dt / substep_count

        for substep_index in range(substep_count):
            start_fraction = substep_index / substep_count
            middle_fraction = (substep_index + 0.5) / substep_count
            end_fraction = (substep_index + 1) / substep_count
            start_stimulus = first_stimulus + (last_stimulus - first_stimulus) * start_fraction
            middle_stimulus = first_stimulus + (last_stimulus - first_stimulus) * middle_fraction
            end_stimulus = first_stimulus + (last_stimulus - first_stimulus) * end_fraction
            time = (index + start_fraction) * dt

            k1 = np.asarray(inflammatory_ode(time, state, start_stimulus, params))
            k2 = np.asarray(
                inflammatory_ode(
                    time + substep / 2.0,
                    state + substep * k1 / 2.0,
                    middle_stimulus,
                    params,
                )
            )
            k3 = np.asarray(
                inflammatory_ode(
                    time + substep / 2.0,
                    state + substep * k2 / 2.0,
                    middle_stimulus,
                    params,
                )
            )
            k4 = np.asarray(
                inflammatory_ode(
                    time + substep,
                    state + substep * k3,
                    end_stimulus,
                    params,
                )
            )
            state = state + (substep / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            if not np.isfinite(state).all():
                raise FloatingPointError("ODE integration produced a non-finite state")
            state = np.maximum(state, 0.0)

        trajectory[index + 1] = state

    if not np.isfinite(trajectory).all():
        raise FloatingPointError("ODE integration produced a non-finite trajectory")

    return pd.DataFrame(
        {
            "time": np.arange(stimulus_values.size, dtype=float) * dt,
            "stimulus": stimulus_values,
            "il6": trajectory[:, 0],
            "crp": trajectory[:, 1],
        }
    )