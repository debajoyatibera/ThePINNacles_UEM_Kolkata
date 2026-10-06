"""Simplified Physics-Informed Neural Network for the CRP/IL-6 system.

This module defines a small CPU-friendly synthetic PINN architecture for a
competition/research proof-of-concept. It is not clinically validated and must
not be used for diagnosis, treatment, or medical decision-making.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from biotwin.models.biological_ode import BiologicalParameters


class CRPIL6PINN(nn.Module):
    """Simple feed-forward network mapping [time, stimulus] to [IL6, CRP]."""

    def __init__(self, hidden_dim: int = 32) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return predictions shaped [N, 2] for IL6 and CRP."""
        if inputs.dim() != 2 or inputs.shape[1] != 2:
            raise ValueError("inputs must have shape [N, 2] with columns [time, stimulus]")
        if not torch.is_tensor(inputs):
            raise TypeError("inputs must be a torch.Tensor")
        return self.network(inputs)


class TimeNormalizedPINN(nn.Module):
    """Wrap a CRP/IL-6 PINN with affine time scaling to the range [0, 1].

    Callers continue to provide physical time in hours. Because scaling occurs
    inside ``forward``, autograd applies the chain rule and physics residuals
    remain expressed per physical hour.
    """

    def __init__(self, model: nn.Module, time_min: float, time_max: float) -> None:
        super().__init__()
        if not torch.isfinite(torch.tensor([time_min, time_max])).all() or time_max <= time_min:
            raise ValueError("time_max must be finite and greater than finite time_min")
        self.model = model
        self.register_buffer("time_min", torch.tensor(float(time_min), dtype=torch.float32))
        self.register_buffer("time_range", torch.tensor(float(time_max - time_min), dtype=torch.float32))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(inputs):
            raise TypeError("inputs must be a torch.Tensor")
        if inputs.dim() != 2 or inputs.shape[1] != 2:
            raise ValueError("inputs must have shape [N, 2] with columns [time, stimulus]")
        normalized_time = (inputs[:, 0] - self.time_min) / self.time_range
        normalized_inputs = torch.stack([normalized_time, inputs[:, 1]], dim=1)
        return self.model(normalized_inputs)


def physics_residual(
    model: nn.Module,
    time: torch.Tensor,
    stimulus: torch.Tensor,
    params: BiologicalParameters,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the IL6 and CRP residuals from the coupled ODE.

    Uses automatic differentiation to obtain the time derivatives of the model
    output and compares them with the same simplified ODE structure used in the
    biological model.
    """
    if not isinstance(params, BiologicalParameters):
        raise ValueError("params must be a BiologicalParameters instance")
    if not torch.is_tensor(time) or not torch.is_tensor(stimulus):
        raise TypeError("time and stimulus must be torch tensors")
    if time.shape != stimulus.shape:
        raise ValueError("time and stimulus must have the same shape")
    if time.dim() != 1:
        raise ValueError("time and stimulus must be one-dimensional tensors")

    time = time.to(dtype=torch.float32).clone().detach().requires_grad_(True)
    stimulus = stimulus.to(dtype=torch.float32)
    model_input = torch.stack([time, stimulus], dim=1)

    model.eval()
    with torch.set_grad_enabled(True):
        predictions = model(model_input)
        if predictions.shape != (time.shape[0], 2):
            raise ValueError("model output must have shape [N, 2]")

        il6_pred = predictions[:, 0]
        crp_pred = predictions[:, 1]

        il6_dt = torch.autograd.grad(
            outputs=il6_pred,
            inputs=time,
            grad_outputs=torch.ones_like(il6_pred),
            retain_graph=True,
            create_graph=True,
        )[0]
        crp_dt = torch.autograd.grad(
            outputs=crp_pred,
            inputs=time,
            grad_outputs=torch.ones_like(crp_pred),
            retain_graph=True,
            create_graph=True,
        )[0]

    residual_il6 = il6_dt - (params.alpha_il6 * stimulus - params.k_il6 * il6_pred)
    residual_crp = crp_dt - (params.alpha_crp * il6_pred - params.k_crp * crp_pred)
    return residual_il6, residual_crp


def physics_loss(
    model: nn.Module,
    time: torch.Tensor,
    stimulus: torch.Tensor,
    params: BiologicalParameters,
) -> torch.Tensor:
    """Return the mean squared residual loss used for the next training phase."""
    residual_il6, residual_crp = physics_residual(model, time, stimulus, params)
    return torch.mean(residual_il6 ** 2) + torch.mean(residual_crp ** 2)
