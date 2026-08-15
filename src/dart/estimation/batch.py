"""Robust post-pass offset/bias estimator using the shared measurement model."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import least_squares

from ..measurements import MeasurementModel, prepare_batch_cache, wrap_angle_rad
from ..types import Estimate, MeasurementMode, RFObservation, TLEContext


@dataclass(frozen=True, slots=True)
class BatchConfig:
    offset_bounds_s: tuple[float, float] = (-120.0, 120.0)
    frequency_bias_bounds_hz: tuple[float, float] = (-100_000.0, 100_000.0)
    doppler_std_hz: float = 500.0
    phase_std_rad: float = 0.1
    robust_loss: str = "soft_l1"
    robust_scale_doppler_hz: float = 700.0
    max_scaled_jacobian_condition: float = 1e12
    offset_starts_s: tuple[float, ...] = (-90.0, -60.0, -30.0, 0.0, 30.0, 60.0, 90.0)
    prior: np.ndarray | None = field(default=None, repr=False)
    prior_std: np.ndarray | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.doppler_std_hz <= 0.0 or self.phase_std_rad <= 0.0:
            raise ValueError("measurement standard deviations must be positive")


@dataclass(frozen=True, slots=True)
class BatchFit:
    estimate: Estimate
    residuals: np.ndarray
    doppler_rmse_hz: float
    weighted_ssr: float
    robust_cost: float
    success: bool
    message: str
    observations_used: int
    jacobian_condition: float
    jacobian_rank: int
    at_bound: bool


def _mode_for(observations: list[RFObservation], requested: MeasurementMode | None) -> MeasurementMode:
    if requested is not None:
        return requested
    return (
        MeasurementMode.DOPPLER_PHASE
        if observations and any(item.phase_rad is not None for item in observations)
        else MeasurementMode.DOPPLER
    )


def fit_batch(
    context: TLEContext,
    observations: list[RFObservation],
    config: BatchConfig = BatchConfig(),
    *,
    mode: MeasurementMode | None = None,
    model: MeasurementModel | None = None,
) -> BatchFit:
    usable = [item for item in observations if item.valid and item.doppler_hz is not None]
    if not usable:
        raise ValueError("no valid Doppler observations")
    selected = _mode_for(usable, mode)
    if selected is MeasurementMode.DOPPLER_PHASE:
        if not any(item.phase_rad is not None for item in usable):
            raise ValueError("no valid Doppler+phase observations")
    model = MeasurementModel() if model is None else model
    dimension = 2 if selected is MeasurementMode.DOPPLER else 3
    lower = [config.offset_bounds_s[0], config.frequency_bias_bounds_hz[0]]
    upper = [config.offset_bounds_s[1], config.frequency_bias_bounds_hz[1]]
    if dimension == 3:
        lower.append(-np.pi)
        upper.append(np.pi)
    lower_array, upper_array = np.asarray(lower), np.asarray(upper)
    cache = prepare_batch_cache(context, usable)
    measured_doppler = np.asarray([item.doppler_hz for item in usable], dtype=float)
    phase_indices = np.asarray(
        [index for index, item in enumerate(usable) if item.phase_rad is not None],
        dtype=int,
    )
    measured_phase = np.asarray(
        [usable[index].phase_rad for index in phase_indices], dtype=float
    )

    def residual(state: np.ndarray) -> np.ndarray:
        predicted = model.predict_many(context, state, selected, cache)
        channels = [(measured_doppler - predicted[:, 0]) / config.doppler_std_hz]
        if selected is MeasurementMode.DOPPLER_PHASE:
            phase_residual = wrap_angle_rad(
                measured_phase - predicted[phase_indices, 1]
            )
            channels.append(phase_residual / config.phase_std_rad)
        result = np.concatenate(channels)
        if config.prior is not None and config.prior_std is not None:
            prior = np.asarray(config.prior, dtype=float)[:dimension]
            prior_residual = state - prior
            if dimension == 3:
                prior_residual[2] = float(wrap_angle_rad(prior_residual[2]))
            result = np.concatenate((result, prior_residual / np.asarray(config.prior_std)[:dimension]))
        return result

    # SGP4/time conversion is not usefully differentiated at scipy's default
    # ~1e-8-second step.  Use parameter-specific absolute central differences
    # so the phase-lag column remains observable at and around offset zero.
    jacobian_steps = np.array([1e-3, 1.0] + ([1e-4] if dimension == 3 else []))

    def residual_difference(right: np.ndarray, left: np.ndarray) -> np.ndarray:
        difference = right - left
        if selected is MeasurementMode.DOPPLER_PHASE:
            start = len(usable)
            stop = start + len(phase_indices)
            difference[start:stop] = (
                wrap_angle_rad(
                    (right[start:stop] - left[start:stop]) * config.phase_std_rad
                )
                / config.phase_std_rad
            )
            if config.prior is not None and config.prior_std is not None:
                phase_prior_index = len(difference) - 1
                prior_scale = float(np.asarray(config.prior_std)[2])
                difference[phase_prior_index] = float(
                    wrap_angle_rad(
                        (right[phase_prior_index] - left[phase_prior_index])
                        * prior_scale
                    )
                    / prior_scale
                )
        return difference

    def jacobian(state: np.ndarray) -> np.ndarray:
        columns = []
        for index, step in enumerate(jacobian_steps):
            right = state.copy()
            left = state.copy()
            right[index] = min(right[index] + step, upper_array[index])
            left[index] = max(left[index] - step, lower_array[index])
            width = right[index] - left[index]
            columns.append(
                residual_difference(residual(right), residual(left)) / width
            )
        return np.column_stack(columns)

    starts = config.offset_starts_s or (0.0,)
    candidates = []
    for offset in starts:
        initial = np.zeros(dimension)
        initial[0] = np.clip(offset, lower_array[0], upper_array[0])
        candidate = least_squares(
            residual,
            initial,
            jac=jacobian,
            bounds=(lower_array, upper_array),
            loss=config.robust_loss,
            f_scale=config.robust_scale_doppler_hz / config.doppler_std_hz,
            x_scale=np.array([30.0, 5_000.0] + ([1.0] if dimension == 3 else [])),
            ftol=1e-10,
            xtol=1e-10,
            gtol=1e-10,
        )
        if np.isfinite(candidate.cost):
            candidates.append(candidate)
    if not candidates:
        raise RuntimeError("all batch least-squares starts returned non-finite costs")
    successful = [candidate for candidate in candidates if candidate.success]
    best = min(successful or candidates, key=lambda candidate: float(candidate.cost))
    raw_residual = residual(best.x)
    doppler_residual_hz = raw_residual[: len(usable)] * config.doppler_std_hz
    doppler_rmse_hz = float(np.sqrt(np.mean(doppler_residual_hz**2)))
    best_ssr = float(np.sum(raw_residual**2))

    jacobian = jacobian(best.x)
    f_scale = config.robust_scale_doppler_hz / config.doppler_std_hz
    squared_scaled = np.square(raw_residual / f_scale)
    if config.robust_loss == "linear":
        weights = np.ones_like(raw_residual)
    elif config.robust_loss == "soft_l1":
        weights = 1.0 / np.sqrt(1.0 + squared_scaled)
    elif config.robust_loss == "huber":
        weights = np.where(squared_scaled <= 1.0, 1.0, 1.0 / np.sqrt(squared_scaled))
    elif config.robust_loss == "cauchy":
        weights = 1.0 / (1.0 + squared_scaled)
    elif config.robust_loss == "arctan":
        weights = 1.0 / (1.0 + squared_scaled**2)
    else:
        weights = np.ones_like(raw_residual)
    information = jacobian.T @ (weights[:, None] * jacobian)
    dof = max(1, len(raw_residual) - dimension)
    mse = float(np.sum(weights * raw_residual**2) / dof)
    covariance = np.linalg.pinv(information, rcond=1e-12) * mse
    parameter_scale = np.array([30.0, 5_000.0] + ([1.0] if dimension == 3 else []))
    scaled_jacobian = np.sqrt(weights)[:, None] * jacobian * parameter_scale
    singular_values = np.linalg.svd(scaled_jacobian, compute_uv=False)
    rank = int(np.linalg.matrix_rank(scaled_jacobian))
    condition = float(
        np.inf if singular_values[-1] == 0.0 else singular_values[0] / singular_values[-1]
    )
    bound_tolerance = 1e-7 * np.maximum(1.0, upper_array - lower_array)
    at_bound = bool(
        np.any(best.x - lower_array <= bound_tolerance)
        or np.any(upper_array - best.x <= bound_tolerance)
    )
    healthy = bool(
        best.success
        and rank == dimension
        and np.isfinite(condition)
        and condition <= config.max_scaled_jacobian_condition
        and not at_bound
    )
    estimate = Estimate(
        epoch=usable[-1].epoch,
        mode=selected,
        offset_s=float(best.x[0]),
        frequency_bias_hz=float(best.x[1]),
        phase_bias_rad=float(wrap_angle_rad(best.x[2])) if dimension == 3 else None,
        covariance=covariance,
        healthy=healthy,
        reason=(
            None
            if healthy
            else f"success={best.success}, rank={rank}/{dimension}, condition={condition:.3g}, at_bound={at_bound}"
        ),
    )
    return BatchFit(
        estimate=estimate,
        residuals=raw_residual,
        doppler_rmse_hz=doppler_rmse_hz,
        weighted_ssr=best_ssr,
        robust_cost=float(best.cost),
        success=bool(best.success),
        message=str(best.message),
        observations_used=len(usable),
        jacobian_condition=condition,
        jacobian_rank=rank,
        at_bound=at_bound,
    )
