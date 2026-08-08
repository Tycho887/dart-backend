"""Numerical operations shared by the production solvers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SPEED_OF_LIGHT_M_S = 299_792_458.0


def doppler_offset_hz(range_rate_m_s, carrier_hz: float, bias_hz: float = 0.0):
    """Convert range rate to carrier-frequency offset."""

    return -np.asarray(range_rate_m_s) * float(carrier_hz) / SPEED_OF_LIGHT_M_S + bias_hz


@dataclass(frozen=True, slots=True)
class WeightedLinearAlgebra:
    """One robust weighting of a raw residual Jacobian."""

    covariance: np.ndarray
    rank: int
    condition: float


def project_strictly_within_bounds(
    preferred: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> np.ndarray:
    """Return a finite least-squares initial point strictly inside valid bounds."""

    if preferred.shape != lower.shape or lower.shape != upper.shape:
        raise ValueError("initial point and bounds must have matching shapes")
    if not (np.isfinite(preferred).all() and np.isfinite(lower).all() and np.isfinite(upper).all()):
        raise ValueError("initial point and bounds must be finite")
    if np.any(lower >= upper):
        raise ValueError("lower bounds must be strictly less than upper bounds")
    interior_lower = np.nextafter(lower, upper)
    interior_upper = np.nextafter(upper, lower)
    if np.any(interior_lower > interior_upper):
        raise ValueError("bounds must contain a representable interior point")
    return np.minimum(np.maximum(preferred, interior_lower), interior_upper)


def robust_weights(residual: np.ndarray, loss: str, scale: float) -> np.ndarray:
    """Return non-negative IRLS weights for already-scaled residuals."""

    if not np.isfinite(residual).all() or not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("robust residuals and scale must be finite with a positive scale")
    squared = np.square(residual / scale)
    if loss == "linear":
        return np.ones_like(residual)
    if loss == "soft_l1":
        return 1.0 / np.sqrt(1.0 + squared)
    if loss == "huber":
        return np.where(squared <= 1.0, 1.0, 1.0 / np.sqrt(squared))
    if loss == "cauchy":
        return 1.0 / (1.0 + squared)
    if loss == "arctan":
        return 1.0 / (1.0 + squared**2)
    raise ValueError(f"unsupported robust loss {loss}")


def weighted_linear_algebra(
    raw_jacobian: np.ndarray,
    scaled_residual: np.ndarray,
    weights: np.ndarray,
) -> WeightedLinearAlgebra:
    """Compute covariance and observability after applying robust weights once."""

    if raw_jacobian.ndim != 2:
        raise ValueError("raw_jacobian must be two-dimensional")
    if len(scaled_residual) != len(raw_jacobian) or len(weights) != len(raw_jacobian):
        raise ValueError("jacobian, residuals, and weights must have matching row counts")
    if not np.isfinite(raw_jacobian).all() or not np.isfinite(scaled_residual).all():
        raise ValueError("jacobian and residuals must be finite")
    if not np.isfinite(weights).all() or np.any(weights < 0.0):
        raise ValueError("robust weights must be finite and non-negative")
    weighted_jacobian = np.sqrt(weights)[:, None] * raw_jacobian
    information = weighted_jacobian.T @ weighted_jacobian
    parameter_count = raw_jacobian.shape[1]
    weighted_ssr = float(np.sum(weights * np.square(scaled_residual)))
    covariance = np.linalg.pinv(information, rcond=1e-12)
    covariance *= weighted_ssr / max(1, len(scaled_residual) - parameter_count)
    singular_values = np.linalg.svd(weighted_jacobian, compute_uv=False)
    rank = int(np.linalg.matrix_rank(weighted_jacobian))
    condition = float(
        np.inf
        if rank < parameter_count or singular_values[-1] == 0.0
        else singular_values[0] / singular_values[-1]
    )
    return WeightedLinearAlgebra(covariance, rank, condition)


def at_any_bound(parameters: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> bool:
    """Report whether a solution is numerically adjacent to any fit bound."""

    tolerance = 1e-7 * np.maximum(1.0, upper - lower)
    return bool(np.any(parameters - lower <= tolerance) or np.any(upper - parameters <= tolerance))
