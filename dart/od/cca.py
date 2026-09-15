"""Classical consider covariance, with Rust as the numerical authority."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from dart.forward_models import _native

from .schema import OptimizerOutput, ParameterRole, ParameterSpec

FloatArray = NDArray[np.float64]


def _covariance(
    value: FloatArray,
    size: int,
    name: str,
) -> FloatArray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (size, size):
        raise ValueError(f"{name} must have shape {(size, size)}, got {matrix.shape}")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must contain only finite values")
    if not np.allclose(matrix, matrix.T, rtol=1e-10, atol=1e-12):
        raise ValueError(f"{name} must be symmetric")
    if size:
        try:
            np.linalg.cholesky(matrix)
        except np.linalg.LinAlgError as exc:
            raise ValueError(f"{name} must be positive definite") from exc
    return np.ascontiguousarray(matrix)


def _partition(output: OptimizerOutput) -> tuple[np.ndarray, np.ndarray]:
    if len(output.parameter_roles) != len(output.parameter_names):
        raise ValueError("parameter_roles must match parameter_names")
    if any(not isinstance(role, ParameterRole) for role in output.parameter_roles):
        raise TypeError("parameter_roles must contain ParameterRole values")
    estimated = np.array(
        [
            i
            for i, role in enumerate(output.parameter_roles)
            if role == ParameterRole.ESTIMATE
        ],
        dtype=np.intp,
    )
    considered = np.array(
        [
            i
            for i, role in enumerate(output.parameter_roles)
            if role == ParameterRole.CONSIDER
        ],
        dtype=np.intp,
    )
    if estimated.size == 0:
        raise ValueError(
            "consider covariance requires at least one estimated parameter"
        )
    return estimated, considered


def _validate_output(output: OptimizerOutput) -> FloatArray:
    jacobian = np.asarray(output.jacobian, dtype=np.float64)
    if jacobian.ndim != 2 or not np.all(np.isfinite(jacobian)):
        raise ValueError("optimizer Jacobian must be a finite two-dimensional matrix")
    if jacobian.shape[1] != len(output.parameter_names):
        raise ValueError("optimizer Jacobian columns must match parameter_names")
    parameters = np.asarray(output.parameters, dtype=np.float64)
    if parameters.shape != (len(output.parameter_names),) or not np.all(
        np.isfinite(parameters)
    ):
        raise ValueError(
            "optimizer parameters must be finite and match parameter_names"
        )
    if len(set(output.parameter_names)) != len(output.parameter_names):
        raise ValueError("optimizer parameter_names must be unique")
    return jacobian


def compute_consider_covariance(
    output: OptimizerOutput,
    prior_estimated_covariance: FloatArray,
    prior_consider_covariance: FloatArray,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """Return unconsidered/consider covariances, sensitivity, and perturbations.

    Prior matrices follow the relative ordering of ``ESTIMATE`` and
    ``CONSIDER`` columns in ``output.parameter_roles``. Fixed columns are
    excluded.
    """

    jacobian = _validate_output(output)
    estimated, considered = _partition(output)
    p_estimated = _covariance(
        prior_estimated_covariance,
        estimated.size,
        "prior_estimated_covariance",
    )
    p_consider = _covariance(
        prior_consider_covariance,
        considered.size,
        "prior_consider_covariance",
    )
    native = _native_consider_covariance(
        jacobian, estimated, considered, p_estimated, p_consider
    )
    unconsidered, consider_covariance, sensitivity, perturbation, _, _ = native
    return unconsidered, consider_covariance, sensitivity, perturbation


def _native_consider_covariance(
    jacobian: FloatArray,
    estimated: np.ndarray,
    considered: np.ndarray,
    p_estimated: FloatArray,
    p_consider: FloatArray,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, FloatArray, int]:
    native = _native.consider_covariance(
        jacobian[:, estimated].tolist(),
        jacobian[:, considered].tolist(),
        p_estimated.tolist(),
        p_consider.tolist(),
    )
    unconsidered, consider_covariance, sensitivity, perturbation, joint, rank = native
    return (
        np.asarray(unconsidered, dtype=np.float64).reshape(
            estimated.size, estimated.size
        ),
        np.asarray(consider_covariance, dtype=np.float64).reshape(
            estimated.size, estimated.size
        ),
        np.asarray(sensitivity, dtype=np.float64).reshape(
            estimated.size, considered.size
        ),
        np.asarray(perturbation, dtype=np.float64).reshape(
            estimated.size, considered.size
        ),
        np.asarray(joint, dtype=np.float64).reshape(
            estimated.size + considered.size, estimated.size + considered.size
        ),
        int(rank),
    )


def full_consider_covariance(
    output: OptimizerOutput, specifications: tuple[ParameterSpec, ...]
) -> tuple[FloatArray, int]:
    """Assemble the joint estimated/consider covariance in profile order."""
    if tuple(spec.name for spec in specifications) != output.parameter_names:
        raise ValueError(
            "covariance specifications must match optimizer parameter order"
        )
    if any(spec.role == ParameterRole.FIXED for spec in specifications):
        raise ValueError("covariance-enabled profiles may not contain fixed parameters")
    if any(spec.prior_standard_uncertainty is None for spec in specifications):
        raise ValueError("covariance-enabled parameters require prior uncertainty")
    estimated, considered = _partition(output)
    sigmas = np.array(
        [spec.prior_standard_uncertainty for spec in specifications], dtype=np.float64
    )
    prior_estimated = np.diag(np.square(sigmas[estimated]))
    prior_consider = np.diag(np.square(sigmas[considered]))
    jacobian = _validate_output(output)
    unconsidered, posterior, sensitivity, perturbation, joint, rank = (
        _native_consider_covariance(
            jacobian, estimated, considered, prior_estimated, prior_consider
        )
    )
    del unconsidered, posterior, sensitivity, perturbation
    covariance = np.zeros((len(specifications), len(specifications)))
    active = np.concatenate((estimated, considered))
    covariance[np.ix_(active, active)] = joint
    if not np.all(np.isfinite(covariance)):
        raise ValueError("full consider covariance contains non-finite values")
    try:
        np.linalg.cholesky(covariance)
    except np.linalg.LinAlgError as exc:
        raise ValueError("full consider covariance must be positive definite") from exc
    return np.ascontiguousarray(covariance), rank
