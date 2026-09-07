"""Classical consider-covariance analysis for linear-loss OD results."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .schema import OptimizerOutput, ParameterRole

FloatArray = NDArray[np.float64]


def _covariance(
    value: FloatArray,
    size: int,
    name: str,
    *,
    positive_definite: bool,
) -> FloatArray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (size, size):
        raise ValueError(f"{name} must have shape {(size, size)}, got {matrix.shape}")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must contain only finite values")
    if not np.allclose(matrix, matrix.T, rtol=1e-10, atol=1e-12):
        raise ValueError(f"{name} must be symmetric")
    eigenvalues = np.linalg.eigvalsh(matrix)
    scale = float(np.max(np.abs(eigenvalues), initial=0.0))
    tolerance = 100.0 * np.finfo(np.float64).eps * max(size, 1) * scale
    if positive_definite:
        try:
            np.linalg.cholesky(matrix)
        except np.linalg.LinAlgError as exc:
            raise ValueError(f"{name} must be positive definite") from exc
    elif np.any(eigenvalues < -tolerance):
        raise ValueError(f"{name} must be positive semidefinite")
    return np.ascontiguousarray(matrix)


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

    if output.loss != "linear":
        raise ValueError("consider covariance requires an OptimizerOutput with linear loss")
    jacobian = np.asarray(output.jacobian, dtype=np.float64)
    if jacobian.ndim != 2 or not np.all(np.isfinite(jacobian)):
        raise ValueError("optimizer Jacobian must be a finite two-dimensional matrix")
    if jacobian.shape[1] != len(output.parameter_names):
        raise ValueError("optimizer Jacobian columns must match parameter_names")
    parameters = np.asarray(output.parameters, dtype=np.float64)
    if parameters.shape != (len(output.parameter_names),) or not np.all(
        np.isfinite(parameters)
    ):
        raise ValueError("optimizer parameters must be finite and match parameter_names")
    if len(set(output.parameter_names)) != len(output.parameter_names):
        raise ValueError("optimizer parameter_names must be unique")
    if len(output.parameter_roles) != len(output.parameter_names):
        raise ValueError("parameter_roles must match parameter_names")
    if any(not isinstance(role, ParameterRole) for role in output.parameter_roles):
        raise TypeError("parameter_roles must contain ParameterRole values")

    estimated = np.array(
        [
            index
            for index, role in enumerate(output.parameter_roles)
            if role == ParameterRole.ESTIMATE
        ],
        dtype=np.intp,
    )
    considered = np.array(
        [
            index
            for index, role in enumerate(output.parameter_roles)
            if role == ParameterRole.CONSIDER
        ],
        dtype=np.intp,
    )
    if estimated.size == 0:
        raise ValueError("consider covariance requires at least one estimated parameter")
    p_estimated = _covariance(
        prior_estimated_covariance,
        estimated.size,
        "prior_estimated_covariance",
        positive_definite=True,
    )
    p_consider = _covariance(
        prior_consider_covariance,
        considered.size,
        "prior_consider_covariance",
        positive_definite=False,
    )

    h_estimated = jacobian[:, estimated]
    h_consider = jacobian[:, considered]
    information = h_estimated.T @ h_estimated + np.linalg.inv(p_estimated)
    try:
        unconsidered = np.linalg.inv(information)
    except np.linalg.LinAlgError as exc:
        raise ValueError("estimated-parameter information matrix is singular") from exc
    sensitivity = -unconsidered @ h_estimated.T @ h_consider
    consider_covariance = unconsidered + sensitivity @ p_consider @ sensitivity.T
    perturbation = sensitivity * np.sqrt(np.diag(p_consider))
    return (
        np.ascontiguousarray(unconsidered),
        np.ascontiguousarray(consider_covariance),
        np.ascontiguousarray(sensitivity),
        np.ascontiguousarray(perturbation),
    )
