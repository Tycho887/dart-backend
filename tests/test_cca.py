"""Consider-covariance analysis contracts."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from dart.od import (
    OptimizerOutput,
    OrbitModel,
    ParameterRole,
    PriorSource,
    compute_consider_covariance,
)
from dart.od.cca import full_consider_covariance
from dart.od.schema import LossKind, ParameterSpec


def output(*, loss: LossKind = "linear") -> OptimizerOutput:
    return OptimizerOutput(
        model_kind=OrbitModel.SGP4,
        prior_source=PriorSource.TLE,
        parameter_names=("e1", "c1", "fixed", "e2", "c2"),
        parameter_roles=(
            ParameterRole.ESTIMATE,
            ParameterRole.CONSIDER,
            ParameterRole.FIXED,
            ParameterRole.ESTIMATE,
            ParameterRole.CONSIDER,
        ),
        parameters=np.zeros(5),
        residuals=np.zeros(3),
        jacobian=np.array(
            [
                [1.0, 0.5, 1000.0, 0.0, 0.2],
                [0.0, 0.3, 1000.0, 2.0, -0.1],
                [1.0, -0.2, 1000.0, 1.0, 0.4],
            ]
        ),
        cost=0.0,
        optimality=0.0,
        success=True,
        status=1,
        message="synthetic",
        function_evaluations=1,
        loss=loss,
    )


def test_consider_covariance_partitions_interleaved_roles() -> None:
    result = output()
    prior_estimated = np.diag([4.0, 9.0])
    prior_consider = np.array([[1.0, 0.4], [0.4, 4.0]])

    unconsidered, considered, sensitivity, perturbation = compute_consider_covariance(
        result, prior_estimated, prior_consider
    )

    h_estimated = result.jacobian[:, [0, 3]]
    h_consider = result.jacobian[:, [1, 4]]
    expected_unconsidered = np.linalg.inv(
        h_estimated.T @ h_estimated + np.linalg.inv(prior_estimated)
    )
    expected_sensitivity = -expected_unconsidered @ h_estimated.T @ h_consider
    np.testing.assert_allclose(unconsidered, expected_unconsidered)
    np.testing.assert_allclose(sensitivity, expected_sensitivity)
    np.testing.assert_allclose(
        considered,
        expected_unconsidered
        + expected_sensitivity @ prior_consider @ expected_sensitivity.T,
    )
    np.testing.assert_allclose(
        perturbation, expected_sensitivity * np.sqrt(np.diag(prior_consider))
    )


def test_consider_covariance_uses_unweighted_robust_jacobian_and_rejects_invalid_priors() -> (
    None
):
    linear = compute_consider_covariance(output(), np.eye(2), np.eye(2))
    robust = compute_consider_covariance(output(loss="huber"), np.eye(2), np.eye(2))
    for actual, expected in zip(robust, linear, strict=True):
        np.testing.assert_allclose(actual, expected)
    with pytest.raises(ValueError, match="positive definite"):
        compute_consider_covariance(output(), np.zeros((2, 2)), np.eye(2))
    with pytest.raises(ValueError, match="symmetric"):
        compute_consider_covariance(
            output(), np.eye(2), np.array([[1.0, 1.0], [0.0, 1.0]])
        )


def test_empty_consider_set_and_full_interleaved_covariance() -> None:
    base = output()
    reduced = replace(
        base,
        parameter_names=("e1", "e2"),
        parameter_roles=(ParameterRole.ESTIMATE, ParameterRole.ESTIMATE),
        parameters=np.zeros(2),
        jacobian=base.jacobian[:, [0, 3]],
    )
    unconsidered, considered, sensitivity, perturbation = compute_consider_covariance(
        reduced, np.eye(2), np.empty((0, 0))
    )
    np.testing.assert_allclose(considered, unconsidered)
    assert sensitivity.shape == perturbation.shape == (2, 0)

    interleaved = replace(
        base,
        parameter_names=("e1", "c1", "e2", "c2"),
        parameter_roles=(
            ParameterRole.ESTIMATE,
            ParameterRole.CONSIDER,
            ParameterRole.ESTIMATE,
            ParameterRole.CONSIDER,
        ),
        parameters=np.zeros(4),
        jacobian=base.jacobian[:, [0, 1, 3, 4]],
    )
    specs = tuple(
        ParameterSpec(name, 0, -10, 10, 1, role, sigma)
        for name, role, sigma in zip(
            interleaved.parameter_names,
            interleaved.parameter_roles,
            (2.0, 1.0, 3.0, 2.0),
            strict=True,
        )
    )
    covariance, rank = full_consider_covariance(interleaved, specs)
    assert rank == 2
    np.testing.assert_allclose(covariance, covariance.T)
    np.linalg.cholesky(covariance)
    np.testing.assert_allclose(covariance[np.ix_([1, 3], [1, 3])], np.diag([1, 4]))
