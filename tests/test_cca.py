"""Consider-covariance analysis contracts."""

from __future__ import annotations

import numpy as np
import pytest

from dart.od import (
    OptimizerOutput,
    OrbitModel,
    ParameterRole,
    PriorSource,
    compute_consider_covariance,
)
from dart.od.schema import LossKind


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


def test_consider_covariance_rejects_robust_loss_and_invalid_priors() -> None:
    with pytest.raises(ValueError, match="linear loss"):
        compute_consider_covariance(output(loss="huber"), np.eye(2), np.eye(2))
    with pytest.raises(ValueError, match="positive definite"):
        compute_consider_covariance(output(), np.zeros((2, 2)), np.eye(2))
    with pytest.raises(ValueError, match="symmetric"):
        compute_consider_covariance(
            output(), np.eye(2), np.array([[1.0, 1.0], [0.0, 1.0]])
        )
