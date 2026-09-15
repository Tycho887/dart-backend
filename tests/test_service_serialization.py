"""Publication preserves native-unit covariance ordering and rejects invalid data."""

from dataclasses import replace

import numpy as np
import pytest
from service_fixtures import configuration, prior_fixture

from dart.od import fit
from dart.service.serialization import result_rows
from dart.service.worker import run_estimate


@pytest.fixture(scope="module")
def v2_fit():
    config = configuration(model_version=2)
    prior = prior_fixture(config)
    output, optimizer, _ = run_estimate(prior, config)
    assert output.success
    return prior, config, optimizer, output


def test_uncertainty_uses_full_diagonal_in_interleaved_parameter_order(v2_fit):
    prior, config, optimizer, output = v2_fit
    # Alternate estimated time, considered frequency, estimated bias, then orbit.
    order = [7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 10]
    optimizer = replace(
        optimizer, parameters=tuple(optimizer.parameters[i] for i in order)
    )
    sigmas = np.array(
        [spec.prior_standard_uncertainty for spec in optimizer.parameters]
    )
    sigmas[[0, 2, 10]] = [2.5, 3.5, 4.5]
    covariance = np.diag(sigmas**2)
    covariance[0, 2] = covariance[2, 0] = 0.5
    output = replace(
        output,
        parameter_names=tuple(output.parameter_names[i] for i in order),
        parameter_roles=tuple(output.parameter_roles[i] for i in order),
        parameters=output.parameters[order],
        jacobian=output.jacobian[:, order],
        covariance=covariance,
    )
    rows, diagnostics = result_rows(prior, config, optimizer, output)
    assert [row["role"] for row in rows[:4]] == [
        "estimate",
        "consider",
        "estimate",
        "consider",
    ]
    assert [row["unit"] for row in rows[:3]] == ["s", "Hz", "Hz"]
    np.testing.assert_allclose([row["standard_uncertainty"] for row in rows], sigmas)
    np.testing.assert_array_equal(diagnostics["covariance"], covariance)
    assert diagnostics["parameter_order"] == tuple(
        row["parameter_name"] for row in rows
    )
    assert diagnostics["covariance_method"] == "classical_consider_v1"


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_rejects_nonfinite_covariance_including_off_diagonal(v2_fit, value):
    prior, config, optimizer, output = v2_fit
    covariance = output.covariance.copy()
    covariance[0, 1] = value
    with pytest.raises(ValueError, match="invalid parameter covariance"):
        result_rows(prior, config, optimizer, replace(output, covariance=covariance))


@pytest.mark.parametrize(
    "changes,message",
    [
        ({"covariance": np.eye(2)}, "invalid parameter covariance"),
        ({"covariance": None}, "covariance availability"),
        ({"covariance_method": None}, "covariance method"),
        ({"covariance_method": "unknown"}, "covariance method"),
        ({"success": False}, "covariance availability"),
        ({"parameter_names": ()}, "parameter order"),
        ({"parameter_roles": ()}, "parameter roles"),
        ({"parameters": np.zeros((11, 1))}, "parameter order"),
    ],
)
def test_rejects_inconsistent_covariance_publication(v2_fit, changes, message):
    prior, config, optimizer, output = v2_fit
    with pytest.raises(ValueError, match=message):
        result_rows(prior, config, optimizer, replace(output, **changes))


def test_unsuccessful_v2_fit_has_no_uncertainty(v2_fit):
    prior, config, optimizer, _ = v2_fit
    output = fit(prior, replace(optimizer, max_evaluations=1))
    assert not output.success
    rows, diagnostics = result_rows(prior, config, optimizer, output)
    assert all(row["standard_uncertainty"] is None for row in rows)
    assert diagnostics["covariance"] is None
    assert diagnostics["covariance_method"] is None
    assert diagnostics["covariance_rank"] is None
