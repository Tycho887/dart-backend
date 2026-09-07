"""Behavioral cross-model inversion tests, separate from the optional sweep."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from cross_model_validation import (
    REGIMES,
    CaseConfig,
    diagnostics,
    evaluate,
    experiment_record,
    make_case,
    predictions,
    run_case,
    shared_state_parameters,
    subset,
)

from dart.od import (
    OptimizerContext,
    OrbitModel,
    ParameterRole,
    ParameterSpec,
    compute_consider_covariance,
    fit,
)


@pytest.mark.parametrize("regime", REGIMES)
@pytest.mark.parametrize("truth_model", OrbitModel)
@pytest.mark.parametrize("fit_model", OrbitModel)
def test_noisy_cross_model_inversion(
    regime: str, truth_model: OrbitModel, fit_model: OrbitModel
) -> None:
    case = make_case(CaseConfig(regime, truth_model, fit_model))
    result, metrics = run_case(case)

    assert result.success, metrics
    assert 0 < result.function_evaluations <= case.optimizer.max_evaluations
    assert np.all(np.isfinite(result.parameters))
    assert np.all(np.isfinite(result.residuals))
    assert np.all(np.isfinite(result.jacobian))
    assert metrics["final_cost"] < metrics["initial_cost"], metrics
    for parameter, value in zip(
        case.optimizer.parameters, result.parameters, strict=True
    ):
        assert parameter.lower_bound <= value <= parameter.upper_bound
    if fit_model == OrbitModel.SGP4 and truth_model == OrbitModel.FULL_STATE:
        assert (
            metrics["final_validation_rms_hz"] < metrics["initial_validation_rms_hz"]
        ), metrics
    if fit_model == truth_model:
        # A near-exact GEO prior can already predict below the noise floor.
        assert metrics["final_validation_rms_hz"] < 3.0 * case.config.sigma_hz, metrics
    # GEO can terminate successfully despite weak state observability.
    if regime == "GEO":
        assert metrics["scaled_condition"] is not None
        assert metrics["scaled_condition"] > 1e4, metrics


@pytest.mark.parametrize("regime", REGIMES)
@pytest.mark.parametrize("model", OrbitModel)
def test_noiseless_same_model_recovers_truth(regime: str, model: OrbitModel) -> None:
    case = make_case(CaseConfig(regime, model, model, noisy=False))
    result, metrics = run_case(case)

    assert result.success, metrics
    assert metrics["final_training_rms_hz"] < 1e-5, metrics
    assert metrics["final_validation_rms_hz"] < 1e-5, metrics
    assert metrics["scaled_condition"] is not None
    if metrics["scaled_condition"] < 1e4:
        assert "scaled_parameter_error" in metrics
        assert np.linalg.norm(metrics["scaled_parameter_error"]) < 1e-3, metrics


@pytest.mark.parametrize("model", OrbitModel)
def test_noise_whitening_and_epoch_holdout(model: OrbitModel) -> None:
    config = CaseConfig(
        truth_model=model, fit_model=model, sigma_hz=2.0, varying_variance=True
    )
    case = make_case(config)
    truth = shared_state_parameters(case)
    evaluation = evaluate(model, truth, case.prior)
    rows = case.prior.observations.observations
    sigma = np.sqrt([row.noise_cov[0][0] for row in rows])
    expected_noise = np.random.default_rng(config.seed).normal(size=len(rows)) * sigma

    np.testing.assert_array_equal(case.noise_hz, expected_noise)
    np.testing.assert_allclose(evaluation.residuals, -expected_noise / sigma, atol=1e-9)
    np.testing.assert_allclose(
        predictions(model, truth, case.prior), case.clean_hz, atol=1e-9
    )
    unit_rows = [replace(row, noise_cov=[[1.0]]) for row in rows]
    unit_prior = replace(
        case.prior,
        observations=replace(case.prior.observations, observations=unit_rows),
    )
    unit = evaluate(model, truth, unit_prior)
    np.testing.assert_allclose(
        evaluation.jacobian * sigma[:, None], unit.jacobian, atol=1e-9
    )
    training_times = {
        row.time.as_unixtime()
        for row in subset(case.prior, case.train).observations.observations
    }
    heldout_times = {
        row.time.as_unixtime()
        for row in subset(case.prior, ~case.train).observations.observations
    }
    assert training_times.isdisjoint(heldout_times)
    assert len(training_times) == 30
    assert len(heldout_times) == 10


@pytest.mark.parametrize("model", OrbitModel)
@pytest.mark.parametrize("group", ["timing", "frequency_bias", "pass_bias"])
def test_injected_nuisance_parameters(model: OrbitModel, group: str) -> None:
    case = make_case(
        CaseConfig(
            truth_model=model,
            fit_model=model,
            parameters=group,
            passes=2,
        )
    )
    result, metrics = run_case(case)

    assert result.success, metrics
    assert metrics["final_validation_rms_hz"] < metrics["initial_validation_rms_hz"], (
        metrics
    )
    tolerances = {"time_offset_s": 0.02, "center_frequency_offset_hz": 5000.0}
    for name, actual in zip(result.parameter_names, result.parameters, strict=True):
        assert abs(actual - case.truth_parameters[name]) < tolerances.get(name, 0.1), (
            metrics
        )
    if group == "pass_bias":
        assert result.parameter_names == ("pass_bias_hz:pass-z", "pass_bias_hz:pass-a")


@pytest.mark.parametrize("model", OrbitModel)
def test_robust_loss_limits_outlier_prediction_error(model: OrbitModel) -> None:
    case = make_case(
        CaseConfig(
            truth_model=model,
            fit_model=model,
            parameters="frequency_bias",
            outliers=True,
        )
    )
    linear, linear_metrics = run_case(case)
    robust_case = replace(case, optimizer=replace(case.optimizer, loss="soft_l1"))
    robust, robust_metrics = run_case(robust_case)
    assert linear.success and robust.success
    assert (
        robust_metrics["final_validation_rms_hz"]
        < 0.5 * linear_metrics["final_validation_rms_hz"]
    )


@pytest.mark.parametrize("model", OrbitModel)
def test_consider_covariance_on_actual_fit(model: OrbitModel) -> None:
    opposite = {
        OrbitModel.SGP4: OrbitModel.FULL_STATE,
        OrbitModel.FULL_STATE: OrbitModel.SGP4,
    }
    case = make_case(
        CaseConfig(
            truth_model=opposite[model], fit_model=model, parameters="frequency_bias"
        )
    )
    parameters = (
        ParameterSpec("pass_bias_hz:pass-z", 0.0, -100.0, 100.0, 10.0),
        ParameterSpec(
            "center_frequency_offset_hz", 1e5, -1e6, 1e6, 1e5, ParameterRole.CONSIDER
        ),
        ParameterSpec("time_offset_s", 0.0, -10.0, 10.0, 1.0),
        ParameterSpec(
            "position_x_m" if model == OrbitModel.FULL_STATE else "equinoctial_f",
            0.0,
            -1.0,
            1.0,
            0.1,
            ParameterRole.FIXED,
        ),
    )
    optimizer = OptimizerContext(model, parameters, max_evaluations=100)
    output = fit(subset(case.prior, case.train), optimizer)
    assert output.success, output.message
    assert output.parameter_names == tuple(p.name for p in parameters)
    assert output.parameter_roles == tuple(p.role for p in parameters)
    np.testing.assert_array_equal(output.parameters[[1, 3]], [1e5, 0.0])
    direct = evaluate(
        model,
        dict(zip(output.parameter_names, output.parameters, strict=True)),
        subset(case.prior, case.train),
    )
    # Frequency, timing, and bias columns have intentionally interleaved roles.
    np.testing.assert_allclose(output.jacobian[:, :3], direct.jacobian[:, [-1, -2, -3]])

    prior_estimated = np.diag([100.0, 1.0])
    prior_consider = np.array([[1e8]])
    unconsidered, considered, sensitivity, perturbation = compute_consider_covariance(
        output,
        prior_estimated,
        prior_consider,
    )
    h_estimated = output.jacobian[:, [0, 2]]
    h_consider = output.jacobian[:, [1]]
    # Independent augmented least-squares oracle; avoid the production normal equations.
    design = np.vstack((h_estimated, np.diag([0.1, 1.0])))
    rhs = np.vstack((-h_consider, np.zeros((2, 1))))
    expected_sensitivity = np.linalg.lstsq(design, rhs, rcond=None)[0]
    inverse = np.linalg.lstsq(design, np.eye(design.shape[0]), rcond=None)[0]
    np.testing.assert_allclose(unconsidered, inverse @ inverse.T, rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(sensitivity, expected_sensitivity, rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(perturbation, expected_sensitivity * 1e4)
    for matrix in (unconsidered, considered, considered - unconsidered):
        np.testing.assert_allclose(matrix, matrix.T, atol=1e-12)
        assert np.linalg.eigvalsh(matrix).min() >= -1e-12
    _, zero_consider, _, _ = compute_consider_covariance(
        output, prior_estimated, np.zeros((1, 1))
    )
    _, larger_consider, _, _ = compute_consider_covariance(
        output, prior_estimated, 4.0 * prior_consider
    )
    np.testing.assert_allclose(zero_consider, unconsidered)
    np.testing.assert_allclose(
        larger_consider - unconsidered, 4.0 * (considered - unconsidered)
    )
    with pytest.raises(ValueError, match="linear loss"):
        compute_consider_covariance(
            replace(output, loss="huber"), prior_estimated, prior_consider
        )


@pytest.mark.parametrize("model", OrbitModel)
def test_evaluation_budget_exhaustion(model: OrbitModel) -> None:
    case = make_case(CaseConfig(truth_model=model, fit_model=model, max_evaluations=1))
    result, metrics = run_case(case)
    assert not result.success, metrics
    assert result.status == 0
    assert result.function_evaluations == 1
    assert metrics["final_cost"] == metrics["initial_cost"]


@pytest.mark.parametrize("model", OrbitModel)
def test_restrictive_bounds_are_distinct_from_recovery(model: OrbitModel) -> None:
    case = make_case(
        CaseConfig(truth_model=model, fit_model=model, parameters="timing", noisy=False)
    )
    bounded = replace(case.optimizer.parameters[0], lower_bound=-0.01, upper_bound=0.01)
    case = replace(case, optimizer=replace(case.optimizer, parameters=(bounded,)))
    result, metrics = run_case(case)
    assert result.success, metrics
    assert metrics["bound_hits"] == ["time_offset_s"]
    assert abs(result.parameters[0] - 0.35) > 0.3
    assert metrics["final_validation_rms_hz"] > 0.1


@pytest.mark.parametrize("model", OrbitModel)
def test_repeated_epochs_expose_rank_deficiency(model: OrbitModel) -> None:
    case = make_case(CaseConfig(truth_model=model, fit_model=model))
    first = case.prior.observations.observations[0]
    repeated = replace(
        case.prior.observations, observations=[replace(first) for _ in range(12)]
    )
    result = fit(replace(case.prior, observations=repeated), case.optimizer)
    report = diagnostics(result, case.optimizer)
    assert np.all(np.isfinite(result.jacobian))
    assert report["rank"] == 1
    assert report["estimated_parameters"] == 6
    assert report["scaled_condition"] is None


def test_sweep_preserves_failure_configuration() -> None:
    record = experiment_record(CaseConfig(regime="invalid-fixture", seed=17))
    assert "error" in record
    assert record["configuration"]["seed"] == 17
    assert record["error"]["type"] == "KeyError"
    assert "invalid-fixture" in record["error"]["message"]
