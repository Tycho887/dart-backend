from __future__ import annotations

import numpy as np
import pytest
import satkit as sk
from dart.services.solver.mean_element import (
    MeanElementsTwoParameterConfig,
    fit_mean_elements_two_parameter,
    rebuild_tle_mean_elements,
)
from dart.services.solver.numerical import weighted_linear_algebra
from sgp4.api import Satrec

from dart_research.simulation import observations_from_truth, phase_shifted_tle_truth
from dart_research.types import MeasurementMode, TLEContext


def test_mean_elements_two_parameter_recovers_doppler_orbit(context, visible_times):
    line1, line2 = context.tle.to_2line()
    source = Satrec.twoline2rv(line1, line2)
    truth_tle = rebuild_tle_mean_elements(context.tle, source.mo + 0.004, source.no_kozai + 2e-5)
    truth_elements = Satrec.twoline2rv(*truth_tle.to_2line())
    truth_context = TLEContext(truth_tle, context.station, context.carrier_hz)
    second_peak = visible_times[len(visible_times) // 2] + sk.duration(seconds=6_000.0)
    second_pass = [
        second_peak + sk.duration(seconds=float(value)) for value in np.arange(-240, 241, 5)
    ]
    epochs = visible_times + second_pass
    truth = phase_shifted_tle_truth(truth_context, epochs, 0.0)
    observations = observations_from_truth(
        truth_context,
        epochs,
        truth,
        mode=MeasurementMode.DOPPLER,
        frequency_bias_hz=200.0,
    )
    pass_ids = ["pass-1"] * len(visible_times) + ["pass-2"] * len(second_pass)

    fit = fit_mean_elements_two_parameter(
        context,
        observations,
        pass_ids,
        MeanElementsTwoParameterConfig(
            doppler_standard_deviation_hz=1.0,
            mean_anomaly_half_width_rad=0.5,
            mean_motion_half_width_rad_min=0.002,
            pass_bias_bounds_hz=(-15_000.0, 15_000.0),
            qmc_samples=8,
            robust_loss="linear",
            robust_scale_hz=700.0,
        ),
    )

    assert fit.success
    assert abs(fit.mean_anomaly_rad - truth_elements.mo) < 2e-4
    assert abs(fit.mean_motion_rad_min - truth_elements.no_kozai) < 2e-6
    np.testing.assert_allclose(fit.pass_biases_hz, 200.0, atol=10.0)


def test_mean_elements_projects_nonzero_pass_bias_bounds(context, visible_times):
    second_peak = visible_times[len(visible_times) // 2] + sk.duration(seconds=6_000.0)
    second_pass = [
        second_peak + sk.duration(seconds=float(value)) for value in np.arange(-240, 241, 5)
    ]
    epochs = visible_times + second_pass
    observations = observations_from_truth(
        context,
        epochs,
        phase_shifted_tle_truth(context, epochs, 0.0),
        mode=MeasurementMode.DOPPLER,
        frequency_bias_hz=200.0,
    )

    fit = fit_mean_elements_two_parameter(
        context,
        observations,
        ["pass-1"] * len(visible_times) + ["pass-2"] * len(second_pass),
        MeanElementsTwoParameterConfig(
            doppler_standard_deviation_hz=1.0,
            mean_anomaly_half_width_rad=0.5,
            mean_motion_half_width_rad_min=0.002,
            pass_bias_bounds_hz=(100.0, 400.0),
            qmc_samples=0,
            robust_loss="linear",
            robust_scale_hz=700.0,
        ),
    )

    assert fit.success
    np.testing.assert_allclose(fit.pass_biases_hz, 200.0, atol=10.0)


def test_mean_elements_requires_two_distinct_passes(context, visible_times):
    observations = observations_from_truth(
        context,
        visible_times,
        phase_shifted_tle_truth(context, visible_times, 0.0),
        mode=MeasurementMode.DOPPLER,
    )

    with pytest.raises(ValueError, match="at least two distinct passes"):
        fit_mean_elements_two_parameter(
            context,
            observations,
            ["pass-1"] * len(observations),
            MeanElementsTwoParameterConfig(
                doppler_standard_deviation_hz=1.0,
                mean_anomaly_half_width_rad=0.5,
                mean_motion_half_width_rad_min=0.002,
                pass_bias_bounds_hz=(-1_000.0, 1_000.0),
                qmc_samples=0,
                robust_loss="linear",
                robust_scale_hz=700.0,
            ),
        )


def test_mean_elements_recovers_circular_mean_anomaly_delta(context, visible_times):
    line1, line2 = context.tle.to_2line()
    source = Satrec.twoline2rv(line1, line2)
    reference_tle = rebuild_tle_mean_elements(
        context.tle,
        2.0 * np.pi - 0.05,
        source.no_kozai,
    )
    truth_tle = rebuild_tle_mean_elements(reference_tle, 0.05, source.no_kozai)
    truth_elements = Satrec.twoline2rv(*truth_tle.to_2line())
    reference_context = TLEContext(reference_tle, context.station, context.carrier_hz)
    truth_context = TLEContext(truth_tle, context.station, context.carrier_hz)
    truth_times = [
        epoch + sk.duration(seconds=float((source.mo - 0.05) / source.no_kozai * 60.0))
        for epoch in visible_times
    ]
    second_truth_times = [epoch + sk.duration(seconds=6_000.0) for epoch in truth_times]
    epochs = truth_times + second_truth_times
    observed = observations_from_truth(
        truth_context,
        epochs,
        phase_shifted_tle_truth(truth_context, epochs, 0.0),
        mode=MeasurementMode.DOPPLER,
        frequency_bias_hz=200.0,
    )
    pairs = [
        (observation, pass_id)
        for observation, pass_id in zip(
            observed,
            ["pass-1"] * len(truth_times) + ["pass-2"] * len(second_truth_times),
            strict=True,
        )
        if observation.valid
    ]
    observations = [observation for observation, _ in pairs]
    pass_ids = [pass_id for _, pass_id in pairs]
    assert len(observations) > 4

    fit = fit_mean_elements_two_parameter(
        reference_context,
        observations,
        pass_ids,
        MeanElementsTwoParameterConfig(
            doppler_standard_deviation_hz=1.0,
            mean_anomaly_half_width_rad=0.2,
            mean_motion_half_width_rad_min=0.002,
            pass_bias_bounds_hz=(-1_000.0, 1_000.0),
            qmc_samples=8,
            robust_loss="linear",
            robust_scale_hz=700.0,
        ),
    )

    assert fit.success
    circular_error = (fit.mean_anomaly_rad - truth_elements.mo + np.pi) % (2.0 * np.pi) - np.pi
    assert abs(circular_error) < 2e-4


def test_mean_elements_recovers_wrapped_anomaly_with_default_qmc_samples(context, visible_times):
    line1, line2 = context.tle.to_2line()
    source = Satrec.twoline2rv(line1, line2)
    reference_tle = rebuild_tle_mean_elements(context.tle, 0.1, source.no_kozai)
    truth_tle = rebuild_tle_mean_elements(reference_tle, 2.0 * np.pi - 0.1, source.no_kozai)
    truth_elements = Satrec.twoline2rv(*truth_tle.to_2line())
    reference_context = TLEContext(reference_tle, context.station, context.carrier_hz)
    truth_context = TLEContext(truth_tle, context.station, context.carrier_hz)
    phase_shift_s = (source.mo - truth_elements.mo) % (2.0 * np.pi) / source.no_kozai * 60.0
    first_peak = visible_times[len(visible_times) // 2] + sk.duration(seconds=float(phase_shift_s))
    first_pass = [
        first_peak + sk.duration(seconds=float(value)) for value in np.arange(-120, 121, 10)
    ]
    second_peak = first_peak + sk.duration(seconds=6_000.0)
    second_pass = [
        second_peak + sk.duration(seconds=float(value)) for value in np.arange(-120, 121, 10)
    ]
    epochs = first_pass + second_pass
    observations = observations_from_truth(
        truth_context,
        epochs,
        phase_shifted_tle_truth(truth_context, epochs, 0.0),
        mode=MeasurementMode.DOPPLER,
        frequency_bias_hz=200.0,
    )
    assert len(observations) == 50
    assert all(observation.valid for observation in observations)

    fit = fit_mean_elements_two_parameter(
        reference_context,
        observations,
        ["pass-1"] * len(first_pass) + ["pass-2"] * len(second_pass),
        MeanElementsTwoParameterConfig(
            doppler_standard_deviation_hz=1.0,
            mean_anomaly_half_width_rad=0.5,
            mean_motion_half_width_rad_min=0.002,
            pass_bias_bounds_hz=(-15_000.0, 15_000.0),
            qmc_samples=64,
            robust_loss="linear",
            robust_scale_hz=700.0,
        ),
    )

    circular_error = (fit.mean_anomaly_rad - truth_elements.mo + np.pi) % (2.0 * np.pi) - np.pi
    assert fit.success
    assert fit.healthy
    assert abs(circular_error) < 2e-4


def test_mean_elements_turns_invalid_broad_mean_motion_candidates_into_value_errors(
    context, visible_times
):
    second_peak = visible_times[len(visible_times) // 2] + sk.duration(seconds=6_000.0)
    second_pass = [
        second_peak + sk.duration(seconds=float(value)) for value in np.arange(-240, 241, 5)
    ]
    epochs = visible_times + second_pass
    observations = observations_from_truth(
        context,
        epochs,
        phase_shifted_tle_truth(context, epochs, 0.0),
        mode=MeasurementMode.DOPPLER,
    )

    try:
        fit = fit_mean_elements_two_parameter(
            context,
            observations,
            ["pass-1"] * len(visible_times) + ["pass-2"] * len(second_pass),
            MeanElementsTwoParameterConfig(
                doppler_standard_deviation_hz=1.0,
                mean_anomaly_half_width_rad=0.5,
                mean_motion_half_width_rad_min=1.0,
                pass_bias_bounds_hz=(-1_000.0, 1_000.0),
                qmc_samples=8,
                robust_loss="linear",
                robust_scale_hz=700.0,
            ),
        )
    except ValueError as error:
        assert "mean-element" in str(error)
    else:
        assert fit.mean_motion_rad_min > 0.0


def test_weighted_linear_algebra_applies_robust_weights_once_to_raw_jacobian():
    diagnostics = weighted_linear_algebra(
        np.asarray([[2.0], [3.0]]),
        np.asarray([2.0, 4.0]),
        np.asarray([0.25, 1.0]),
    )

    np.testing.assert_allclose(diagnostics.covariance, [[1.7]])
    assert diagnostics.rank == 1
    assert diagnostics.condition == 1.0


def test_mean_elements_two_parameter_rejects_underconstrained_input(context, visible_times):
    observations = observations_from_truth(
        context,
        visible_times[:3],
        phase_shifted_tle_truth(context, visible_times[:3], 0.0),
        mode=MeasurementMode.DOPPLER,
    )
    with pytest.raises(ValueError, match="fitted parameters"):
        fit_mean_elements_two_parameter(
            context,
            observations,
            ["pass-1", "pass-2", "pass-2"],
            MeanElementsTwoParameterConfig(
                doppler_standard_deviation_hz=1.0,
                mean_anomaly_half_width_rad=0.5,
                mean_motion_half_width_rad_min=0.002,
                pass_bias_bounds_hz=(-15_000.0, 15_000.0),
                qmc_samples=0,
                robust_loss="linear",
                robust_scale_hz=700.0,
            ),
        )
