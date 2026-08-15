from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import satkit as sk

from dart.estimation import (
    BatchConfig,
    MeanElementConfig,
    PassiveRFUKF,
    UKFConfig,
    fit_batch,
    fit_mean_elements,
    rebuild_tle_mean_elements,
)
from dart.simulation import observations_from_truth, phase_shifted_tle_truth
from dart.types import MeasurementMode, RFObservation, TLEContext
from sgp4.api import Satrec


def _synthetic(context, visible_times, mode, offset=8.0, bias=300.0, phase=1.2):
    truth = phase_shifted_tle_truth(context, visible_times, offset)
    return observations_from_truth(
        context,
        visible_times,
        truth,
        mode=mode,
        frequency_bias_hz=bias,
        phase_bias_rad=phase,
    )


def test_batch_recovers_doppler_offset(context, visible_times):
    observations = _synthetic(context, visible_times, MeasurementMode.DOPPLER)
    fit = fit_batch(
        context,
        observations,
        BatchConfig(
            doppler_std_hz=5.0,
            offset_starts_s=(-30.0, 0.0, 30.0),
            robust_loss="linear",
        ),
    )
    assert fit.success
    assert abs(fit.estimate.offset_s - 8.0) < 0.05
    assert abs(fit.estimate.frequency_bias_hz - 300.0) < 1.0
    expected_rmse_hz = float(
        np.sqrt(np.mean(np.square(fit.residuals[: len(observations)])))
        * 5.0
    )
    assert fit.doppler_rmse_hz == pytest.approx(expected_rmse_hz)


def test_ukf_supports_intermittent_phase(context, visible_times):
    observations = _synthetic(context, visible_times, MeasurementMode.DOPPLER_PHASE)
    filter_ = PassiveRFUKF(
        context,
        initial_offset_s=5.0,
        phase_capable=True,
        config=UKFConfig(
            doppler_std_hz=5.0,
            phase_std_rad=0.1,
            gate_probability=0.999999,
            initial_offset_std_s=5.0,
            initial_frequency_std_hz=1_000.0,
        ),
    )
    accepted = 0
    estimate = None
    for index, observation in enumerate(observations):
        if index % 4 == 0:
            observation = RFObservation(
                epoch=observation.epoch,
                station_id=observation.station_id,
                doppler_hz=observation.doppler_hz,
                phase_rad=None,
                valid=observation.valid,
                sequence=observation.sequence,
            )
        estimate = filter_.step(observation)
        accepted += int(estimate.accepted)
    assert estimate is not None
    assert accepted > len(observations) // 2
    assert abs(estimate.offset_s - 8.0) < 1.0
    assert abs(estimate.frequency_bias_hz - 300.0) < 50.0
    assert np.min(np.linalg.eigvalsh(estimate.covariance)) > 0.0


def test_batch_supports_intermittent_phase(context, visible_times):
    observations = _synthetic(context, visible_times, MeasurementMode.DOPPLER_PHASE)
    intermittent = [
        replace(observation, phase_rad=None) if index % 4 == 0 else observation
        for index, observation in enumerate(observations)
    ]
    fit = fit_batch(
        context,
        intermittent,
        BatchConfig(
            doppler_std_hz=1.0,
            phase_std_rad=0.05,
            robust_loss="linear",
        ),
    )
    assert fit.success
    assert fit.observations_used == len(observations)
    assert abs(fit.estimate.offset_s - 8.0) < 0.05
    assert abs(fit.estimate.frequency_bias_hz - 300.0) < 1.0
    assert abs(float(np.angle(np.exp(1j * (fit.estimate.phase_bias_rad - 1.2))))) < 0.01


def test_batch_phase_jacobian_is_circular_at_branch_cut(context, visible_times):
    observations = _synthetic(
        context,
        visible_times,
        MeasurementMode.DOPPLER_PHASE,
        phase=np.pi - 1e-6,
    )
    fit = fit_batch(
        context,
        observations,
        BatchConfig(
            doppler_std_hz=1.0,
            phase_std_rad=0.1,
            robust_loss="linear",
            offset_starts_s=(0.0,),
        ),
    )
    assert fit.success
    assert abs(fit.estimate.offset_s - 8.0) < 0.05
    phase_error = np.angle(
        np.exp(1j * (fit.estimate.phase_bias_rad - (np.pi - 1e-6)))
    )
    assert abs(float(phase_error)) < 0.01


def test_static_ukf_prediction_does_not_inflate_covariance(context):
    filter_ = PassiveRFUKF(
        context,
        initial_offset_s=4.0,
        config=UKFConfig(gate_probability=None),
    )
    state_before = filter_.x.copy()
    covariance_before = filter_.P.copy()
    filter_.predict()
    np.testing.assert_allclose(filter_.x, state_before, atol=1e-12)
    np.testing.assert_allclose(filter_.P, covariance_before, rtol=1e-12, atol=1e-10)


def test_ukf_rejects_stale_epoch(context, visible_times):
    observations = _synthetic(context, visible_times[:2], MeasurementMode.DOPPLER)
    filter_ = PassiveRFUKF(context, 8.0)
    filter_.step(observations[0])
    stale = filter_.step(observations[0])
    assert not stale.accepted
    assert stale.reason == "stale measurement epoch"


def test_mean_element_fit_retains_basic_dart_od(context, visible_times):
    line1, line2 = context.tle.to_2line()
    source = Satrec.twoline2rv(line1, line2)
    truth_m = source.mo + 0.004
    truth_n = source.no_kozai + 2e-5
    truth_tle = rebuild_tle_mean_elements(context.tle, truth_m, truth_n)
    truth_elements = Satrec.twoline2rv(*truth_tle.to_2line())
    truth_context = TLEContext(
        truth_tle, context.station, context.carrier_hz, context.baseline
    )
    second_peak = visible_times[len(visible_times) // 2] + sk.duration(seconds=6_000.0)
    second_pass = [
        second_peak + sk.duration(seconds=float(value))
        for value in np.arange(-240, 241, 5)
    ]
    multipass_times = visible_times + second_pass
    truth_states = phase_shifted_tle_truth(truth_context, multipass_times, 0.0)
    observations = observations_from_truth(
        truth_context,
        multipass_times,
        truth_states,
        mode=MeasurementMode.DOPPLER,
        frequency_bias_hz=200.0,
    )
    fit = fit_mean_elements(
        context,
        observations,
        MeanElementConfig(qmc_samples=8, robust_loss="linear"),
    )
    assert fit.success
    assert fit.passes_found == 2
    assert abs(fit.mean_anomaly_rad - truth_elements.mo) < 2e-4
    assert abs(fit.mean_motion_rad_min - truth_elements.no_kozai) < 2e-6
    np.testing.assert_allclose(fit.pass_biases_hz, 200.0, atol=10.0)

    phase_observations = observations_from_truth(
        truth_context,
        multipass_times,
        truth_states,
        mode=MeasurementMode.DOPPLER_PHASE,
        frequency_bias_hz=200.0,
        phase_bias_rad=1.0,
    )
    phase_fit = fit_mean_elements(
        context,
        phase_observations,
        MeanElementConfig(
            doppler_std_hz=1.0,
            phase_std_rad=0.05,
            qmc_samples=8,
            robust_loss="linear",
        ),
        mode=MeasurementMode.DOPPLER_PHASE,
        initial_mean_elements=(fit.mean_anomaly_rad, fit.mean_motion_rad_min),
    )
    assert phase_fit.success
    assert abs(phase_fit.mean_anomaly_rad - truth_elements.mo) < 2e-4
    assert abs(phase_fit.mean_motion_rad_min - truth_elements.no_kozai) < 2e-6
    assert phase_fit.phase_biases_rad is not None
