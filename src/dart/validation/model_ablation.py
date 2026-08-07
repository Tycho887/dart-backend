"""Held-out synthetic comparison of scalar offset and mean-element models."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace

import numpy as np
import satkit as sk
from sgp4.api import Satrec

from ..estimation import (
    BatchConfig,
    MeanElementConfig,
    fit_batch,
    fit_mean_elements,
    rebuild_tle_mean_elements,
)
from ..geometry import tle_relative_geometry
from ..measurements import wrap_angle_rad
from ..simulation import observations_from_truth, phase_shifted_tle_truth
from ..types import MeasurementMode, TLEContext


@dataclass(frozen=True, slots=True)
class ModelAblationResult:
    truth_family: str
    fitted_model: str
    channel_mode: str
    seed: int
    fit_passes: int
    evaluation_samples: int
    median_position_error_km: float
    p90_position_error_km: float
    worst_position_error_km: float
    success: bool

    def as_dict(self) -> dict:
        return asdict(self)


def visible_pass_windows(
    context: TLEContext,
    *,
    count: int = 3,
    search_hours: float = 48.0,
    minimum_peak_elevation_deg: float = 20.0,
) -> list[list]:
    start = context.tle.epoch
    coarse = [
        start + sk.duration(seconds=60.0 * index)
        for index in range(int(search_hours * 60.0))
    ]
    elevation = np.asarray(
        [
            tle_relative_geometry(context.tle, context.station, epoch).elevation_rad
            for epoch in coarse
        ]
    )
    visible = np.flatnonzero(elevation > 0.0)
    groups = np.split(visible, np.flatnonzero(np.diff(visible) > 1) + 1)
    windows = []
    for group in groups:
        if not len(group):
            continue
        peak_index = int(group[np.argmax(elevation[group])])
        if elevation[peak_index] < np.radians(minimum_peak_elevation_deg):
            continue
        peak = coarse[peak_index]
        windows.append(
            [peak + sk.duration(seconds=float(value)) for value in np.arange(-240, 241, 5)]
        )
        if len(windows) == count:
            return windows
    raise RuntimeError(f"found only {len(windows)} suitable passes")


def run_model_ablation(
    context: TLEContext,
    *,
    seed: int,
    truth_family: str,
    doppler_std_hz: float = 50.0,
    true_offset_s: float = 5.0,
    mean_anomaly_delta_rad: float = 0.004,
    mean_motion_delta_rad_min: float = 2e-5,
    phase_std_rad: float = 0.1,
) -> list[ModelAblationResult]:
    windows = visible_pass_windows(context)
    all_times = [epoch for window in windows for epoch in window]
    line1, line2 = context.tle.to_2line()
    source = Satrec.twoline2rv(line1, line2)
    if truth_family == "offset":
        truth_context = context
        truth_states = phase_shifted_tle_truth(context, all_times, true_offset_s)
    elif truth_family == "mean_elements":
        truth_tle = rebuild_tle_mean_elements(
            context.tle,
            source.mo + mean_anomaly_delta_rad,
            source.no_kozai + mean_motion_delta_rad_min,
        )
        truth_context = TLEContext(
            truth_tle, context.station, context.carrier_hz, context.baseline
        )
        truth_states = phase_shifted_tle_truth(truth_context, all_times, 0.0)
    else:
        raise ValueError(f"unknown truth family {truth_family!r}")
    observations = observations_from_truth(
        truth_context,
        all_times,
        truth_states,
        mode=MeasurementMode.DOPPLER_PHASE,
        frequency_bias_hz=300.0,
        phase_bias_rad=1.2,
    )
    rng = np.random.default_rng(seed)
    doppler_noise = rng.normal(0.0, doppler_std_hz, len(observations))
    phase_noise = rng.normal(0.0, phase_std_rad, len(observations))
    complete_phase = [
        replace(
            observation,
            doppler_hz=float(observation.doppler_hz + doppler_noise[index]),
            phase_rad=float(wrap_angle_rad(observation.phase_rad + phase_noise[index])),
        )
        for index, observation in enumerate(observations)
    ]
    doppler_only = [replace(observation, phase_rad=None) for observation in complete_phase]
    fit_count = len(windows[0]) + len(windows[1])
    evaluation_times = windows[2]
    evaluation_truth = truth_states[fit_count:]
    doppler_fit_observations = doppler_only[:fit_count]
    phase_fit_observations = complete_phase[:fit_count]
    offset_doppler_fit = fit_batch(
        context,
        doppler_fit_observations,
        BatchConfig(
            doppler_std_hz=doppler_std_hz,
            phase_std_rad=phase_std_rad,
            robust_loss="linear",
            offset_starts_s=(-30.0, 0.0, 30.0),
        ),
    )
    element_doppler_fit = fit_mean_elements(
        context,
        doppler_fit_observations,
        MeanElementConfig(
            doppler_std_hz=doppler_std_hz,
            phase_std_rad=phase_std_rad,
            qmc_samples=16,
            robust_loss="linear",
        ),
    )
    offset_phase_fit = fit_batch(
        context,
        phase_fit_observations,
        BatchConfig(
            doppler_std_hz=doppler_std_hz,
            phase_std_rad=phase_std_rad,
            robust_loss="linear",
            offset_starts_s=(offset_doppler_fit.estimate.offset_s, -30.0, 0.0, 30.0),
        ),
        mode=MeasurementMode.DOPPLER_PHASE,
    )
    element_phase_fit = fit_mean_elements(
        context,
        phase_fit_observations,
        MeanElementConfig(
            doppler_std_hz=doppler_std_hz,
            phase_std_rad=phase_std_rad,
            qmc_samples=16,
            robust_loss="linear",
        ),
        mode=MeasurementMode.DOPPLER_PHASE,
        initial_mean_elements=(
            element_doppler_fit.mean_anomaly_rad,
            element_doppler_fit.mean_motion_rad_min,
        ),
    )
    candidates = (
        (
            "time_offset",
            "doppler",
            context.tle,
            offset_doppler_fit.estimate.offset_s,
            offset_doppler_fit.success,
        ),
        (
            "mean_anomaly_motion",
            "doppler",
            element_doppler_fit.corrected_tle,
            0.0,
            element_doppler_fit.success,
        ),
        (
            "time_offset",
            "doppler_phase",
            context.tle,
            offset_phase_fit.estimate.offset_s,
            offset_phase_fit.success,
        ),
        (
            "mean_anomaly_motion",
            "doppler_phase",
            element_phase_fit.corrected_tle,
            0.0,
            element_phase_fit.success,
        ),
    )
    result = []
    for label, channel_mode, tle, offset, success in candidates:
        predicted = np.asarray(
            [
                tle_relative_geometry(tle, context.station, epoch, offset).satellite_position_gcrf_m
                for epoch in evaluation_times
            ]
        )
        errors = np.linalg.norm(predicted - evaluation_truth[:, :3], axis=1) / 1000.0
        result.append(
            ModelAblationResult(
                truth_family=truth_family,
                fitted_model=label,
                channel_mode=channel_mode,
                seed=seed,
                fit_passes=2,
                evaluation_samples=len(errors),
                median_position_error_km=float(np.median(errors)),
                p90_position_error_km=float(np.percentile(errors, 90.0)),
                worst_position_error_km=float(np.max(errors)),
                success=bool(success),
            )
        )
    return result
