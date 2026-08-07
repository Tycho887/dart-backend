"""Doppler orbit refinement in SGP4 mean anomaly and mean motion."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import satkit as sk
from scipy.optimize import least_squares
from scipy.stats import qmc
from sgp4.api import Satrec, WGS72
from sgp4.exporter import export_tle

from ..measurements import MeasurementModel, prepare_batch_cache, wrap_angle_rad
from ..types import MeasurementMode, RFObservation, TLEContext


@dataclass(frozen=True, slots=True)
class MeanElementConfig:
    doppler_std_hz: float = 500.0
    phase_std_rad: float = 0.1
    mean_anomaly_half_width_rad: float = 0.5
    mean_motion_half_width_rad_min: float = 0.002
    pass_bias_bound_hz: float = 15_000.0
    pass_gap_s: float = 900.0
    qmc_samples: int = 32
    robust_loss: str = "soft_l1"
    robust_scale_doppler_hz: float = 700.0


@dataclass(frozen=True, slots=True)
class MeanElementFit:
    corrected_tle: object
    mean_anomaly_rad: float
    mean_motion_rad_min: float
    pass_biases_hz: np.ndarray
    phase_biases_rad: np.ndarray | None
    covariance: np.ndarray
    residuals_hz: np.ndarray
    phase_residuals_rad: np.ndarray | None
    success: bool
    message: str
    passes_found: int


def rebuild_tle_mean_elements(
    tle,
    mean_anomaly_rad: float,
    mean_motion_rad_min: float,
):
    """Return a Satkit TLE with only M and n replaced."""

    line1, line2 = tle.to_2line()
    source = Satrec.twoline2rv(line1, line2)
    source.sgp4init(
        WGS72,
        "i",
        source.satnum,
        source.jdsatepoch + source.jdsatepochF - 2_433_281.5,
        source.bstar,
        source.ndot,
        source.nddot,
        source.ecco,
        source.argpo,
        source.inclo,
        float(mean_anomaly_rad) % (2.0 * np.pi),
        float(mean_motion_rad_min),
        source.nodeo,
    )
    return sk.TLE.from_lines(list(export_tle(source)))


def _pass_indices(observations: list[RFObservation], gap_s: float) -> tuple[np.ndarray, int]:
    epochs = np.array([float(item.epoch.as_unixtime()) for item in observations])
    indices = np.insert(np.cumsum(np.diff(epochs) > gap_s), 0, 0)
    return indices, int(indices[-1] + 1)


def fit_mean_elements(
    context: TLEContext,
    observations: list[RFObservation],
    config: MeanElementConfig = MeanElementConfig(),
    *,
    mode: MeasurementMode = MeasurementMode.DOPPLER,
    initial_mean_elements: tuple[float, float] | None = None,
    model: MeasurementModel | None = None,
) -> MeanElementFit:
    """Fit mean elements and pass-constant RF nuisance parameters."""

    usable = [item for item in observations if item.valid and item.doppler_hz is not None]
    if len(usable) < 4:
        raise ValueError("at least four valid Doppler observations are required")
    if mode is MeasurementMode.DOPPLER_PHASE:
        usable = [item for item in usable if item.phase_rad is not None]
        if len(usable) < 4:
            raise ValueError("at least four complete Doppler+phase observations are required")
    model = MeasurementModel() if model is None else model
    cache = prepare_batch_cache(context, usable)
    pass_index, pass_count = _pass_indices(usable, config.pass_gap_s)
    measured = np.array([item.doppler_hz for item in usable], dtype=float)
    measured_phase = (
        np.array([item.phase_rad for item in usable], dtype=float)
        if mode is MeasurementMode.DOPPLER_PHASE
        else None
    )
    line1, line2 = context.tle.to_2line()
    source = Satrec.twoline2rv(line1, line2)
    base_m = float(source.mo)
    base_n = float(source.no_kozai)
    phase_capable = mode is MeasurementMode.DOPPLER_PHASE
    dimension = 2 + pass_count + (pass_count if phase_capable else 0)
    lower = np.array(
        [
            max(0.0, base_m - config.mean_anomaly_half_width_rad),
            base_n - config.mean_motion_half_width_rad_min,
            *([-config.pass_bias_bound_hz] * pass_count),
            *([-np.pi] * pass_count if phase_capable else []),
        ]
    )
    upper = np.array(
        [
            min(2.0 * np.pi, base_m + config.mean_anomaly_half_width_rad),
            base_n + config.mean_motion_half_width_rad_min,
            *([config.pass_bias_bound_hz] * pass_count),
            *([np.pi] * pass_count if phase_capable else []),
        ]
    )

    def channel_residuals(state: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
        candidate_tle = rebuild_tle_mean_elements(context.tle, state[0], state[1])
        candidate_context = TLEContext(
            tle=candidate_tle,
            station=context.station,
            carrier_hz=context.carrier_hz,
            baseline=context.baseline,
        )
        # Offset and global bias are zero here; per-pass biases are applied
        # after the common orbital prediction.
        prediction_state = np.array([0.0, 0.0] + ([0.0] if phase_capable else []))
        predicted = model.predict_many(
            candidate_context,
            prediction_state,
            mode,
            cache,
        )
        frequency_biases = state[2 : 2 + pass_count]
        doppler = predicted[:, 0] + frequency_biases[pass_index] - measured
        if not phase_capable:
            return doppler, None
        phase_biases = state[2 + pass_count :]
        phase = wrap_angle_rad(
            predicted[:, 1] + phase_biases[pass_index] - measured_phase
        )
        return doppler, phase

    def residual(state: np.ndarray) -> np.ndarray:
        doppler, phase = channel_residuals(state)
        data = [doppler / config.doppler_std_hz]
        if phase is not None:
            data.append(phase / config.phase_std_rad)
        regularization = np.array(
            [
                1e-6 * (state[0] - base_m),
                1e-5 * (state[1] - base_n),
                *(1e-4 * state[2 : 2 + pass_count]),
                *(1e-4 * state[2 + pass_count :] if phase_capable else []),
            ]
        )
        return np.concatenate((*data, regularization))

    steps = np.array(
        [1e-5, 1e-7] + [1.0] * pass_count + ([1e-4] * pass_count if phase_capable else [])
    )

    def jacobian(state: np.ndarray) -> np.ndarray:
        columns = []
        for index, step in enumerate(steps):
            right, left = state.copy(), state.copy()
            right[index] = min(upper[index], right[index] + step)
            left[index] = max(lower[index], left[index] - step)
            right_residual = residual(right)
            left_residual = residual(left)
            difference = right_residual - left_residual
            if phase_capable:
                start, stop = len(usable), 2 * len(usable)
                difference[start:stop] = (
                    wrap_angle_rad(
                        (right_residual[start:stop] - left_residual[start:stop])
                        * config.phase_std_rad
                    )
                    / config.phase_std_rad
                )
            columns.append(difference / (right[index] - left[index]))
        return np.column_stack(columns)

    initial_orbit = initial_mean_elements or (base_m, base_n)
    baseline = np.array(
        [*initial_orbit]
        + [0.0] * pass_count
        + ([0.0] * pass_count if phase_capable else [])
    )
    sampler = qmc.LatinHypercube(d=dimension, seed=42)
    candidates = qmc.scale(sampler.random(config.qmc_samples), lower, upper)
    candidates = np.vstack((baseline, candidates))
    initial = min(candidates, key=lambda value: float(np.sum(residual(value) ** 2)))
    result = least_squares(
        residual,
        initial,
        jac=jacobian,
        bounds=(lower, upper),
        method="trf",
        loss=config.robust_loss,
        f_scale=config.robust_scale_doppler_hz / config.doppler_std_hz,
        x_scale=np.array(
            [0.05, 0.001]
            + [2_000.0] * pass_count
            + ([1.0] * pass_count if phase_capable else [])
        ),
        ftol=1e-11,
        xtol=1e-11,
        gtol=1e-11,
    )
    observation_residual, phase_residual = channel_residuals(result.x)
    channel_rows = len(usable) * (2 if phase_capable else 1)
    dof = max(1, channel_rows - dimension)
    covariance = np.linalg.pinv(result.jac.T @ result.jac, rcond=1e-12)
    covariance *= float(np.sum(result.fun[:channel_rows] ** 2) / dof)
    corrected = rebuild_tle_mean_elements(context.tle, result.x[0], result.x[1])
    return MeanElementFit(
        corrected_tle=corrected,
        mean_anomaly_rad=float(result.x[0]),
        mean_motion_rad_min=float(result.x[1]),
        pass_biases_hz=np.asarray(result.x[2 : 2 + pass_count]).copy(),
        phase_biases_rad=(
            np.asarray(result.x[2 + pass_count :]).copy() if phase_capable else None
        ),
        covariance=covariance,
        residuals_hz=observation_residual,
        phase_residuals_rad=(
            None if phase_residual is None else np.asarray(phase_residual).copy()
        ),
        success=bool(result.success),
        message=str(result.message),
        passes_found=pass_count,
    )
