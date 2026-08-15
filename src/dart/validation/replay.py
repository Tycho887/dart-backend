"""Causal FOREST replay with batch and direct-GPS comparison."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from ..estimation import BatchConfig, PassiveRFUKF, UKFConfig, fit_batch
from ..geometry import tle_positions_itrf
from ..io.forest import ForestPass
from ..io.gps import GPSReference


@dataclass(frozen=True, slots=True)
class ErrorSummary:
    fixes: int
    best_km: float
    median_km: float
    p90_km: float
    worst_km: float
    rms_km: float

    @classmethod
    def from_values(cls, values: np.ndarray) -> "ErrorSummary":
        if not len(values):
            nan = float("nan")
            return cls(0, nan, nan, nan, nan, nan)
        values = np.asarray(values, dtype=float)
        return cls(
            fixes=len(values),
            best_km=float(np.min(values)),
            median_km=float(np.median(values)),
            p90_km=float(np.percentile(values, 90.0)),
            worst_km=float(np.max(values)),
            rms_km=float(np.sqrt(np.mean(values**2))),
        )


@dataclass(frozen=True, slots=True)
class ForecastSummary:
    lower_h: int
    upper_h: int
    prior: ErrorSummary
    ukf: ErrorSummary
    batch: ErrorSummary


@dataclass(frozen=True, slots=True)
class ReplayPassResult:
    satellite: str
    contact_id: str
    station: str
    start_utc_s: float
    end_utc_s: float
    observations: int
    ukf_updates: int
    ukf_rejected: int
    ukf_healthy: bool
    ukf_offset_s: float
    ukf_offset_std_s: float
    ukf_frequency_bias_hz: float
    batch_offset_s: float
    batch_offset_std_s: float
    batch_offset_variance_s2: float
    batch_covariance: tuple[tuple[float, ...], ...]
    batch_frequency_bias_hz: float
    batch_doppler_rmse_hz: float
    batch_condition: float
    batch_rank: int
    batch_at_bound: bool
    batch_healthy: bool
    in_pass_prior: ErrorSummary
    in_pass_ukf_online: ErrorSummary
    in_pass_ukf_postpass: ErrorSummary
    in_pass_batch: ErrorSummary
    forecasts: tuple[ForecastSummary, ...]

    def as_dict(self) -> dict:
        return asdict(self)


def _position_errors_km(tle, gps: GPSReference, offset_s: float) -> np.ndarray:
    if not len(gps):
        return np.array([], dtype=float)
    predicted = tle_positions_itrf(tle, gps.utc_s, offset_s)
    return np.linalg.norm(predicted - gps.position_itrf_m, axis=1) / 1000.0


def _online_position_errors_km(tle, gps: GPSReference, estimates: list) -> np.ndarray:
    """Score each GPS fix using only the latest estimate available at that epoch."""

    if not len(gps):
        return np.array([], dtype=float)
    estimate_times = np.asarray(
        [float(estimate.epoch.as_unixtime()) for estimate in estimates], dtype=float
    )
    offsets = np.asarray([estimate.offset_s for estimate in estimates], dtype=float)
    result = []
    for epoch, truth in zip(gps.utc_s, gps.position_itrf_m):
        index = int(np.searchsorted(estimate_times, epoch, side="right") - 1)
        offset = 0.0 if index < 0 else float(offsets[index])
        predicted = tle_positions_itrf(tle, [epoch], offset)[0]
        result.append(np.linalg.norm(predicted - truth) / 1000.0)
    return np.asarray(result)


def replay_pass(
    forest_pass: ForestPass,
    gps: GPSReference,
    *,
    ukf_config: UKFConfig = UKFConfig(),
    batch_config: BatchConfig = BatchConfig(),
) -> ReplayPassResult:
    """Replay recorded Doppler without pretending historical steering was causal."""

    ukf = PassiveRFUKF(
        forest_pass.context,
        initial_offset_s=0.0,
        phase_capable=False,
        config=ukf_config,
    )
    accepted = rejected = 0
    final = None
    estimates = []
    for observation in forest_pass.observations:
        estimate = ukf.step(observation)
        estimates.append(estimate)
        final = estimate
        if estimate.accepted:
            accepted += 1
        else:
            rejected += 1
    if final is None:
        raise ValueError("FOREST pass contains no observations")

    batch = fit_batch(
        forest_pass.context,
        list(forest_pass.observations),
        batch_config,
    )
    in_pass = gps.between(forest_pass.start_utc_s, forest_pass.end_utc_s)
    tle = forest_pass.context.tle
    prior_error = _position_errors_km(tle, in_pass, 0.0)
    ukf_online_error = _online_position_errors_km(tle, in_pass, estimates)
    ukf_postpass_error = _position_errors_km(tle, in_pass, final.offset_s)
    batch_error = _position_errors_km(tle, in_pass, batch.estimate.offset_s)

    forecast_rows = []
    for lower_h, upper_h in ((0, 1), (1, 3), (3, 6), (6, 12), (12, 24)):
        subset = gps.between(
            forest_pass.end_utc_s + lower_h * 3600.0,
            forest_pass.end_utc_s + upper_h * 3600.0,
        )
        forecast_rows.append(
            ForecastSummary(
                lower_h,
                upper_h,
                ErrorSummary.from_values(_position_errors_km(tle, subset, 0.0)),
                ErrorSummary.from_values(
                    _position_errors_km(tle, subset, final.offset_s)
                ),
                ErrorSummary.from_values(
                    _position_errors_km(tle, subset, batch.estimate.offset_s)
                ),
            )
        )

    return ReplayPassResult(
        satellite=forest_pass.satellite,
        contact_id=forest_pass.contact_id,
        station=forest_pass.ground_station,
        start_utc_s=forest_pass.start_utc_s,
        end_utc_s=forest_pass.end_utc_s,
        observations=len(forest_pass.observations),
        ukf_updates=accepted,
        ukf_rejected=rejected,
        ukf_healthy=(accepted >= 20 and accepted / max(1, accepted + rejected) >= 0.25),
        ukf_offset_s=final.offset_s,
        ukf_offset_std_s=final.offset_std_s,
        ukf_frequency_bias_hz=final.frequency_bias_hz,
        batch_offset_s=batch.estimate.offset_s,
        batch_offset_std_s=batch.estimate.offset_std_s,
        batch_offset_variance_s2=float(max(0.0, batch.estimate.covariance[0, 0])),
        batch_covariance=tuple(
            tuple(float(value) for value in covariance_row)
            for covariance_row in batch.estimate.covariance
        ),
        batch_frequency_bias_hz=batch.estimate.frequency_bias_hz,
        batch_doppler_rmse_hz=batch.doppler_rmse_hz,
        batch_condition=batch.jacobian_condition,
        batch_rank=batch.jacobian_rank,
        batch_at_bound=batch.at_bound,
        batch_healthy=batch.estimate.healthy,
        in_pass_prior=ErrorSummary.from_values(prior_error),
        in_pass_ukf_online=ErrorSummary.from_values(ukf_online_error),
        in_pass_ukf_postpass=ErrorSummary.from_values(ukf_postpass_error),
        in_pass_batch=ErrorSummary.from_values(batch_error),
        forecasts=tuple(forecast_rows),
    )
