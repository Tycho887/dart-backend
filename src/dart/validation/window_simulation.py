"""Paired simulations using the epochs and geometry of recorded contacts."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import zlib

import numpy as np

from ..estimation import BatchConfig, PassiveRFUKF, UKFConfig, fit_batch
from ..geometry import tle_relative_geometry
from ..io.forest import ForestPass
from ..measurements import MeasurementModel, prepare_batch_cache, wrap_angle_rad
from ..simulation import observations_from_truth, phase_shifted_tle_truth, propagate_truth
from ..types import MeasurementMode, PhaseBaseline, TLEContext


@dataclass(frozen=True, slots=True)
class WindowSimulationResult:
    satellite: str
    contact_id: str
    station: str
    samples: int
    seed: int
    truth_tier: str
    channel_mode: str
    estimator: str
    estimator_variant: str
    true_offset_s: float
    estimated_offset_s: float
    offset_error_s: float
    median_position_error_km: float
    worst_position_error_km: float
    accepted: int
    rejected: int
    healthy: bool

    def as_dict(self) -> dict:
        return asdict(self)


def _truth_states(context: TLEContext, times: list, offset_s: float, tier: str) -> np.ndarray:
    shifted_initial = phase_shifted_tle_truth(context, [times[0]], offset_s)[0]
    if tier == "closure":
        return phase_shifted_tle_truth(context, times, offset_s)
    if tier in ("independent_dynamics", "empirical_residual"):
        return propagate_truth(shifted_initial, times)
    raise ValueError(f"unknown truth tier {tier!r}")


def _position_errors_km(
    context: TLEContext, times: list, truth_states: np.ndarray, offset_s: float
) -> np.ndarray:
    predicted = np.asarray(
        [
            tle_relative_geometry(
                context.tle, context.station, epoch, offset_s
            ).satellite_position_gcrf_m
            for epoch in times
        ]
    )
    return np.linalg.norm(predicted - truth_states[:, :3], axis=1) / 1000.0


def simulate_window(
    forest_pass: ForestPass,
    *,
    seed: int,
    tier: str,
    true_offset_s: float = 5.0,
    frequency_bias_hz: float = 300.0,
    phase_bias_rad: float = 1.2,
    doppler_std_hz: float = 500.0,
    phase_std_rad: float = 0.1,
    baseline_enu_m: tuple[float, float, float] = (59.0, 0.0, 0.0),
    doppler_noise_override: np.ndarray | None = None,
    operational: bool = False,
) -> list[WindowSimulationResult]:
    """Run paired Doppler and complete-phase trials with identical Doppler noise."""

    base = forest_pass.context
    context = TLEContext(
        base.tle,
        base.station,
        base.carrier_hz,
        PhaseBaseline(*baseline_enu_m),
    )
    times = [observation.epoch for observation in forest_pass.observations]
    truth_states = _truth_states(context, times, true_offset_s, tier)
    noiseless = observations_from_truth(
        context,
        times,
        truth_states,
        mode=MeasurementMode.DOPPLER_PHASE,
        frequency_bias_hz=frequency_bias_hz,
        phase_bias_rad=phase_bias_rad,
    )
    contact_seed = zlib.crc32(forest_pass.contact_id.encode("utf-8"))
    seed_sequence = np.random.SeedSequence([seed, contact_seed])
    doppler_rng, phase_rng = [
        np.random.default_rng(item) for item in seed_sequence.spawn(2)
    ]
    doppler_noise = (
        doppler_rng.normal(0.0, doppler_std_hz, len(noiseless))
        if doppler_noise_override is None
        else np.asarray(doppler_noise_override, dtype=float)
    )
    if len(doppler_noise) != len(noiseless):
        raise ValueError("Doppler noise override has the wrong length")
    phase_noise = phase_rng.normal(0.0, phase_std_rad, len(noiseless))
    complete_phase = [
        replace(
            observation,
            doppler_hz=float(observation.doppler_hz + doppler_noise[index]),
            phase_rad=float(wrap_angle_rad(observation.phase_rad + phase_noise[index])),
            valid=True,
        )
        for index, observation in enumerate(noiseless)
    ]
    doppler_only = [replace(observation, phase_rad=None) for observation in complete_phase]
    result = []
    batch_loss = "soft_l1" if operational else "linear"
    gate_probability = 0.9973 if operational else None
    doppler_batch = fit_batch(
        context,
        doppler_only,
        BatchConfig(
            doppler_std_hz=doppler_std_hz,
            phase_std_rad=phase_std_rad,
            robust_loss=batch_loss,
            offset_starts_s=(-30.0, 0.0, 30.0),
        ),
        mode=MeasurementMode.DOPPLER,
    )
    for mode, observations in (
        (MeasurementMode.DOPPLER, doppler_only),
        (MeasurementMode.DOPPLER_PHASE, complete_phase),
    ):
        batch = (
            doppler_batch
            if mode is MeasurementMode.DOPPLER
            else fit_batch(
                context,
                observations,
                BatchConfig(
                    doppler_std_hz=doppler_std_hz,
                    phase_std_rad=phase_std_rad,
                    robust_loss=batch_loss,
                    offset_starts_s=(
                        doppler_batch.estimate.offset_s,
                        -30.0,
                        0.0,
                        30.0,
                    ),
                ),
                mode=mode,
            )
        )
        ukf = PassiveRFUKF(
            context,
            initial_offset_s=0.0,
            phase_capable=mode is MeasurementMode.DOPPLER_PHASE,
            config=UKFConfig(
                doppler_std_hz=doppler_std_hz,
                phase_std_rad=phase_std_rad,
                gate_probability=gate_probability,
            ),
        )
        accepted = rejected = 0
        final = None
        for observation in observations:
            final = ukf.step(observation)
            accepted += int(final.accepted)
            rejected += int(not final.accepted)
        assert final is not None
        ukf_healthy = bool(
            final.healthy
            and accepted >= 20
            and accepted / max(1, accepted + rejected) >= 0.25
        )
        for estimator, offset, healthy, accepted_count, rejected_count in (
            ("batch", batch.estimate.offset_s, batch.estimate.healthy, len(observations), 0),
            ("static_ukf", final.offset_s, ukf_healthy, accepted, rejected),
        ):
            errors = _position_errors_km(context, times, truth_states, offset)
            result.append(
                WindowSimulationResult(
                    satellite=forest_pass.satellite,
                    contact_id=forest_pass.contact_id,
                    station=forest_pass.ground_station,
                    samples=len(observations),
                    seed=seed,
                    truth_tier=tier,
                    channel_mode=mode.value,
                    estimator=estimator,
                    estimator_variant=(
                        "operational_robust" if operational else "matched_gaussian"
                    ),
                    true_offset_s=true_offset_s,
                    estimated_offset_s=float(offset),
                    offset_error_s=float(offset - true_offset_s),
                    median_position_error_km=float(np.median(errors)),
                    worst_position_error_km=float(np.max(errors)),
                    accepted=accepted_count,
                    rejected=rejected_count,
                    healthy=bool(healthy),
                )
            )
    return result


def calibration_residuals(forest_passes: list[ForestPass]) -> list[np.ndarray]:
    """Return pass-centred RF residual sequences without using GPS truth."""

    model = MeasurementModel()
    result = []
    for forest_pass in forest_passes:
        observations = list(forest_pass.observations)
        fit = fit_batch(forest_pass.context, observations)
        cache = prepare_batch_cache(forest_pass.context, observations)
        predicted = model.predict_many(
            forest_pass.context,
            np.array([fit.estimate.offset_s, fit.estimate.frequency_bias_hz]),
            MeasurementMode.DOPPLER,
            cache,
        )[:, 0]
        measured = np.asarray([item.doppler_hz for item in observations], dtype=float)
        residual = measured - predicted
        result.append(residual - np.median(residual))
    return result


def sample_residual_blocks(
    sequences: list[np.ndarray],
    length: int,
    *,
    seed: int,
    block_size: int = 30,
) -> np.ndarray:
    """Block-bootstrap correlated residuals from calibration-only passes."""

    if not sequences:
        raise ValueError("at least one calibration residual sequence is required")
    rng = np.random.default_rng(seed)
    blocks = []
    remaining = length
    while remaining:
        sequence = sequences[int(rng.integers(0, len(sequences)))]
        take = min(block_size, remaining, len(sequence))
        start = int(rng.integers(0, len(sequence) - take + 1))
        blocks.append(sequence[start : start + take])
        remaining -= take
    return np.concatenate(blocks)
