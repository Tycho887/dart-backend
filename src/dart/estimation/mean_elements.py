"""Production Doppler fit for two SGP4 mean elements and per-pass biases."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import satkit as sk
from scipy.optimize import OptimizeResult, least_squares
from scipy.stats import qmc
from sgp4.api import WGS72, Satrec
from sgp4.exporter import export_tle

from ..measurements import BatchMeasurementCache, predict_cached_tle_doppler_hz, prepare_batch_cache
from ..types import RFObservation, TLEContext
from .numerics import (
    at_any_bound,
    project_strictly_within_bounds,
    robust_weights,
    weighted_linear_algebra,
)

_TWO_PI = 2.0 * np.pi


@dataclass(frozen=True, slots=True)
class MeanElementsTwoParameterConfig:
    doppler_standard_deviation_hz: float
    mean_anomaly_half_width_rad: float
    mean_motion_half_width_rad_min: float
    pass_bias_bounds_hz: tuple[float, float]
    qmc_samples: int
    robust_loss: str
    robust_scale_hz: float


@dataclass(frozen=True, slots=True)
class MeanElementsTwoParameterFit:
    corrected_tle: sk.TLE
    mean_anomaly_rad: float
    mean_motion_rad_min: float
    pass_biases_hz: np.ndarray
    predicted_doppler_hz: np.ndarray
    residual_doppler_hz: np.ndarray
    robust_weights: np.ndarray
    covariance: np.ndarray
    success: bool
    healthy: bool
    message: str
    weighted_ssr: float
    robust_cost: float
    jacobian_rank: int
    jacobian_condition: float
    at_bound: bool


def rebuild_tle_mean_elements(
    tle: sk.TLE,
    mean_anomaly_rad: float,
    mean_motion_rad_min: float,
) -> sk.TLE:
    """Return a TLE with exactly mean anomaly and positive mean motion replaced."""

    if not np.isfinite([mean_anomaly_rad, mean_motion_rad_min]).all():
        raise ValueError("mean-element candidate values must be finite")
    if mean_motion_rad_min <= 0.0:
        raise ValueError("mean motion must be strictly positive")
    line1, line2 = tle.to_2line()
    source = Satrec.twoline2rv(line1, line2)
    try:
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
            _normalize_angle(mean_anomaly_rad),
            float(mean_motion_rad_min),
            source.nodeo,
        )
        rebuilt = sk.TLE.from_lines(list(export_tle(source)))
    except RuntimeError as exc:
        raise ValueError("invalid mean-element orbital candidate") from exc
    if isinstance(rebuilt, list):
        if len(rebuilt) != 1:
            raise ValueError("mean-element reconstruction produced multiple TLEs")
        return rebuilt[0]
    return rebuilt


@dataclass(frozen=True, slots=True)
class _MeanElementsProblem:
    context: TLEContext
    observations: list[RFObservation]
    pass_indices: np.ndarray
    config: MeanElementsTwoParameterConfig
    cache: BatchMeasurementCache
    base_mean_anomaly_rad: float
    lower: np.ndarray
    upper: np.ndarray

    def __post_init__(self) -> None:
        if any(
            not observation.valid or observation.doppler_hz is None
            for observation in self.observations
        ):
            raise ValueError("mean_elements_two_parameter requires valid Doppler observations")

    def candidate_tle(self, parameters: np.ndarray) -> sk.TLE:
        return rebuild_tle_mean_elements(
            self.context.tle,
            self.base_mean_anomaly_rad + parameters[0],
            parameters[1],
        )

    def predicted_doppler_hz(self, parameters: np.ndarray) -> np.ndarray:
        predicted = self.unbiased_doppler_hz(parameters[:2])
        return predicted + parameters[2:][self.pass_indices]

    def unbiased_doppler_hz(self, orbital_parameters: np.ndarray) -> np.ndarray:
        candidate_context = TLEContext(
            tle=self.candidate_tle(orbital_parameters),
            station=self.context.station,
            carrier_hz=self.context.carrier_hz,
            baseline=self.context.baseline,
        )
        return predict_cached_tle_doppler_hz(candidate_context, self.cache)

    def observed_doppler_hz(self) -> np.ndarray:
        return np.asarray(
            [observation.doppler_hz for observation in self.observations],
            dtype=float,
        )

    def profiled_pass_biases_hz(self, orbital_parameters: np.ndarray) -> np.ndarray:
        """Profile the least-squares Doppler bias for each ordered pass."""

        unbiased = self.unbiased_doppler_hz(orbital_parameters)
        observed = self.observed_doppler_hz()
        residual_hz = observed - unbiased
        bias_count = len(self.lower) - 2
        biases = np.asarray(
            [np.mean(residual_hz[self.pass_indices == index]) for index in range(bias_count)]
        )
        return project_strictly_within_bounds(biases, self.lower[2:], self.upper[2:])

    def residual(self, parameters: np.ndarray) -> np.ndarray:
        residual_hz = self.predicted_doppler_hz(parameters) - self.observed_doppler_hz()
        return residual_hz / self.config.doppler_standard_deviation_hz

    def jacobian(self, parameters: np.ndarray) -> np.ndarray:
        steps = np.asarray([1e-5, 1e-7] + [1.0] * len(np.unique(self.pass_indices)))
        columns = [
            self.jacobian_column(parameters, index, step) for index, step in enumerate(steps)
        ]
        return np.column_stack(columns)

    def jacobian_column(self, parameters: np.ndarray, index: int, step: float) -> np.ndarray:
        right = parameters.copy()
        left = parameters.copy()
        right[index] = min(self.upper[index], right[index] + step)
        left[index] = max(self.lower[index], left[index] - step)
        if right[index] == left[index]:
            raise ValueError("cannot compute a derivative at a zero-width bound")
        return (self.residual(right) - self.residual(left)) / (right[index] - left[index])


def _normalize_angle(value: float) -> float:
    return float(value % _TWO_PI)


def _ordered_pass_ids(pass_ids: list[str]) -> list[str]:
    return list(dict.fromkeys(pass_ids))


def _pass_indices(pass_ids: list[str], ordered_pass_ids: list[str]) -> np.ndarray:
    index_by_pass = {pass_id: index for index, pass_id in enumerate(ordered_pass_ids)}
    return np.asarray([index_by_pass[pass_id] for pass_id in pass_ids], dtype=int)


def _validate_config(config: MeanElementsTwoParameterConfig) -> None:
    if config.doppler_standard_deviation_hz <= 0.0 or config.robust_scale_hz <= 0.0:
        raise ValueError("Doppler standard deviation and robust scale must be positive")
    if config.mean_anomaly_half_width_rad <= 0.0 or config.mean_anomaly_half_width_rad > np.pi:
        raise ValueError("mean anomaly half width must be in (0, pi]")
    if config.mean_motion_half_width_rad_min <= 0.0:
        raise ValueError("mean motion half width must be positive")
    if config.pass_bias_bounds_hz[0] >= config.pass_bias_bounds_hz[1]:
        raise ValueError("invalid pass bias bounds")


def _mean_motion_bounds(
    base_mean_motion_rad_min: float,
    half_width_rad_min: float,
) -> tuple[float, float]:
    lower = max(np.nextafter(0.0, np.inf), base_mean_motion_rad_min - half_width_rad_min)
    upper = base_mean_motion_rad_min + half_width_rad_min
    if not np.isfinite([lower, upper]).all() or lower >= upper:
        raise ValueError("mean-motion bounds must contain strictly positive values")
    return float(lower), float(upper)


def _parameter_bounds(
    base_mean_motion_rad_min: float,
    pass_count: int,
    config: MeanElementsTwoParameterConfig,
) -> tuple[np.ndarray, np.ndarray]:
    mean_motion_lower, mean_motion_upper = _mean_motion_bounds(
        base_mean_motion_rad_min,
        config.mean_motion_half_width_rad_min,
    )
    lower = np.asarray(
        [
            -config.mean_anomaly_half_width_rad,
            mean_motion_lower,
            *([config.pass_bias_bounds_hz[0]] * pass_count),
        ]
    )
    upper = np.asarray(
        [
            config.mean_anomaly_half_width_rad,
            mean_motion_upper,
            *([config.pass_bias_bounds_hz[1]] * pass_count),
        ]
    )
    return lower, upper


def _initial_candidates(problem: _MeanElementsProblem, orbital_baseline: np.ndarray) -> np.ndarray:
    orbital_lower = problem.lower[:2]
    orbital_upper = problem.upper[:2]
    orbital_candidates = [
        project_strictly_within_bounds(orbital_baseline, orbital_lower, orbital_upper)
    ]
    if problem.config.qmc_samples:
        sampler = qmc.LatinHypercube(d=len(orbital_baseline), seed=42)
        samples = qmc.scale(
            sampler.random(problem.config.qmc_samples),
            orbital_lower,
            orbital_upper,
        )
        orbital_candidates.extend(
            project_strictly_within_bounds(sample, orbital_lower, orbital_upper)
            for sample in samples
        )
    candidates = []
    for orbital_candidate in orbital_candidates:
        try:
            pass_biases = problem.profiled_pass_biases_hz(orbital_candidate)
        except (RuntimeError, ValueError):
            continue
        candidates.append(np.concatenate((orbital_candidate, pass_biases)))
    return np.asarray(candidates)


def _candidate_cost(problem: _MeanElementsProblem, candidate: np.ndarray) -> float:
    try:
        return float(np.sum(np.square(problem.residual(candidate))))
    except (RuntimeError, ValueError):
        return float("inf")


def _best_start(problem: _MeanElementsProblem, candidates: np.ndarray) -> np.ndarray:
    if not len(candidates):
        raise ValueError("no valid mean-element initial candidate lies within requested bounds")
    scored = [(candidate, _candidate_cost(problem, candidate)) for candidate in candidates]
    viable = [item for item in scored if np.isfinite(item[1])]
    if not viable:
        raise ValueError("no valid mean-element initial candidate lies within requested bounds")
    return min(viable, key=lambda item: item[1])[0]


def _solve_problem(
    problem: _MeanElementsProblem,
    initial: np.ndarray,
) -> OptimizeResult:
    try:
        return least_squares(
            problem.residual,
            initial,
            jac=problem.jacobian,
            bounds=(problem.lower, problem.upper),
            method="trf",
            loss=problem.config.robust_loss,
            f_scale=problem.config.robust_scale_hz / problem.config.doppler_standard_deviation_hz,
            x_scale=np.asarray([0.05, 0.001] + [2_000.0] * len(np.unique(problem.pass_indices))),
            ftol=1e-11,
            xtol=1e-11,
            gtol=1e-11,
        )
    except (RuntimeError, ValueError) as exc:
        raise ValueError("mean-element solver could not evaluate an orbital candidate") from exc


def _fit_diagnostics(
    problem: _MeanElementsProblem,
    result: OptimizeResult,
) -> tuple[np.ndarray, np.ndarray, int, float, bool]:
    scaled_residual = problem.residual(result.x)
    weights = robust_weights(
        scaled_residual,
        problem.config.robust_loss,
        problem.config.robust_scale_hz / problem.config.doppler_standard_deviation_hz,
    )
    linear_algebra = weighted_linear_algebra(problem.jacobian(result.x), scaled_residual, weights)
    return (
        weights,
        linear_algebra.covariance,
        linear_algebra.rank,
        linear_algebra.condition,
        at_any_bound(result.x, problem.lower, problem.upper),
    )


def fit_mean_elements_two_parameter(
    context: TLEContext,
    observations: list[RFObservation],
    pass_ids: list[str],
    config: MeanElementsTwoParameterConfig,
) -> MeanElementsTwoParameterFit:
    """Fit circular mean-anomaly delta, positive mean motion, and pass biases."""

    _validate_config(config)
    if len(observations) != len(pass_ids):
        raise ValueError("pass_ids must align with observations")
    ordered_pass_ids = _ordered_pass_ids(pass_ids)
    if len(ordered_pass_ids) < 2:
        raise ValueError("mean_elements_two_parameter requires at least two distinct passes")
    parameter_count = 2 + len(ordered_pass_ids)
    if len(observations) <= parameter_count:
        raise ValueError(
            f"at least {parameter_count + 1} valid Doppler observations are required "
            f"for {parameter_count} fitted parameters"
        )
    line1, line2 = context.tle.to_2line()
    source = Satrec.twoline2rv(line1, line2)
    base_mean_anomaly_rad = float(source.mo)
    base_mean_motion_rad_min = float(source.no_kozai)
    lower, upper = _parameter_bounds(base_mean_motion_rad_min, len(ordered_pass_ids), config)
    problem = _MeanElementsProblem(
        context=context,
        observations=observations,
        pass_indices=_pass_indices(pass_ids, ordered_pass_ids),
        config=config,
        cache=prepare_batch_cache(context, observations),
        base_mean_anomaly_rad=base_mean_anomaly_rad,
        lower=lower,
        upper=upper,
    )
    orbital_baseline = np.asarray([0.0, base_mean_motion_rad_min])
    initial = _best_start(problem, _initial_candidates(problem, orbital_baseline))
    result = _solve_problem(problem, initial)
    weights, covariance, rank, condition, at_bound = _fit_diagnostics(problem, result)
    predicted = problem.predicted_doppler_hz(result.x)
    residual = predicted - problem.observed_doppler_hz()
    healthy = bool(
        result.success
        and rank == parameter_count
        and np.isfinite(condition)
        and condition <= 1e12
        and not at_bound
    )
    return MeanElementsTwoParameterFit(
        corrected_tle=problem.candidate_tle(result.x),
        mean_anomaly_rad=_normalize_angle(base_mean_anomaly_rad + result.x[0]),
        mean_motion_rad_min=float(result.x[1]),
        pass_biases_hz=np.asarray(result.x[2:]).copy(),
        predicted_doppler_hz=predicted,
        residual_doppler_hz=residual,
        robust_weights=weights,
        covariance=covariance,
        success=bool(result.success),
        healthy=healthy,
        message=str(result.message),
        weighted_ssr=float(np.sum(weights * np.square(residual))),
        robust_cost=float(result.cost),
        jacobian_rank=rank,
        jacobian_condition=condition,
        at_bound=at_bound,
    )
