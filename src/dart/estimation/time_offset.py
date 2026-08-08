"""Explicit Doppler-only forward models for production time-offset fitting."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import satkit as sk
from scipy.optimize import OptimizeResult, least_squares
from scipy.stats import qmc

from ..contracts import (
    Measurement,
    OptimizerModel,
    TimeOffsetFrequencyPassBiasMetaparameters,
    TimeOffsetMetaparameters,
    TimeOffsetPassBiasMetaparameters,
)
from ..geometry import station_from_itrf, tle_relative_geometry
from ..measurements import doppler_offset_hz
from ..types import Station
from .numerics import (
    at_any_bound,
    project_strictly_within_bounds,
    robust_weights,
    weighted_linear_algebra,
)


@dataclass(frozen=True, slots=True)
class DopplerSample:
    """A contract sample with its resolved station geometry."""

    source_index: int
    measurement: Measurement
    station: Station


@dataclass(frozen=True, slots=True)
class DopplerFit:
    model: OptimizerModel
    parameter_order: tuple[str, ...]
    parameters: np.ndarray
    effective_carrier_frequency_hz: float
    predicted_doppler_hz: np.ndarray
    residual_doppler_hz: np.ndarray
    source_indices: np.ndarray
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


@dataclass(frozen=True, slots=True)
class _SolveResult:
    result: OptimizeResult
    weights: np.ndarray
    covariance: np.ndarray
    rank: int
    condition: float
    at_bound: bool


@dataclass(frozen=True, slots=True)
class _DopplerProblem:
    tle: sk.TLE
    samples: list[DopplerSample]
    nominal_carrier_frequency_hz: float
    metaparameters: (
        TimeOffsetMetaparameters
        | TimeOffsetPassBiasMetaparameters
        | TimeOffsetFrequencyPassBiasMetaparameters
    )

    def predicted_doppler_hz(
        self,
        time_offset_s: float,
        effective_carrier_frequency_hz: float,
    ) -> np.ndarray:
        values = []
        for sample in self.samples:
            geometry = tle_relative_geometry(
                self.tle,
                sample.station,
                sk.time.from_datetime(sample.measurement.time_tag),
                time_offset_s,
            )
            values.append(
                float(doppler_offset_hz(geometry.range_rate_m_s, effective_carrier_frequency_hz))
            )
        return np.asarray(values)

    def observed_doppler_hz(self) -> np.ndarray:
        return np.asarray([sample.measurement.doppler_hz for sample in self.samples], dtype=float)

    def normalize(self, residual_hz: np.ndarray) -> np.ndarray:
        return residual_hz / self.metaparameters.doppler_standard_deviation_hz

    def jacobian(self, parameters: np.ndarray) -> np.ndarray:
        columns = [
            self.jacobian_column(parameters, index, step)
            for index, step in enumerate(self.jacobian_steps())
        ]
        return np.column_stack(columns)

    def jacobian_column(self, parameters: np.ndarray, index: int, step: float) -> np.ndarray:
        right = parameters.copy()
        left = parameters.copy()
        right[index] += step
        left[index] -= step
        return (self.residual(right) - self.residual(left)) / (2.0 * step)

    def residual(self, parameters: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def jacobian_steps(self) -> np.ndarray:
        raise NotImplementedError

    def predicted_for_parameters(self, parameters: np.ndarray) -> np.ndarray:
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class _TimeOffsetProblem(_DopplerProblem):
    def residual(self, parameters: np.ndarray) -> np.ndarray:
        predicted = self.predicted_doppler_hz(parameters[0], self.nominal_carrier_frequency_hz)
        return self.normalize(predicted - self.observed_doppler_hz())

    def jacobian_steps(self) -> np.ndarray:
        return np.asarray([1e-3])

    def predicted_for_parameters(self, parameters: np.ndarray) -> np.ndarray:
        return self.predicted_doppler_hz(parameters[0], self.nominal_carrier_frequency_hz)


@dataclass(frozen=True, slots=True)
class _TimeOffsetPassBiasProblem(_DopplerProblem):
    pass_indices: np.ndarray

    def residual(self, parameters: np.ndarray) -> np.ndarray:
        predicted = self.predicted_doppler_hz(parameters[0], self.nominal_carrier_frequency_hz)
        predicted += parameters[1:][self.pass_indices]
        return self.normalize(predicted - self.observed_doppler_hz())

    def jacobian_steps(self) -> np.ndarray:
        return np.asarray([1e-3] + [1.0] * (1 + int(np.max(self.pass_indices))))

    def predicted_for_parameters(self, parameters: np.ndarray) -> np.ndarray:
        prediction = self.predicted_doppler_hz(parameters[0], self.nominal_carrier_frequency_hz)
        return prediction + parameters[1:][self.pass_indices]


@dataclass(frozen=True, slots=True)
class _TimeOffsetFrequencyPassBiasProblem(_DopplerProblem):
    pass_indices: np.ndarray

    def residual(self, parameters: np.ndarray) -> np.ndarray:
        effective_carrier_frequency_hz = self.nominal_carrier_frequency_hz + parameters[1]
        predicted = self.predicted_doppler_hz(parameters[0], effective_carrier_frequency_hz)
        predicted += parameters[2:][self.pass_indices]
        return self.normalize(predicted - self.observed_doppler_hz())

    def jacobian_steps(self) -> np.ndarray:
        return np.asarray([1e-3, 1.0] + [1.0] * (1 + int(np.max(self.pass_indices))))

    def predicted_for_parameters(self, parameters: np.ndarray) -> np.ndarray:
        effective_carrier_frequency_hz = self.nominal_carrier_frequency_hz + parameters[1]
        prediction = self.predicted_doppler_hz(parameters[0], effective_carrier_frequency_hz)
        return prediction + parameters[2:][self.pass_indices]


def samples_from_measurements(measurements: list[Measurement]) -> list[DopplerSample]:
    """Resolve station geometry once while preserving contract input order."""

    samples = []
    for index, measurement in enumerate(measurements):
        station = station_from_itrf(
            measurement.station_id,
            np.asarray(measurement.station_position_itrf_m.as_list(), dtype=float),
        )
        samples.append(DopplerSample(index, measurement, station))
    return samples


def valid_doppler_samples(samples: list[DopplerSample]) -> list[DopplerSample]:
    return [
        sample
        for sample in samples
        if sample.measurement.valid and sample.measurement.doppler_hz is not None
    ]


def ordered_pass_ids(samples: list[DopplerSample]) -> list[str]:
    return list(dict.fromkeys(sample.measurement.pass_id for sample in samples))


def _pass_indices(samples: list[DopplerSample], pass_ids: list[str]) -> np.ndarray:
    index_by_pass = {pass_id: index for index, pass_id in enumerate(pass_ids)}
    return np.asarray([index_by_pass[sample.measurement.pass_id] for sample in samples], dtype=int)


def _require_identifiable_samples(sample_count: int, parameter_count: int) -> None:
    if sample_count <= parameter_count:
        raise ValueError(
            f"at least {parameter_count + 1} valid Doppler observations are required "
            f"for {parameter_count} fitted parameters"
        )


def _best_start(
    problem: _TimeOffsetProblem | _TimeOffsetPassBiasProblem | _TimeOffsetFrequencyPassBiasProblem,
    candidates: np.ndarray,
) -> np.ndarray:
    best = candidates[0]
    best_cost = float(np.sum(np.square(problem.residual(best))))
    for candidate in candidates[1:]:
        cost = float(np.sum(np.square(problem.residual(candidate))))
        if cost < best_cost:
            best = candidate
            best_cost = cost
    return best


def _solve(
    problem: _TimeOffsetProblem | _TimeOffsetPassBiasProblem | _TimeOffsetFrequencyPassBiasProblem,
    baseline: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> _SolveResult:
    candidates = [project_strictly_within_bounds(baseline, lower, upper)]
    if problem.metaparameters.qmc_samples:
        sampler = qmc.LatinHypercube(d=len(baseline), seed=42)
        qmc_candidates = qmc.scale(
            sampler.random(problem.metaparameters.qmc_samples),
            lower,
            upper,
        )
        candidates.extend(
            project_strictly_within_bounds(candidate, lower, upper) for candidate in qmc_candidates
        )
    initial = _best_start(problem, np.asarray(candidates))
    result = least_squares(
        problem.residual,
        initial,
        jac=problem.jacobian,
        bounds=(lower, upper),
        loss=problem.metaparameters.robust_loss,
        f_scale=(
            problem.metaparameters.robust_scale_hz
            / problem.metaparameters.doppler_standard_deviation_hz
        ),
        x_scale="jac",
        ftol=1e-10,
        xtol=1e-10,
        gtol=1e-10,
    )
    scaled_residual = problem.residual(result.x)
    weights = robust_weights(
        scaled_residual,
        problem.metaparameters.robust_loss,
        problem.metaparameters.robust_scale_hz
        / problem.metaparameters.doppler_standard_deviation_hz,
    )
    diagnostics = weighted_linear_algebra(problem.jacobian(result.x), scaled_residual, weights)
    return _SolveResult(
        result,
        weights,
        diagnostics.covariance,
        diagnostics.rank,
        diagnostics.condition,
        at_any_bound(result.x, lower, upper),
    )


def _fit(
    model: OptimizerModel,
    parameter_order: tuple[str, ...],
    problem: _TimeOffsetProblem | _TimeOffsetPassBiasProblem | _TimeOffsetFrequencyPassBiasProblem,
    baseline: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    effective_carrier_frequency_hz: float,
) -> DopplerFit:
    solved = _solve(problem, baseline, lower, upper)
    predicted = problem.predicted_for_parameters(solved.result.x)
    residual = predicted - problem.observed_doppler_hz()
    healthy = bool(
        solved.result.success
        and solved.rank == len(solved.result.x)
        and np.isfinite(solved.condition)
        and solved.condition <= 1e12
        and not solved.at_bound
    )
    return DopplerFit(
        model=model,
        parameter_order=parameter_order,
        parameters=np.asarray(solved.result.x).copy(),
        effective_carrier_frequency_hz=effective_carrier_frequency_hz,
        predicted_doppler_hz=predicted,
        residual_doppler_hz=residual,
        source_indices=np.asarray([sample.source_index for sample in problem.samples], dtype=int),
        robust_weights=solved.weights,
        covariance=solved.covariance,
        success=bool(solved.result.success),
        healthy=healthy,
        message=str(solved.result.message),
        weighted_ssr=float(np.sum(solved.weights * np.square(residual))),
        robust_cost=float(solved.result.cost),
        jacobian_rank=solved.rank,
        jacobian_condition=solved.condition,
        at_bound=solved.at_bound,
    )


def fit_time_offset(
    tle: sk.TLE,
    nominal_carrier_frequency_hz: float,
    samples: list[DopplerSample],
    metaparameters: TimeOffsetMetaparameters,
) -> DopplerFit:
    """Fit one global time offset with a fixed nominal carrier frequency."""

    if metaparameters.model is not OptimizerModel.TIME_OFFSET:
        raise ValueError("fit_time_offset requires model time_offset")
    _require_identifiable_samples(len(samples), 1)
    problem = _TimeOffsetProblem(tle, samples, nominal_carrier_frequency_hz, metaparameters)
    lower = np.asarray([metaparameters.time_offset_bounds_s[0]])
    upper = np.asarray([metaparameters.time_offset_bounds_s[1]])
    return _fit(
        OptimizerModel.TIME_OFFSET,
        ("time_offset_s",),
        problem,
        np.asarray([0.0]),
        lower,
        upper,
        nominal_carrier_frequency_hz,
    )


def fit_time_offset_pass_bias(
    tle: sk.TLE,
    nominal_carrier_frequency_hz: float,
    samples: list[DopplerSample],
    metaparameters: TimeOffsetPassBiasMetaparameters,
) -> DopplerFit:
    """Fit one global time offset and an independently ordered Doppler bias per pass."""

    if metaparameters.model is not OptimizerModel.TIME_OFFSET_PASS_BIAS:
        raise ValueError("fit_time_offset_pass_bias requires model time_offset_pass_bias")
    pass_ids = ordered_pass_ids(samples)
    _require_identifiable_samples(len(samples), 1 + len(pass_ids))
    problem = _TimeOffsetPassBiasProblem(
        tle,
        samples,
        nominal_carrier_frequency_hz,
        metaparameters,
        _pass_indices(samples, pass_ids),
    )
    lower = np.asarray(
        [metaparameters.time_offset_bounds_s[0]]
        + [metaparameters.pass_bias_bounds_hz[0]] * len(pass_ids)
    )
    upper = np.asarray(
        [metaparameters.time_offset_bounds_s[1]]
        + [metaparameters.pass_bias_bounds_hz[1]] * len(pass_ids)
    )
    return _fit(
        OptimizerModel.TIME_OFFSET_PASS_BIAS,
        ("time_offset_s", *(f"pass_bias_hz:{pass_id}" for pass_id in pass_ids)),
        problem,
        np.zeros(1 + len(pass_ids)),
        lower,
        upper,
        nominal_carrier_frequency_hz,
    )


def fit_time_offset_frequency_pass_bias(
    tle: sk.TLE,
    nominal_carrier_frequency_hz: float,
    samples: list[DopplerSample],
    metaparameters: TimeOffsetFrequencyPassBiasMetaparameters,
) -> DopplerFit:
    """Fit time offset, centre-frequency correction, and ordered per-pass biases."""

    if metaparameters.model is not OptimizerModel.TIME_OFFSET_FREQUENCY_PASS_BIAS:
        raise ValueError(
            "fit_time_offset_frequency_pass_bias requires model time_offset_frequency_pass_bias"
        )
    pass_ids = ordered_pass_ids(samples)
    _require_identifiable_samples(len(samples), 2 + len(pass_ids))
    problem = _TimeOffsetFrequencyPassBiasProblem(
        tle,
        samples,
        nominal_carrier_frequency_hz,
        metaparameters,
        _pass_indices(samples, pass_ids),
    )
    frequency_lower = max(
        metaparameters.center_frequency_correction_bounds_hz[0],
        -nominal_carrier_frequency_hz + 1.0,
    )
    frequency_upper = metaparameters.center_frequency_correction_bounds_hz[1]
    if frequency_lower >= frequency_upper:
        raise ValueError("center frequency correction bounds do not allow a positive carrier")
    lower = np.asarray(
        [metaparameters.time_offset_bounds_s[0], frequency_lower]
        + [metaparameters.pass_bias_bounds_hz[0]] * len(pass_ids)
    )
    upper = np.asarray(
        [metaparameters.time_offset_bounds_s[1], frequency_upper]
        + [metaparameters.pass_bias_bounds_hz[1]] * len(pass_ids)
    )
    fit = _fit(
        OptimizerModel.TIME_OFFSET_FREQUENCY_PASS_BIAS,
        (
            "time_offset_s",
            "center_frequency_correction_hz",
            *(f"pass_bias_hz:{pass_id}" for pass_id in pass_ids),
        ),
        problem,
        np.zeros(2 + len(pass_ids)),
        lower,
        upper,
        nominal_carrier_frequency_hz,
    )
    return DopplerFit(
        model=fit.model,
        parameter_order=fit.parameter_order,
        parameters=fit.parameters,
        effective_carrier_frequency_hz=nominal_carrier_frequency_hz + fit.parameters[1],
        predicted_doppler_hz=fit.predicted_doppler_hz,
        residual_doppler_hz=fit.residual_doppler_hz,
        source_indices=fit.source_indices,
        robust_weights=fit.robust_weights,
        covariance=fit.covariance,
        success=fit.success,
        healthy=fit.healthy,
        message=fit.message,
        weighted_ssr=fit.weighted_ssr,
        robust_cost=fit.robust_cost,
        jacobian_rank=fit.jacobian_rank,
        jacobian_condition=fit.jacobian_condition,
        at_bound=fit.at_bound,
    )
