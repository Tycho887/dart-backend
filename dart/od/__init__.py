"""Batch orbit determination using DART's authoritative forward models."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable
from dataclasses import replace

import numpy as np
import satkit as sk
from numpy.typing import NDArray
from scipy.optimize import OptimizeResult, least_squares

from dart.forward_models import (
    ForwardModelEvaluation,
    corrected_tle_lines,
    evaluate_full_state_augmented,
    evaluate_sgp4_epoch,
    prepare_sgp4_tle,
    tle_state_gcrf,
)
from dart.orbit import CartesianOrbit, OrbitSolution, Sgp4Orbit

from .cca import compute_consider_covariance, full_consider_covariance
from .schema import (
    OptimizerContext,
    OptimizerOutput,
    OrbitModel,
    ParameterRole,
    ParameterSpec,
    PriorSource,
    PriorStateData,
)

_SGP4_PARAMETER_NAMES = (
    "mean_motion_rev_per_day",
    "equinoctial_f",
    "equinoctial_g",
    "equinoctial_h",
    "equinoctial_k",
    "mean_longitude_deg",
    "bstar",
)
_FULL_STATE_PARAMETER_NAMES = (
    "position_x_m",
    "position_y_m",
    "position_z_m",
    "velocity_x_m_s",
    "velocity_y_m_s",
    "velocity_z_m_s",
)
_SHARED_PARAMETER_NAMES = ("time_offset_s", "center_frequency_offset_hz")
_DRAG_PARAMETER_NAME = "cd_a_over_m_m2_kg"
_LOSSES = {"linear", "soft_l1", "huber", "cauchy", "arctan"}
FloatArray = NDArray[np.float64]
Evaluator = Callable[[FloatArray], ForwardModelEvaluation]


def _source_tle_lines(raw: str) -> list[str]:
    """Guard fixed-width inputs before satkit's unchecked Python line dispatch."""
    lines = [line.rstrip() for line in raw.splitlines() if line.strip()]
    if len(lines) not in (2, 3):
        raise ValueError("source ephemeris must contain exactly one TLE")
    for number, line in enumerate(lines[-2:], start=1):
        if len(line) != 69 or not line.isascii() or not line.startswith(f"{number} "):
            raise ValueError("source ephemeris contains an invalid fixed-width TLE")
    return lines


def _canonical_tle(data: PriorStateData) -> tuple[str, str]:
    if data.prepared_tle is not None:
        return _derived_tle(data.prepared_tle.tle_lines, data.ephemeris.tle or "")
    if data.derived_tle_lines is not None:
        return _derived_tle(data.derived_tle_lines, data.ephemeris.tle or "")
    raw = data.ephemeris.tle
    if raw is None or not raw.strip():
        raise ValueError("source ephemeris does not contain a TLE")
    try:
        parsed = sk.TLE.from_lines(_source_tle_lines(raw))
    except Exception as exc:
        raise ValueError("source ephemeris contains an invalid TLE") from exc
    if isinstance(parsed, list):
        if len(parsed) != 1:
            raise ValueError("source ephemeris must contain exactly one TLE")
        parsed = parsed[0]
    # Keep the delivered identity columns; satkit's Python serializer normalizes
    # classification/designator fields that the native serializer preserves.
    lines = _source_tle_lines(raw)[-2:]
    return str(lines[0]), str(lines[1])


def _derived_tle(lines: tuple[str, str], original: str) -> tuple[str, str]:
    if len(lines) != 2:
        raise ValueError("derived TLE must contain two lines")
    _source_tle_lines("\n".join(lines))
    source = _source_tle_lines(original)[-2:]
    if lines[0][2:17] != source[0][2:17] or lines[1][2:7] != source[1][2:7]:
        raise ValueError("derived TLE spacecraft identifiers differ from source")
    # Do not reserialize a product already validated by the Rust core.
    return lines


def prepare_sgp4_prior(
    data: PriorStateData, optimizer: OptimizerContext | None = None
) -> PriorStateData:
    """Center an SGP4 prior once, before scans/fits, retaining its source report.

    Cartesian callers do not use this step. The optional optimizer widens the
    preservation interval to cover both clock and TLE-epoch search bounds.
    """
    _validate_data(data)
    times = [o.time for o in data.observations.observations]
    window = _preparation_window(data, optimizer, times)
    # Always derive from the selected source, even if a caller changes the
    # observations or source on a previously prepared input.
    lines = _canonical_tle(replace(data, prepared_tle=None))
    report = prepare_sgp4_tle(lines, times, window=window)
    return replace(data, prepared_tle=report, preservation_window=window)


def _preparation_window(
    data: PriorStateData, optimizer: OptimizerContext | None, times: list[sk.time]
) -> tuple[sk.time, sk.time] | None:
    if optimizer is None:
        return data.preservation_window
    margin = sum(
        max(abs(p.lower_bound), abs(p.upper_bound))
        if p.role == ParameterRole.ESTIMATE
        else abs(p.initial)
        for p in optimizer.parameters
        if p.name in {"time_offset_s", "tle_epoch_offset_s"}
    )
    start = min(times) - sk.duration(seconds=margin)
    stop = max(times) + sk.duration(seconds=margin)
    if data.preservation_window is not None:
        start = min(start, data.preservation_window[0])
        stop = max(stop, data.preservation_window[1])
    # A singleton without a timing search uses the helper's one-orbit interval.
    return (start, stop) if start < stop else None


def _pass_parameter_names(data: PriorStateData) -> tuple[str, ...]:
    mapping = data.observations.contact_to_pass_idx
    if set(mapping.values()) != set(range(len(mapping))):
        raise ValueError("contact pass indices must be unique and contiguous from zero")
    contacts = sorted(mapping, key=mapping.__getitem__)
    return tuple(f"pass_bias_hz:{contact_id}" for contact_id in contacts)


def _canonical_parameter_names(
    data: PriorStateData,
    model: OrbitModel,
    include_drag: bool = False,
) -> tuple[str, ...]:
    pass_names = _pass_parameter_names(data)
    if model == OrbitModel.SGP4:
        return (
            _SGP4_PARAMETER_NAMES
            + _SHARED_PARAMETER_NAMES
            + pass_names
            + ("tle_epoch_offset_s",)
        )
    drag_names = (_DRAG_PARAMETER_NAME,) if include_drag else ()
    return (
        _FULL_STATE_PARAMETER_NAMES + _SHARED_PARAMETER_NAMES + pass_names + drag_names
    )


def _validate_parameter(parameter: ParameterSpec) -> None:
    if not isinstance(parameter.role, ParameterRole):
        raise TypeError(f"parameter {parameter.name!r} role must be a ParameterRole")
    values = (
        parameter.initial,
        parameter.lower_bound,
        parameter.upper_bound,
        parameter.scale,
    )
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"parameter {parameter.name!r} values must be finite")
    if parameter.lower_bound >= parameter.upper_bound:
        raise ValueError(
            f"parameter {parameter.name!r} lower bound must be below its upper bound"
        )
    if (
        parameter.lower_bound > parameter.initial
        or parameter.initial > parameter.upper_bound
    ):
        raise ValueError(
            f"parameter {parameter.name!r} initial value is outside its bounds"
        )
    if parameter.scale <= 0:
        raise ValueError(f"parameter {parameter.name!r} scale must be positive")
    if parameter.name == _DRAG_PARAMETER_NAME and parameter.lower_bound < 0:
        raise ValueError("Cd A/m bounds must be nonnegative (m²/kg)")
    if parameter.prior_standard_uncertainty is not None and (
        not math.isfinite(parameter.prior_standard_uncertainty)
        or parameter.prior_standard_uncertainty <= 0
    ):
        raise ValueError(
            f"parameter {parameter.name!r} prior standard uncertainty must be positive"
        )


def _validate_optimizer(
    data: PriorStateData,
    optimizer: OptimizerContext,
) -> None:
    if not isinstance(optimizer.model, OrbitModel):
        raise TypeError("optimizer model must be an OrbitModel")
    names = tuple(parameter.name for parameter in optimizer.parameters)
    canonical = _canonical_parameter_names(data, optimizer.model, include_drag=True)
    if len(set(names)) != len(names):
        raise ValueError("optimizer parameter names must be unique")
    unknown = tuple(name for name in names if name not in canonical)
    if unknown:
        raise ValueError(
            f"unsupported optimizer parameters {unknown}; canonical names are {canonical}"
        )
    for parameter in optimizer.parameters:
        _validate_parameter(parameter)
    if optimizer.loss not in _LOSSES:
        raise ValueError(f"unsupported loss: {optimizer.loss}")
    if optimizer.x_scale not in ("profile", "jac"):
        raise ValueError(f"unsupported x_scale: {optimizer.x_scale}")
    settings = (
        optimizer.loss_scale,
        optimizer.ftol,
        optimizer.xtol,
        optimizer.gtol,
    )
    if not all(math.isfinite(value) and value > 0 for value in settings):
        raise ValueError(
            "loss scale and optimizer tolerances must be finite and positive"
        )
    if optimizer.max_evaluations <= 0:
        raise ValueError("max_evaluations must be positive")


def _validate_data(data: PriorStateData) -> None:
    if not isinstance(data.epoch, sk.time):
        raise TypeError("prior-state epoch must be a satkit.time")
    data.observations.validate()
    for contact in data.observations.contacts.values():
        if contact.spacecraft_id != data.ephemeris.spacecraft_id:
            raise ValueError("observation and source-ephemeris spacecraft IDs differ")
        # A deliberately selected fit prior may differ from the ephemeris
        # originally used to track any individual contact.


def _materialize_orbit(
    data: PriorStateData, model: OrbitModel, values: dict[str, float]
) -> OrbitSolution:
    _validate_data(data)
    object_ids = {contact.cospar for contact in data.observations.contacts.values()}
    if len(object_ids) != 1 or not next(iter(object_ids)).strip():
        raise ValueError("orbit products require one explicit contact COSPAR identity")
    object_id = next(iter(object_ids))
    lines = _canonical_tle(data)
    identity = json.dumps(
        [
            model,
            data.ephemeris.ephemeris_id,
            lines,
            data.epoch.as_unixtime(),
            values,
            None
            if data.nominal_state_gcrf_si is None
            else data.nominal_state_gcrf_si.tolist(),
        ],
        sort_keys=True,
        allow_nan=False,
    )
    solution_id = hashlib.sha256(identity.encode()).hexdigest()
    if model == OrbitModel.SGP4:
        return _sgp4_product(data, values, object_id, solution_id, lines)
    if model != OrbitModel.FULL_STATE:
        raise ValueError(f"unsupported orbit model: {model}")
    nominal, _ = _full_state(data, lines)
    correction = np.array(
        [values.get(name, 0.0) for name in _FULL_STATE_PARAMETER_NAMES]
    )
    state = tuple(float(value) for value in nominal + correction)
    return CartesianOrbit(
        object_id,
        data.ephemeris,
        solution_id,
        data.epoch,
        state,
        cd_a_over_m_m2_kg=values.get(_DRAG_PARAMETER_NAME, 0.0),
    )


def _sgp4_product(
    data: PriorStateData,
    values: dict[str, float],
    object_id: str,
    solution_id: str,
    lines: tuple[str, str],
) -> Sgp4Orbit:
    offsets = tuple(values.get(name, 0.0) for name in _SGP4_PARAMETER_NAMES)
    if "tle_epoch_offset_s" in values:
        lines = corrected_tle_lines(lines, offsets, values["tle_epoch_offset_s"])
        offsets = (0.0,) * len(_SGP4_PARAMETER_NAMES)
    return Sgp4Orbit(object_id, data.ephemeris, solution_id, lines, offsets)


def resolve_prior(
    data: PriorStateData,
    model: OrbitModel,
    *,
    cd_a_over_m_m2_kg: float = 0.0,
) -> OrbitSolution:
    """Materialize the selected uncorrected prior for baseline evaluation."""
    if model == OrbitModel.SGP4:
        if cd_a_over_m_m2_kg != 0:
            raise ValueError("Cd A/m is only supported for Cartesian propagation")
        data = prepare_sgp4_prior(data)
    values = {_DRAG_PARAMETER_NAME: cd_a_over_m_m2_kg} if cd_a_over_m_m2_kg else {}
    return _materialize_orbit(data, model, values)


def resolve_solution(data: PriorStateData, output: OptimizerOutput) -> OrbitSolution:
    """Combine a successful fit's named corrections with its original prior.

    Callers retain source identity and observations. SGP4 outputs carry the exact
    prepared baseline, so replay never refits or applies offsets at another epoch.
    """
    if not output.success:
        raise ValueError("cannot publish an orbit from an unsuccessful fit")
    if len(set(output.parameter_names)) != len(output.parameter_names):
        raise ValueError("fit parameter names must be unique")
    if not np.all(np.isfinite(output.parameters)):
        raise ValueError("fit parameters must be finite")
    values = dict(zip(output.parameter_names, output.parameters.tolist(), strict=True))
    unknown = values.keys() - set(
        _canonical_parameter_names(data, output.model_kind, include_drag=True)
    )
    if unknown:
        raise ValueError(f"unknown fit parameters: {sorted(unknown)}")
    if output.prepared_tle is not None:
        source = _canonical_tle(replace(data, prepared_tle=None))
        if source != output.prepared_tle.original_tle_lines:
            raise ValueError(
                "fit prepared TLE does not match the supplied source prior"
            )
        data = replace(data, prepared_tle=output.prepared_tle)
    return _materialize_orbit(data, output.model_kind, values)


def _full_state(
    data: PriorStateData,
    tle_lines: tuple[str, str],
) -> tuple[FloatArray, PriorSource]:
    if data.nominal_state_gcrf_si is None:
        return (
            tle_state_gcrf(tle_lines, data.epoch),
            PriorSource.TLE_DERIVED_FULL_STATE,
        )
    state = np.asarray(data.nominal_state_gcrf_si, dtype=np.float64)
    if state.shape != (6,) or not np.all(np.isfinite(state)):
        raise ValueError("nominal GCRF state must contain six finite values")
    return np.ascontiguousarray(state), PriorSource.FULL_STATE


def _full_state_evaluator(
    data: PriorStateData,
    include_drag: bool = False,
    fixed_drag: bool = False,
) -> tuple[Evaluator, PriorSource]:
    tle_lines = _canonical_tle(data)
    state, source = _full_state(data, tle_lines)

    def evaluate(x: FloatArray) -> ForwardModelEvaluation:
        return evaluate_full_state_augmented(
            x[:-1] if fixed_drag else x,
            state,
            data.epoch,
            data.observations,
            include_drag=include_drag and not fixed_drag,
            cd_a_over_m_m2_kg=float(x[-1]) if fixed_drag else 0.0,
        )

    return evaluate, source


def _sgp4_evaluator(data: PriorStateData) -> tuple[Evaluator, PriorSource]:
    tle_lines = _canonical_tle(data)

    def evaluate(x: FloatArray) -> ForwardModelEvaluation:
        return evaluate_sgp4_epoch(x, tle_lines, data.observations)

    return evaluate, PriorSource.TLE


def _cached_functions(evaluator: Evaluator) -> tuple[Callable, Callable]:
    cached_x: FloatArray | None = None
    cached_evaluation: ForwardModelEvaluation | None = None

    def evaluate(x: FloatArray) -> ForwardModelEvaluation:
        nonlocal cached_x, cached_evaluation
        values = np.asarray(x, dtype=np.float64)
        if cached_x is None or not np.array_equal(values, cached_x):
            cached_x = values.copy()
            cached_evaluation = evaluator(values)
        assert cached_evaluation is not None
        return cached_evaluation

    return (
        lambda x: evaluate(x).residuals.copy(),
        lambda x: evaluate(x).jacobian.copy(),
    )


def _fixed_cost(residuals: FloatArray, loss: str, scale: float) -> float:
    z = np.square(residuals / scale)
    if loss == "linear":
        rho = z
    elif loss == "soft_l1":
        rho = 2.0 * (np.sqrt(1.0 + z) - 1.0)
    elif loss == "huber":
        rho = np.where(z <= 1.0, z, 2.0 * np.sqrt(z) - 1.0)
    elif loss == "cauchy":
        rho = np.log1p(z)
    else:
        rho = np.arctan(z)
    return float(0.5 * scale**2 * np.sum(rho))


def _fit_estimated(
    evaluator: Evaluator,
    optimizer: OptimizerContext,
    names: tuple[str, ...],
) -> tuple[FloatArray, OptimizeResult]:
    parameters = tuple(
        p for p in optimizer.parameters if p.role == ParameterRole.ESTIMATE
    )
    indices = np.array([names.index(p.name) for p in parameters], dtype=np.intp)
    initial = np.zeros(len(names), dtype=np.float64)
    for parameter in optimizer.parameters:
        initial[names.index(parameter.name)] = parameter.initial

    def expand(estimated: FloatArray) -> FloatArray:
        values = initial.copy()
        values[indices] = estimated
        return values

    def projected(estimated: FloatArray) -> ForwardModelEvaluation:
        evaluation = evaluator(expand(estimated))
        return ForwardModelEvaluation(
            evaluation.residuals, np.ascontiguousarray(evaluation.jacobian[:, indices])
        )

    result = _optimize(projected, parameters, optimizer)
    return expand(result.x), result


def _optimize(
    evaluator: Evaluator,
    parameters: tuple[ParameterSpec, ...],
    optimizer: OptimizerContext,
) -> OptimizeResult:
    initial = np.array([p.initial for p in parameters], dtype=np.float64)
    if not parameters:
        return OptimizeResult(
            x=initial,
            cost=_fixed_cost(
                evaluator(initial).residuals, optimizer.loss, optimizer.loss_scale
            ),
            optimality=0.0,
            success=True,
            status=1,
            message="No parameters were configured for estimation.",
            nfev=1,
            njev=0,
        )
    residuals, jacobian = _cached_functions(evaluator)
    return least_squares(
        residuals,
        initial,
        jac=jacobian,
        bounds=(
            [p.lower_bound for p in parameters],
            [p.upper_bound for p in parameters],
        ),
        x_scale="jac" if optimizer.x_scale == "jac" else [p.scale for p in parameters],
        loss=optimizer.loss,
        f_scale=optimizer.loss_scale,
        max_nfev=optimizer.max_evaluations,
        ftol=optimizer.ftol,
        xtol=optimizer.xtol,
        gtol=optimizer.gtol,
        method="trf",
        tr_solver="exact",
    )


def fit(data: PriorStateData, optimizer: OptimizerContext) -> OptimizerOutput:
    """Fit the selected orbit model to normalized Doppler observations."""

    _validate_data(data)
    _validate_optimizer(data, optimizer)
    include_drag = any(p.name == _DRAG_PARAMETER_NAME for p in optimizer.parameters)
    if optimizer.model == OrbitModel.SGP4:
        data = prepare_sgp4_prior(data, optimizer)
        evaluator, prior_source = _sgp4_evaluator(data)
    else:
        evaluator, prior_source = _full_state_evaluator(data, include_drag=include_drag)

    parameters = optimizer.parameters
    canonical_names = _canonical_parameter_names(data, optimizer.model, include_drag)
    configured_indices = np.array(
        [canonical_names.index(p.name) for p in parameters], dtype=np.intp
    )
    optimization_evaluator = evaluator
    if any(
        p.name == _DRAG_PARAMETER_NAME and p.role != ParameterRole.ESTIMATE
        for p in parameters
    ):
        # The optimizer projects onto estimated columns, so it does not need the
        # terminal fixed/considered drag derivative. Recompute it once below.
        optimization_evaluator, _ = _full_state_evaluator(
            data, include_drag=True, fixed_drag=True
        )
    canonical_final, result = _fit_estimated(
        optimization_evaluator, optimizer, canonical_names
    )
    final = evaluator(canonical_final)
    output = OptimizerOutput(
        model_kind=optimizer.model,
        prior_source=prior_source,
        parameter_names=tuple(parameter.name for parameter in parameters),
        parameters=np.ascontiguousarray(canonical_final[configured_indices]),
        residuals=final.residuals,
        jacobian=np.ascontiguousarray(final.jacobian[:, configured_indices]),
        cost=float(result.cost),
        optimality=float(result.optimality),
        success=bool(result.success),
        status=int(result.status),
        message=str(result.message),
        function_evaluations=int(result.nfev),
        parameter_roles=tuple(parameter.role for parameter in parameters),
        loss=optimizer.loss,
        jacobian_evaluations=result.njev,
        prepared_tle=data.prepared_tle if optimizer.model == OrbitModel.SGP4 else None,
    )
    return _fit_covariance(output, parameters)


def _fit_covariance(
    output: OptimizerOutput, parameters: tuple[ParameterSpec, ...]
) -> OptimizerOutput:
    covariance_enabled = bool(parameters) and all(
        parameter.role != ParameterRole.FIXED
        and parameter.prior_standard_uncertainty is not None
        for parameter in parameters
    )
    if output.success and covariance_enabled:
        covariance, rank = full_consider_covariance(output, parameters)
        output = replace(
            output,
            covariance=covariance,
            covariance_rank=rank,
            covariance_method="classical_consider_v1",
        )
    return output


__all__ = [
    "OrbitModel",
    "OptimizerContext",
    "OptimizerOutput",
    "ParameterRole",
    "ParameterSpec",
    "PriorSource",
    "PriorStateData",
    "compute_consider_covariance",
    "fit",
    "prepare_sgp4_prior",
    "resolve_prior",
    "resolve_solution",
]
