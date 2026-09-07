"""Batch orbit determination using DART's authoritative forward models."""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
import satkit as sk
from numpy.typing import NDArray
from scipy.optimize import least_squares

from dart.forward_models import (
    ForwardModelEvaluation,
    evaluate_full_state_augmented,
    evaluate_sgp4_augmented,
    tle_state_gcrf,
)

from .cca import compute_consider_covariance
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
_LOSSES = {"linear", "soft_l1", "huber", "cauchy", "arctan"}
FloatArray = NDArray[np.float64]
Evaluator = Callable[[FloatArray], ForwardModelEvaluation]


def _canonical_tle(data: PriorStateData) -> tuple[str, str]:
    raw = data.ephemeris.tle
    if raw is None or not raw.strip():
        raise ValueError("source ephemeris does not contain a TLE")
    try:
        parsed = sk.TLE.from_lines([line for line in raw.splitlines() if line.strip()])
    except Exception as exc:
        raise ValueError("source ephemeris contains an invalid TLE") from exc
    if isinstance(parsed, list):
        if len(parsed) != 1:
            raise ValueError("source ephemeris must contain exactly one TLE")
        parsed = parsed[0]
    lines = parsed.to_2line()
    return str(lines[0]), str(lines[1])


def _pass_parameter_names(data: PriorStateData) -> tuple[str, ...]:
    mapping = data.observations.contact_to_pass_idx
    if set(mapping.values()) != set(range(len(mapping))):
        raise ValueError("contact pass indices must be unique and contiguous from zero")
    contacts = sorted(mapping, key=mapping.__getitem__)
    return tuple(f"pass_bias_hz:{contact_id}" for contact_id in contacts)


def _canonical_parameter_names(
    data: PriorStateData,
    model: OrbitModel,
) -> tuple[str, ...]:
    pass_names = _pass_parameter_names(data)
    if model == OrbitModel.SGP4:
        return _SGP4_PARAMETER_NAMES + _SHARED_PARAMETER_NAMES + pass_names
    return _FULL_STATE_PARAMETER_NAMES + _SHARED_PARAMETER_NAMES + pass_names


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


def _validate_optimizer(
    data: PriorStateData,
    optimizer: OptimizerContext,
) -> None:
    if not isinstance(optimizer.model, OrbitModel):
        raise TypeError("optimizer model must be an OrbitModel")
    names = tuple(parameter.name for parameter in optimizer.parameters)
    canonical = _canonical_parameter_names(data, optimizer.model)
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
        if contact.ephemeris_id != data.ephemeris.ephemeris_id:
            raise ValueError("observation and source ephemeris IDs differ")


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


def _full_state_evaluator(data: PriorStateData) -> tuple[Evaluator, PriorSource]:
    tle_lines = _canonical_tle(data)
    state, source = _full_state(data, tle_lines)

    def evaluate(x: FloatArray) -> ForwardModelEvaluation:
        return evaluate_full_state_augmented(x, state, data.epoch, data.observations)

    return evaluate, source


def _sgp4_evaluator(data: PriorStateData) -> tuple[Evaluator, PriorSource]:
    tle_lines = _canonical_tle(data)

    def evaluate(x: FloatArray) -> ForwardModelEvaluation:
        return evaluate_sgp4_augmented(x, tle_lines, data.observations)

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
        lambda x: evaluate(x).residuals,
        lambda x: evaluate(x).jacobian,
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


def fit(data: PriorStateData, optimizer: OptimizerContext) -> OptimizerOutput:
    """Fit the selected orbit model to normalized Doppler observations."""

    _validate_data(data)
    _validate_optimizer(data, optimizer)
    if optimizer.model == OrbitModel.SGP4:
        evaluator, prior_source = _sgp4_evaluator(data)
    else:
        evaluator, prior_source = _full_state_evaluator(data)

    parameters = optimizer.parameters
    canonical_names = _canonical_parameter_names(data, optimizer.model)
    canonical_indices = {name: index for index, name in enumerate(canonical_names)}
    configured_indices = np.array(
        [canonical_indices[parameter.name] for parameter in parameters], dtype=np.intp
    )
    estimated_parameters = tuple(
        parameter
        for parameter in parameters
        if parameter.role == ParameterRole.ESTIMATE
    )
    estimated_indices = np.array(
        [canonical_indices[parameter.name] for parameter in estimated_parameters],
        dtype=np.intp,
    )
    canonical_initial = np.zeros(len(canonical_names), dtype=np.float64)
    for parameter, index in zip(parameters, configured_indices, strict=True):
        canonical_initial[index] = parameter.initial

    def expand(estimated: FloatArray) -> FloatArray:
        values = canonical_initial.copy()
        values[estimated_indices] = estimated
        return values

    def projected_evaluator(estimated: FloatArray) -> ForwardModelEvaluation:
        evaluation = evaluator(expand(estimated))
        return ForwardModelEvaluation(
            residuals=evaluation.residuals,
            jacobian=np.ascontiguousarray(evaluation.jacobian[:, estimated_indices]),
        )

    estimated_initial = np.array(
        [parameter.initial for parameter in estimated_parameters], dtype=np.float64
    )
    if estimated_parameters:
        residuals, jacobian = _cached_functions(projected_evaluator)
        result = least_squares(
            residuals,
            estimated_initial,
            jac=jacobian,
            bounds=(
                [parameter.lower_bound for parameter in estimated_parameters],
                [parameter.upper_bound for parameter in estimated_parameters],
            ),
            x_scale=[parameter.scale for parameter in estimated_parameters],
            loss=optimizer.loss,
            f_scale=optimizer.loss_scale,
            max_nfev=optimizer.max_evaluations,
            ftol=optimizer.ftol,
            xtol=optimizer.xtol,
            gtol=optimizer.gtol,
            method="trf",
        )
        canonical_final = expand(np.asarray(result.x, dtype=np.float64))
        cost = float(result.cost)
        optimality = float(result.optimality)
        success = bool(result.success)
        status = int(result.status)
        message = str(result.message)
        function_evaluations = int(result.nfev)
        jacobian_evaluations = None if result.njev is None else int(result.njev)
    else:
        canonical_final = canonical_initial
        fixed = evaluator(canonical_final)
        cost = _fixed_cost(fixed.residuals, optimizer.loss, optimizer.loss_scale)
        optimality = 0.0
        success = True
        status = 1
        message = "No parameters were configured for estimation."
        function_evaluations = 1
        jacobian_evaluations = 0
    final = evaluator(canonical_final)
    return OptimizerOutput(
        model_kind=optimizer.model,
        prior_source=prior_source,
        parameter_names=tuple(parameter.name for parameter in parameters),
        parameters=np.ascontiguousarray(canonical_final[configured_indices]),
        residuals=final.residuals,
        jacobian=np.ascontiguousarray(final.jacobian[:, configured_indices]),
        cost=cost,
        optimality=optimality,
        success=success,
        status=status,
        message=message,
        function_evaluations=function_evaluations,
        parameter_roles=tuple(parameter.role for parameter in parameters),
        loss=optimizer.loss,
        jacobian_evaluations=jacobian_evaluations,
    )


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
]
