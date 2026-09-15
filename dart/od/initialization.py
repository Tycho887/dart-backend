"""Doppler-only initialization through the authoritative Rust evaluator."""

from collections.abc import Sequence
from dataclasses import replace

import numpy as np
from scipy.optimize import OptimizeResult, least_squares, lsq_linear

from . import (
    _canonical_parameter_names,
    _fixed_cost,
    _sgp4_evaluator,
    _validate_data,
    _validate_optimizer,
    prepare_sgp4_prior,
)
from .schema import (
    FloatArray,
    OptimizerContext,
    OrbitModel,
    ParameterRole,
    ParameterSpec,
    PriorStateData,
)


def _timing_parameters(data: PriorStateData, optimizer: OptimizerContext) -> list[str]:
    _validate_data(data)
    _validate_optimizer(data, optimizer)
    if optimizer.model != OrbitModel.SGP4 or optimizer.loss not in {
        "linear",
        "soft_l1",
    }:
        raise ValueError(
            "timing initialization requires SGP4 and linear or soft_l1 loss"
        )
    mapping = data.observations.contact_to_pass_idx
    names = [
        timing_parameter_name(optimizer),
        *[f"pass_bias_hz:{cid}" for cid in sorted(mapping, key=mapping.__getitem__)],
    ]
    estimated = {
        p.name for p in optimizer.parameters if p.role == ParameterRole.ESTIMATE
    }
    if estimated != set(names):
        raise ValueError(
            "timing initialization must estimate only time offset and all pass biases"
        )
    return names


def timing_parameter_name(optimizer: OptimizerContext) -> str:
    names = {p.name for p in optimizer.parameters} & {
        "time_offset_s",
        "tle_epoch_offset_s",
    }
    if len(names) != 1:
        raise ValueError(
            "timing requires exactly one clock-offset or TLE-epoch parameter"
        )
    return names.pop()


def _time_candidates(lower: float, upper: float, step_s: float) -> FloatArray:
    if not np.isfinite(step_s) or step_s <= 0:
        raise ValueError("timing scan step must be finite and positive")
    if not lower <= 0 <= upper:
        raise ValueError("timing bounds must contain zero")
    return np.unique(np.r_[np.arange(lower, upper, step_s), upper, 0.0])


def _seed_optimizer(
    optimizer: OptimizerContext, names: list[str], values: FloatArray
) -> OptimizerContext:
    initial = dict(zip(names, values, strict=True))
    return replace(
        optimizer,
        parameters=tuple(
            replace(p, initial=initial.get(p.name, p.initial))
            for p in optimizer.parameters
        ),
    )


def initialize_sgp4_time(
    prior: PriorStateData,
    optimizer: OptimizerContext,
    step_s: float = 10,
) -> tuple[OptimizerContext, FloatArray]:
    """Scan timing bounds with pass biases fitted under the configured loss.

    Columns are [offset_s, bias/contact in pass-index order, cost]. All other
    initial values and settings remain fixed. Equal costs select the first
    candidate in ascending offset order. OEM data is never accepted here.
    """
    scan_names = _timing_parameters(prior, optimizer)
    prior = prepare_sgp4_prior(prior, optimizer)
    specs = {p.name: p for p in optimizer.parameters}
    timing = specs[scan_names[0]]
    offsets = _time_candidates(timing.lower_bound, timing.upper_bound, step_s)
    names = _canonical_parameter_names(prior, optimizer.model)
    indices = [names.index(name) for name in scan_names]
    values = np.array([specs[n].initial if n in specs else 0.0 for n in names])
    bias_indices = indices[1:]
    values[bias_indices] = 0
    bounds = (
        [specs[n].lower_bound for n in scan_names[1:]],
        [specs[n].upper_bound for n in scan_names[1:]],
    )
    evaluate, _ = _sgp4_evaluator(prior)
    scan = np.empty((len(offsets), len(scan_names) + 1))
    for row, offset in enumerate(offsets):
        values[indices[0]] = offset
        evaluation = evaluate(values)
        bias_jacobian = evaluation.jacobian[:, bias_indices]
        result = _scan_bias_fit(evaluation.residuals, bias_jacobian, bounds, optimizer)
        if not result.success:
            raise ValueError(
                f"timing scan bias fit failed at {offset} s: {result.message}"
            )
        residuals = evaluation.residuals + bias_jacobian @ result.x
        scan[row] = (
            offset,
            *result.x,
            _fixed_cost(residuals, optimizer.loss, optimizer.loss_scale),
        )
    if not np.all(np.isfinite(scan)):
        raise ValueError("timing scan produced nonfinite values")
    return _seed_optimizer(
        optimizer, scan_names, scan[np.argmin(scan[:, -1]), :-1]
    ), scan


def _scan_bias_fit(
    residuals: FloatArray,
    jacobian: FloatArray,
    bounds: tuple[list[float], list[float]],
    optimizer: OptimizerContext,
) -> OptimizeResult:
    """The affine bias problem needs no additional orbit evaluations."""
    result = lsq_linear(jacobian, -residuals, bounds=bounds)
    if optimizer.loss == "linear" or not result.success:
        return result
    return least_squares(
        lambda bias: residuals + jacobian @ bias,
        result.x,
        jac=lambda bias: jacobian.copy(),
        bounds=bounds,
        x_scale="jac",
        loss=optimizer.loss,
        f_scale=optimizer.loss_scale,
        max_nfev=optimizer.max_evaluations,
        ftol=optimizer.ftol,
        xtol=optimizer.xtol,
        gtol=optimizer.gtol,
    )


def initialize_sgp4_phase(
    data: PriorStateData, optimizer: OptimizerContext
) -> tuple[OptimizerContext, FloatArray]:
    """Scan -30..30 degrees at 1-degree steps, with a bounded median bias.

    Returns columns [delta L (deg), bias/contact (Hz), cost], in pass-index order.
    Equal costs choose the first candidate in ascending phase order. All other
    initial values, roles, scales and bounds are preserved. No GPS is accepted.
    """
    bias_names = _phase_parameters(data, optimizer)
    data = prepare_sgp4_prior(data, optimizer)
    specs = {p.name: p for p in optimizer.parameters}
    phase_name = "mean_longitude_deg"
    names = _canonical_parameter_names(data, optimizer.model)
    values = np.array([specs[n].initial if n in specs else 0.0 for n in names])
    phase_index = names.index(phase_name)
    bias_indices = [names.index(name) for name in bias_names]
    values[bias_indices] = 0.0
    pass_indices = np.array([o.pass_index for o in data.observations.observations])
    evaluate, _ = _sgp4_evaluator(data)
    scan = np.empty((61, len(bias_names) + 2))
    for row, delta in enumerate(range(-30, 31)):
        values[phase_index] = delta
        evaluation = evaluate(values)
        biases = _median_biases(
            evaluation.residuals,
            evaluation.jacobian[:, bias_indices],
            pass_indices,
            [specs[name] for name in bias_names],
        )
        residuals = evaluation.residuals + evaluation.jacobian[:, bias_indices] @ biases
        cost = _fixed_cost(residuals, optimizer.loss, optimizer.loss_scale)
        scan[row] = delta, *biases, cost
    if not np.all(np.isfinite(scan)):
        raise ValueError("phase scan produced nonfinite values")
    best = scan[np.argmin(scan[:, -1])]
    return _seed_optimizer(optimizer, [phase_name, *bias_names], best[:-1]), scan


def _phase_parameters(data: PriorStateData, optimizer: OptimizerContext) -> list[str]:
    _validate_data(data)
    _validate_optimizer(data, optimizer)
    if optimizer.model != OrbitModel.SGP4:
        raise ValueError("phase initialization requires SGP4")
    mapping = data.observations.contact_to_pass_idx
    contacts = sorted(mapping, key=mapping.__getitem__)
    specs = {p.name: p for p in optimizer.parameters}
    phase_name = "mean_longitude_deg"
    bias_names = [f"pass_bias_hz:{cid}" for cid in contacts]
    for name in (phase_name, *bias_names):
        if name not in specs or specs[name].role != ParameterRole.ESTIMATE:
            raise ValueError("phase and pass bias must both be estimated")
    phase = specs[phase_name]
    if phase.lower_bound > -30 or phase.upper_bound < 30:
        raise ValueError("phase bounds must include the complete -30..30 degree scan")
    return bias_names


def _median_biases(
    residuals: FloatArray,
    jacobian: FloatArray,
    pass_indices: np.ndarray,
    specs: Sequence[ParameterSpec],
) -> FloatArray:
    biases = []
    for index, spec in enumerate(specs):
        mask = pass_indices == index
        sensitivity = jacobian[mask, index]
        if not mask.any() or np.any(sensitivity == 0):
            raise ValueError(
                "each pass needs observations with nonzero bias sensitivity"
            )
        biases.append(
            float(
                np.clip(
                    -np.median(residuals[mask] / sensitivity),
                    spec.lower_bound,
                    spec.upper_bound,
                )
            )
        )
    return np.asarray(biases)
