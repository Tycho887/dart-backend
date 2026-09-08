"""Doppler-only initialization through the authoritative Rust evaluator."""

from collections.abc import Sequence
from dataclasses import replace

import numpy as np

from . import (
    _canonical_parameter_names,
    _fixed_cost,
    _sgp4_evaluator,
    _validate_data,
    _validate_optimizer,
)
from .schema import (
    FloatArray,
    OptimizerContext,
    OrbitModel,
    ParameterRole,
    ParameterSpec,
    PriorStateData,
)


def initialize_sgp4_phase(
    data: PriorStateData, optimizer: OptimizerContext
) -> tuple[OptimizerContext, FloatArray]:
    """Scan -30..30 degrees at 1-degree steps, with a bounded median bias.

    Returns columns [delta L (deg), bias/contact (Hz), cost], in pass-index order.
    Equal costs choose the first candidate in ascending phase order. All other
    initial values, roles, scales and bounds are preserved. No GPS is accepted.
    """
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
    names = _canonical_parameter_names(data, optimizer.model)
    values = np.array([specs[n].initial if n in specs else 0.0 for n in names])
    phase_index = names.index(phase_name)
    bias_indices = [names.index(name) for name in bias_names]
    values[bias_indices] = 0.0
    pass_indices = np.array([o.pass_index for o in data.observations.observations])
    evaluate, _ = _sgp4_evaluator(data)
    scan = np.empty((61, len(contacts) + 2))
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
    initial = dict(zip([phase_name, *bias_names], best[:-1], strict=True))
    return replace(
        optimizer,
        parameters=tuple(
            replace(p, initial=initial.get(p.name, p.initial))
            for p in optimizer.parameters
        ),
    ), scan


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
