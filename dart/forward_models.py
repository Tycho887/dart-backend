"""Python boundary for DART's authoritative Rust forward models.

The functions in this module are intentionally evaluators, not optimizers.
Their outputs can be passed directly to SciPy, an MCP controller, or another
estimation loop without duplicating the numerical model in Python.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import TypeAlias

import numpy as np
import satkit as sk
from numpy.typing import ArrayLike, NDArray

from .io import ForwardModelContext

_native = import_module("dart._forward_models")

FloatArray: TypeAlias = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class ForwardModelEvaluation:
    """Whitened predicted-minus-observed residuals and their Jacobian."""

    residuals: FloatArray
    jacobian: FloatArray


@dataclass(frozen=True, slots=True)
class _NativeInputs:
    receivers: list[tuple[float, float, float]]
    epochs_unix: list[float]
    observed_hz: list[float]
    variances_hz2: list[float]
    receiver_ids: list[int]
    pass_indices: list[int]
    center_frequency_hz: float
    num_passes: int


def _native_inputs(context: ForwardModelContext) -> _NativeInputs:
    context.validate()
    return _NativeInputs(
        receivers=[
            (receiver.latitude_deg, receiver.longitude_deg, receiver.altitude)
            for receiver in context.receivers
        ],
        epochs_unix=[
            observation.time.as_unixtime() for observation in context.observations
        ],
        observed_hz=[observation.observed[0] for observation in context.observations],
        variances_hz2=[
            observation.noise_cov[0][0] for observation in context.observations
        ],
        receiver_ids=[observation.receiver_id for observation in context.observations],
        pass_indices=[observation.pass_index for observation in context.observations],
        center_frequency_hz=context.center_frequency_hz,
        num_passes=context.num_passes,
    )


def _evaluation(
    result: tuple[list[float], list[list[float]]],
) -> ForwardModelEvaluation:
    residuals, jacobian = result
    return ForwardModelEvaluation(
        residuals=np.ascontiguousarray(residuals, dtype=np.float64),
        jacobian=np.ascontiguousarray(jacobian, dtype=np.float64),
    )


def _parameter_vector(x: ArrayLike) -> list[float]:
    values = np.asarray(x, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("parameter vector must be one-dimensional")
    return values.tolist()


def _unix_seconds(value: sk.time) -> float:
    if not isinstance(value, sk.time):
        raise TypeError("epoch must be a satkit.time")
    return float(value.as_unixtime())


def evaluate_sgp4(
    x: ArrayLike,
    tle_lines: tuple[str, str],
    context: ForwardModelContext,
) -> ForwardModelEvaluation:
    """Evaluate the SGP4 Doppler objective and its analytic outer Jacobian.

    ``x`` is ordered as mean-motion offset (rev/day), equinoctial f, g, h, k,
    mean-longitude offset (degrees), B* offset, followed by one Doppler bias
    (Hz) per pass.
    """

    inputs = _native_inputs(context)
    if len(tle_lines) != 2:
        raise ValueError("tle_lines must contain line 1 and line 2")
    return _evaluation(
        _native.evaluate_sgp4(
            _parameter_vector(x),
            tle_lines[0],
            tle_lines[1],
            inputs,
        )
    )


def evaluate_sgp4_augmented(
    x: ArrayLike,
    tle_lines: tuple[str, str],
    context: ForwardModelContext,
) -> ForwardModelEvaluation:
    """Evaluate SGP4 with global time/frequency offsets and pass biases.

    ``x`` is ordered as seven SGP4 offsets, time offset (s), center-frequency
    offset (Hz), then one Doppler bias (Hz) per pass.
    """

    inputs = _native_inputs(context)
    if len(tle_lines) != 2:
        raise ValueError("tle_lines must contain line 1 and line 2")
    return _evaluation(
        _native.evaluate_sgp4_augmented(
            _parameter_vector(x),
            tle_lines[0],
            tle_lines[1],
            inputs,
        )
    )


def evaluate_full_state(
    x: ArrayLike,
    nominal_state_gcrf_si: ArrayLike,
    epoch: sk.time,
    context: ForwardModelContext,
) -> ForwardModelEvaluation:
    """Evaluate the propagated Cartesian Doppler objective and Jacobian.

    ``x`` is ordered as three GCRF position offsets (m), three GCRF velocity
    offsets (m/s), followed by one Doppler bias (Hz) per pass. The nominal
    state uses the same GCRF metres/metres-per-second convention.
    """

    inputs = _native_inputs(context)
    nominal = _parameter_vector(nominal_state_gcrf_si)
    return _evaluation(
        _native.evaluate_full_state(
            _parameter_vector(x),
            nominal,
            _unix_seconds(epoch),
            inputs,
        )
    )


def evaluate_full_state_augmented(
    x: ArrayLike,
    nominal_state_gcrf_si: ArrayLike,
    epoch: sk.time,
    context: ForwardModelContext,
) -> ForwardModelEvaluation:
    """Evaluate full-state Doppler with global time/frequency offsets.

    ``x`` is ordered as six Cartesian corrections, time offset (s),
    center-frequency offset (Hz), then one Doppler bias (Hz) per pass.
    """

    inputs = _native_inputs(context)
    nominal = _parameter_vector(nominal_state_gcrf_si)
    return _evaluation(
        _native.evaluate_full_state_augmented(
            _parameter_vector(x),
            nominal,
            _unix_seconds(epoch),
            inputs,
        )
    )


def tle_state_gcrf(
    tle_lines: tuple[str, str],
    epoch: sk.time,
) -> FloatArray:
    """Propagate a TLE into a six-component GCRF state in SI units."""

    if len(tle_lines) != 2:
        raise ValueError("tle_lines must contain line 1 and line 2")
    state = _native.tle_state_gcrf(
        tle_lines[0],
        tle_lines[1],
        _unix_seconds(epoch),
    )
    return np.ascontiguousarray(state, dtype=np.float64)


__all__ = [
    "ForwardModelEvaluation",
    "evaluate_full_state",
    "evaluate_full_state_augmented",
    "evaluate_sgp4",
    "evaluate_sgp4_augmented",
    "tle_state_gcrf",
]
