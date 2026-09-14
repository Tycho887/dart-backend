"""Python boundary for DART's authoritative Rust forward models.

Evaluators feed SciPy or other estimation loops without duplicating the
numerical model in Python. TLE re-epoching delegates fitting and preservation
checks to Rust using the published satkit fitter.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, fields
from importlib import import_module
from typing import TypeAlias

import numpy as np
import satkit as sk
from numpy.typing import ArrayLike, NDArray

from .io import ForwardModelContext

_native = import_module("dart._forward_models")

FloatArray: TypeAlias = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class ReepochedTle:
    """Serialized candidate and its preservation diagnostics in SI units."""

    original_tle_lines: tuple[str, str]
    tle_lines: tuple[str, str]
    epoch_unix_s: float
    serialized_epoch_unix_s: float
    window_start_unix_s: float
    window_stop_unix_s: float
    fit_status: str
    converged: bool
    position_rms_m: float
    position_max_m: float
    velocity_rms_m_s: float
    velocity_max_m_s: float


class ReepochError(ValueError):
    """Rejected re-epoching; diagnostics exist when a candidate was validated."""

    def __init__(self, message: str, diagnostics: ReepochedTle | None = None):
        super().__init__(message)
        self.diagnostics = diagnostics


def _reepoch_result(native: object) -> ReepochedTle:
    values = {field.name: getattr(native, field.name) for field in fields(ReepochedTle)}
    for name in ("original_tle_lines", "tle_lines"):
        values[name] = tuple(values[name])
    return ReepochedTle(**values)


def reepoch_tle(
    tle_lines: tuple[str, str],
    epoch: sk.time,
    window_start: sk.time,
    window_stop: sk.time,
) -> ReepochedTle:
    """Fit with stock Rust satkit and reject nonconvergence or preservation loss.

    Checks 241 epochs after serialization: position RMS/max <10/20 m and
    velocity RMS/max <0.01/0.02 m/s. No observations or OEM enter this fit.
    """
    if len(tle_lines) != 2:
        raise ReepochError("tle_lines must contain line 1 and line 2")
    try:
        native = _native.reepoch_tle(
            tle_lines,
            _unix_seconds(epoch),
            _unix_seconds(window_start),
            _unix_seconds(window_stop),
        )
    except ValueError as exc:
        diagnostics = _reepoch_result(exc.args[1]) if len(exc.args) == 2 else None
        raise ReepochError(str(exc.args[0]), diagnostics) from exc
    return _reepoch_result(native)


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


def sgp4_states_gcrf(
    orbit_offsets: ArrayLike,
    tle_lines: tuple[str, str],
    epochs: Sequence[sk.time],
) -> FloatArray:
    """Return (N, 6) GCRF states in m and m/s in requested epoch order.

    The seven offsets use the same mean-equinoctial/B* convention as
    :func:`evaluate_sgp4`. Repeated epochs are preserved; empty inputs fail.
    """
    if len(tle_lines) != 2:
        raise ValueError("tle_lines must contain line 1 and line 2")
    return np.ascontiguousarray(
        _native.sgp4_states_gcrf(
            _parameter_vector(orbit_offsets),
            *tle_lines,
            [_unix_seconds(epoch) for epoch in epochs],
        ),
        dtype=np.float64,
    )


def full_state_states_gcrf(
    state_gcrf_si: ArrayLike,
    epoch: sk.time,
    epochs: Sequence[sk.time],
) -> FloatArray:
    """Return (N, 6) hifi GCRF states in m and m/s, preserving epoch order.

    Uses the same Rust satkit propagation settings as evaluate_full_state.
    Epochs must be nonempty and must not precede the initial-state epoch.
    """
    return np.ascontiguousarray(
        _native.full_state_states_gcrf(
            _parameter_vector(state_gcrf_si),
            _unix_seconds(epoch),
            [_unix_seconds(value) for value in epochs],
        ),
        dtype=np.float64,
    )


def transform_states(
    states: ArrayLike,
    epochs: Sequence[sk.time],
    from_frame: str,
    to_frame: str = "GCRF",
) -> FloatArray:
    """Transform (N, 6) SI states with Rust satkit, including velocity terms."""
    values = np.asarray(states, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 6:
        raise ValueError("states must have shape (N, 6)")
    return np.asarray(
        _native.transform_states(
            values.tolist(), [_unix_seconds(t) for t in epochs], from_frame, to_frame
        ),
        dtype=np.float64,
    )


__all__ = [
    "ReepochedTle",
    "ReepochError",
    "reepoch_tle",
    "clear_frame_cache",
    "transform_states",
    "ForwardModelEvaluation",
    "evaluate_full_state",
    "evaluate_full_state_augmented",
    "evaluate_sgp4",
    "evaluate_sgp4_augmented",
    "tle_state_gcrf",
    "sgp4_states_gcrf",
    "full_state_states_gcrf",
]


def clear_frame_cache() -> None:
    """Invalidate Rust rotations after changing satkit Earth-orientation data.

    Keep runtime reference data fixed during fitting or trajectory evaluation.
    Normal callers do not need to clear the bounded process-wide cache.
    """
    _native.clear_frame_cache()
