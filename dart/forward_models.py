"""Python boundary for DART's authoritative Rust forward models.

Evaluators feed SciPy or other estimation loops without duplicating the
numerical model in Python. TLE re-epoching seeds with stock Rust satkit, refines
with SciPy against Rust state sensitivities, and validates serialization in Rust.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, fields, replace
from functools import lru_cache
from importlib import import_module
from typing import TypeAlias

import numpy as np
import satkit as sk
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import OptimizeResult, least_squares

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
    stock_converged: bool = False
    refinement_status: int = 0
    refinement_message: str = ""
    refinement_evaluations: int = 0


class ReepochError(ValueError):
    """Rejected re-epoching; diagnostics exist when a candidate was validated."""

    def __init__(self, message: str, diagnostics: ReepochedTle | None = None):
        super().__init__(message)
        self.diagnostics = diagnostics


def _reepoch_result(native: object) -> ReepochedTle:
    values = {
        field.name: getattr(native, field.name)
        for field in fields(ReepochedTle)
        if hasattr(native, field.name)
    }
    for name in ("original_tle_lines", "tle_lines"):
        values[name] = tuple(values[name])
    return ReepochedTle(**values)


def reepoch_tle(
    tle_lines: tuple[str, str],
    epoch: sk.time,
    window_start: sk.time,
    window_stop: sk.time,
) -> ReepochedTle:
    """Seed with stock satkit, refine through Rust SGP4, validate serialization.

    Checks 241 epochs after serialization: position RMS/max <10/20 m and
    velocity RMS/max <0.01/0.02 m/s. No observations or OEM enter this fit.
    """
    if len(tle_lines) != 2:
        raise ReepochError("tle_lines must contain line 1 and line 2")
    seed = _satkit_seed(tle_lines, epoch, window_start, window_stop)
    seed_report = _reepoch_result(seed)
    if seed_report.fit_status == "AlreadyCentered":
        return replace(
            seed_report, refinement_message="already centered; no fit required"
        )
    try:
        refinement = _refine_reepoch(seed)
    except ValueError as exc:
        diagnostics = replace(
            seed_report,
            converged=False,
            stock_converged=seed_report.converged,
            refinement_message=str(exc),
        )
        raise ReepochError(f"TLE refinement failed: {exc}", diagnostics) from exc
    details = dict(
        stock_converged=seed_report.converged,
        refinement_status=int(refinement.status),
        refinement_message=str(refinement.message),
        refinement_evaluations=int(refinement.nfev),
    )
    try:
        native = _native.validate_reepoch_refinement(
            seed, refinement.x.tolist(), bool(refinement.success)
        )
    except ValueError as exc:
        diagnostics = _reepoch_result(exc.args[1]) if len(exc.args) == 2 else None
        if diagnostics is not None:
            diagnostics = replace(diagnostics, **details)
        raise ReepochError(str(exc.args[0]), diagnostics) from exc
    return replace(_reepoch_result(native), **details)


def prepare_sgp4_tle(
    tle_lines: tuple[str, str],
    timestamps: Sequence[sk.time],
    *,
    window: tuple[sk.time, sk.time] | None = None,
) -> ReepochedTle:
    """Prepare a fixed SGP4 baseline at the arithmetic mean observation epoch.

    Duplicate timestamps contribute separately. The preservation interval includes
    all timestamps, at least one orbital period about their mean, and an optional
    wider scoring/search window. Identical preparations reuse an immutable report.
    Evaluators continue to apply parameter offsets to the supplied, fixed baseline.
    """
    epochs = [_unix_seconds(t) for t in timestamps]
    interval = None if window is None else tuple(_unix_seconds(t) for t in window)
    try:
        epoch, start, stop = _native.sgp4_preparation_epochs(
            tle_lines, epochs, interval
        )
    except ValueError as exc:
        raise ReepochError(str(exc)) from exc
    # Match cache-key precision to satkit's microsecond Instants.
    epoch, start, stop = (round(value, 6) for value in (epoch, start, stop))
    return _prepared_sgp4_tle(tle_lines, epoch, start, stop)


@lru_cache(maxsize=128)
def _prepared_sgp4_tle(
    lines: tuple[str, str], epoch: float, start: float, stop: float
) -> ReepochedTle:
    report = reepoch_tle(
        lines,
        sk.time.from_unixtime(epoch),
        sk.time.from_unixtime(start),
        sk.time.from_unixtime(stop),
    )
    if abs(report.serialized_epoch_unix_s - epoch) > 0.0005:
        raise ReepochError("serialized TLE epoch differs from timestamp mean", report)
    return report


def _satkit_seed(
    lines: tuple[str, str],
    epoch: sk.time,
    start: sk.time,
    stop: sk.time,
) -> object:
    try:
        return _native.reepoch_tle(
            lines, _unix_seconds(epoch), _unix_seconds(start), _unix_seconds(stop)
        )
    except ValueError as exc:
        if len(exc.args) == 2:
            report = _reepoch_result(exc.args[1])
            if report.fit_status == "AlreadyCentered":
                raise ReepochError(str(exc.args[0]), report) from exc
            # A rejected stock candidate is a seed, never an accepted product.
            return exc.args[1]
        raise ReepochError(str(exc)) from exc


def _refine_reepoch(seed: object) -> OptimizeResult:
    report = _reepoch_result(seed)
    times = np.linspace(report.window_start_unix_s, report.window_stop_unix_s, 241)[::2]

    def evaluate(x: FloatArray) -> ForwardModelEvaluation:
        return _evaluation(
            _native.tle_position_evaluation(
                x.tolist(), report.original_tle_lines, report.tle_lines, times.tolist()
            )
        )

    bounds = np.array([0.2, 0.1, 0.1, 0.1, 0.1, 30.0, 1.0])
    return least_squares(
        lambda x: evaluate(x).residuals,
        np.zeros(7),
        jac=lambda x: evaluate(x).jacobian,
        bounds=(-bounds, bounds),
        x_scale=[0.001, 0.001, 0.001, 0.001, 0.001, 0.1, 0.001],
        loss="linear",
        max_nfev=200,
        ftol=1e-10,
        xtol=1e-10,
        gtol=1e-10,
    )


def corrected_tle_lines(
    tle_lines: tuple[str, str],
    orbit_offsets: ArrayLike,
    epoch_offset_s: float = 0,
) -> tuple[str, str]:
    """Serialize orbit corrections and E' = E + epoch_offset_s through Rust."""
    lines = _native.corrected_tle_lines(
        tle_lines, _parameter_vector(orbit_offsets), epoch_offset_s
    )
    return str(lines[0]), str(lines[1])


def evaluate_sgp4_epoch(
    x: ArrayLike,
    tle_lines: tuple[str, str],
    context: ForwardModelContext,
) -> ForwardModelEvaluation:
    """Legacy augmented vector followed by tle_epoch_offset_s; epochs stay UTC.

    Epoch adjustment changes only the TLE epoch. The legacy time_offset_s
    parameter retains its separate measurement-clock meaning.
    """
    return _evaluation(
        _native.evaluate_sgp4_epoch(
            _parameter_vector(x), *tle_lines, _native_inputs(context)
        )
    )


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
    "prepare_sgp4_tle",
    "corrected_tle_lines",
    "evaluate_sgp4_epoch",
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
