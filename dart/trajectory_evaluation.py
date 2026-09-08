"""Post-fit trajectory alignment against an explicitly supplied reference.

Positive delta evaluates orbit(t + delta) against reference(t). This diagnostic
does not shift measurement/station epochs and must not be interpreted as a
measurement clock calibration or used for Doppler-only selection.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import satkit as sk
from scipy.optimize import minimize_scalar

from dart.forward_models import _native
from dart.orbit import OrbitSolution, StateHistory, propagate


class MissingReferenceCoverage(ValueError):
    """A complete requested scoring interval has no reference coverage."""


@dataclass(frozen=True)
class OffsetScore:
    offset_s: float
    sample_count: int
    position_rms_m: float
    velocity_rms_m_s: float
    rtn_rms_m: tuple[float, float, float]


def window_samples(
    segments: Sequence[StateHistory],
    start: sk.time,
    stop: sk.time,
) -> StateHistory:
    """Require full segment coverage and select actual samples without interpolation."""
    if stop <= start:
        raise ValueError("scoring window must have positive duration")
    ordered = sorted(segments, key=lambda s: s.epochs[0])
    cursor = start
    for segment in ordered:
        if segment.epochs[-1] < cursor:
            continue
        if segment.epochs[0] > cursor:
            break
        cursor = max(cursor, segment.epochs[-1])
    if cursor < stop:
        raise MissingReferenceCoverage(
            "reference does not cover the full requested window"
        )
    samples = [
        (t, state)
        for s in ordered
        for t, state in zip(s.epochs, s.states, strict=True)
        if start <= t <= stop
    ]
    if not samples:
        raise MissingReferenceCoverage("no actual reference samples in window")
    epochs, states = zip(*samples, strict=True)
    if len({s.object_id for s in ordered}) != 1 or len(
        {t.as_unixtime() for t in epochs}
    ) != len(epochs):
        raise ValueError("reference identities differ or epochs overlap")
    return StateHistory(
        ordered[0].object_id, ordered[0].source_id, tuple(epochs), np.array(states)
    )


def score_offset(
    orbit: OrbitSolution, reference: StateHistory, offset_s: float
) -> OffsetScore:
    if not np.isfinite(offset_s):
        raise ValueError("trajectory offset must be finite")
    if orbit.object_id != reference.object_id:
        raise ValueError("orbit and reference identities differ")
    epochs = tuple(t + sk.duration(seconds=float(offset_s)) for t in reference.epochs)
    predicted = propagate(orbit, epochs)
    difference = predicted.states - reference.states
    rms = np.sqrt(np.mean(difference**2, axis=0))
    rtn = np.asarray(
        _native.position_errors_rtn(
            predicted.states.tolist(), reference.states.tolist()
        )
    )
    return OffsetScore(
        float(offset_s),
        len(epochs),
        float(np.linalg.norm(rms[:3])),
        float(np.linalg.norm(rms[3:])),
        tuple(np.sqrt(np.mean(rtn**2, axis=0)).tolist()),
    )


def evaluate_trajectory(
    orbit: OrbitSolution,
    reference_segments: Sequence[StateHistory],
    start: sk.time,
    stop: sk.time,
    offsets_s: Sequence[float] = (0.0,),
) -> tuple[OffsetScore, ...]:
    reference = window_samples(reference_segments, start, stop)
    return tuple(score_offset(orbit, reference, delta) for delta in offsets_s)


def timing_sweep(
    orbit: OrbitSolution,
    reference: StateHistory,
    *,
    lower_s: float = -1.0,
    upper_s: float = 1.0,
    step_s: float = 0.01,
) -> tuple[tuple[OffsetScore, ...], OffsetScore]:
    """Grid plus bounded interior refinement; keep endpoint optima explicitly."""
    if (
        not np.all(np.isfinite([lower_s, upper_s, step_s]))
        or lower_s >= upper_s
        or step_s <= 0
    ):
        raise ValueError("invalid sweep bounds or spacing")
    grid = np.arange(lower_s, upper_s, step_s)
    special = [
        x for x in (0.0, -0.707, -0.350, 0.350, 0.707) if lower_s <= x <= upper_s
    ]
    grid = np.unique(np.round(np.r_[grid, upper_s, special], 12))
    scores = tuple(score_offset(orbit, reference, float(delta)) for delta in grid)
    index = int(np.argmin([s.position_rms_m for s in scores]))
    best = scores[index]
    if 0 < index < len(grid) - 1:
        result = minimize_scalar(
            lambda delta: score_offset(orbit, reference, delta).position_rms_m,
            bounds=(grid[index - 1], grid[index + 1]),
            method="bounded",
            options={"xatol": 1e-4},
        )
        refined = score_offset(orbit, reference, float(result.x))
        if result.success and refined.position_rms_m < best.position_rms_m:
            best = refined
    return scores, best
