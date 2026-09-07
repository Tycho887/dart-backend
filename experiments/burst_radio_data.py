"""Synthetic fixtures, visibility and paired realizations for the offline study."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import satkit as sk
from scipy.optimize import brentq

from dart.forward_models import FloatArray, full_state_states_gcrf, tle_state_gcrf
from dart.io import ForwardModelContext, ForwardObservation
from tests.cross_model_validation import synthetic_tle

EPOCH = sk.time(2025, 1, 1)
START = float(EPOCH.as_unixtime())
DAY = 86400
THERMAL = 200.0
MECHANICAL = 30000.0
REGIMES = ("LEO", "MEO", "GEO", "GTO", "PROBA3")
STATION = sk.itrfcoord(latitude_deg=40.4168, longitude_deg=-3.7038, altitude=650.0)
ELEMENTS = (
    "mean_motion",
    "eccen",
    "inclination",
    "raan",
    "arg_of_perigee",
    "mean_anomaly",
)


@dataclass(frozen=True)
class Session:
    rise: float
    peak: float
    setting: float
    start: float
    stop: float
    peak_elevation_deg: float


@dataclass(frozen=True)
class Fixture:
    regime: str
    lines: tuple[str, str]
    state: FloatArray
    period: float


def times(unix: FloatArray) -> list[sk.time]:
    return [sk.time.from_unixtime(float(value)) for value in unix]


def tle_lines(tle: sk.TLE) -> tuple[str, str]:
    first, second = tle.to_2line()
    return str(first), str(second)


def parse_tle(lines: tuple[str, str]) -> sk.TLE:
    tle = sk.TLE.from_lines(list(lines))
    assert isinstance(tle, sk.TLE)
    return tle


def fixture(regime: str) -> Fixture:
    base = regime if regime in {"LEO", "MEO", "GEO"} else "LEO"
    tle = parse_tle(synthetic_tle(base))
    tle.epoch = EPOCH
    if regime in {"GTO", "PROBA3"}:
        perigee, apogee, inclination = {
            "GTO": (250, 35786, 27),
            "PROBA3": (600, 60530, 59),
        }[regime]
        rp, ra = (6378.137 + perigee) * 1000, (6378.137 + apogee) * 1000
        axis = (rp + ra) / 2
        tle.mean_motion = np.sqrt(3.986004418e14 / axis**3) * DAY / (2 * np.pi)
        tle.eccen = (ra - rp) / (ra + rp)
        tle.inclination = inclination
        tle.raan, tle.arg_of_perigee, tle.mean_anomaly = 0.0, 270.0, 0.0
    if regime == "GEO":
        # Solve longitude using Rust SGP4 states and satkit's frame operation.
        def longitude(anomaly: float) -> float:
            tle.mean_anomaly = anomaly
            state = tle_state_gcrf(tle_lines(tle), EPOCH)
            position = sk.frametransform.qgcrf2itrf(EPOCH) * state[:3]
            return float(np.degrees(np.arctan2(position[1], position[0])))

        guess = (tle.mean_anomaly - longitude(tle.mean_anomaly)) % 360
        tle.mean_anomaly = brentq(longitude, guess - 5, guess + 5, xtol=1e-8)
    lines = tle_lines(tle)
    return Fixture(regime, lines, tle_state_gcrf(lines, EPOCH), DAY / tle.mean_motion)


def elevation(fix: Fixture, unix: FloatArray) -> FloatArray:
    epochs = times(unix)
    states = full_state_states_gcrf(fix.state, EPOCH, epochs)
    enu = np.array(
        [
            sk.itrfcoord(sk.frametransform.qgcrf2itrf(epoch) * state[:3]).to_enu(
                STATION
            )
            for epoch, state in zip(epochs, states, strict=True)
        ]
    )
    return np.degrees(np.arctan2(enu[:, 2], np.linalg.norm(enu[:, :2], axis=1)))


def clipped_session(
    rise: float, peak: float, setting: float, maximum: float
) -> Session:
    return Session(
        rise, peak, setting, max(rise, peak - 600), min(setting, peak + 600), maximum
    )


def daily_sessions(
    end: float, evaluator: Callable[[FloatArray], FloatArray]
) -> list[Session]:
    centers = np.arange(START + DAY / 2, end - 600 + 1, DAY)
    maxima = evaluator(centers)
    return [
        Session(START, center, end, center - 600, center + 600, maximum)
        for center, maximum in zip(centers, maxima, strict=True)
    ]


def visibility_sessions(
    evaluator: Callable[[FloatArray], FloatArray],
    *,
    days: int = 30,
    maximum: int = 60,
    continuous_geo: bool = False,
) -> list[Session]:
    """60 s brackets, 1 s boundary/peak refinement; sessions use inclusive epochs."""
    end = START + days * DAY
    grid = np.arange(START, end + 1, 60.0)
    heights = evaluator(grid)
    if continuous_geo and np.all(heights >= 10):
        return daily_sessions(end, evaluator)[:maximum]
    # Include nearby sub-mask local maxima so short grazing passes aren't lost
    # merely because no coarse node was above the mask.
    peaks = (
        np.flatnonzero((heights[1:-1] >= heights[:-2]) & (heights[1:-1] > heights[2:]))
        + 1
    )
    visible = heights >= 10
    edges = np.diff(np.r_[False, visible, False].astype(int))
    brackets = [
        (int(left), int(right))
        for left, right in zip(
            np.flatnonzero(edges == 1), np.flatnonzero(edges == -1) - 1, strict=True
        )
    ]
    covered = {p for left, right in brackets for p in peaks if left <= p <= right}
    brackets.extend((int(p), int(p)) for p in peaks if p not in covered)
    brackets.sort()
    if maximum == 1:
        return first_session(grid, heights, brackets, evaluator)
    return refine_sessions(grid, heights, brackets, evaluator)[:maximum]


def first_session(
    grid: FloatArray,
    heights: FloatArray,
    brackets: list[tuple[int, int]],
    evaluator: Callable[[FloatArray], FloatArray],
) -> list[Session]:
    for bracket in brackets:
        sessions = refine_sessions(grid, heights, [bracket], evaluator)
        if sessions:
            return sessions
    return []


def refine_sessions(
    grid: FloatArray,
    heights: FloatArray,
    brackets: list[tuple[int, int]],
    evaluator: Callable[[FloatArray], FloatArray],
) -> list[Session]:
    windows = []
    for left, right in brackets:
        peak = left + int(np.argmax(heights[left : right + 1]))
        nodes = np.unique(
            np.concatenate(
                [
                    np.arange(
                        grid[max(0, index - 1)], grid[min(grid.size - 1, index + 1)] + 1
                    )
                    for index in (left, peak, right)
                ]
            )
        )
        windows.append(nodes)
    if not windows:
        return []
    nodes = np.unique(np.concatenate(windows))
    values = evaluator(nodes)
    sessions = []
    for window in windows:
        heights_fine = values[np.searchsorted(nodes, window)]
        above = window[heights_fine >= 10]
        if above.size == 0:
            continue
        peak_index = int(np.argmax(heights_fine))
        sessions.append(
            clipped_session(
                float(above[0]),
                float(window[peak_index]),
                float(above[-1]),
                float(heights_fine[peak_index]),
            )
        )
    return sessions


def context(
    sessions: list[Session], observed: FloatArray | None = None
) -> ForwardModelContext:
    epochs = np.concatenate(
        [np.arange(session.start, session.stop + 1) for session in sessions]
    )
    if observed is None:
        observed = np.zeros(epochs.size)
    if observed.size != epochs.size:
        raise ValueError("observation count differs from the session grid")
    indices = np.concatenate(
        [np.full(int(s.stop - s.start) + 1, i) for i, s in enumerate(sessions)]
    )
    rows = [
        ForwardObservation.from_scalar(
            float(epoch),
            float(value),
            THERMAL**2,
            receiver_id=0,
            pass_index=int(index),
            contact_id=f"session-{index:02}",
        )
        for epoch, value, index in zip(epochs, observed, indices, strict=True)
    ]
    return ForwardModelContext(
        center_frequency_hz=400e6,
        receivers=[STATION],
        contact_to_pass_idx={f"session-{i:02}": i for i in range(len(sessions))},
        observations=rows,
    )


def noise_realization(
    sessions: list[Session], probability: float, timing: str, seed: int
) -> FloatArray:
    """Labels never leave this function; seeded draws do not depend on prefix size."""
    if not 0 <= probability <= 1 or timing not in {"independent", "bursty"}:
        raise ValueError("invalid mixture probability or timing")
    pieces = []
    for index, session in enumerate(sessions):
        rng = np.random.default_rng(np.random.SeedSequence([seed, index]))
        count = int(session.stop - session.start) + 1
        thermal = rng.normal(0, THERMAL, count)
        size = count if timing == "independent" else (count + 29) // 30
        mechanical = rng.random(size) < probability
        offsets = rng.normal(0, np.sqrt(MECHANICAL**2 - THERMAL**2), size) * mechanical
        if timing == "bursty":
            offsets = np.repeat(offsets, 30)[:count]
        pieces.append(thermal + offsets)
    return np.concatenate(pieces)


def perturbed_lines(
    base: tuple[str, str], direction: FloatArray, factor: float
) -> tuple[str, str]:
    tle = parse_tle(base)
    for name, offset in zip(ELEMENTS, factor * direction, strict=True):
        setattr(tle, name, getattr(tle, name) + float(offset))
    if tle.mean_motion <= 0 or not 0 <= tle.eccen < 1 or not 0 <= tle.inclination < 180:
        raise ValueError("perturbed elements outside valid domain")
    for name in ELEMENTS[3:]:
        setattr(tle, name, getattr(tle, name) % 360)
    return tle_lines(tle)


def common_prior(
    fix: Fixture, error_km: float, seed: int
) -> tuple[tuple[str, str], FloatArray]:
    rng = np.random.default_rng(seed)
    tle = parse_tle(fix.lines)
    scales = np.array([tle.mean_motion * 0.001, 0.001, 0.1, 0.1, 0.1, 0.1])
    for _ in range(200):
        direction = rng.normal(size=6) * scales
        try:
            return scaled_prior(fix, error_km, direction)
        except ValueError:
            continue
    raise ValueError("could not generate a valid common prior in 200 draws")


def scaled_prior(
    fix: Fixture, error_km: float, direction: FloatArray
) -> tuple[tuple[str, str], FloatArray]:
    def error(factor: float) -> float:
        state = tle_state_gcrf(perturbed_lines(fix.lines, direction, factor), EPOCH)
        return float(np.linalg.norm(state[:3] - fix.state[:3]) / 1000 - error_km)

    upper = 1.0
    while error(upper) < 0 and upper < 1024:
        upper *= 2
    factor = brentq(error, 0, upper, xtol=1e-10)
    lines = perturbed_lines(fix.lines, direction, factor)
    state = tle_state_gcrf(lines, EPOCH)
    # TLE text quantization prevents an exact continuous root; tolerance is 10 m.
    if abs(np.linalg.norm(state[:3] - fix.state[:3]) / 1000 - error_km) > 0.01:
        raise ValueError("serialized prior missed requested position error")
    return lines, state
