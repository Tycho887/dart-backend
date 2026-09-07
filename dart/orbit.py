"""Reproducible orbit solutions and sampled GCRF state histories.

Numerical propagation remains in Rust. These types contain no observations,
provider credentials, or GPS evaluation data.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import satkit as sk
from numpy.typing import NDArray

from dart.forward_models import full_state_states_gcrf, sgp4_states_gcrf
from dart.io.contact import EphemerisMetadata

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class Sgp4Orbit:
    object_id: str
    source: EphemerisMetadata
    solution_id: str
    tle_lines: tuple[str, str]
    offsets: tuple[float, ...]


@dataclass(frozen=True)
class CartesianOrbit:
    object_id: str
    source: EphemerisMetadata
    solution_id: str
    epoch: sk.time
    state_gcrf_si: tuple[float, ...]


OrbitSolution = Sgp4Orbit | CartesianOrbit


@dataclass(frozen=True)
class StateHistory:
    """Nonempty states in requested epoch order; repeats are permitted.

    Only normalized Earth-centered GCRF histories inhabit this type. OEM source
    metadata and covariance are retained separately by the OEM adapter.
    """

    object_id: str
    source_id: str
    epochs: tuple[sk.time, ...]
    states: FloatArray
    frame: ClassVar[str] = "GCRF"
    center: ClassVar[str] = "EARTH"
    time_system: ClassVar[str] = "UTC"
    position_unit: ClassVar[str] = "m"
    velocity_unit: ClassVar[str] = "m/s"

    def __post_init__(self) -> None:
        if not self.object_id.strip() or not self.source_id.strip():
            raise ValueError("state history requires object identity and provenance")
        if not self.epochs or not all(isinstance(t, sk.time) for t in self.epochs):
            raise ValueError("state history requires satkit epochs")
        states = np.array(self.states, dtype=np.float64, copy=True)
        if states.shape != (len(self.epochs), 6):
            raise ValueError("state history must have shape (N, 6)")
        if not np.all(np.isfinite(states)):
            raise ValueError("state history must contain finite states")
        states.flags.writeable = False
        object.__setattr__(self, "states", states)


def propagate(solution: OrbitSolution, epochs: Sequence[sk.time]) -> StateHistory:
    """Sample the fitted physical orbit at unshifted absolute epochs.

    Measurement clock offsets and Doppler biases are not orbital corrections.
    Full-state propagation uses the same default force settings as the fitter
    and rejects epochs before its initial state.
    """
    times = tuple(epochs)
    if isinstance(solution, Sgp4Orbit):
        states = sgp4_states_gcrf(solution.offsets, solution.tle_lines, times)
    elif isinstance(solution, CartesianOrbit):
        states = full_state_states_gcrf(solution.state_gcrf_si, solution.epoch, times)
    else:
        raise TypeError("unsupported orbit solution")
    return StateHistory(solution.object_id, solution.solution_id, times, states)
