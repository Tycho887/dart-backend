from __future__ import annotations

import numpy as np
import pytest
import satkit as sk

from dart.types import PhaseBaseline, Station, TLEContext


TEST_TLE_LINES = [
    "0 STARLINK-30477",
    "1 57912U 23146X   24099.49439401  .00006757  00000+0  51475-3 0  9997",
    "2 57912  43.0018 157.5807 0001420 272.5369  87.5310 15.02537576 31746",
]


@pytest.fixture
def context() -> TLEContext:
    return TLEContext(
        tle=sk.TLE.from_lines(TEST_TLE_LINES),
        station=Station("test", 42.0, -71.0, 100.0),
        carrier_hz=2.2e9,
        baseline=PhaseBaseline(59.0, 0.0, 0.0),
    )


@pytest.fixture
def visible_times(context) -> list:
    # Peak of a high-elevation pass for TEST_TLE_LINES at the test station.
    peak = sk.time.from_unixtime(1_712_653_465.642464)
    return [peak + sk.duration(seconds=float(value)) for value in np.arange(-240, 241, 5)]

