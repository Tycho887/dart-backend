from __future__ import annotations

import numpy as np
import pytest
import satkit as sk

from dart.frames import Station, station_state_gcrf, tle_relative_geometry, tle_state_gcrf
from dart.services.solver.mean_element import TLEContext

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
    )


def test_station_state_includes_earth_rotation(context):
    position, velocity = station_state_gcrf(context.station, context.tle.epoch)
    assert 6.0e6 < np.linalg.norm(position) < 6.5e6
    assert 200.0 < np.linalg.norm(velocity) < 500.0


def test_positive_offset_advances_orbital_phase(context):
    epoch = context.tle.epoch
    before, _ = tle_state_gcrf(context.tle, epoch, 0.0)
    after, _ = tle_state_gcrf(context.tle, epoch, 10.0)
    assert np.linalg.norm(after - before) > 50_000.0
    geometry = tle_relative_geometry(context.tle, context.station, epoch, 0.0)
    assert np.isfinite(geometry.range_rate_m_s)
    assert np.isclose(np.linalg.norm(geometry.line_of_sight_enu), 1.0)
