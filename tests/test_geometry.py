from __future__ import annotations

import numpy as np

from dart.geometry import station_state_gcrf, tle_relative_geometry, tle_state_gcrf


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

