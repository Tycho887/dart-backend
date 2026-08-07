from __future__ import annotations

import numpy as np

from dart.constants import SPEED_OF_LIGHT_M_S
from dart.measurements import (
    absolute_received_to_offset_hz,
    doppler_offset_hz,
    interferometric_phase_rad,
    offset_to_absolute_received_hz,
    wrap_angle_rad,
)
from dart.types import MeasurementMode, RFObservation


def test_doppler_is_canonical_carrier_offset():
    carrier = 2.2e9
    expected = -7_500.0 * carrier / SPEED_OF_LIGHT_M_S + 125.0
    assert np.isclose(doppler_offset_hz(7_500.0, carrier, 125.0), expected)
    absolute = offset_to_absolute_received_hz(expected, carrier)
    assert np.isclose(absolute_received_to_offset_hz(absolute, carrier), expected)


def test_phase_projection_and_wrapping():
    phase = interferometric_phase_rad(
        np.array([1.0, 0.0, 0.0]), 2.2e9, np.array([59.0, 0.0, 0.0]), 8.0
    )
    assert -np.pi <= phase < np.pi
    assert np.isclose(wrap_angle_rad(np.pi + 0.2), -np.pi + 0.2)


def test_observation_selects_present_channels(context):
    epoch = context.tle.epoch
    doppler = RFObservation(epoch, "test", 4.0)
    combined = RFObservation(epoch, "test", 4.0, 0.2)
    assert doppler.mode is MeasurementMode.DOPPLER
    assert combined.mode is MeasurementMode.DOPPLER_PHASE
    np.testing.assert_allclose(combined.measurement(), [4.0, 0.2])

