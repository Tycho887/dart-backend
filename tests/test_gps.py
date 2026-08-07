from __future__ import annotations

import numpy as np

from dart.io.gps import gps_sow_to_utc


def test_gps_seconds_of_week_uses_packet_only_to_resolve_week():
    packet_utc_s = np.array([1_778_000_000.0])
    sow_s = np.array([123_456.1])
    result = gps_sow_to_utc(packet_utc_s, sow_s)
    # Recovered epoch is in the packet's GPS week and has the specified SOW.
    assert abs(result[0] - packet_utc_s[0]) < 7 * 24 * 3600
    reconstructed = (result[0] + 18.0 - 315_964_800.0) % 604_800.0
    assert np.isclose(reconstructed, sow_s[0], atol=1e-6)

