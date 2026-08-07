from __future__ import annotations

import datetime as dt

import polars as pl

from dart.io.forest import load_forest_passes

from conftest import TEST_TLE_LINES


def test_forest_loader_normalizes_pass(tmp_path):
    path = tmp_path / "forest16.parquet"
    frame = pl.DataFrame(
        {
            "timestamp": [
                dt.datetime(2026, 5, 3, 12, 0, tzinfo=dt.timezone.utc),
                dt.datetime(2026, 5, 3, 12, 0, 1, tzinfo=dt.timezone.utc),
            ],
            "contact_id": ["contact", "contact"],
            "groundStation": ["TROLL", "TROLL"],
            "station_lat": [72.0, 72.0],
            "station_lon": [2.0, 2.0],
            "station_alt": [100.0, 100.0],
            "expected_frequency": [2.2e9, 2.2e9],
            "tle_line1": [TEST_TLE_LINES[1], TEST_TLE_LINES[1]],
            "tle_line2": [TEST_TLE_LINES[2], TEST_TLE_LINES[2]],
            "antenna1_position_azimuth": [10.0, 11.0],
            "antenna1_position_elevation": [20.0, 21.0],
            "lr1_receiver1_actualCarrierFrequencyOffset": [100.0, 101.0],
        }
    )
    frame.write_parquet(path)
    passes = load_forest_passes(path, "FOREST-16")
    assert len(passes) == 1
    assert len(passes[0].observations) == 2
    assert passes[0].observations[0].phase_rad is None
    assert passes[0].context.carrier_hz == 2.2e9
