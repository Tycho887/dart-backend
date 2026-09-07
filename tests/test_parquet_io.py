from datetime import UTC, datetime

import polars as pl
import pytest

from dart.io import parquet


def test_read_measurements_normalizes_recorded_adx_columns(tmp_path):
    path = tmp_path / "measurements.parquet"
    pl.DataFrame(
        {
            "timestamp": [datetime(2026, 1, 1, tzinfo=UTC)],
            "contact_id": ["contact-1"],
            "spacecraft_id": ["spacecraft-1"],
            "system_id": ["system-1"],
            "antenna_name": ["SGS1"],
            "antenna1_tracking_epochOffset": [0.0],
            "antenna1_position_azimuth": [10.0],
            "antenna1_position_elevation": [20.0],
            "lr1_receiver1_carrierLockState": ["Locked"],
            "lr1_receiver1_ebN0": [12.0],
            "lr1_receiver1_actualCarrierFrequencyOffset": [100.0],
        }
    ).write_parquet(path)

    result = parquet.read_measurements(path)

    assert result["doppler_hz"].to_list() == [100.0]
    assert "lr1_receiver1_actualCarrierFrequencyOffset" not in result.columns


def test_read_measurements_rejects_incomplete_frame(tmp_path):
    path = tmp_path / "measurements.parquet"
    pl.DataFrame({"timestamp": [datetime(2026, 1, 1, tzinfo=UTC)]}).write_parquet(path)

    with pytest.raises(ValueError, match="missing columns"):
        parquet.read_measurements(path)
