import datetime as dt
from pathlib import Path

import polars as pl
import pytest
from test_ksat_track import CREATED, mock_sources

from dart.io import meos
from dart.tdm import AngleColumns, AngleRequest, AngleResult, write_angle_tdm


def request(**overrides) -> AngleRequest:
    values = {
        "contact_id": "contact-1",
        "band": "S",
        "tracking_mode": "PROGRAM",
    }
    values.update(overrides)
    return AngleRequest(**values)


def frame(**overrides) -> pl.DataFrame:
    values = {
        "timestamp": [
            dt.datetime(2026, 8, 28, 10, 0, 2, tzinfo=dt.UTC),
            dt.datetime(2026, 8, 28, 10, 0, 1, tzinfo=dt.UTC),
        ],
        "contact_id": ["contact-1", "contact-1"],
        "antenna_name": ["SG221", "SG221"],
        "antenna1_position_azimuth": [41.5, 41.25],
        "antenna1_position_elevation": [35.75, 35.5],
    }
    values.update(overrides)
    return pl.DataFrame(values)


def test_angle_fetches_bounded_data_and_writes_standard_tdm(monkeypatch, tmp_path):
    calls = mock_sources(monkeypatch, frame())

    result = write_angle_tdm(request(), tmp_path, creation_date=CREATED)

    assert isinstance(result, AngleResult)
    assert result.filename == "ANGLE_SG221_2024-149A_2026-08-28T12-34-56.tdm"
    assert result.path == tmp_path / result.filename
    assert result.path.read_text(encoding="ascii") == result.text
    assert "COMMENT Antenna Pointing Angles" in result.text
    assert "COMMENT Pedestal offset: Lg=4.0 meters" in result.text
    assert "COMMENT TLT calibration date: 2026-08-01" in result.text
    assert "COMMENT TRACKING_MODE = PROGRAM" in result.text
    assert "RECEIVE_BAND = S" in result.text
    assert "ANGLE_TYPE = AZEL" in result.text
    first = "2026-08-28T10:00:01.000000"
    assert f"ANGLE_1 = {first} 41.250000" in result.text
    assert f"ANGLE_2 = {first} 35.500000" in result.text
    assert result.warnings == (
        (
            "confirm that ADX columns 'antenna1_position_azimuth' and "
            "'antenna1_position_elevation' contain controller pointing readback "
            "for the target site"
        ),
    )
    args, kwargs = calls[0]
    assert args[1:3] == (
        dt.datetime(2026, 8, 28, 10, tzinfo=dt.UTC),
        dt.datetime(2026, 8, 28, 10, 10, tzinfo=dt.UTC),
    )
    assert args[3] == (
        "timestamp",
        "contact_id",
        "antenna_name",
        "antenna1_position_azimuth",
        "antenna1_position_elevation",
    )
    assert kwargs["order_by"] == "timestamp"


def test_missing_calibration_is_unknown_and_warned(monkeypatch):
    mock_sources(monkeypatch, frame())

    def missing(*args):
        raise LookupError("no reviewed MEOS TRACK calibration for SG221/S")

    monkeypatch.setattr(meos, "get_track_calibration", missing)

    result = write_angle_tdm(request(), creation_date=CREATED)

    assert "COMMENT Pedestal offset: UNKNOWN" in result.text
    assert "COMMENT TLT calibration date: UNKNOWN" in result.text
    assert any("calibration comments use UNKNOWN" in item for item in result.warnings)


@pytest.mark.parametrize("angle_type", ["XEYN", "XSYE", "RADEC"])
def test_unconfirmed_angle_types_are_unavailable(angle_type):
    with pytest.raises(NotImplementedError, match="only AZEL"):
        request(angle_type=angle_type)


@pytest.mark.parametrize("mode", ["", "program", "MANUAL"])
def test_undocumented_tracking_modes_are_rejected(mode):
    with pytest.raises(ValueError, match="tracking_mode"):
        request(tracking_mode=mode)


def test_incomplete_angle_pair_is_rejected(monkeypatch):
    incomplete = frame(antenna1_position_elevation=[35.75, None])
    mock_sources(monkeypatch, incomplete)

    with pytest.raises(ValueError, match="only one pointing coordinate"):
        write_angle_tdm(request(), creation_date=CREATED)


def test_non_finite_angle_is_rejected(monkeypatch):
    invalid = frame(antenna1_position_azimuth=[float("nan"), 41.25])
    mock_sources(monkeypatch, invalid)

    with pytest.raises(ValueError, match="non-finite angle"):
        write_angle_tdm(request(), creation_date=CREATED)


def test_conflicting_values_at_one_epoch_are_rejected(monkeypatch):
    epoch = dt.datetime(2026, 8, 28, 10, 0, 1, tzinfo=dt.UTC)
    conflicting = frame(
        timestamp=[epoch, epoch],
        antenna1_position_azimuth=[41.25, 42.0],
        antenna1_position_elevation=[35.5, 35.5],
    )
    mock_sources(monkeypatch, conflicting)

    with pytest.raises(ValueError, match="conflicting ANGLE"):
        write_angle_tdm(request(), creation_date=CREATED)


def test_empty_measurement_selection_is_rejected(monkeypatch):
    empty = frame(
        antenna1_position_azimuth=[None, None],
        antenna1_position_elevation=[None, None],
    )
    mock_sources(monkeypatch, empty)

    with pytest.raises(ValueError, match="no complete ANGLE"):
        write_angle_tdm(request(), creation_date=CREATED)


def test_custom_columns_and_station_mismatch(monkeypatch):
    custom = frame().rename(
        {
            "timestamp": "sample_time",
            "antenna1_position_azimuth": "readback_az",
            "antenna1_position_elevation": "readback_el",
        }
    ).with_columns(pl.lit("WRONG").alias("antenna_name"))
    mock_sources(monkeypatch, custom)
    configured = request(
        columns=AngleColumns(
            timestamp="sample_time", angle_1="readback_az", angle_2="readback_el"
        )
    )

    with pytest.raises(ValueError, match="ADX station"):
        write_angle_tdm(configured, creation_date=CREATED)


def test_existing_standard_filename_requires_overwrite(monkeypatch, tmp_path):
    mock_sources(monkeypatch, frame())
    write_angle_tdm(request(), tmp_path, creation_date=CREATED)

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_angle_tdm(request(), tmp_path, creation_date=CREATED)
    assert write_angle_tdm(
        request(), tmp_path, creation_date=CREATED, overwrite=True
    ).path.is_file()


def test_writer_stays_within_complexity_budget():
    source = Path(__file__).parents[1] / "dart" / "tdm" / "angle.py"
    assert len(source.read_text().splitlines()) <= 300
