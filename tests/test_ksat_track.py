import datetime as dt
from pathlib import Path

import polars as pl
import pytest

from dart.io import azure, ctrl_config, kogs, meos
from dart.tdm.ranging import FrequencySource, TrackRequest, write_track_tdm

CREATED = dt.datetime(2026, 8, 28, 12, 34, 56, 999, tzinfo=dt.UTC)


def request(**overrides) -> TrackRequest:
    values = {
        "contact_id": "contact-1",
        "band": "S",
        "integration_interval_s": 1.0,
        "turnaround_numerator": 240,
        "turnaround_denominator": 221,
        "transmit": FrequencySource("s_band_uplink_p1_1"),
        "receive": FrequencySource(
            "s_band_downlink_p1_1",
            "lr1_receiver1_actualCarrierFrequencyOffset",
        ),
    }
    values.update(overrides)
    return TrackRequest(**values)


def mock_sources(monkeypatch, frame=None):
    monkeypatch.setenv("KOGS_API_KEY", "k" * 40)
    monkeypatch.setattr(
        kogs,
        "get_contact",
        lambda *args, **kwargs: {
            "contact": {
                "id": "contact-1",
                "spacecraft_id": "spacecraft-1",
                "system_id": "system-1",
                "station_id": "station-1",
                "start_time": "2026-08-28T10:00:00Z",
                "end_time": "2026-08-28T10:10:00Z",
                "ephemeris_id": "ephemeris-1",
            }
        },
    )
    monkeypatch.setattr(
        kogs,
        "get_antenna",
        lambda *args, **kwargs: {
            "antenna": {
                "id": "system-1",
                "name": "SG221",
                "station": "station-1",
                "location": {
                    "latitude": 78.22504,
                    "longitude": 15.39198,
                    "altitude": 485.1,
                },
            },
            "expanded": {"stations": [{"id": "station-1", "name": "Svalbard"}]},
        },
    )
    monkeypatch.setattr(
        kogs,
        "get_spacecraft",
        lambda *args, **kwargs: {
            "spacecraft": {
                "id": "spacecraft-1",
                "name": "TESTSAT",
                "satellite_catalog_number": 60543,
            }
        },
    )
    tle = (
        "TESTSAT\n"
        "1 60543U 24149A   24240.50000000  .00000000  00000-0  00000-0 0  9999\n"
        "2 60543  51.6400 100.0000 0005000 100.0000 260.0000 15.50000000123456\n"
    )
    monkeypatch.setattr(
        kogs,
        "get_ephemeris",
        lambda *args, **kwargs: {"ephemeris": {"inline": {"tle": tle}}},
    )
    frequencies = {
        "s_band_uplink_p1_1": 2_085_500_000.0,
        "s_band_downlink_p1_1": 2_269_750_000.0,
    }
    monkeypatch.setattr(
        ctrl_config, "get_link_frequency", lambda _, link, __: frequencies[link]
    )
    calibration = meos.TrackCalibration(4.0, dt.date(2026, 8, 1), -0.125)
    monkeypatch.setattr(meos, "get_track_calibration", lambda *args: calibration)
    telemetry = (
        frame
        if frame is not None
        else pl.DataFrame(
            {
                "timestamp": [
                    dt.datetime(2026, 8, 28, 10, 0, 2),
                    dt.datetime(2026, 8, 28, 10, 0, 1),
                ],
                "contact_id": ["contact-1", "contact-1"],
                "antenna_name": ["SG221", "SG221"],
                "lr1_receiver1_actualCarrierFrequencyOffset": [-50.25, 100.5],
            }
        )
    )
    calls = []

    def fetch(*args, **kwargs):
        calls.append((args, kwargs))
        return telemetry

    monkeypatch.setattr(azure, "fetch_contact_columns", fetch)
    return calls


def test_mode_4_fetches_bounded_data_and_writes_standard_tdm(monkeypatch, tmp_path):
    calls = mock_sources(monkeypatch)

    result = write_track_tdm(request(), tmp_path, creation_date=CREATED)

    assert result.filename == "TRACK_SG221_2024-149A_2026-08-28T12-34-56.tdm"
    assert result.path == tmp_path / result.filename
    assert result.path.read_text(encoding="ascii") == result.text
    assert "CCSDS_TDM_VERS = 2.0" in result.text
    assert "COMMENT S-band TLT calibration date: 2026-08-01" in result.text
    assert "CORRECTION_DOPPLER = -0.125" in result.text
    assert result.text.count("TRANSMIT_FREQ_1 =") == 1
    assert (
        "TRANSMIT_FREQ_1 = 2026-08-28T10:00:01.000000 2085500000.000000" in result.text
    )
    assert (
        "RECEIVE_FREQ_1 = 2026-08-28T10:00:01.000000 2269750100.500000" in result.text
    )
    args, kwargs = calls[0]
    assert args[1:3] == (
        dt.datetime(2026, 8, 28, 10, tzinfo=dt.UTC),
        dt.datetime(2026, 8, 28, 10, 10, tzinfo=dt.UTC),
    )
    assert kwargs["order_by"] == "timestamp"


def test_transmit_frequency_is_repeated_only_when_presteering_changes(monkeypatch):
    frame = pl.DataFrame(
        {
            "timestamp": [
                dt.datetime(2026, 8, 28, 10, 0, 1),
                dt.datetime(2026, 8, 28, 10, 0, 2),
            ],
            "contact_id": ["contact-1", "contact-1"],
            "antenna_name": ["SG221", "SG221"],
            "tx_offset": [0.0, 10.0],
            "rx_offset": [100.0, 90.0],
        }
    )
    mock_sources(monkeypatch, frame)
    configured = request(
        transmit=FrequencySource("s_band_uplink_p1_1", "tx_offset"),
        receive=FrequencySource("s_band_downlink_p1_1", "rx_offset"),
    )

    result = write_track_tdm(configured, creation_date=CREATED)

    assert result.text.count("TRANSMIT_FREQ_1 =") == 2
    assert "2085500010.000000" in result.text


@pytest.mark.parametrize("mode", [1, 2, 3])
def test_unavailable_modes_fail_before_io(mode):
    with pytest.raises(NotImplementedError):
        request(mode=mode)


def test_station_mismatch_is_rejected(monkeypatch):
    frame = pl.DataFrame(
        {
            "timestamp": [dt.datetime(2026, 8, 28, 10, 0, 1)],
            "contact_id": ["contact-1"],
            "antenna_name": ["WRONG"],
            "lr1_receiver1_actualCarrierFrequencyOffset": [10.0],
        }
    )
    mock_sources(monkeypatch, frame)

    with pytest.raises(ValueError, match="ADX station"):
        write_track_tdm(request(), creation_date=CREATED)


def test_empty_measurement_selection_is_rejected(monkeypatch):
    frame = pl.DataFrame(
        {
            "timestamp": [dt.datetime(2026, 8, 28, 10, 0, 1)],
            "contact_id": ["contact-1"],
            "antenna_name": ["SG221"],
            "lr1_receiver1_actualCarrierFrequencyOffset": [None],
        }
    )
    mock_sources(monkeypatch, frame)

    with pytest.raises(ValueError, match="contains a null value"):
        write_track_tdm(request(), creation_date=CREATED)


def test_existing_standard_filename_requires_overwrite(monkeypatch, tmp_path):
    mock_sources(monkeypatch)
    write_track_tdm(request(), tmp_path, creation_date=CREATED)

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_track_tdm(request(), tmp_path, creation_date=CREATED)
    assert write_track_tdm(
        request(), tmp_path, creation_date=CREATED, overwrite=True
    ).path.is_file()


def test_writer_stays_within_complexity_budget():
    source = Path(__file__).parents[1] / "dart" / "tdm" / "ranging.py"
    assert len(source.read_text().splitlines()) <= 300


def test_meos_lookup_fails_closed():
    with pytest.raises(LookupError, match="no reviewed MEOS"):
        meos.get_track_calibration(
            "UNCONFIGURED", "S", dt.datetime(2026, 1, 1, tzinfo=dt.UTC)
        )
