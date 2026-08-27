"""Tests for strict TOML configuration and KSAT export orchestration."""

import datetime
from pathlib import Path

import pytest

import dart.io.ksat_export as ksat_export
from dart.io.ksat_adx import KsatExportResult
from dart.io.ksat_export import (
    export_ksat_contact,
    load_ksat_export_config,
    parse_utc_datetime,
)
from dart.io.ksat_metadata import KsatMetadataResult
from dart.io.ksat_tdm import KsatProduct, KsatSite, KsatSpacecraft


UTC = datetime.timezone.utc

VALID_CONFIG = """
[site]
name = "Svalbard"
tlt_calibration_date = 2026-01-02
tlt_band = "X"

[kogs]
spacecraft_id = "spacecraft-uuid"
system_id = "system-uuid"
station_id = "station-uuid"

[spacecraft]
identifier = "2026-001A"
name = "Example"

[header]
summary = "Customer delivery"
comments = ["Validated configuration"]

[adx]
track_timestamp = "integration_end"
tracking_mode = "tracking_mode"

[adx.range_delay]
column = "range_ps"
unit = "ps"

[adx.transmit_frequency]
column = "tx_ghz"
unit = "GHz"

[adx.receive_frequency]
column = "rx_mhz"
unit = "MHz"

[adx.carrier_power]
column = "carrier_power"
unit = "dBW"

[track]
mode = 3
transmit_band = "X"
receive_band = "X"
integration_interval_s = 1.0
turnaround_numerator = 880
turnaround_denominator = 749
transmit_delay_s = 0.0
receive_delay_s = 0.0
correction_range_s = 0.0
correction_doppler_hz = 0.0

[angle]
receive_band = "X"

[sigmet]
transmit_band = "X"
receive_band = "X"
"""


def write_config(tmp_path: Path, text: str = VALID_CONFIG) -> Path:
    path = tmp_path / "ksat.toml"
    path.write_text(text, encoding="utf-8")
    return path


def test_load_config_builds_typed_values_and_discovers_products(tmp_path):
    config = load_ksat_export_config(write_config(tmp_path))

    assert config.site.name == "Svalbard"
    assert config.site.tlt_calibration_date == datetime.date(2026, 1, 2)
    assert config.kogs.spacecraft_id == "spacecraft-uuid"
    assert config.kogs.system_id == "system-uuid"
    assert config.kogs.station_id == "station-uuid"
    assert config.geocoder.url.endswith("/reverse")
    assert config.spacecraft.identifier == "2026-001A"
    assert config.header_summary == "Customer delivery"
    assert config.header_comments == ("Validated configuration",)
    assert config.columns.range_delay.column == "range_ps"
    assert config.columns.range_delay.unit == "ps"
    assert config.track.mode == 3
    assert config.angle.receive_band == "X"
    assert config.sigmet.transmit_band == "X"
    assert config.products == (
        KsatProduct.TRACK,
        KsatProduct.ANGLE,
        KsatProduct.SIGMET,
    )


@pytest.mark.parametrize(
    ("text", "match"),
    [
        ("unexpected = true\n" + VALID_CONFIG, "unknown key.*root"),
        (VALID_CONFIG.replace('name = "Svalbard"', 'nickname = "Svalbard"'), "site"),
        (VALID_CONFIG.replace('unit = "ps"', 'unit = "km"', 1), "adx.range_delay.unit"),
        (VALID_CONFIG.replace('receive_band = "X"', 'receive_band = "L"', 1), "band"),
        (
            VALID_CONFIG.replace('column = "range_ps"\nunit = "ps"', 'column = "range_ps"'),
            "missing required key.*unit",
        ),
    ],
)
def test_load_config_rejects_unknown_keys_invalid_units_and_incomplete_fields(
    tmp_path, text, match
):
    with pytest.raises(ValueError, match=match):
        load_ksat_export_config(write_config(tmp_path, text))


def test_load_config_requires_mappings_for_configured_product(tmp_path):
    text = VALID_CONFIG.replace('track_timestamp = "integration_end"\n', "")

    with pytest.raises(ValueError, match=r"\[track\].*adx.track_timestamp"):
        load_ksat_export_config(write_config(tmp_path, text))


@pytest.mark.parametrize(
    "text",
    [
        VALID_CONFIG.replace('station_id = "station-uuid"\n', ""),
        VALID_CONFIG.replace('tlt_band = "X"\n', ""),
        VALID_CONFIG.replace(
            'name = "Svalbard"\n', 'location = "fabricated place"\n'
        ),
    ],
)
def test_load_config_rejects_incomplete_or_obsolete_metadata(tmp_path, text):
    with pytest.raises(ValueError):
        load_ksat_export_config(write_config(tmp_path, text))


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("2026-08-25T10:30:00Z", datetime.datetime(2026, 8, 25, 10, 30, tzinfo=UTC)),
        (
            "2026-08-25T12:30:00+02:00",
            datetime.datetime(2026, 8, 25, 10, 30, tzinfo=UTC),
        ),
    ],
)
def test_parse_utc_datetime_normalizes_offsets(value, expected):
    assert parse_utc_datetime(value) == expected


@pytest.mark.parametrize("value", ["2026-08-25T10:30:00", "not-a-time", ""])
def test_parse_utc_datetime_rejects_naive_or_invalid_values(value):
    with pytest.raises(ValueError):
        parse_utc_datetime(value)


def test_export_service_forwards_one_bounded_contact_and_options(monkeypatch, tmp_path):
    config = load_ksat_export_config(write_config(tmp_path))
    calls = []
    metadata = KsatMetadataResult(
        site=KsatSite(identifier="D32"),
        spacecraft=KsatSpacecraft(identifier="2026-001A"),
        warnings=("location unavailable",),
    )
    monkeypatch.setattr(
        ksat_export, "load_ksat_contact_metadata", lambda **kwargs: metadata
    )

    def fake_export(selection, header, **kwargs):
        calls.append((selection, header, kwargs))
        return KsatExportResult()

    monkeypatch.setattr(ksat_export, "export_ksat_tdm_bundle", fake_export)
    output = tmp_path / "delivery"
    start = datetime.datetime(2026, 8, 25, 10, tzinfo=UTC)
    stop = datetime.datetime(2026, 8, 25, 11, tzinfo=UTC)

    result = export_ksat_contact(
        config,
        contact_id="contact-1",
        start_time=start,
        stop_time=stop,
        output_dir=output,
        products=("angle", "sigmet"),
        overwrite=True,
        timeout_seconds=12.5,
    )

    assert result == KsatExportResult(warnings=("location unavailable",))
    selection, header, kwargs = calls[0]
    assert selection.contact_ids == ("contact-1",)
    assert selection.start_time == start
    assert selection.stop_time == stop
    assert selection.columns is config.columns
    assert header.site is metadata.site
    assert header.spacecraft is metadata.spacecraft
    assert header.creation_date.tzinfo is UTC
    assert kwargs["products"] == (KsatProduct.ANGLE, KsatProduct.SIGMET)
    assert kwargs["output_dir"] == output
    assert kwargs["overwrite"] is True
    assert kwargs["timeout_seconds"] == 12.5


def test_export_service_defaults_to_all_configured_products(monkeypatch, tmp_path):
    config = load_ksat_export_config(write_config(tmp_path))
    calls = []
    monkeypatch.setattr(
        ksat_export,
        "load_ksat_contact_metadata",
        lambda **kwargs: KsatMetadataResult(
            KsatSite("D32"), KsatSpacecraft("2026-001A")
        ),
    )
    monkeypatch.setattr(
        ksat_export,
        "export_ksat_tdm_bundle",
        lambda *args, **kwargs: calls.append(kwargs) or KsatExportResult(),
    )

    export_ksat_contact(
        config,
        contact_id="contact-1",
        start_time=datetime.datetime(2026, 8, 25, 10, tzinfo=UTC),
        stop_time=datetime.datetime(2026, 8, 25, 11, tzinfo=UTC),
        output_dir=tmp_path,
    )

    assert calls[0]["products"] == config.products
