"""Tests for authoritative KSAT header metadata enrichment."""

import datetime as dt

import pytest
import requests

import dart.io.ksat_metadata as metadata
from dart.io.ksat_metadata import (
    GeocoderConfig,
    KogsMetadataConfig,
    KsatSiteOverrides,
    load_ksat_contact_metadata,
    reverse_geocode_location,
)
from dart.io.ksat_tdm import KsatSpacecraft


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


def test_reverse_geocoder_builds_deduplicated_english_location(monkeypatch):
    calls = []

    def fake_get(url, *, params, headers, timeout):
        calls.append((url, params, headers, timeout))
        return FakeResponse(
            {
                "address": {
                    "town": "Longyearbyen",
                    "state": "Svalbard",
                    "county": "Svalbard",
                    "country": "Norway",
                }
            }
        )

    monkeypatch.setattr(metadata.requests, "get", fake_get)
    config = GeocoderConfig(user_agent="DART tests", timeout_seconds=3.5)

    assert reverse_geocode_location(78.22504, 15.39198, config) == (
        "Longyearbyen, Svalbard, Norway"
    )
    _, params, headers, timeout = calls[0]
    assert params["zoom"] == "18"
    assert params["accept-language"] == "en"
    assert headers["Accept-Language"] == "en"
    assert timeout == 3.5


@pytest.mark.parametrize(
    "address",
    [
        {"town": "Tromsø", "country": "Norway"},
        {"state": "Colorado", "country": "USA"},
    ],
)
def test_reverse_geocoder_rejects_non_ascii_or_missing_locality(monkeypatch, address):
    monkeypatch.setattr(
        metadata.requests, "get", lambda *args, **kwargs: FakeResponse({"address": address})
    )

    with pytest.raises(ValueError, match="usable ASCII locality/country"):
        reverse_geocode_location(1.0, 2.0, GeocoderConfig())


def _contact_payload(**overrides):
    contact = {
        "id": "contact-1",
        "spacecraft_id": "spacecraft-1",
        "system_id": "system-1",
        "station_id": "station-1",
    }
    contact.update(overrides)
    return {"contact": contact}


def _antenna_payload():
    return {
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
        "expanded": {
            "stations": [{"id": "station-1", "name": "SVALSAT"}]
        },
    }


def _configs():
    return {
        "contact_id": "contact-1",
        "kogs": KogsMetadataConfig("spacecraft-1", "system-1", "station-1"),
        "geocoder": GeocoderConfig(),
        "site": KsatSiteOverrides(),
        "spacecraft": KsatSpacecraft("60543", catalog_id="60543"),
    }


def _mock_kogs(monkeypatch, *, contact=None):
    monkeypatch.setenv("KOGS_API_KEY", "secret")
    monkeypatch.setattr(
        metadata,
        "get_contact",
        lambda *args, **kwargs: contact or _contact_payload(),
    )
    monkeypatch.setattr(
        metadata, "get_antenna", lambda *args, **kwargs: _antenna_payload()
    )
    monkeypatch.setattr(
        metadata,
        "get_spacecraft",
        lambda *args, **kwargs: {
            "id": "spacecraft-1",
            "name": "AWESAT-1",
            "satellite_catalog_number": 60543,
        },
    )


def test_metadata_uses_kogs_coordinates_and_satkit_ecef(monkeypatch):
    _mock_kogs(monkeypatch)
    monkeypatch.setattr(
        metadata,
        "reverse_geocode_location",
        lambda *args: "Longyearbyen, Svalbard, Norway",
    )

    result = load_ksat_contact_metadata(**_configs())

    assert result.site.identifier == "SG221"
    assert result.site.location == "Longyearbyen, Svalbard, Norway"
    assert result.site.ecef_x_m == pytest.approx(1_259_031.370, abs=0.001)
    assert result.site.ecef_y_m == pytest.approx(346_605.420, abs=0.001)
    assert result.site.ecef_z_m == pytest.approx(6_222_586.863, abs=0.001)
    assert result.spacecraft.name == "AWESAT-1"
    assert result.warnings == (
        "pedestal offset is UNKNOWN; no reviewed value was configured",
        "TLT calibration date is UNKNOWN; no reviewed value was configured",
    )


def test_metadata_geocoder_failure_is_a_warning(monkeypatch):
    _mock_kogs(monkeypatch)
    monkeypatch.setattr(
        metadata,
        "reverse_geocode_location",
        lambda *args: (_ for _ in ()).throw(requests.Timeout("timed out")),
    )
    configs = _configs()
    configs["site"] = KsatSiteOverrides(
        pedestal_offset_m=4.0,
        tlt_calibration_date=dt.date(2026, 1, 2),
        tlt_band="X",
    )

    result = load_ksat_contact_metadata(**configs)

    assert result.site.location is None
    assert result.warnings == ("ground-station location is UNKNOWN: timed out",)


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"spacecraft_id": "wrong"}, "spacecraft identity mismatch"),
        ({"system_id": "wrong"}, "system identity mismatch"),
        ({"station_id": "wrong"}, "station identity mismatch"),
    ],
)
def test_metadata_rejects_contact_identity_mismatch_before_other_requests(
    monkeypatch, overrides, match
):
    monkeypatch.setenv("KOGS_API_KEY", "secret")
    monkeypatch.setattr(
        metadata,
        "get_contact",
        lambda *args, **kwargs: _contact_payload(**overrides),
    )
    monkeypatch.setattr(
        metadata,
        "get_antenna",
        lambda *args, **kwargs: pytest.fail("antenna request must not run"),
    )

    with pytest.raises(ValueError, match=match):
        load_ksat_contact_metadata(**_configs())


def test_metadata_requires_kogs_credentials(monkeypatch):
    monkeypatch.delenv("KOGS_API_KEY", raising=False)

    with pytest.raises(ValueError, match="KOGS_API_KEY"):
        load_ksat_contact_metadata(**_configs())
