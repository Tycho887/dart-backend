from datetime import UTC, datetime

import pytest

from dart.io import kogs
from dart.io.contact import EphemerisMetadata

API_KEY = "a" * 40


class Response:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


def test_get_contact_is_typed_and_has_timeout(monkeypatch):
    calls = []
    payload = {
        "contact": {
            "id": "contact-1",
            "spacecraft_id": "spacecraft-1",
            "system_id": "system-1",
            "station_id": "station-1",
            "ephemeris_id": "ephemeris-1",
            "start_time": "2026-01-01T00:00:00Z",
            "end_time": "2026-01-01T00:05:00Z",
            "state": "scheduled",
        }
    }

    def request(method, url, **kwargs):
        calls.append((method, url, kwargs))
        return Response(payload)

    monkeypatch.setattr(kogs.requests, "request", request)
    contact = kogs.get_contact(API_KEY, "contact-1", timeout_seconds=4.5)

    assert contact.id == "contact-1"
    assert contact.start == datetime(2026, 1, 1, tzinfo=UTC)
    assert calls[0][0:2] == (
        "GET",
        "https://mgmt.kogs.api.ksat.no/24.08/contacts/contact-1",
    )
    assert calls[0][2]["timeout"] == 4.5


def test_auth_accepts_raw_or_canonical_key_without_double_prefixing():
    assert kogs.headers(API_KEY)["Authorization"] == f"KSAT1-PLAIN {API_KEY}"
    assert kogs.headers(f"KSAT1-PLAIN {API_KEY}")["Authorization"] == (
        f"KSAT1-PLAIN {API_KEY}"
    )


@pytest.mark.parametrize(
    "value",
    ["", "too-short", "Bearer " + "e" * 40, "KSAT1-PLAIN " + "f" * 39, " " + "g" * 40],
)
def test_auth_rejects_malformed_credentials(value):
    with pytest.raises((TypeError, ValueError)):
        kogs.headers(value)


def test_load_contact_metadata_cross_checks_sources(monkeypatch):
    contact = kogs.Contact(
        "contact-1",
        "spacecraft-1",
        "system-1",
        "station-1",
        "profile-1",
        "ephemeris-1",
        datetime(2026, 1, 1, tzinfo=UTC),
        datetime(2026, 1, 1, 0, 5, tzinfo=UTC),
        "scheduled",
    )
    antenna = kogs.Antenna(
        "system-1", "SGS1", "station-1", "Svalbard", 78.2, 15.4, 100.0, ("S",)
    )
    spacecraft = kogs.Spacecraft("spacecraft-1", "TESTSAT", "60543")
    ephemeris = EphemerisMetadata(
        "ephemeris-1",
        "spacecraft-1",
        "TLE",
        "KOGS",
        None,
        None,
        None,
        None,
        None,
        "TESTSAT\n"
        "1 60543U 24149A   24240.50000000  .00000000  00000-0  00000-0 0  9999\n"
        "2 60543  51.6400 100.0000 0005000 100.0000 260.0000 15.50000000123456\n",
        None,
        None,
        False,
        None,
    )
    monkeypatch.setattr(kogs, "get_contact", lambda *args, **kwargs: contact)
    monkeypatch.setattr(kogs, "get_antenna", lambda *args, **kwargs: antenna)
    monkeypatch.setattr(kogs, "get_spacecraft", lambda *args, **kwargs: spacecraft)
    monkeypatch.setattr(kogs, "get_ephemeris", lambda *args, **kwargs: ephemeris)

    metadata = kogs.load_contact_metadata(API_KEY, "contact-1")

    assert metadata.contact_id == "contact-1"
    assert metadata.ephemeris.tle == ephemeris.tle
    assert metadata.catalog == "60543"
    assert metadata.cospar == "2024-149A"


def test_mutations_require_confirmed_contract(monkeypatch):
    monkeypatch.setattr(kogs.requests, "request", lambda *args, **kwargs: pytest.fail())

    with pytest.raises(kogs.KogsError, match="not been confirmed"):
        kogs.cancel_contact(API_KEY, "contact-1")


def test_list_contacts_uses_utc_bounds(monkeypatch):
    calls = []
    payload = {
        "data": [
            {
                "id": "contact-1",
                "spacecraft_id": "spacecraft-1",
                "system_id": "system-1",
                "station_id": "station-1",
                "ephemeris_id": "ephemeris-1",
                "start_time": "2026-01-01T00:00:00Z",
                "end_time": "2026-01-01T00:05:00Z",
                "state": "scheduled",
            }
        ]
    }

    def request(method, url, **kwargs):
        calls.append((method, url, kwargs))
        return Response(payload)

    monkeypatch.setattr(kogs.requests, "request", request)
    contacts = kogs.list_contacts(
        API_KEY,
        datetime(2026, 1, 1, tzinfo=UTC),
        datetime(2026, 1, 2, tzinfo=UTC),
        station_ids=("station-1",),
    )

    assert [contact.id for contact in contacts] == ["contact-1"]
    assert calls[0][2]["params"]["start_time"] == "2026-01-01T00:00:00Z"


def test_get_station_returns_typed_location(monkeypatch):
    monkeypatch.setattr(
        kogs.requests,
        "request",
        lambda *args, **kwargs: Response(
            {
                "station": {
                    "id": "station-1",
                    "name": "Svalbard",
                    "location": {
                        "latitude": 78.2,
                        "longitude": 15.4,
                        "altitude": 100,
                    },
                }
            }
        ),
    )

    station = kogs.get_station(API_KEY, "station-1")

    assert station.name == "Svalbard"
    assert station.altitude == 100.0
