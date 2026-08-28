import pytest

from dart.io import kogs


class FakeResponse:
    def raise_for_status(self):
        pass

    def json(self):
        return {"contact": {"id": "c1"}}


def test_get_contact_has_request_timeout(monkeypatch):
    calls = []

    def fake_get(url, *, headers, timeout):
        calls.append((url, headers, timeout))
        return FakeResponse()

    monkeypatch.setattr(kogs.requests, "get", fake_get)

    raw_key = "a" * 40
    result = kogs.get_contact(raw_key, "c1")

    assert result == {"contact": {"id": "c1"}}
    assert calls == [
        (
            "https://mgmt.kogs.api.ksat.no/24.08/contacts/c1",
            {
                "Authorization": f"KSAT1-PLAIN {raw_key}",
                "Accept": "application/json",
            },
            kogs.KOGS_REQUEST_TIMEOUT_SECONDS,
        )
    ]


def test_get_contact_accepts_explicit_timeout(monkeypatch):
    calls = []
    monkeypatch.setattr(
        kogs.requests,
        "get",
        lambda url, *, headers, timeout: calls.append(timeout) or FakeResponse(),
    )

    kogs.get_contact("KSAT1-PLAIN " + "b" * 40, "c1", timeout_seconds=4.5)

    assert calls == [4.5]


def test_get_ephemeris_accepts_explicit_timeout(monkeypatch):
    calls = []
    monkeypatch.setattr(
        kogs.requests,
        "get",
        lambda url, *, headers, timeout: calls.append((url, timeout)) or FakeResponse(),
    )

    kogs.get_ephemeris("c" * 40, "eph-1", timeout_seconds=6.5)

    assert calls == [("https://mgmt.kogs.api.ksat.no/24.08/ephemeris/eph-1", 6.5)]


def test_auth_header_accepts_raw_or_canonical_key_without_double_prefixing():
    raw_key = "d" * 40

    assert kogs.generate_auth_header(raw_key)["Authorization"] == (
        f"KSAT1-PLAIN {raw_key}"
    )
    assert kogs.generate_auth_header(f"KSAT1-PLAIN {raw_key}")["Authorization"] == (
        f"KSAT1-PLAIN {raw_key}"
    )


@pytest.mark.parametrize(
    "value",
    [
        "",
        "too-short",
        "Bearer " + "e" * 40,
        "KSAT1-PLAIN " + "f" * 39,
        " " + "g" * 40,
    ],
)
def test_auth_header_rejects_malformed_credentials(value):
    with pytest.raises((TypeError, ValueError)):
        kogs.generate_auth_header(value)


def test_link_frequency_uses_exact_spacecraft_and_direction(monkeypatch, tmp_path):
    config = tmp_path / "ctrl-config" / "v2" / "spacecrafts"
    config.mkdir(parents=True)
    (config / "TESTSAT.yml").write_text(
        "links:\n  selected:\n    direction: down\n    frequency: 2269750000\n"
    )
    monkeypatch.setattr(kogs, "_PROJECT_ROOT", tmp_path)

    assert kogs.get_link_frequency("TESTSAT", "selected", "down") == 2_269_750_000
    with pytest.raises(ValueError, match="not a uplink"):
        kogs.get_link_frequency("TESTSAT", "selected", "up")


def test_link_frequency_requires_matching_control_config(monkeypatch, tmp_path):
    monkeypatch.setattr(kogs, "_PROJECT_ROOT", tmp_path)

    with pytest.raises(FileNotFoundError, match="No config found"):
        kogs.get_link_frequency("MISSING", "downlink", "down")


def test_ephemeris_identity_uses_and_cross_checks_tle_and_omm():
    tle = (
        "TESTSAT\n"
        "1 60543U 24149A   24240.50000000  .00000000  00000-0  00000-0 0  9999\n"
        "2 60543  51.6400 100.0000 0005000 100.0000 260.0000 15.50000000123456\n"
    )
    data = kogs.parse_ephemeris(
        {
            "inline": {
                "tle": tle,
                "omm": "OBJECT_ID = 2024-149A\nNORAD_CAT_ID = 60543\n",
            }
        }
    )

    assert kogs.parse_ephemeris_identity(data) == ("2024-149A", "60543")
