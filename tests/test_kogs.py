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

    assert calls == [
        ("https://mgmt.kogs.api.ksat.no/24.08/ephemeris/eph-1", 6.5)
    ]


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
