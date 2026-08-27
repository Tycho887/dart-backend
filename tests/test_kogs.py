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

    result = kogs.get_contact("secret auth", "c1")

    assert result == {"contact": {"id": "c1"}}
    assert calls == [
        (
            "https://mgmt.kogs.api.ksat.no/24.08/contacts/c1",
            {"Authorization": "secret auth", "Accept": "application/json"},
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

    kogs.get_contact("secret auth", "c1", timeout_seconds=4.5)

    assert calls == [4.5]


def test_get_ephemeris_accepts_explicit_timeout(monkeypatch):
    calls = []
    monkeypatch.setattr(
        kogs.requests,
        "get",
        lambda url, *, headers, timeout: calls.append((url, timeout)) or FakeResponse(),
    )

    kogs.get_ephemeris("secret auth", "eph-1", timeout_seconds=6.5)

    assert calls == [
        ("https://mgmt.kogs.api.ksat.no/24.08/ephemeris/eph-1", 6.5)
    ]
