from datetime import UTC, datetime, timedelta

import pytest
import requests

from dart.controller.controller import OffsetWrite
from dart.controller.UKF import FilterIdentity
from dart.io.orbital import (
    OrbitalContract,
    OrbitalTransportError,
    RequestsOrbitalWriter,
)

NOW = datetime(2026, 8, 31, 10, tzinfo=UTC)


class Response:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


def contract(**overrides):
    values = {
        "base_url": "http://orbital.test/",
        "endpoint_path": "/confirmed/offset/path",
        "connect_timeout_s": 1,
        "read_timeout_s": 2,
        "sign": -1,
        "idempotency_header": "Idempotency-Key",
        "acknowledgement_field": "status",
        "acknowledgement_value": "accepted",
        "command_id_field": "command_id",
        "absolute_semantics_confirmed": True,
    }
    values.update(overrides)
    return OrbitalContract(**values)


def command():
    return OffsetWrite(
        "command-1",
        FilterIdentity("contact", "antenna", "ephemeris"),
        0.125,
        NOW,
        NOW + timedelta(seconds=10),
        "tracking",
    )


def test_refuses_unconfirmed_contract():
    with pytest.raises(ValueError, match="not confirmed"):
        RequestsOrbitalWriter(contract(absolute_semantics_confirmed=False))


def test_exact_path_encoding_timeout_and_acknowledgement(monkeypatch):
    calls = []

    def post(url, *, headers, data, timeout):
        calls.append((url, headers, data, timeout))
        return Response({"status": "accepted", "command_id": "command-1"})

    monkeypatch.setattr(requests, "post", post)

    acknowledgement = RequestsOrbitalWriter(contract()).write_offset(command())

    assert acknowledgement.accepted
    assert calls == [
        (
            "http://orbital.test/confirmed/offset/path",
            {
                "Accept": "application/json",
                "Content-Type": "text/plain",
                "Idempotency-Key": "command-1",
            },
            "-0.125",
            (1, 2),
        )
    ]


def test_rejects_uncorrelated_or_malformed_acknowledgement(monkeypatch):
    monkeypatch.setattr(
        requests,
        "post",
        lambda *args, **kwargs: Response(
            {"status": "accepted", "command_id": "old-command"}
        ),
    )
    writer = RequestsOrbitalWriter(contract())

    with pytest.raises(OrbitalTransportError, match="not correlated"):
        writer.write_offset(command())
