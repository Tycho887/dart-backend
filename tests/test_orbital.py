import pytest
import requests

from dart.io.orbital import OrbitalContract, OrbitalError, write_offset


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


def test_refuses_unconfirmed_contract(monkeypatch):
    monkeypatch.setattr(requests, "post", lambda *args, **kwargs: pytest.fail())
    with pytest.raises(ValueError, match="not confirmed"):
        write_offset(contract(absolute_semantics_confirmed=False), "command-1", 0.125)


def test_write_offset_validates_correlated_acknowledgement(monkeypatch):
    calls = []

    def post(url, **kwargs):
        calls.append((url, kwargs))
        return Response({"status": "accepted", "command_id": "command-1"})

    monkeypatch.setattr(requests, "post", post)
    acknowledgement = write_offset(contract(), "command-1", 0.125)

    assert acknowledgement.accepted
    assert calls[0][0] == "http://orbital.test/confirmed/offset/path"
    assert calls[0][1]["data"] == "-0.125"


def test_write_offset_rejects_uncorrelated_acknowledgement(monkeypatch):
    monkeypatch.setattr(
        requests,
        "post",
        lambda *args, **kwargs: Response({"status": "accepted", "command_id": "old"}),
    )
    with pytest.raises(OrbitalError, match="not correlated"):
        write_offset(contract(), "command-1", 0.125)
