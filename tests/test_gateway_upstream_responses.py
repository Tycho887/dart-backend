"""Gateway behavior when an upstream response cannot be normalized."""

from __future__ import annotations

import asyncio

import httpx
import pytest
from pydantic import BaseModel

from dart.services.http import ProblemType
from dart.services.orchestrator import api as gateway_api


class _KogsPayloadClient:
    def __init__(self, payload: object) -> None:
        self.payload = payload

    def get_json(self, _path: str) -> object:
        return self.payload


class _MalformedDatasetPacket(BaseModel):
    raw_count: str = "not-a-count"


class _MalformedDatasetProvider:
    def fetch(self, _query: object, _frequency_hz: float) -> _MalformedDatasetPacket:
        return _MalformedDatasetPacket()


class _ConfigurationDatasetProvider:
    def fetch(self, _query: object, _frequency_hz: float) -> object:
        raise gateway_api.AdxConfigurationError("AZURE_CLIENT_ID is not configured")


async def _request(
    method: str,
    path: str,
    *,
    payload: object | None = None,
) -> httpx.Response:
    transport = httpx.ASGITransport(app=gateway_api.app, raise_app_exceptions=False)
    headers = {"Authorization": "Bearer orchestrator-test-token"}
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        return await client.request(method, path, json=payload, headers=headers)


def _assert_upstream_problem(response: httpx.Response, instance: str) -> None:
    assert response.status_code == 502
    assert response.headers["content-type"].startswith("application/problem+json")
    assert response.json() == {
        "type": ProblemType.UPSTREAM.value,
        "title": "Upstream service failure",
        "status": 502,
        "detail": "An upstream service failed.",
        "instance": instance,
    }


def _assert_configuration_problem(response: httpx.Response, instance: str) -> None:
    assert response.status_code == 503
    assert response.headers["content-type"].startswith("application/problem+json")
    assert response.json() == {
        "type": ProblemType.CONFIGURATION.value,
        "title": "Service unavailable",
        "status": 503,
        "detail": "KOGS_API_KEY is not configured",
        "instance": instance,
    }


@pytest.mark.parametrize("value", ["not-number", "NaN", "Infinity", "-Infinity"])
def test_contact_metadata_maps_invalid_kogs_number_to_upstream_problem(
    monkeypatch, value: str
) -> None:
    monkeypatch.setenv("DART_ORCHESTRATOR_BEARER_TOKEN", "orchestrator-test-token")
    monkeypatch.setenv("DART_AUTO_MIGRATE", "false")
    monkeypatch.setattr(
        gateway_api,
        "KogsClient",
        lambda: _KogsPayloadClient({"contact": {"setup_duration": value}}),
    )

    response = asyncio.run(_request("GET", "/v0/metadata/contact/contact-1"))

    _assert_upstream_problem(response, "/v0/metadata/contact/contact-1")


@pytest.mark.parametrize(
    "path",
    ["/v0/metadata/contact/contact-1", "/v0/metadata/ephemeris/ephemeris-1"],
)
def test_metadata_502_does_not_reflect_hostile_kogs_values(monkeypatch, path: str) -> None:
    hostile_value = "hostile-kogs-value-" + ("x" * 4_000)
    if "contact" in path:
        payload = {"contact": {"setup_duration": hostile_value}}
    else:
        payload = {"inline": hostile_value}
    monkeypatch.setenv("DART_ORCHESTRATOR_BEARER_TOKEN", "orchestrator-test-token")
    monkeypatch.setenv("DART_AUTO_MIGRATE", "false")
    monkeypatch.setattr(gateway_api, "KogsClient", lambda: _KogsPayloadClient(payload))

    response = asyncio.run(_request("GET", path))

    _assert_upstream_problem(response, path)
    assert hostile_value not in response.text


def test_ephemeris_metadata_maps_invalid_kogs_structure_to_upstream_problem(monkeypatch) -> None:
    monkeypatch.setenv("DART_ORCHESTRATOR_BEARER_TOKEN", "orchestrator-test-token")
    monkeypatch.setenv("DART_AUTO_MIGRATE", "false")
    monkeypatch.setattr(
        gateway_api,
        "KogsClient",
        lambda: _KogsPayloadClient({"inline": "not-an-object"}),
    )

    response = asyncio.run(_request("GET", "/v0/metadata/ephemeris/ephemeris-1"))

    assert response.status_code == 502
    assert response.headers["content-type"].startswith("application/problem+json")
    body = response.json()
    assert body["type"] == ProblemType.UPSTREAM.value
    assert body["instance"] == "/v0/metadata/ephemeris/ephemeris-1"


def test_dataset_query_maps_invalid_provider_packet_to_upstream_problem(monkeypatch) -> None:
    monkeypatch.setenv("DART_ORCHESTRATOR_BEARER_TOKEN", "orchestrator-test-token")
    monkeypatch.setenv("DART_AUTO_MIGRATE", "false")
    monkeypatch.setattr(gateway_api, "AdxKogsProvider", lambda: _MalformedDatasetProvider())

    response = asyncio.run(
        _request(
            "POST",
            "/v0/datasets/query",
            payload={
                "query": {"contact_ids": ["contact-1"]},
                "nominal_carrier_frequency_hz": 2.2e9,
            },
        )
    )

    assert response.status_code == 502
    assert response.headers["content-type"].startswith("application/problem+json")
    body = response.json()
    assert body["type"] == ProblemType.UPSTREAM.value
    assert body["instance"] == "/v0/datasets/query"


def test_dataset_query_maps_provider_configuration_error_to_service_unavailable(
    monkeypatch,
) -> None:
    monkeypatch.setenv("DART_ORCHESTRATOR_BEARER_TOKEN", "orchestrator-test-token")
    monkeypatch.setenv("DART_AUTO_MIGRATE", "false")
    monkeypatch.setattr(gateway_api, "AdxKogsProvider", lambda: _ConfigurationDatasetProvider())

    response = asyncio.run(
        _request(
            "POST",
            "/v0/datasets/query",
            payload={
                "query": {"contact_ids": ["contact-1"]},
                "nominal_carrier_frequency_hz": 2.2e9,
            },
        )
    )

    assert response.status_code == 503
    assert response.headers["content-type"].startswith("application/problem+json")
    assert response.json()["type"] == ProblemType.CONFIGURATION.value
    assert response.json()["instance"] == "/v0/datasets/query"


@pytest.mark.parametrize(
    "path",
    ["/v0/metadata/contact/contact-1", "/v0/metadata/ephemeris/ephemeris-1"],
)
def test_kogs_configuration_error_maps_to_service_unavailable(monkeypatch, path: str) -> None:
    monkeypatch.setenv("DART_ORCHESTRATOR_BEARER_TOKEN", "orchestrator-test-token")
    monkeypatch.setenv("DART_AUTO_MIGRATE", "false")

    def unavailable_client() -> object:
        raise gateway_api.KogsConfigurationError("KOGS_API_KEY is not configured")

    monkeypatch.setattr(gateway_api, "KogsClient", unavailable_client)

    response = asyncio.run(_request("GET", path))

    _assert_configuration_problem(response, path)
