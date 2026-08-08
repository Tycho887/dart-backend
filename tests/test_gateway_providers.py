from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

import pytest
from azure.kusto.data.exceptions import KustoError, KustoUnsupportedApiError

from dart.contracts import Cartesian3, DatasetQuery
from dart.services.orchestrator import acquisition as providers


class FakeBuilder:
    @staticmethod
    def with_aad_application_key_authentication(*_args: str) -> object:
        return object()


class FakeKustoClient:
    def __init__(self, response: object | Exception) -> None:
        self.response = response

    def __enter__(self) -> FakeKustoClient:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def execute_query(self, *_args: object, **_kwargs: object) -> object:
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


class FakeKogsClient:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def station_position(self, system_id: str) -> tuple[str, Cartesian3]:
        self.events.append(f"kogs:{system_id}")
        return system_id, Cartesian3(x=1.0, y=2.0, z=3.0)


class EventHeartbeat:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def __call__(self) -> None:
        self.events.append("heartbeat")


def _row(system_id: str, sequence: int) -> dict[str, Any]:
    return {
        "timestamp": datetime(2026, 8, 8, 0, sequence, tzinfo=UTC),
        "contact_id": "contact-1",
        "spacecraft_id": "spacecraft-1",
        "system_id": system_id,
        "antenna1_tracking_epochOffset": 0.0,
        "antenna1_position_azimuth": 10.0,
        "antenna1_position_elevation": 20.0,
        "lr1_receiver1_ebN0": 5.0,
        "lr1_receiver1_actualCarrierFrequencyOffset": 100.0,
    }


def _configure_provider(
    monkeypatch: pytest.MonkeyPatch,
    response: object | Exception,
) -> None:
    monkeypatch.setenv("AZURE_ADX_CLUSTER_ENDPOINT", "https://adx.example.test")
    monkeypatch.setenv("AZURE_CLIENT_ID", "client")
    monkeypatch.setenv("AZURE_CLIENT_SECRET", "secret")
    monkeypatch.setenv("AZURE_TENANT_ID", "tenant")
    monkeypatch.setattr(providers, "KustoConnectionStringBuilder", FakeBuilder)
    monkeypatch.setattr(providers, "KustoClient", lambda _builder: FakeKustoClient(response))


@pytest.mark.parametrize("failure", [KustoError(), KustoUnsupportedApiError()])
def test_provider_translates_base_kusto_failures(
    monkeypatch: pytest.MonkeyPatch,
    failure: Exception,
) -> None:
    _configure_provider(monkeypatch, failure)

    with pytest.raises(providers.AdxServiceError, match="protocol"):
        providers.AdxKogsProvider().fetch(DatasetQuery(contact_ids=["contact-1"]), 2.2e9)


@pytest.mark.parametrize(
    "response",
    [
        object(),
        SimpleNamespace(primary_results=[]),
        SimpleNamespace(primary_results=[{}]),
    ],
)
def test_provider_rejects_invalid_primary_result_shapes(
    monkeypatch: pytest.MonkeyPatch,
    response: object,
) -> None:
    _configure_provider(monkeypatch, response)

    with pytest.raises(providers.AdxServiceError, match="primary result table"):
        providers.AdxKogsProvider().fetch(DatasetQuery(contact_ids=["contact-1"]), 2.2e9)


def test_worker_fetch_heartbeats_between_adx_and_distinct_kogs_lookups(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = SimpleNamespace(
        primary_results=[[_row("station-a", 0), _row("station-b", 1), _row("station-a", 2)]]
    )
    _configure_provider(monkeypatch, response)
    events: list[str] = []
    monkeypatch.setattr(providers, "KogsClient", lambda: FakeKogsClient(events))

    packet = providers.AdxKogsProvider().fetch_with_heartbeat(
        DatasetQuery(contact_ids=["contact-1"]),
        2.2e9,
        EventHeartbeat(events),
    )

    assert packet.presented_count == 3
    assert events == ["heartbeat", "kogs:station-a", "heartbeat", "kogs:station-b"]
