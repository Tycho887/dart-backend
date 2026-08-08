"""Credential-required release gate for the real ADX/KOGS integration."""

import os

import pytest
from azure.kusto.data import KustoClient, KustoConnectionStringBuilder

from dart.contracts import DatasetQuery
from dart.gateway.providers import ADX_DATABASE, AdxKogsProvider

pytestmark = pytest.mark.live_integration

REQUIRED_ENVIRONMENT = (
    "AZURE_ADX_CLUSTER_ENDPOINT",
    "AZURE_CLIENT_ID",
    "AZURE_CLIENT_SECRET",
    "AZURE_TENANT_ID",
    "KOGS_API_KEY",
)
LEGACY_NOMINAL_CARRIER_FREQUENCY_HZ = 2.2e9


def latest_qualifying_contact_id() -> str:
    builder = KustoConnectionStringBuilder.with_aad_application_key_authentication(
        os.environ["AZURE_ADX_CLUSTER_ENDPOINT"],
        os.environ["AZURE_CLIENT_ID"],
        os.environ["AZURE_CLIENT_SECRET"],
        os.environ["AZURE_TENANT_ID"],
    )
    query = """
contacts
| where antenna1_position_elevation >= 1.0
| where lr1_receiver1_ebN0 >= 0.0
| where lr1_receiver1_actualCarrierFrequencyOffset between (-100000.0 .. 100000.0)
| summarize system_count=dcount(system_id), latest_timestamp=max(timestamp) by contact_id
| where system_count == 1
| top 1 by latest_timestamp desc
| project contact_id
"""
    with KustoClient(builder) as client:
        response = client.execute_query(ADX_DATABASE, query)
    table = response.primary_results[0]
    assert len(table) > 0, "ADX telemetry has no qualifying contact"
    return str(table[0]["contact_id"])


def test_real_adx_kogs_query_produces_hashed_tdm():
    missing = [name for name in REQUIRED_ENVIRONMENT if not os.getenv(name)]
    assert not missing, "missing required live integration variables: " + ", ".join(missing)
    contact_id = os.getenv("DART_TEST_CONTACT_ID") or latest_qualifying_contact_id()
    carrier_frequency_hz = float(
        os.getenv("DART_TEST_CARRIER_FREQUENCY_HZ", LEGACY_NOMINAL_CARRIER_FREQUENCY_HZ)
    )
    assert carrier_frequency_hz > 0.0

    packet = AdxKogsProvider().fetch(
        DatasetQuery(contact_ids=[contact_id]),
        carrier_frequency_hz,
    )

    assert packet.presented_count > 0
    assert packet.presented_count == len(packet.measurements)
    assert packet.tdm.content.startswith("CCSDS_TDM_VERS = ")
    assert packet.provenance == {
        "telemetry": "Azure Data Explorer",
        "metadata": "KOGS",
    }
