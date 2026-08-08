"""Credential-required release gate for the real ADX/KOGS integration."""

import os

import pytest

from dart.contracts import DatasetQuery
from dart.services.orchestrator.acquisition import AdxKogsProvider

pytestmark = pytest.mark.live_integration

REQUIRED_ENVIRONMENT = (
    "AZURE_ADX_CLUSTER_ENDPOINT",
    "AZURE_CLIENT_ID",
    "AZURE_CLIENT_SECRET",
    "AZURE_TENANT_ID",
    "KOGS_API_KEY",
    "DART_TEST_CONTACT_ID",
)
LEGACY_NOMINAL_CARRIER_FREQUENCY_HZ = 2.2e9


def test_real_adx_kogs_query_produces_hashed_tdm():
    missing = [name for name in REQUIRED_ENVIRONMENT if not os.getenv(name)]
    assert not missing, "missing required live integration variables: " + ", ".join(missing)
    carrier_frequency_hz = float(
        os.getenv("DART_TEST_CARRIER_FREQUENCY_HZ", LEGACY_NOMINAL_CARRIER_FREQUENCY_HZ)
    )
    assert carrier_frequency_hz > 0.0

    packet = AdxKogsProvider().fetch(
        DatasetQuery(contact_ids=[os.environ["DART_TEST_CONTACT_ID"]]),
        carrier_frequency_hz,
    )

    assert packet.presented_count > 0
    assert packet.presented_count == len(packet.measurements)
    assert packet.tdm.content.startswith("CCSDS_TDM_VERS = ")
    assert packet.provenance == {
        "telemetry": "Azure Data Explorer",
        "metadata": "KOGS",
    }
