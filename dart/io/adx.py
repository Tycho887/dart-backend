"""Azure Data Explorer access for raw and canonical tracking measurements."""

from __future__ import annotations

import os
import re
from datetime import UTC, datetime, timedelta

import polars as pl
from azure.kusto.data import (
    ClientRequestProperties,
    KustoClient,
    KustoConnectionStringBuilder,
)
from azure.kusto.data.helpers import dataframe_from_result_table

from .contact import ContactMetadata
from .measurement import MEASUREMENT_COLUMNS, canonical_measurements

ADX_QUERY_TIMEOUT_SECONDS = 30.0

_ADX_MEASUREMENT_COLUMNS = (
    "timestamp",
    "contact_id",
    "spacecraft_id",
    "system_id",
    "antenna_name",
    "antenna1_tracking_epochOffset",
    "antenna1_position_azimuth",
    "antenna1_position_elevation",
    "lr1_receiver1_carrierLockState",
    "lr1_receiver1_ebN0",
    "lr1_receiver1_actualCarrierFrequencyOffset",
)
_CANONICAL_NAMES = dict(zip(_ADX_MEASUREMENT_COLUMNS, MEASUREMENT_COLUMNS, strict=True))
_KUSTO_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_CONTACT_IDENTIFIER = re.compile(r"^[A-Za-z0-9_-]+$")


def get_client(
    endpoint: str,
    client_id: str,
    client_secret: str,
    tenant_id: str,
    *,
    proxy: str = "",
) -> KustoClient:
    """Construct an ADX client from explicit credentials."""

    if not all((endpoint, client_id, client_secret, tenant_id)):
        raise ValueError("ADX endpoint and credentials are required")
    connection = KustoConnectionStringBuilder.with_aad_application_key_authentication(
        connection_string=endpoint,
        aad_app_id=client_id,
        app_key=client_secret,
        authority_id=tenant_id,
    )
    client = KustoClient(connection)
    if proxy:
        client.set_proxy(proxy)
    return client


def client_from_env() -> KustoClient:
    """Construct an ADX client from the standard environment variables."""

    return get_client(
        endpoint=os.getenv("AZURE_ADX_CLUSTER_ENDPOINT", ""),
        client_id=os.getenv("AZURE_CLIENT_ID", ""),
        client_secret=os.getenv("AZURE_CLIENT_SECRET", ""),
        tenant_id=os.getenv("AZURE_TENANT_ID", ""),
        proxy=os.getenv("HTTP_PROXY", ""),
    )


def _properties(timeout_seconds: float) -> ClientRequestProperties:
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    properties = ClientRequestProperties()
    properties.set_option(
        ClientRequestProperties.request_timeout_option_name,
        timedelta(seconds=timeout_seconds),
    )
    return properties


def _kusto_datetime(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("ADX bounds must be timezone-aware")
    return (
        value.astimezone(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")
    )


def _query(client: KustoClient, query: str, timeout_seconds: float) -> pl.DataFrame:
    response = client.execute_query("telemetry", query, _properties(timeout_seconds))
    raw = dataframe_from_result_table(response.primary_results[0])
    return pl.from_pandas(raw)


def fetch_columns(
    client: KustoClient,
    contact_id: str,
    start: datetime,
    stop: datetime,
    columns: tuple[str, ...],
    *,
    order_by: str,
    contact_column: str = "contact_id",
    timeout_seconds: float = ADX_QUERY_TIMEOUT_SECONDS,
) -> pl.DataFrame:
    """Fetch selected ADX columns for one contact and bounded UTC interval."""

    if not _CONTACT_IDENTIFIER.fullmatch(contact_id):
        raise ValueError(f"invalid contact identifier {contact_id!r}")
    selected = tuple(dict.fromkeys(columns))
    if not selected or any(not _KUSTO_IDENTIFIER.fullmatch(name) for name in selected):
        raise ValueError("ADX columns must be non-empty Kusto identifiers")
    if order_by not in selected or contact_column not in selected:
        raise ValueError("order_by and contact_column must be selected ADX columns")
    if stop <= start:
        raise ValueError("ADX stop must be later than start")
    query = (
        "contacts\n"
        f"| where {contact_column} == '{contact_id}'\n"
        f"| where {order_by} between (datetime({_kusto_datetime(start)}) .. "
        f"datetime({_kusto_datetime(stop)}))\n"
        f"| project {', '.join(selected)}\n"
        f"| order by {order_by} asc"
    )
    frame = _query(client, query, timeout_seconds)
    return frame.sort(order_by) if order_by in frame.columns else frame


def fetch_measurements(
    client: KustoClient,
    contact: ContactMetadata,
    *,
    timeout_seconds: float = ADX_QUERY_TIMEOUT_SECONDS,
) -> pl.DataFrame:
    """Return one pass as a canonical, unfiltered measurement frame."""

    frame = fetch_columns(
        client,
        contact.contact_id,
        contact.start,
        contact.stop,
        _ADX_MEASUREMENT_COLUMNS,
        order_by="timestamp",
        timeout_seconds=timeout_seconds,
    )
    missing = set(_ADX_MEASUREMENT_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(
            f"ADX response is missing columns: {', '.join(sorted(missing))}"
        )
    return canonical_measurements(frame.rename(_CANONICAL_NAMES))
