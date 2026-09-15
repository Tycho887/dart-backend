"""Offline smoke check run with production dependencies during image builds."""

import io
from importlib import import_module
from types import SimpleNamespace
from unittest.mock import Mock

import polars as pl
from azure.kusto.data import KustoClient
from azure.kusto.data.response import KustoResultTable

from dart.io import adx


def main() -> None:
    for module in ("dart.service.worker", "dart.service.gateway", "dart.io.oem"):
        import_module(module)

    # Mock only the network response. Exercise the real SDK pandas conversion,
    # Polars Arrow conversion, and raw-artifact Parquet serialization.
    table = KustoResultTable(
        {
            "TableName": "PrimaryResult",
            "Columns": [
                {"ColumnName": "timestamp", "ColumnType": "datetime"},
                {"ColumnName": "contact_id", "ColumnType": "string"},
                {"ColumnName": "doppler_hz", "ColumnType": "real"},
            ],
            "Rows": [
                ["2026-01-01T00:00:00Z", "test-pass", 100.0],
                ["2026-01-01T00:00:01Z", "test-pass", None],
            ],
        }
    )
    client = Mock(spec=KustoClient)
    client.execute_query.return_value = SimpleNamespace(primary_results=[table])
    frame = adx._query(client, "synthetic response", timeout_seconds=30)
    assert frame["timestamp"].dt.epoch("us").to_list() == [
        1767225600000000,
        1767225601000000,
    ]
    timestamp_type = frame.schema["timestamp"]
    assert isinstance(timestamp_type, pl.Datetime)
    assert timestamp_type.time_zone == "UTC"
    assert frame["contact_id"].to_list() == ["test-pass", "test-pass"]
    assert frame["doppler_hz"].to_list() == [100.0, None]
    buffer = io.BytesIO()
    frame.write_parquet(buffer)
    buffer.seek(0)
    assert pl.read_parquet(buffer).equals(frame)
    print("Runtime imports, ADX conversion, and Parquet smoke check passed")


if __name__ == "__main__":
    main()
