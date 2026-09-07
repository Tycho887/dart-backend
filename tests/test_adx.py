from datetime import UTC, datetime

import pandas as pd

from dart.io import adx


class Result:
    primary_results = [object()]


class Client:
    def __init__(self):
        self.calls = []

    def execute_query(self, database, query, properties):
        self.calls.append((database, query, properties))
        return Result()


def raw_frame():
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-01-01T00:00:02Z", "2026-01-01T00:00:01Z"]),
            "contact_id": ["contact-1", "contact-1"],
            "spacecraft_id": ["spacecraft-1", "spacecraft-1"],
            "system_id": ["system-1", "system-1"],
            "antenna_name": ["SGS1", "SGS1"],
            "antenna1_tracking_epochOffset": [0.0, 0.0],
            "antenna1_position_azimuth": [20.0, 10.0],
            "antenna1_position_elevation": [40.0, 30.0],
            "lr1_receiver1_carrierLockState": ["Locked", "Locked"],
            "lr1_receiver1_ebN0": [12.0, 11.0],
            "lr1_receiver1_actualCarrierFrequencyOffset": [200.0, 100.0],
        }
    )


def contact_metadata():
    from test_io_load import metadata

    return metadata("contact-1", "2026-01-01T00:00:00Z")


def test_fetch_measurements_is_bounded_canonical_and_unfiltered(monkeypatch):
    client = Client()
    monkeypatch.setattr(adx, "dataframe_from_result_table", lambda _: raw_frame())

    frame = adx.fetch_measurements(client, contact_metadata())

    assert tuple(frame.columns) == adx.MEASUREMENT_COLUMNS
    assert frame["doppler_hz"].to_list() == [100.0, 200.0]
    query = client.calls[0][1]
    assert "contact_id == 'contact-1'" in query
    assert "between (datetime(2026-01-01T00:00:00.000000Z)" in query
    assert "carrierLockState ==" not in query
    assert "actualCarrierFrequencyOffset >=" not in query


def test_fetch_columns_rejects_unsafe_identifiers():
    client = Client()
    start = datetime(2026, 1, 1, tzinfo=UTC)

    try:
        adx.fetch_columns(
            client,
            "contact'; drop table contacts",
            start,
            start.replace(hour=1),
            ("timestamp", "contact_id"),
            order_by="timestamp",
        )
    except ValueError as error:
        assert "invalid contact identifier" in str(error)
    else:
        raise AssertionError("unsafe identifier was accepted")
