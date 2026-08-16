"""Tests for the env-backed ADX logic in ``dart.io.azure``.

Env sanity + client/query-routing tests run offline; the ``test_live_*``
tests hit the real ADX cluster with the real credentials and are part of
the default suite (no marker gating).

Secrets are read from ``DART_SECRETS_ENV`` (default
``/opt/dart/secrets/test.env``); when the file is missing, the env and live
tests skip with a clear message.
"""

import datetime
import importlib
import os

import pandas as pd
import polars as pl
import pytest
from azure.kusto.data import KustoClient
from azure.kusto.data.helpers import dataframe_from_result_table
from dotenv import dotenv_values, load_dotenv

import dart.io.azure as azure
from dart.io.azure import TrackingContext

SECRETS_ENV = os.environ.get("DART_SECRETS_ENV", "/opt/dart/secrets/test.env")

# Keys the module reads at import time (HTTP_PROXY is optional and tolerated empty).
REQUIRED_KEYS = {
    "AZURE_ADX_CLUSTER_ENDPOINT",
    "AZURE_CLIENT_ID",
    "AZURE_CLIENT_SECRET",
    "AZURE_TENANT_ID",
}
ALL_ENV_KEYS = REQUIRED_KEYS | {"HTTP_PROXY"}

# Columns projected by the KQL query in fetch_tracking_data.
PROJECTED_COLUMNS = {
    "timestamp",
    "contact_id",
    "antenna_name",
    "spacecraft_id",
    "system_id",
    "antenna1_tracking_epochOffset",
    "antenna1_position_azimuth",
    "antenna1_position_elevation",
    "lr1_receiver1_carrierLockState",
    "lr1_receiver1_ebN0",
    "lr1_receiver1_actualCarrierFrequencyOffset",
}


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def secrets_env() -> dict:
    """The parsed secrets file; skips the whole group when it is absent."""
    if not os.path.exists(SECRETS_ENV):
        pytest.skip(
            f"secrets file {SECRETS_ENV} not found; "
            "point DART_SECRETS_ENV at it to run the env + live ADX tests"
        )
    values = dotenv_values(SECRETS_ENV)
    missing = REQUIRED_KEYS - {k for k, v in values.items() if v and str(v).strip()}
    if missing:
        pytest.fail(f"{SECRETS_ENV} is missing required keys: {sorted(missing)}")
    return values


@pytest.fixture
def azure_with_env(secrets_env):
    """Load the secrets file into the process env, reload ``dart.io.azure``
    so its import-time env bindings pick them up, then restore + reload on
    teardown so the module state seen by the rest of the suite is unchanged."""
    saved = {k: os.environ.get(k) for k in secrets_env}
    try:
        load_dotenv(SECRETS_ENV, override=True)
        yield importlib.reload(azure)
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        importlib.reload(azure)


# ---------------------------------------------------------------------------
# env sanity (offline)
# ---------------------------------------------------------------------------

def test_secrets_file_has_required_keys(secrets_env):
    for key in REQUIRED_KEYS:
        value = secrets_env.get(key)
        assert value is not None and str(value).strip(), (
            f"{key} is missing or empty in {SECRETS_ENV}"
        )


def test_secrets_file_endpoint_is_https(secrets_env):
    assert secrets_env["AZURE_ADX_CLUSTER_ENDPOINT"].startswith("https://")


def test_env_class_populated_from_secrets(azure_with_env, secrets_env):
    """The import-time plumbing: env class attributes == file values."""
    for key in ALL_ENV_KEYS:
        assert getattr(azure_with_env.env, key) == secrets_env.get(key, ""), key


def test_get_client_constructs_from_env(azure_with_env, secrets_env):
    """get_client() plumbs the file's credentials into a KustoClient
    (construction is lazy — no network is touched here)."""
    client = azure_with_env.get_client()
    assert isinstance(client, KustoClient)
    assert client._kcsb.data_source == secrets_env["AZURE_ADX_CLUSTER_ENDPOINT"]
    expected_proxy = secrets_env.get("HTTP_PROXY", "") or None
    assert client._proxy_url == expected_proxy


# ---------------------------------------------------------------------------
# query routing (offline, mocked client)
# ---------------------------------------------------------------------------

class FakeResponse:
    def __init__(self):
        self.primary_results = [object()]


class FakeKustoClient:
    """with-able stand-in that records (db, query) and returns a canned table."""

    def __init__(self):
        self.calls = []
        self.response = FakeResponse()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute_query(self, db, query):
        self.calls.append((db, query))
        return self.response


def fake_dataframe_from_result_table(primary_results, fake):
    assert primary_results is fake.response.primary_results[0]
    return pd.DataFrame(
        {
            "timestamp": [
                datetime.datetime(2024, 1, 1, 0, 0, 10),
                datetime.datetime(2024, 1, 1, 0, 0, 20),
            ],
            "contact_id": ["c1", "c1"],
            "antenna_name": ["Svalbard", "Troll"],
            "spacecraft_id": ["sc-1234", "sc-1234"],
            "system_id": ["sys-1", "sys-2"],
            "antenna1_tracking_epochOffset": [0.5, 0.6],
            "antenna1_position_azimuth": [10.0, 20.0],
            "antenna1_position_elevation": [30.0, 40.0],
            "lr1_receiver1_carrierLockState": ["Locked", "Locked"],
            "lr1_receiver1_ebN0": [8.0, 9.0],
            "lr1_receiver1_actualCarrierFrequencyOffset": [-1000.0, 2000.0],
        }
    )


def _mock_client(monkeypatch):
    fake = FakeKustoClient()
    monkeypatch.setattr(azure, "get_client", lambda: fake)
    monkeypatch.setattr(
        azure,
        "dataframe_from_result_table",
        lambda table: fake_dataframe_from_result_table(table, fake),
    )
    return fake


def test_fetch_tracking_data_case1_query(monkeypatch):
    fake = _mock_client(monkeypatch)
    ctx = TrackingContext(
        spacecraft_uuid="sc-1234",
        start_time_iso="2024-01-01T00:00:00Z",
        end_time_iso="2024-01-01T01:00:00Z",
        contact_uuid_list=None,
        ephemeris_id=None,
        mode="sgp4",
        lock_requirement=True,
        min_elevation=5.0,
        minimum_ebn0=3.0,
        min_doppler=10.0,
        max_doppler=2000.0,
    )

    result = azure.fetch_tracking_data(ctx)

    assert len(fake.calls) == 1
    db, query = fake.calls[0]
    assert db == "telemetry"
    assert "spacecraft_id == 'sc-1234'" in query
    assert "timestamp between (datetime(2024-01-01T00:00:00Z) .. datetime(2024-01-01T01:00:00Z))" in query
    assert "antenna1_position_elevation >= 5.0" in query
    assert "lr1_receiver1_ebN0 >= 3.0" in query
    assert "lr1_receiver1_actualCarrierFrequencyOffset >= 10.0" in query
    assert "lr1_receiver1_actualCarrierFrequencyOffset <= 2000.0" in query
    assert "lr1_receiver1_carrierLockState == 'Locked'" in query
    assert query.rstrip().endswith("order by timestamp asc")

    assert isinstance(result, pl.DataFrame)
    assert result["contact_id"].to_list() == ["c1", "c1"]
    assert result["system_id"].to_list() == ["sys-1", "sys-2"]


@pytest.mark.parametrize(
    "contact_values",
    (["c1", "c2", "c3"], "c1  c2 c3"),
)
def test_fetch_tracking_data_case2_query(monkeypatch, contact_values):
    fake = _mock_client(monkeypatch)
    ctx = TrackingContext.from_payload(
        {
            "contact_UUID_List": contact_values,
            "Mode": "sgp4",
            "lockRequirement": False,
        }
    )

    azure.fetch_tracking_data(ctx)

    assert len(fake.calls) == 1
    _, query = fake.calls[0]
    assert "contact_id in ('c1', 'c2', 'c3')" in query
    assert "spacecraft_id ==" not in query
    assert "carrierLockState == 'Locked'" not in query  # lock_requirement=False


def test_fetch_tracking_data_case3_raises():
    ctx = TrackingContext(
        spacecraft_uuid=None,
        start_time_iso=None,
        end_time_iso=None,
        contact_uuid_list=None,
        ephemeris_id=None,
        mode="sgp4",
        lock_requirement=False,
    )
    with pytest.raises(ValueError, match="Insufficient context"):
        azure.fetch_tracking_data(ctx)


# ---------------------------------------------------------------------------
# TrackingContext.from_payload (offline, schema normalization)
# ---------------------------------------------------------------------------

def test_tracking_context_from_payload_full():
    payload = {
        "spacecraft_UUID": "sc-1",
        "start_time": "2024-01-01T00:00:00Z",
        "end_time": "2024-01-01T01:00:00Z",
        "contact_UUID_List": ["c1", "c2"],
        "EphemerisID": "eph-1",
        "lockRequirement": True,
        "Mode": "sgp4",
        "minElevation": 5.0,
        "minimumEbN0": 2.5,
        "minDoppler": 10.0,
        "maxDoppler": 5000.0,
        "angleConstraint": 2.0,
        "penaltyWeight": 100.0,
        "qmcSamples": 50,
        "lossFunction": "soft_l1",
        "fScale": 300.0,
        "modelType": "linear",
        "criterion": "aic",
        "useQmc": True,
        "method": "trf",
    }

    ctx = TrackingContext.from_payload(payload)

    assert ctx.spacecraft_uuid == "sc-1"
    assert ctx.start_time_iso == "2024-01-01T00:00:00Z"
    assert ctx.end_time_iso == "2024-01-01T01:00:00Z"
    assert ctx.contact_uuid_list == "c1,c2"
    assert ctx.ephemeris_id == "eph-1"
    assert ctx.lock_requirement is True
    assert ctx.mode == "sgp4"
    assert ctx.min_elevation == 5.0
    assert ctx.minimum_ebn0 == 2.5
    assert ctx.min_doppler == 10.0
    assert ctx.max_doppler == 5000.0
    assert ctx.angle_constraint == 2.0
    assert ctx.penalty_weight == 100.0
    assert ctx.qmc_samples == 50
    assert ctx.loss_function == "soft_l1"
    assert ctx.f_scale == 300.0
    assert ctx.model_type == "linear"
    assert ctx.criterion == "aic"
    assert ctx.use_qmc is True
    assert ctx.method == "trf"


def test_tracking_context_from_payload_quirks():
    """Document existing normalization quirks: payload zeroes are skipped
    (dataclass defaults survive) but explicit ``False`` booleans are kept."""
    payload = {
        "spacecraft_UUID": "sc-1",
        "start_time": "2024-01-01T00:00:00Z",
        "end_time": "2024-01-01T01:00:00Z",
        "lockRequirement": False,
        "Mode": "sgp4",
        "minElevation": 0,  # skipped -> default 1.0 kept
        "minDoppler": 0,    # skipped -> default 1.0 kept
        "useQmc": False,    # bool -> kept
    }

    ctx = TrackingContext.from_payload(payload)

    assert ctx.contact_uuid_list is None
    assert ctx.lock_requirement is False
    assert ctx.min_elevation == 1.0
    assert ctx.min_doppler == 1.0
    assert ctx.use_qmc is False


# ---------------------------------------------------------------------------
# live tests — real ADX cluster with the real credentials (default suite)
# ---------------------------------------------------------------------------

def _probe_contact_ids(client) -> list[str]:
    response = client.execute(
        "telemetry", "contacts | where isnotempty(contact_id) | project contact_id | take 5"
    )
    df = dataframe_from_result_table(response.primary_results[0])
    return [str(value) for value in df["contact_id"].tolist()]


def test_live_adx_connectivity(azure_with_env):
    """Credentials + cluster reachability end to end."""
    client = azure_with_env.get_client()
    response = client.execute("telemetry", ".show version")
    assert response.primary_results
    result_table = response.primary_results[0]
    assert len(result_table) > 0
    assert result_table.rows


def test_live_fetch_case2_real_contacts(azure_with_env):
    """fetch_tracking_data routed on real contact IDs, executed against the
    real cluster, converted to a polars DataFrame with the projected schema.
    An empty result set is valid — the filters may exclude every row."""
    client = azure_with_env.get_client()
    contact_ids = _probe_contact_ids(client)
    assert contact_ids, "telemetry DB has no contacts to query"

    ctx = TrackingContext(
        spacecraft_uuid=None,
        start_time_iso=None,
        end_time_iso=None,
        contact_uuid_list=" ".join(contact_ids),
        ephemeris_id=None,
        mode="sgp4",
        lock_requirement=False,
    )

    result = azure_with_env.fetch_tracking_data(ctx)

    assert isinstance(result, pl.DataFrame)
    assert PROJECTED_COLUMNS <= set(result.columns)


def test_live_fetch_case1_real_spacecraft(azure_with_env):
    """fetch_tracking_data routed on a real spacecraft + epoch window,
    executed against the real cluster. Empty result set is valid."""
    client = azure_with_env.get_client()
    contact_ids = _probe_contact_ids(client)
    assert contact_ids, "telemetry DB has no contacts to query"
    ids_csv = ", ".join(f"'{cid}'" for cid in contact_ids)

    response = client.execute(
        "telemetry",
        f"contacts | where contact_id in ({ids_csv}) "
        "| where isnotempty(spacecraft_id) and isnotnull(timestamp) "
        "| project spacecraft_id, timestamp | take 1",
    )
    df = dataframe_from_result_table(response.primary_results[0])
    assert len(df) > 0, "no contact carries a timestamp to build the window from"
    spacecraft_uuid = str(df["spacecraft_id"].iloc[0])
    epoch = pd.Timestamp(df["timestamp"].iloc[0])
    if epoch.tzinfo is None:
        epoch = epoch.tz_localize("UTC")
    else:
        epoch = epoch.tz_convert("UTC")

    ctx = TrackingContext(
        spacecraft_uuid=spacecraft_uuid,
        start_time_iso=(epoch - pd.Timedelta(hours=1)).isoformat(),
        end_time_iso=(epoch + pd.Timedelta(hours=1)).isoformat(),
        contact_uuid_list=None,
        ephemeris_id=None,
        mode="sgp4",
        lock_requirement=False,
    )

    result = azure_with_env.fetch_tracking_data(ctx)

    assert isinstance(result, pl.DataFrame)
    assert PROJECTED_COLUMNS <= set(result.columns)
