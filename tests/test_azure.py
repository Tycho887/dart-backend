"""Tests for the env-backed ADX logic in ``dart.io.azure``.

Env sanity + client/query-routing tests run offline; the ``test_live_*``
tests hit the real ADX cluster with the real credentials and are part of
the default suite (no marker gating).

Secrets are read from ``DART_SECRETS_ENV`` (default
``/opt/dart/secrets/test.env``); when the file or required keys are missing,
the env and live tests skip with a clear message.
"""

import datetime
import os

import pandas as pd
import polars as pl
import pytest
from azure.kusto.data import ClientRequestProperties, KustoClient
from dotenv import dotenv_values, load_dotenv

from dart.io import azure
from dart.io.azure import TrackingContext
from dart.io.kogs import get_contact, parse_reservation
from dart.io.auth import create_api_auth

SECRETS_ENV = os.environ.get("DART_SECRETS_ENV", "/opt/dart/secrets/test.env")

# Keys the ADX client reads from the environment (HTTP_PROXY is optional).
REQUIRED_KEYS = {
    "AZURE_ADX_CLUSTER_ENDPOINT",
    "AZURE_CLIENT_ID",
    "AZURE_CLIENT_SECRET",
    "AZURE_TENANT_ID",
}
LIVE_TEST_TIMEOUT_SECONDS = 45


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
        pytest.skip(f"{SECRETS_ENV} is missing required keys: {sorted(missing)}")
    return values


@pytest.fixture
def azure_with_env(secrets_env):
    """Load the secrets file into the process env and restore it on teardown.

    ``dart.io.azure.client_from_env`` reads ``os.environ`` lazily, so no module
    reload is needed."""
    saved = {k: os.environ.get(k) for k in secrets_env}
    try:
        load_dotenv(SECRETS_ENV, override=True)
        yield azure
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


class RedactedAuth(str):
    """String-compatible auth value that pytest cannot leak via ``repr``."""

    def __repr__(self):
        return "<redacted KOGS auth>"


@pytest.fixture(scope="module")
def kogs_auth(secrets_env) -> str:
    key = secrets_env.get("KOGS_API_KEY")
    if not key or not str(key).strip():
        pytest.skip("KOGS_API_KEY missing from the secrets file")
    return RedactedAuth(create_api_auth(key))


@pytest.fixture(scope="module")
def test_contact_id(secrets_env) -> str:
    cid = secrets_env.get("DART_TEST_CONTACT_ID")
    if not cid or not str(cid).strip():
        pytest.skip("DART_TEST_CONTACT_ID missing from the secrets file")
    return str(cid)


@pytest.fixture(scope="module")
def live_reservation(kogs_auth, test_contact_id):
    reservation = parse_reservation(get_contact(kogs_auth, test_contact_id)["contact"])
    assert reservation.spacecraft_id, "test contact carries no spacecraft_id"
    assert reservation.start_time, "test contact carries no start_time"
    assert reservation.end_time, "test contact carries no end_time"
    return reservation


def _reservation_window(reservation) -> tuple[str, str]:
    """Pad the scheduled contact to retain setup/teardown telemetry."""
    start = pd.Timestamp(reservation.start_time) - pd.Timedelta(minutes=10)
    end = pd.Timestamp(reservation.end_time) + pd.Timedelta(minutes=10)
    return start.isoformat(), end.isoformat()


def _live_query_properties() -> ClientRequestProperties:
    return azure._query_properties(azure.ADX_QUERY_TIMEOUT_SECONDS)


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


def test_client_from_env_constructs_from_env(azure_with_env, secrets_env):
    """client_from_env() plumbs the environment credentials into a KustoClient
    (construction is lazy — no network is touched here)."""
    client = azure_with_env.client_from_env()
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
        self.properties = []
        self.response = FakeResponse()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute_query(self, db, query, properties=None):
        self.calls.append((db, query))
        self.properties.append(properties)
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
    monkeypatch.setattr(azure, "client_from_env", lambda: fake)
    monkeypatch.setattr(
        azure,
        "dataframe_from_result_table",
        lambda table: fake_dataframe_from_result_table(table, fake),
    )
    return fake


def test_fetch_contact_columns_is_raw_bounded_and_projected(monkeypatch):
    fake = _mock_client(monkeypatch)

    frame = azure.fetch_contact_columns(
        "contact-1",
        datetime.datetime(2026, 1, 1, tzinfo=datetime.UTC),
        datetime.datetime(2026, 1, 1, 1, tzinfo=datetime.UTC),
        ("timestamp", "contact_id", "antenna_name"),
        order_by="timestamp",
    )

    assert len(frame) == 2
    query = fake.calls[0][1]
    assert "contact_id == 'contact-1'" in query
    assert "timestamp between (datetime(2026-01-01T00:00:00.000000Z)" in query
    assert "project timestamp, contact_id, antenna_name" in query
    assert "carrierLockState" not in query
    assert "ebN0" not in query


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
    assert (
        "timestamp between (datetime(2024-01-01T00:00:00Z) .. datetime(2024-01-01T01:00:00Z))"
        in query
    )
    assert "antenna1_position_elevation >= 5.0" in query
    assert "lr1_receiver1_ebN0 >= 3.0" in query
    assert "lr1_receiver1_actualCarrierFrequencyOffset >= 10.0" in query
    assert "lr1_receiver1_actualCarrierFrequencyOffset <= 2000.0" in query
    assert "lr1_receiver1_carrierLockState == 'Locked'" in query
    assert query.rstrip().endswith("order by timestamp asc")
    timeout = fake.properties[0].get_option(
        ClientRequestProperties.request_timeout_option_name, None
    )
    assert timeout == datetime.timedelta(seconds=azure.ADX_QUERY_TIMEOUT_SECONDS)

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


def test_fetch_tracking_data_contact_time_window(monkeypatch):
    fake = _mock_client(monkeypatch)
    ctx = TrackingContext(
        spacecraft_uuid=None,
        start_time_iso="2024-01-01T00:00:00Z",
        end_time_iso="2024-01-01T01:00:00Z",
        contact_uuid_list="c1",
        ephemeris_id=None,
        mode="sgp4",
        lock_requirement=False,
    )

    azure.fetch_tracking_data(ctx)

    _, query = fake.calls[0]
    assert "contact_id in ('c1')" in query
    assert (
        "timestamp between (datetime(2024-01-01T00:00:00Z) .. "
        "datetime(2024-01-01T01:00:00Z))" in query
    )


def test_fetch_tracking_data_contact_partial_time_window_fails(monkeypatch):
    _mock_client(monkeypatch)
    ctx = TrackingContext(
        spacecraft_uuid=None,
        start_time_iso="2024-01-01T00:00:00Z",
        end_time_iso=None,
        contact_uuid_list="c1",
        ephemeris_id=None,
        mode="sgp4",
        lock_requirement=False,
    )

    with pytest.raises(ValueError, match="require both"):
        azure.fetch_tracking_data(ctx)


def test_fetch_tracking_data_rejects_nonpositive_timeout(monkeypatch):
    _mock_client(monkeypatch)
    ctx = TrackingContext(
        spacecraft_uuid=None,
        start_time_iso=None,
        end_time_iso=None,
        contact_uuid_list="c1",
        ephemeris_id=None,
        mode="sgp4",
        lock_requirement=False,
    )

    with pytest.raises(ValueError, match="timeout_seconds must be positive"):
        azure.fetch_tracking_data(ctx, timeout_seconds=0)


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
        "minDoppler": 0,  # skipped -> default 1.0 kept
        "useQmc": False,  # bool -> kept
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


@pytest.mark.timeout(LIVE_TEST_TIMEOUT_SECONDS)
def test_live_adx_connectivity(azure_with_env):
    """Credentials + cluster reachability end to end."""
    client = azure_with_env.client_from_env()
    response = client.execute("telemetry", ".show version", _live_query_properties())
    assert response.primary_results
    result_table = response.primary_results[0]
    assert len(result_table) > 0
    assert result_table.rows


@pytest.mark.timeout(LIVE_TEST_TIMEOUT_SECONDS)
def test_live_fetch_case2_real_contacts(
    azure_with_env, test_contact_id, live_reservation
):
    """fetch_tracking_data routed on a real bounded contact, executed against the
    real cluster, converted to a polars DataFrame with the projected schema.
    An empty result set is valid — the filters may exclude every row."""
    start_time, end_time = _reservation_window(live_reservation)

    ctx = TrackingContext(
        spacecraft_uuid=None,
        start_time_iso=start_time,
        end_time_iso=end_time,
        contact_uuid_list=test_contact_id,
        ephemeris_id=None,
        mode="sgp4",
        lock_requirement=False,
    )

    result = azure_with_env.fetch_tracking_data(ctx)

    assert isinstance(result, pl.DataFrame)
    assert PROJECTED_COLUMNS <= set(result.columns)


@pytest.mark.timeout(LIVE_TEST_TIMEOUT_SECONDS)
def test_live_fetch_case1_real_spacecraft(azure_with_env, live_reservation):
    """fetch_tracking_data routed on a real spacecraft + epoch window,
    executed against the real cluster. Empty result set is valid."""
    start_time, end_time = _reservation_window(live_reservation)

    ctx = TrackingContext(
        spacecraft_uuid=live_reservation.spacecraft_id,
        start_time_iso=start_time,
        end_time_iso=end_time,
        contact_uuid_list=None,
        ephemeris_id=None,
        mode="sgp4",
        lock_requirement=False,
    )

    result = azure_with_env.fetch_tracking_data(ctx)

    assert isinstance(result, pl.DataFrame)
    assert PROJECTED_COLUMNS <= set(result.columns)
