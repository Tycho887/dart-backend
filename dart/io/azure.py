"""ADX/Kusto telemetry client (ported from lib/IO/azure.py) plus the query
context model (ported from depr/load.py)."""
import os
import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import polars as pl
from azure.kusto.data import (
    ClientRequestProperties,
    KustoClient,
    KustoConnectionStringBuilder,
)
from azure.kusto.data.helpers import dataframe_from_result_table
from dotenv import load_dotenv

from dart.io.utils import _join_list, _safe_bool, _safe_str, setup_logger

logger = setup_logger()
load_dotenv()

ADX_QUERY_TIMEOUT_SECONDS = 30.0

class env:
    AZURE_ADX_CLUSTER_ENDPOINT: str = os.getenv("AZURE_ADX_CLUSTER_ENDPOINT", "")
    AZURE_CLIENT_ID: str = os.getenv("AZURE_CLIENT_ID", "")
    AZURE_CLIENT_SECRET: str = os.getenv("AZURE_CLIENT_SECRET", "")
    AZURE_TENANT_ID: str = os.getenv("AZURE_TENANT_ID", "")
    HTTP_PROXY: str = os.getenv("HTTP_PROXY", "")

def get_client() -> KustoClient:
    logger.info(
        f'Connecting to ADX cluster {env.AZURE_ADX_CLUSTER_ENDPOINT}, '
        f'client {env.AZURE_CLIENT_ID}, tenant {env.AZURE_TENANT_ID}'
    )
    kcsb = KustoConnectionStringBuilder.with_aad_application_key_authentication(
      connection_string=env.AZURE_ADX_CLUSTER_ENDPOINT,
      aad_app_id=env.AZURE_CLIENT_ID,
      app_key=env.AZURE_CLIENT_SECRET,
      authority_id=env.AZURE_TENANT_ID
    )
    client = KustoClient(kcsb)

    if env.HTTP_PROXY:
        client.set_proxy(env.HTTP_PROXY)
    return client

def _query_properties(timeout_seconds: float) -> ClientRequestProperties:
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    properties = ClientRequestProperties()
    properties.set_option(
        ClientRequestProperties.request_timeout_option_name,
        timedelta(seconds=timeout_seconds),
    )
    return properties

_KUSTO_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_CONTACT_IDENTIFIER = re.compile(r"^[A-Za-z0-9_-]+$")


def _kusto_datetime(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("ADX bounds must be timezone-aware")
    return (
        value.astimezone(UTC)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )


def fetch_contact_columns(
    contact_id: str,
    start_time: datetime,
    stop_time: datetime,
    columns: tuple[str, ...],
    *,
    order_by: str,
    contact_column: str = "contact_id",
    timeout_seconds: float = ADX_QUERY_TIMEOUT_SECONDS,
) -> pl.DataFrame:
    """Fetch selected raw telemetry for one contact and bounded UTC interval."""
    if not _CONTACT_IDENTIFIER.fullmatch(contact_id):
        raise ValueError(f"invalid contact identifier {contact_id!r}")
    selected = tuple(dict.fromkeys(columns))
    if not selected or any(not _KUSTO_IDENTIFIER.fullmatch(name) for name in selected):
        raise ValueError("ADX columns must be non-empty Kusto identifiers")
    if order_by not in selected or contact_column not in selected:
        raise ValueError("order_by and contact_column must be selected ADX columns")
    if stop_time <= start_time:
        raise ValueError("ADX stop_time must be later than start_time")
    start = _kusto_datetime(start_time)
    stop = _kusto_datetime(stop_time)
    query = (
        "contacts\n"
        f"| where {contact_column} == '{contact_id}'\n"
        f"| where {order_by} between (datetime({start}) .. datetime({stop}))\n"
        f"| project {', '.join(selected)}\n"
        f"| order by {order_by} asc"
    )
    with get_client() as client:
        response = client.execute_query(
            "telemetry", query, _query_properties(timeout_seconds)
        )
        raw = dataframe_from_result_table(response.primary_results[0])
    frame = pl.from_pandas(raw)
    return frame.sort(order_by) if order_by in frame.columns else frame


def fetch_tracking_data(
    ctx, *, timeout_seconds: float = ADX_QUERY_TIMEOUT_SECONDS
) -> pl.DataFrame:
    """
    Routes the database request based on the context object case,
    injects KQL-level filtering, and returns a unified Polars DataFrame.
    """
    prefix = "lr1_receiver1"

    print(ctx)

    # 1. Determine the Routing Case and Initial Dataset
    if ctx.spacecraft_uuid and ctx.start_time_iso and ctx.end_time_iso:
        # Case 1: Time-bounded spacecraft query
        target_clause = (
            f"where spacecraft_id == '{ctx.spacecraft_uuid}' "
            f"and timestamp between (datetime({ctx.start_time_iso}) .. datetime({ctx.end_time_iso}))"
        )
        logger.info(f"Executing Case 1 query for Spacecraft UUID: {ctx.spacecraft_uuid}")

    elif ctx.contact_uuid_list:
        # Case 2: Specific Contact IDs
        contact_ids = re.split(r"[,\s]+", ctx.contact_uuid_list)
        formatted_ids = ", ".join(f"'{cid.strip()}'" for cid in contact_ids if cid.strip())

        target_clause = f"where contact_id in ({formatted_ids})"
        if bool(ctx.start_time_iso) != bool(ctx.end_time_iso):
            raise ValueError(
                "Contact queries require both start_time_iso and end_time_iso "
                "when either timestamp is provided."
            )
        if ctx.start_time_iso and ctx.end_time_iso:
            target_clause += (
                f" and timestamp between (datetime({ctx.start_time_iso}) .. "
                f"datetime({ctx.end_time_iso}))"
            )
        logger.info(f"Executing Case 2 query for Contact IDs: {ctx.contact_uuid_list}")

    else:
        # Case 3 placeholder / Fallback error
        raise ValueError("Insufficient context parameters to route the database query.")

    # 2. Apply strict database-level filtering
    filters = [
        f"antenna1_position_elevation >= {ctx.min_elevation}",
        f"{prefix}_ebN0 >= {ctx.minimum_ebn0}",
        f"{prefix}_actualCarrierFrequencyOffset >= {ctx.min_doppler}",
        f"{prefix}_actualCarrierFrequencyOffset <= {ctx.max_doppler}"
    ]

    if ctx.lock_requirement:
        filters.append(f"{prefix}_carrierLockState == 'Locked'")

    filter_clause = " and ".join(filters)

    print(f"Target clause: {target_clause}")
    print(f"Filter clause: {filter_clause}")

    # 3. Construct Unified Query
    query = (
        f"contacts\n"
        f"| {target_clause}\n"
        f"| where {filter_clause}\n"
        f"| project timestamp, contact_id, antenna_name, spacecraft_id, system_id, "
        f"antenna1_tracking_epochOffset, antenna1_position_azimuth, antenna1_position_elevation, "
        f"{prefix}_carrierLockState, {prefix}_ebN0, "
        f"{prefix}_actualCarrierFrequencyOffset\n"
        f"| order by timestamp asc"
    )

    # 4. Execute Query
    with get_client() as client:
        db = "telemetry"
        logger.debug(f"Executing KQL Query:\n{query}")
        response = client.execute_query(db, query, _query_properties(timeout_seconds))
        raw_df_pd = dataframe_from_result_table(response.primary_results[0])

    df_final = pl.from_pandas(raw_df_pd)

    if df_final.is_empty():
        logger.warning("Query returned an empty dataset. Review database threshold parameters.")

    logger.info(f"Returning {len(df_final)} validated samples from database.")

    return df_final


def fetch_contact_tracking_data(
    contact_id: str,
    *,
    require_lock: bool = False,
    min_elevation_deg: float = 1.0,
    min_doppler_hz: float = 1.0,
    max_doppler_hz: float = 1e5,
    timeout_seconds: float = ADX_QUERY_TIMEOUT_SECONDS,
) -> pl.DataFrame:
    """Fetch one contact for the asynchronous service.

    Unlike the legacy ``TrackingContext`` adapter, this query has no hidden
    Eb/N0 filter and interprets Doppler limits as absolute magnitudes. The
    caller validates ``contact_id`` as a UUID before reaching this boundary.
    """
    if min_doppler_hz < 0 or max_doppler_hz <= 0 or min_doppler_hz > max_doppler_hz:
        raise ValueError("invalid absolute Doppler bounds")
    prefix = "lr1_receiver1"
    filters = [
        f"contact_id == '{contact_id}'",
        f"antenna1_position_elevation >= {float(min_elevation_deg)}",
        (
            f"abs({prefix}_actualCarrierFrequencyOffset) between "
            f"({float(min_doppler_hz)} .. {float(max_doppler_hz)})"
        ),
    ]
    if require_lock:
        filters.append(f"{prefix}_carrierLockState == 'Locked'")
    query = (
        "contacts\n"
        f"| where {' and '.join(filters)}\n"
        "| project timestamp, contact_id, antenna_name, spacecraft_id, system_id, "
        "antenna1_tracking_epochOffset, antenna1_position_azimuth, antenna1_position_elevation, "
        f"{prefix}_carrierLockState, {prefix}_ebN0, "
        f"{prefix}_actualCarrierFrequencyOffset\n"
        "| order by timestamp asc"
    )
    with get_client() as client:
        response = client.execute_query(
            "telemetry", query, _query_properties(timeout_seconds)
        )
        raw_df_pd = dataframe_from_result_table(response.primary_results[0])
    return pl.from_pandas(raw_df_pd)


@dataclass
class TrackingContext:
    """
    Encapsulates and normalizes the incoming Grafana payload.
    """
    # Query Mode 1
    spacecraft_uuid: str | None
    start_time_iso: str | None
    end_time_iso: str | None

    # Query Mode 2
    contact_uuid_list: str | None

    # Query Mode 3
    ephemeris_id: str | None

    # Other
    mode: str
    lock_requirement: bool

    # Database filter parameters
    min_elevation: float = 1.0
    minimum_ebn0: float = 0.0
    min_doppler: float = 1.0
    max_doppler: float = 1e5

    # Optimization modes (Mapped 1:1 with internal Config)
    angle_constraint: float = 1.0
    penalty_weight: float = 1e4
    qmc_samples: int = 20
    loss_function: str = "linear"
    f_scale: float = 500.0
    model_type: str = "auto"
    criterion: str = "bic"
    use_qmc: bool = True
    method: str = "dogbox"

    @classmethod
    def from_payload(cls, payload: dict):
        kwargs = {
            "spacecraft_uuid": _safe_str(payload.get("spacecraft_UUID")),
            "start_time_iso": _safe_str(payload.get("start_time")),
            "end_time_iso": _safe_str(payload.get("end_time")),
            "contact_uuid_list": _join_list(payload.get("contact_UUID_List")),
            "ephemeris_id": _safe_str(payload.get("EphemerisID")),
            "lock_requirement": _safe_bool(payload.get("lockRequirement")),
            "mode": str(payload.get("Mode", ""))
        }

        optional_fields = {
            "min_elevation": ("minElevation", float),
            "minimum_ebn0": ("minimumEbN0", float),
            "min_doppler": ("minDoppler", float),
            "max_doppler": ("maxDoppler", float),
            "angle_constraint": ("angleConstraint", float),
            "penalty_weight": ("penaltyWeight", float),
            "qmc_samples": ("qmcSamples", int),
            "loss_function": ("lossFunction", str),
            "f_scale": ("fScale", float),
            "model_type": ("modelType", str),
            "criterion": ("criterion", str),
            "use_qmc": ("useQmc", _safe_bool),
            "method": ("method", str)
        }

        for dc_field, (payload_key, cast_func) in optional_fields.items():
            val = payload.get(payload_key)
            if val is not None:
                casted_val = cast_func(val)
                if casted_val == 0 and not isinstance(casted_val, bool):
                    continue
                kwargs[dc_field] = casted_val

        return cls(**kwargs)
