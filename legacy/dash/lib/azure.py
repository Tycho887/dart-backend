import os
import re
from datetime import datetime
import polars as pl
from azure.kusto.data import KustoConnectionStringBuilder
from azure.kusto.data import KustoClient
from azure.kusto.data.helpers import dataframe_from_result_table

from lib.db_logger import ProcessLogger
from lib.settings import load_environment

logger = ProcessLogger("azure")

load_environment()

class env:
    AZURE_ADX_CLUSTER_ENDPOINT: str = os.getenv("AZURE_ADX_CLUSTER_ENDPOINT", "")
    AZURE_CLIENT_ID: str = os.getenv("AZURE_CLIENT_ID", "")
    AZURE_CLIENT_SECRET: str = os.getenv("AZURE_CLIENT_SECRET", "")
    AZURE_TENANT_ID: str = os.getenv("AZURE_TENANT_ID", "")
    HTTP_PROXY: str = os.getenv("HTTP_PROXY", "")


def _kql_string(value: str) -> str:
    """Escape a value for a single-quoted KQL string literal."""

    return value.replace("'", "''")


def _kql_datetime(value: str) -> str:
    """Validate an ISO-8601 timestamp before inserting it into KQL."""

    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"Invalid ISO-8601 timestamp: {value!r}") from exc
    return parsed.isoformat()

def get_client() -> KustoClient:
    required = {
        "AZURE_ADX_CLUSTER_ENDPOINT": env.AZURE_ADX_CLUSTER_ENDPOINT,
        "AZURE_CLIENT_ID": env.AZURE_CLIENT_ID,
        "AZURE_CLIENT_SECRET": env.AZURE_CLIENT_SECRET,
        "AZURE_TENANT_ID": env.AZURE_TENANT_ID,
    }
    missing = [name for name, value in required.items() if not value]
    if missing:
        raise RuntimeError(f"Missing Azure configuration: {', '.join(missing)}")

    logger.info(
        200,
        f"Connecting to ADX cluster {env.AZURE_ADX_CLUSTER_ENDPOINT}"
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
        logger.debug(100, f"HTTP proxy set to {env.HTTP_PROXY}")
    return client

def fetch_tracking_data(ctx) -> pl.DataFrame:
    """
    Routes the database request based on the context object case, 
    injects KQL-level filtering, and returns a unified Polars DataFrame.
    """
    prefix = "lr1_receiver1"
    
    logger.debug(101, f"TrackingContext received: {ctx}")

    # 1. Determine the Routing Case and Initial Dataset
    if ctx.spacecraft_uuid and ctx.start_time_iso and ctx.end_time_iso:
        spacecraft_id = _kql_string(ctx.spacecraft_uuid)
        start_time = _kql_datetime(ctx.start_time_iso)
        end_time = _kql_datetime(ctx.end_time_iso)
        target_clause = (
            f"where spacecraft_id == '{spacecraft_id}' "
            f"and timestamp between (datetime({start_time}) .. datetime({end_time}))"
        )
        logger.info(
            200,
            f"Executing Case 1 query for Spacecraft UUID: {ctx.spacecraft_uuid}",
            extra={"start": ctx.start_time_iso, "end": ctx.end_time_iso}
        )
        
    elif ctx.contact_uuid_list:
        contact_ids = re.split(r"[\s,]+", ctx.contact_uuid_list)
        formatted_ids = ", ".join(
            f"'{_kql_string(contact_id)}'"
            for contact_id in contact_ids
            if contact_id
        )
        
        target_clause = f"where contact_id in ({formatted_ids})"
        logger.info(
            200,
            f"Executing Case 2 query for Contact IDs: {ctx.contact_uuid_list}",
            extra={"contact_count": len(contact_ids)}
        )
        
    else:
        logger.error(400, "Insufficient context parameters to route the database query")
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

    logger.debug(102, f"Target clause: {target_clause}")
    logger.debug(103, f"Filter clause: {filter_clause}")

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
        logger.debug(104, f"Executing KQL Query:\n{query}")
        response = client.execute_query(db, query)
        raw_df_pd = dataframe_from_result_table(response.primary_results[0])

    df_final = pl.from_pandas(raw_df_pd)
    
    if df_final.is_empty():
        logger.warning(
            404,
            "Query returned an empty dataset. Review database threshold parameters.",
            extra={"spacecraft": getattr(ctx, "spacecraft_uuid", None)}
        )
        
    logger.info(200, f"Returning {len(df_final)} validated samples from database.")
    
    return df_final
