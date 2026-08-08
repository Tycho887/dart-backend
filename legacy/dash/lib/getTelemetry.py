import polars as pl
import os
from lib.parseKogs import (
    get_contact, parse_reservation,
    get_TLE, parse_ephemeris,
    get_antenna, parse_response,
    get_spacecraft, parse_satellite
)
from lib.utils import create_api_auth
from lib.db_logger import ProcessLogger
from lib.settings import load_environment

logger = ProcessLogger("getTelemetry")
load_environment()


def _api_auth() -> str:
    """Build the KOGS authorization value without retaining the key in memory globally."""

    api_key = os.getenv("KOGS_API_KEY")
    if not api_key:
        raise RuntimeError("KOGS_API_KEY is not configured")
    return create_api_auth(api_key)

def build_contact_lookup(unique_contacts: list[str]) -> pl.DataFrame:
    """Builds a lookup table mapping contact_id to ephemeris_uuid."""
    auth = _api_auth()
    logger.info(200, f"Building contact lookup for {len(unique_contacts)} contacts")
    records = []
    for cid in unique_contacts:
        try:
            logger.debug(300, f"Fetching contact {cid}")
            resp = get_contact(auth, cid)["contact"]
            contact_data = parse_reservation(resp)
            
            eph_id = contact_data.ephemeris_id
            if not eph_id:
                logger.warning(400, f"Contact {cid} parsed successfully, but 'ephemeris_id' is missing")
                raise ValueError(f"Contact {cid} parsed successfully, but 'ephemeris_id' is missing from the payload.")
                
            records.append({"contact_id": cid, "ephemeris_uuid": eph_id})
            logger.info(201, f"Contact {cid} resolved to ephemeris {eph_id}")
        except Exception as e:
            logger.error(500, f"Failed to fetch or parse contact data for {cid}: {e}")
            raise RuntimeError(f"Failed to fetch or parse contact data for {cid}: {str(e)}") from e
            
    logger.info(200, f"Contact lookup complete: {len(records)} records")
    return pl.DataFrame(records, schema={"contact_id": pl.Utf8, "ephemeris_uuid": pl.Utf8})

def build_ephemeris_lookup(unique_ephemeris: list[str]) -> pl.DataFrame:
    """Builds a lookup table mapping ephemeris_uuid to TLE lines."""
    auth = _api_auth()
    logger.info(200, f"Building ephemeris lookup for {len(unique_ephemeris)} ephemeris IDs")
    records = []
    for eid in unique_ephemeris:
        if not eid:
            logger.debug(301, "Skipping empty ephemeris ID")
            continue
        try:
            logger.debug(302, f"Fetching TLE for ephemeris {eid}")
            resp = get_TLE(auth, eid)
            eph_data = parse_ephemeris(resp)
            
            tle_lines = ["", "", ""]
            if eph_data.inline_tle:
                lines = eph_data.inline_tle.strip().split("\n")
                if len(lines) >= 3:
                    tle_lines = lines[:3]
                elif len(lines) >= 2:
                    tle_lines = [""] + lines[:2]
            
            if not tle_lines[1] or not tle_lines[2]:
                logger.warning(401, f"Incomplete TLE lines parsed for ephemeris {eid}")
                raise ValueError(f"Incomplete TLE lines parsed for ephemeris {eid}.")
                    
            records.append({
                "ephemeris_uuid": eid,
                "tle_line_1": tle_lines[1],
                "tle_line_2": tle_lines[2]
            })
            logger.info(202, f"Ephemeris {eid} parsed successfully")
        except Exception as e:
            logger.error(501, f"Failed to fetch or parse TLE for ephemeris {eid}: {e}")
            raise RuntimeError(f"Failed to fetch or parse TLE for ephemeris {eid}: {str(e)}") from e
            
    logger.info(200, f"Ephemeris lookup complete: {len(records)} records")
    return pl.DataFrame(records, schema={"ephemeris_uuid": pl.Utf8, "tle_line_1": pl.Utf8, "tle_line_2": pl.Utf8})

def build_antenna_lookup(unique_systems: list[str]) -> pl.DataFrame:
    """Builds a lookup table mapping system_id to station coordinates."""
    auth = _api_auth()
    logger.info(200, f"Building antenna lookup for {len(unique_systems)} systems")
    records = []
    for sys_id in unique_systems:
        try:
            logger.debug(303, f"Fetching antenna data for system {sys_id}")
            resp = get_antenna(auth, sys_id)
            ant_data = parse_response(resp)
            
            if None in (ant_data.latitude, ant_data.longitude, ant_data.altitude):
                logger.warning(402, f"Incomplete antenna coordinates for system {sys_id}")
                raise ValueError(f"Incomplete antenna coordinates for system {sys_id}")
                
            records.append({
                "system_id": sys_id,
                "station_lat": ant_data.latitude,
                "station_lon": ant_data.longitude,
                "station_alt": ant_data.altitude
            })
            logger.info(
                203,
                f"Antenna {sys_id} resolved: lat={ant_data.latitude}, lon={ant_data.longitude}, alt={ant_data.altitude}"
            )
        except Exception as e:
            logger.error(502, f"Failed to fetch antenna data for system {sys_id}: {e}")
            raise RuntimeError(f"Failed to fetch antenna data for system {sys_id}: {str(e)}") from e
            
    logger.info(200, f"Antenna lookup complete: {len(records)} records")
    return pl.DataFrame(records, schema={
        "system_id": pl.Utf8, 
        "station_lat": pl.Float64, 
        "station_lon": pl.Float64, 
        "station_alt": pl.Float64
    })

def build_spacecraft_lookup(unique_spacecraft: list[str], get_frequency_fn=None) -> pl.DataFrame:
    """Builds a lookup table mapping spacecraft_id to name and frequency."""
    auth = _api_auth()
    logger.info(200, f"Building spacecraft lookup for {len(unique_spacecraft)} spacecraft")
    records = []
    for sc_id in unique_spacecraft:
        try:
            logger.debug(304, f"Fetching spacecraft data for {sc_id}")
            resp = get_spacecraft(auth, sc_id)
            sc_data = parse_satellite(resp)
            
            freq = get_frequency_fn(sc_id) if get_frequency_fn else 2.2e9
            
            if not sc_data.name:
                logger.warning(403, f"Spacecraft name missing for {sc_id}")
                raise ValueError(f"Spacecraft name missing for {sc_id}")
                
            records.append({
                "spacecraft_id": sc_id,
                "spacecraft_name": sc_data.name,
                "satellite_frequency": freq
            })
            logger.info(204, f"Spacecraft {sc_id} resolved: name={sc_data.name}")
        except Exception as e:
            logger.error(503, f"Failed to fetch spacecraft data for {sc_id}: {e}")
            raise RuntimeError(f"Failed to fetch spacecraft data for {sc_id}: {str(e)}") from e
            
    logger.info(200, f"Spacecraft lookup complete: {len(records)} records")
    return pl.DataFrame(records, schema={
        "spacecraft_id": pl.Utf8, 
        "spacecraft_name": pl.Utf8, 
        "satellite_frequency": pl.Float64
    })

def augment_telemetry_dataframe(df: pl.DataFrame, get_frequency_fn=None) -> pl.DataFrame:
    """Executes the batch metadata fetching and relational joins."""
    logger.info(200, "Starting telemetry dataframe augmentation")
    
    unique_contacts = df["contact_id"].drop_nulls().unique().to_list()
    unique_systems = df["system_id"].drop_nulls().unique().to_list()
    unique_spacecraft = df["spacecraft_id"].drop_nulls().unique().to_list()
    
    logger.debug(
        305,
        f"Unique contacts: {len(unique_contacts)}, systems: {len(unique_systems)}, spacecraft: {len(unique_spacecraft)}"
    )
    
    contact_df = build_contact_lookup(unique_contacts)
    antenna_df = build_antenna_lookup(unique_systems)
    spacecraft_df = build_spacecraft_lookup(unique_spacecraft, get_frequency_fn)
    
    unique_ephemeris = contact_df["ephemeris_uuid"].drop_nulls().unique().to_list()
    logger.debug(306, f"Unique ephemeris to resolve: {len(unique_ephemeris)}")
    ephemeris_df = build_ephemeris_lookup(unique_ephemeris)
    
    logger.info(205, "Executing relational joins on telemetry dataframe")
    df = df.join(contact_df, on="contact_id", how="left")
    
    df = (
        df.join(ephemeris_df, on="ephemeris_uuid", how="left")
          .join(antenna_df, on="system_id", how="left")
          .join(spacecraft_df, on="spacecraft_id", how="left")
    )
    
    # Report join quality
    missing_contacts = df["ephemeris_uuid"].is_null().sum()
    missing_tle = df["tle_line_1"].is_null().sum()
    missing_ant = df["station_lat"].is_null().sum()
    missing_sc = df["spacecraft_name"].is_null().sum()
    
    if missing_contacts:
        logger.warning(404, f"Join produced {missing_contacts} rows with missing contact metadata")
    if missing_tle:
        logger.warning(405, f"Join produced {missing_tle} rows with missing TLE data")
    if missing_ant:
        logger.warning(406, f"Join produced {missing_ant} rows with missing antenna coordinates")
    if missing_sc:
        logger.warning(407, f"Join produced {missing_sc} rows with missing spacecraft names")
    
    logger.info(200, "Telemetry dataframe augmentation complete")
    return df
