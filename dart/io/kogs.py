"""KOGS API client and payload parsers (ported from lib/IO/kogs.py).

Each ``get_*`` hits a KOGS endpoint and returns the raw JSON payload; each
``parse_*`` normalizes a payload into a dataclass of strings/floats. The
loaders combine both into ``dart.schema`` transport structs.
"""
from dataclasses import dataclass
import math
from pathlib import Path
import re
from typing import Optional, Dict, Any
import requests
import satkit
import yaml
from dart.io.utils import _safe_float, _safe_str, _join_field, _join_list, _iso_to_unix, create_api_auth

KOGS_REQUEST_TIMEOUT_SECONDS = 30.0
_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def unwrap_payload(value: object, key: str) -> dict:
    """Return a raw KOGS object whether or not its response is keyed."""
    if not isinstance(value, dict):
        raise ValueError(f"KOGS {key} response is not an object")
    nested = value.get(key, value)
    if not isinstance(nested, dict):
        raise ValueError(f"KOGS {key} response contains no {key}")
    return nested


def _config_path(kind: str, name: str) -> Path:
    if not name or Path(name).name != name:
        raise ValueError(f"invalid {kind} configuration name {name!r}")
    return _PROJECT_ROOT / "ctrl-config" / "v2" / kind / f"{name}.yml"


def antenna_has_config(system_name: str) -> bool:
    return _config_path("system", system_name).is_file()


def spacecraft_has_config(spacecraft_name: str) -> bool:
    return _config_path("spacecrafts", spacecraft_name).is_file()


def get_ip_address(system_name: str) -> str:
    """Find the IP address for the system's qradio's REST API from ctrl-config"""
    # NOTE: For this to work, the file MUST be ran through docker or from the root of the project dir
    path = _config_path("system", system_name)

    if not antenna_has_config(system_name):
        raise FileNotFoundError(f"No config found for system '{system_name}'.")

    with open(path, "r") as f:
        data = yaml.safe_load(f)

    defaults = data.get("defaults")
    if isinstance(defaults, dict):
        if "qradio" in defaults:
            return defaults["qradio"].get("rest")
    elif isinstance(defaults, list):
        for entry in defaults:
            if isinstance(entry, dict) and "qradio" in entry:
                return entry["qradio"].get("rest")

    raise ValueError(
        f"'qradio' with 'rest' not found in the config for '{system_name}'."
    )


def get_link_frequency(spacecraft_name: str, link_name: str, direction: str) -> float:
    """Return one positive ctrl-config link frequency after checking direction."""
    if direction not in {"up", "down"}:
        raise ValueError("link direction must be 'up' or 'down'")
    path = _config_path("spacecrafts", spacecraft_name)
    if not path.is_file():
        raise FileNotFoundError(f"No config found for spacecraft {spacecraft_name!r}.")
    with path.open(encoding="utf-8") as stream:
        data = yaml.safe_load(stream) or {}
    link = data.get("links", {}).get(link_name)
    if not isinstance(link, dict):
        raise ValueError(f"link {link_name!r} is missing from {spacecraft_name}.yml")
    if link.get("direction") != direction:
        raise ValueError(f"link {link_name!r} is not a {direction}link")
    try:
        frequency = float(link["frequency"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"link {link_name!r} has no numeric frequency") from exc
    if not math.isfinite(frequency) or frequency <= 0:
        raise ValueError(f"link {link_name!r} frequency must be positive and finite")
    return frequency


def get_observed_frequency(spacecraft_name: str) -> float:
    """Return the legacy primary S-band downlink frequency."""
    return get_link_frequency(spacecraft_name, "s_band_downlink_p1_1", "down")


def parse_ephemeris_identity(data: "EphemerisData") -> tuple[str, str]:
    """Return matching COSPAR and catalog identities from TLE/OMM content."""
    identities: list[tuple[str, str]] = []
    if data.inline_tle:
        tle = satkit.TLE.from_lines(
            [line for line in data.inline_tle.splitlines() if line.strip()]
        )
        designator = str(tle.intl_desig).strip()
        year = int(designator[:2])
        full_year = 1900 + year if year >= 57 else 2000 + year
        identities.append(
            (f"{full_year}-{designator[2:5]}{designator[5:]}", str(tle.satnum))
        )
    if data.inline_omm:
        fields = dict(
            re.findall(
                r"(?m)^\s*(OBJECT_ID|NORAD_CAT_ID)\s*=\s*([^\s]+)",
                data.inline_omm,
            )
        )
        if len(fields) == 2:
            identities.append((fields["OBJECT_ID"], fields["NORAD_CAT_ID"]))
    if not identities:
        raise ValueError("contact ephemeris contains no usable COSPAR identity")
    if len(set(identities)) != 1:
        raise ValueError("contact TLE and OMM identities do not match")
    return identities[0]

def generate_auth_header(auth: str) -> dict:
    return {
        'Authorization': create_api_auth(auth),
        'Accept': 'application/json'
    }

def validate(resp: requests.Response) -> dict:
    """Raises HTTP Errors, if one occured. Otherwise, data is returned"""
    resp.raise_for_status()
    return resp.json()

def _get(
    url: str,
    auth: str,
    *,
    timeout_seconds: float = KOGS_REQUEST_TIMEOUT_SECONDS,
) -> dict:
    return validate(
        requests.get(
            url,
            headers=generate_auth_header(auth),
            timeout=timeout_seconds,
        )
    )


def get_contact(
    auth: str,
    contact_id: str,
    *,
    timeout_seconds: float = KOGS_REQUEST_TIMEOUT_SECONDS,
) -> dict:
    """https://ksat.stoplight.io/docs/internal-apis-1/b8a89c7f20ff1-get-contact"""
    url = f'https://mgmt.kogs.api.ksat.no/24.08/contacts/{contact_id}'
    return _get(url, auth, timeout_seconds=timeout_seconds)

def get_spacecraft(
    auth: str,
    spacecraft_id: str,
    *,
    timeout_seconds: float = KOGS_REQUEST_TIMEOUT_SECONDS,
) -> dict:
    """https://ksat.stoplight.io/docs/internal-apis-1/70211f511b112-get-spacecraft"""
    url = f'https://mgmt.kogs.api.ksat.no/24.08/spacecrafts/{spacecraft_id}'
    return _get(url, auth, timeout_seconds=timeout_seconds)

def get_station(
    auth: str,
    station_id: str,
    *,
    timeout_seconds: float = KOGS_REQUEST_TIMEOUT_SECONDS,
) -> dict:
    """https://ksat.stoplight.io/docs/internal-apis-1/cac2d50049631-get-station"""
    url = f'https://mgmt.kogs.api.ksat.no/24.08/stations/{station_id}'
    return _get(url, auth, timeout_seconds=timeout_seconds)

def get_antenna(
    auth: str,
    system_id: str,
    *,
    timeout_seconds: float = KOGS_REQUEST_TIMEOUT_SECONDS,
) -> dict:
    """https://ksat.stoplight.io/docs/internal-apis-1/0c3a140c61f03-find-system-antenna"""
    url = f'https://mgmt.kogs.api.ksat.no/24.08/systems/antennas/{system_id}'
    return _get(url, auth, timeout_seconds=timeout_seconds)

def get_TLE(
    auth: str,
    ephemeris_id: str,
    *,
    timeout_seconds: float = KOGS_REQUEST_TIMEOUT_SECONDS,
) -> dict:
    """https://ksat.stoplight.io/docs/internal-apis-1/294893d364844-locate-ephemeris-entry"""
    url = f"https://mgmt.kogs.api.ksat.no/24.08/ephemeris/{ephemeris_id}"
    return _get(url, auth, timeout_seconds=timeout_seconds)

def get_ephemeris(
    auth: str,
    ephemeris_id: str,
    *,
    timeout_seconds: float = KOGS_REQUEST_TIMEOUT_SECONDS,
) -> dict:
    """https://ksat.stoplight.io/docs/internal-apis-1/831bff601c741-fetch-known-ephemeris"""
    url = f"https://mgmt.kogs.api.ksat.no/24.08/ephemeris/{ephemeris_id}"
    return _get(url, auth, timeout_seconds=timeout_seconds)

@dataclass
class AntennaData:
    antenna_id: Optional[str]
    antenna_name: Optional[str]
    station_id: Optional[str]
    station_name: Optional[str]
    latitude: Optional[float]
    longitude: Optional[float]
    altitude: Optional[float]
    setup_duration: Optional[float]
    teardown_duration: Optional[float]
    operator_id: Optional[str]
    ops_unit_id: Optional[str]
    ops_unit_name: Optional[str]
    lifecycle_state: Optional[str]
    diameter: Optional[float]
    bands_types: Optional[str]
    bands_directions: Optional[str]
    bands_polarizations: Optional[str]
    partner: Optional[str]

@dataclass
class EphemerisData:
    ephemeris_uuid: Optional[str]
    spacecraft_uuid: Optional[str]
    kind: Optional[str]
    origin: Optional[str]
    tenant_uuid: Optional[str]
    epoch: Optional[str]
    epoch_unix: Optional[float]
    last_useable_at: Optional[str]
    last_useable_at_unix: Optional[float]
    submitted_at: Optional[str]
    submitted_at_unix: Optional[float]
    submitted_by: Optional[str]
    inline_tle: Optional[str]
    inline_omm: Optional[str]
    inline_oem: Optional[str]
    is_cui: Optional[str]
    payload: Optional[str]

@dataclass
class SatelliteData:
    id: Optional[str]
    name: Optional[str]
    satellite_catalog_number: Optional[float]
    catalog_number_assignment: Optional[str]
    lifecycle_state: Optional[str]
    kind: Optional[str]
    orbit: Optional[str]
    norad_id: Optional[float]

@dataclass
class ReservationData:
    id: Optional[str]
    state: Optional[str]
    criticality: Optional[str]
    signature_outcome: Optional[str]
    signature_comment: Optional[str]
    signature_contains_human_override: Optional[str]
    signature_last_impacting_principal_kind: Optional[str]
    created_at: Optional[str]
    created_at_unix: Optional[float]
    updated_at: Optional[str]
    updated_at_unix: Optional[float]
    setup_duration: Optional[float]
    teardown_duration: Optional[float]
    ephemeris_mode: Optional[str]
    mission_profile_id: Optional[str]
    spacecraft_id: Optional[str]
    system_id: Optional[str]
    station_id: Optional[str]
    tenant_id: Optional[str]
    start_time: Optional[str]
    start_time_unix: Optional[float]
    end_time: Optional[str]
    end_time_unix: Optional[float]
    reservation_id: Optional[str]
    ephemeris_id: Optional[str]
    external_ref: Optional[str]
    properties_is_test: Optional[str]
    properties_is_internal: Optional[str]
    properties_cfes: Optional[str]

def parse_response(resp: Dict[str, Any]) -> AntennaData:
    antenna = resp.get("antenna", {}) or {}
    expanded = resp.get("expanded", {}) or {}

    # Basic antenna fields
    antenna_id = _safe_str(antenna.get("id"))
    antenna_name = _safe_str(antenna.get("name"))
    station_id = _safe_str(antenna.get("station"))
    operator_id = _safe_str(antenna.get("operator"))
    ops_unit_id = _safe_str(antenna.get("ops_unit"))
    lifecycle_state = _safe_str(antenna.get("lifecycle_state"))
    diameter = _safe_float(antenna.get("diameter"))
    setup_duration = _safe_float(antenna.get("setup_duration"))
    teardown_duration = _safe_float(antenna.get("teardown_duration"))
    partner = _safe_str(antenna.get("partner"))

    # Location from antenna.location if present, otherwise try station location
    lat = _safe_float(None)
    lon = _safe_float(None)
    alt = _safe_float(None)
    ant_loc = antenna.get("location")
    if isinstance(ant_loc, dict):
        lat = _safe_float(ant_loc.get("latitude"))
        lon = _safe_float(ant_loc.get("longitude"))
        alt = _safe_float(ant_loc.get("altitude"))

    # Find station name by matching id in expanded.stations
    station_name = None
    stations = expanded.get("stations") or []
    if station_id and isinstance(stations, list):
        for s in stations:
            if _safe_str(s.get("id")) == station_id:
                station_name = _safe_str(s.get("name"))
                # if antenna location missing, use station location
                if lat is None and isinstance(s.get("location"), dict):
                    lat = _safe_float(s["location"].get("latitude"))
                    lon = _safe_float(s["location"].get("longitude"))
                    alt = _safe_float(s["location"].get("altitude"))
                break
    # fallback: if expanded.stations has at least one entry and no match, take first name
    if station_name is None and stations:
        first = stations[0]
        station_name = _safe_str(first.get("name"))

    # Find ops unit name by matching id in expanded.ops_units
    ops_unit_name = None
    ops_units = expanded.get("ops_units") or []
    if ops_unit_id and isinstance(ops_units, list):
        for o in ops_units:
            if _safe_str(o.get("id")) == ops_unit_id:
                ops_unit_name = _safe_str(o.get("name"))
                break
    if ops_unit_name is None and ops_units:
        ops_unit_name = _safe_str(ops_units[0].get("name"))

    # Bands: flatten into comma-separated strings
    bands = antenna.get("bands") or []
    bands_types = _join_field(bands, "type")
    bands_directions = _join_field(bands, "direction")
    bands_polarizations = _join_field(bands, "polarization")

    return AntennaData(
        antenna_id=antenna_id,
        antenna_name=antenna_name,
        station_id=station_id,
        station_name=station_name,
        latitude=lat,
        longitude=lon,
        altitude=alt,
        setup_duration=setup_duration,
        teardown_duration=teardown_duration,
        operator_id=operator_id,
        ops_unit_id=ops_unit_id,
        ops_unit_name=ops_unit_name,
        lifecycle_state=lifecycle_state,
        diameter=diameter,
        bands_types=bands_types,
        bands_directions=bands_directions,
        bands_polarizations=bands_polarizations,
        partner=_safe_str(partner),
    )


def parse_ephemeris(resp: Dict[str, Any]) -> EphemerisData:
    # top-level simple fields
    ephemeris_uuid = _safe_str(resp.get("ephemeris_uuid"))
    spacecraft_uuid = _safe_str(resp.get("spacecraft_uuid"))
    kind = _safe_str(resp.get("kind"))
    origin = _safe_str(resp.get("origin"))
    tenant_uuid = _safe_str(resp.get("tenant_uuid"))

    # timestamps (keep original string and also provide unix float)
    epoch = _safe_str(resp.get("epoch"))
    epoch_unix = _iso_to_unix(epoch)

    last_useable_at = _safe_str(resp.get("last_useable_at"))
    last_useable_at_unix = _iso_to_unix(last_useable_at)

    submitted_at = _safe_str(resp.get("submitted_at"))
    submitted_at_unix = _iso_to_unix(submitted_at)

    submitted_by = _safe_str(resp.get("submitted_by"))

    # inline block (tle/omm/oem) — keep as strings
    inline = resp.get("inline") or {}
    inline_tle = _safe_str(inline.get("tle"))
    inline_omm = _safe_str(inline.get("omm"))
    inline_oem = _safe_str(inline.get("oem"))

    # boolean -> string (to satisfy "str or float" requirement)
    is_cui = _safe_str(resp.get("is_cui"))

    # payload: if it's a dict/complex, stringify it; otherwise keep string/None
    payload_raw = resp.get("payload")
    if payload_raw is None:
        payload = None
    elif isinstance(payload_raw, (str, int, float, bool)):
        payload = _safe_str(payload_raw)
    else:
        # convert complex payload to compact JSON-like string
        try:
            import json
            payload = json.dumps(payload_raw, separators=(",", ":"), ensure_ascii=False)
        except Exception:
            payload = _safe_str(payload_raw)

    return EphemerisData(
        ephemeris_uuid=ephemeris_uuid,
        spacecraft_uuid=spacecraft_uuid,
        kind=kind,
        origin=origin,
        tenant_uuid=tenant_uuid,
        epoch=epoch,
        epoch_unix=epoch_unix,
        last_useable_at=last_useable_at,
        last_useable_at_unix=last_useable_at_unix,
        submitted_at=submitted_at,
        submitted_at_unix=submitted_at_unix,
        submitted_by=submitted_by,
        inline_tle=inline_tle,
        inline_omm=inline_omm,
        inline_oem=inline_oem,
        is_cui=is_cui,
        payload=payload,
    )

def parse_satellite(resp: Dict[str, Any]) -> SatelliteData:
    """
    Parse a satellite API response dict into SatelliteData.
    All numeric-like fields are converted to float when possible.
    """
    if not isinstance(resp, dict):
        raise TypeError("resp must be a dict")

    sid = _safe_str(resp.get("id"))
    name = _safe_str(resp.get("name"))

    # numeric fields: try to coerce to float
    sat_cat_num = _safe_float(resp.get("satellite_catalog_number"))
    norad = _safe_float(resp.get("norad_id"))

    catalog_assignment = _safe_str(resp.get("catalog_number_assignment"))
    lifecycle_state = _safe_str(resp.get("lifecycle_state"))
    kind = _safe_str(resp.get("kind"))
    orbit = _safe_str(resp.get("orbit"))

    return SatelliteData(
        id=sid,
        name=name,
        satellite_catalog_number=sat_cat_num,
        catalog_number_assignment=catalog_assignment,
        lifecycle_state=lifecycle_state,
        kind=kind,
        orbit=orbit,
        norad_id=norad,
    )

def parse_reservation(resp: Dict[str, Any]) -> ReservationData:
    if not isinstance(resp, dict):
        raise TypeError("resp must be a dict")

    rid = _safe_str(resp.get("id"))
    state = _safe_str(resp.get("state"))
    criticality = _safe_str(resp.get("criticality"))

    signature = resp.get("signature") or {}
    sig_outcome = _safe_str(signature.get("outcome"))
    sig_comment = _safe_str(signature.get("comment"))
    sig_contains_human_override = _safe_str(signature.get("contains_human_override"))
    sig_last_kind = _safe_str(signature.get("last_impacting_principal_kind"))

    created_at = _safe_str(resp.get("created_at"))
    created_at_unix = _iso_to_unix(created_at)

    updated_at = _safe_str(resp.get("updated_at"))
    updated_at_unix = _iso_to_unix(updated_at)

    setup_duration = _safe_float(resp.get("setup_duration"))
    teardown_duration = _safe_float(resp.get("teardown_duration"))

    ephemeris_props = resp.get("ephemeris_properties") or {}
    ephemeris_mode = _safe_str(ephemeris_props.get("ephemeris_mode"))

    mission_profile_id = _safe_str(resp.get("mission_profile_id"))
    spacecraft_id = _safe_str(resp.get("spacecraft_id"))
    system_id = _safe_str(resp.get("system_id"))
    station_id = _safe_str(resp.get("station_id"))
    tenant_id = _safe_str(resp.get("tenant_id"))

    start_time = _safe_str(resp.get("start_time"))
    start_time_unix = _iso_to_unix(start_time)

    end_time = _safe_str(resp.get("end_time"))
    end_time_unix = _iso_to_unix(end_time)

    reservation_id = _safe_str(resp.get("reservation_id"))
    ephemeris_id = _safe_str(resp.get("ephemeris_id"))
    external_ref = _safe_str(resp.get("external_ref"))

    properties = resp.get("properties") or {}
    prop_is_test = _safe_str(properties.get("is_test"))
    prop_is_internal = _safe_str(properties.get("is_internal"))
    prop_cfes = _join_list(properties.get("cfes"))

    data = ReservationData(
        id=rid,
        state=state,
        criticality=criticality,
        signature_outcome=sig_outcome,
        signature_comment=sig_comment,
        signature_contains_human_override=sig_contains_human_override,
        signature_last_impacting_principal_kind=sig_last_kind,
        created_at=created_at,
        created_at_unix=created_at_unix,
        updated_at=updated_at,
        updated_at_unix=updated_at_unix,
        setup_duration=setup_duration,
        teardown_duration=teardown_duration,
        ephemeris_mode=ephemeris_mode,
        mission_profile_id=mission_profile_id,
        spacecraft_id=spacecraft_id,
        system_id=system_id,
        station_id=station_id,
        tenant_id=tenant_id,
        start_time=start_time,
        start_time_unix=start_time_unix,
        end_time=end_time,
        end_time_unix=end_time_unix,
        reservation_id=reservation_id,
        ephemeris_id=ephemeris_id,
        external_ref=external_ref,
        properties_is_test=prop_is_test,
        properties_is_internal=prop_is_internal,
        properties_cfes=prop_cfes,
    )

    return data
