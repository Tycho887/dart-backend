"""Typed access to the KOGS management API.

This module alone formats KOGS credentials and understands KOGS payloads.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import re
from dataclasses import dataclass
from typing import Any, Protocol, Sequence

import requests
import satkit as sk

from .contact import ContactMetadata, EphemerisMetadata

KOGS_BASE_URL = "https://mgmt.kogs.api.ksat.no/24.08"
KOGS_REQUEST_TIMEOUT_SECONDS = 30.0
_AUTH_PREFIX = "KSAT1-PLAIN "


class KogsError(RuntimeError):
    """KOGS returned invalid data or an unsafe operation was requested."""


@dataclass(frozen=True, slots=True)
class Contact:
    id: str
    spacecraft_id: str
    system_id: str
    station_id: str
    mission_profile_id: str
    ephemeris_id: str
    start: dt.datetime
    end: dt.datetime
    state: str
    external_ref: str = ""

    @property
    def duration_s(self) -> float:
        return (self.end - self.start).total_seconds()


@dataclass(frozen=True, slots=True)
class Antenna:
    id: str
    name: str
    station_id: str
    station_name: str
    latitude: float
    longitude: float
    altitude: float
    bands: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Spacecraft:
    id: str
    name: str
    catalog: str


class BookingPlan(Protocol):
    source: Contact
    target_antenna_id: str
    mission_profile_id: str
    identity: str


def api_key_from_env(variable: str = "KOGS_API_KEY") -> str:
    """Load and validate the KOGS API key from one environment variable."""

    value = os.getenv(variable, "")
    if not value:
        raise ValueError(f"{variable} is not configured")
    _authorization(value)
    return value


def _authorization(api_key: str) -> str:
    if not isinstance(api_key, str):
        raise TypeError("KOGS API key must be a string")
    if api_key != api_key.strip():
        raise ValueError("KOGS API key must not contain surrounding whitespace")
    if api_key.startswith(_AUTH_PREFIX):
        token = api_key.removeprefix(_AUTH_PREFIX)
    elif any(character.isspace() for character in api_key):
        raise ValueError("unsupported KOGS authorization scheme")
    else:
        token = api_key
    if not token or any(character.isspace() for character in token):
        raise ValueError("KOGS API key must be one non-whitespace token")
    if len(token) != 40:
        raise ValueError("KOGS API key must contain exactly 40 characters")
    if not token.isascii() or not token.isprintable():
        raise ValueError("KOGS API key must contain printable ASCII characters")
    return f"{_AUTH_PREFIX}{token}"


def headers(api_key: str, *, json_content: bool = False) -> dict[str, str]:
    result = {"Authorization": _authorization(api_key), "Accept": "application/json"}
    if json_content:
        result["Content-Type"] = "application/json"
    return result


def _request(
    method: str,
    path: str,
    api_key: str,
    *,
    base_url: str = KOGS_BASE_URL,
    timeout_seconds: float = KOGS_REQUEST_TIMEOUT_SECONDS,
    params: dict[str, Any] | None = None,
    json_body: dict[str, object] | None = None,
) -> dict:
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    response = requests.request(
        method,
        f"{base_url.rstrip('/')}/{path.lstrip('/')}",
        headers=headers(api_key, json_content=json_body is not None),
        timeout=timeout_seconds,
        params=params,
        json=json_body,
    )
    response.raise_for_status()
    try:
        payload = response.json()
    except requests.JSONDecodeError as exc:
        raise KogsError("KOGS returned malformed JSON") from exc
    if not isinstance(payload, dict):
        raise KogsError("KOGS response must be an object")
    return payload


def _unwrap(payload: object, key: str) -> dict:
    if not isinstance(payload, dict):
        raise KogsError(f"KOGS {key} response is not an object")
    value = payload.get(key, payload)
    if not isinstance(value, dict):
        raise KogsError(f"KOGS {key} response contains no {key}")
    return value


def _text(value: object) -> str | None:
    return None if value is None else str(value)


def _required_text(value: object, field: str) -> str:
    result = _text(value)
    if not result:
        raise KogsError(f"KOGS response is missing {field}")
    return result


def _number(value: object, field: str) -> float:
    try:
        return float(str(value))
    except (TypeError, ValueError) as exc:
        raise KogsError(f"KOGS response has invalid {field}") from exc


def _time(value: object, field: str, *, required: bool = True) -> dt.datetime | None:
    if value is None and not required:
        return None
    try:
        parsed = dt.datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError as exc:
        raise KogsError(f"KOGS response has invalid {field}") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        parsed = parsed.replace(tzinfo=dt.UTC)
    return parsed.astimezone(dt.UTC)


def _contact(payload: object) -> Contact:
    value = _unwrap(payload, "contact")
    start = _time(value.get("start_time"), "start_time")
    end = _time(value.get("end_time"), "end_time")
    assert start is not None and end is not None
    if end <= start:
        raise KogsError("KOGS contact stop must be later than its start")
    return Contact(
        id=_required_text(value.get("id"), "contact id"),
        spacecraft_id=_required_text(value.get("spacecraft_id"), "spacecraft_id"),
        system_id=_required_text(value.get("system_id"), "system_id"),
        station_id=_required_text(value.get("station_id"), "station_id"),
        mission_profile_id=_text(value.get("mission_profile_id")) or "",
        ephemeris_id=_required_text(value.get("ephemeris_id"), "ephemeris_id"),
        start=start,
        end=end,
        state=_text(value.get("state")) or "",
        external_ref=_text(value.get("external_ref")) or "",
    )


def _antenna(payload: object) -> Antenna:
    value = _unwrap(payload, "antenna")
    expanded = payload.get("expanded", {}) if isinstance(payload, dict) else {}
    station_id = _required_text(value.get("station"), "antenna station")
    station_name = "UNKNOWN"
    location = value.get("location")
    stations = expanded.get("stations", []) if isinstance(expanded, dict) else []
    if not isinstance(location, dict):
        location = {}
    for station in stations if isinstance(stations, list) else []:
        if isinstance(station, dict) and str(station.get("id")) == station_id:
            station_name = _text(station.get("name")) or station_name
            if not location and isinstance(station.get("location"), dict):
                location = station["location"]
            break
    bands = value.get("bands", [])
    band_names = tuple(
        str(item["type"])
        for item in bands
        if isinstance(bands, list) and isinstance(item, dict) and item.get("type")
    )
    return Antenna(
        id=_required_text(value.get("id"), "antenna id"),
        name=_required_text(value.get("name"), "antenna name"),
        station_id=station_id,
        station_name=station_name,
        latitude=_number(location.get("latitude"), "antenna latitude"),
        longitude=_number(location.get("longitude"), "antenna longitude"),
        altitude=_number(location.get("altitude"), "antenna altitude"),
        bands=band_names,
    )


def _spacecraft(payload: object) -> Spacecraft:
    value = _unwrap(payload, "spacecraft")
    raw_catalog = value.get("satellite_catalog_number") or value.get("norad_id")
    catalog_number = _number(raw_catalog, "spacecraft catalog ID")
    if not catalog_number.is_integer():
        raise KogsError("KOGS spacecraft catalog ID must be integral")
    catalog = str(int(catalog_number))
    if len(catalog) < 5:
        catalog = catalog.zfill(5)
    if len(catalog) not in {5, 9}:
        raise KogsError("KOGS catalog ID must contain 5 or 9 digits")
    return Spacecraft(
        id=_required_text(value.get("id"), "spacecraft id"),
        name=_required_text(value.get("name"), "spacecraft name"),
        catalog=catalog,
    )


def _ephemeris(payload: object) -> EphemerisMetadata:
    value = _unwrap(payload, "ephemeris")
    inline = value.get("inline") or {}
    if not isinstance(inline, dict):
        raise KogsError("KOGS ephemeris inline field must be an object")
    raw_payload = value.get("payload")
    serialized_payload = None
    if raw_payload is not None:
        serialized_payload = raw_payload if isinstance(raw_payload, str) else json.dumps(
            raw_payload, separators=(",", ":"), ensure_ascii=False
        )
    is_cui = value.get("is_cui")
    return EphemerisMetadata(
        ephemeris_id=_required_text(
            value.get("ephemeris_uuid") or value.get("id"), "ephemeris id"
        ),
        spacecraft_id=_required_text(value.get("spacecraft_uuid"), "ephemeris spacecraft"),
        kind=_text(value.get("kind")) or "",
        origin=_text(value.get("origin")),
        tenant_id=_text(value.get("tenant_uuid")),
        epoch=_time(value.get("epoch"), "ephemeris epoch", required=False),
        last_usable_at=_time(value.get("last_useable_at"), "last_useable_at", required=False),
        submitted_at=_time(value.get("submitted_at"), "submitted_at", required=False),
        submitted_by=_text(value.get("submitted_by")),
        tle=_text(inline.get("tle")),
        omm=_text(inline.get("omm")),
        oem=_text(inline.get("oem")),
        is_cui=is_cui if isinstance(is_cui, bool) else None,
        payload=serialized_payload,
    )


def _ephemeris_identity(ephemeris: EphemerisMetadata) -> tuple[str, str | None]:
    identities: list[tuple[str, str | None]] = []
    if ephemeris.tle:
        parsed = sk.TLE.from_lines(
            [line for line in ephemeris.tle.splitlines() if line.strip()]
        )
        tle = parsed[0] if isinstance(parsed, list) else parsed
        designator = str(tle.intl_desig).strip()
        year = int(designator[:2])
        full_year = 1900 + year if year >= 57 else 2000 + year
        identities.append((f"{full_year}-{designator[2:5]}{designator[5:]}", str(tle.satnum)))
    for document in (ephemeris.omm, ephemeris.oem):
        if not document:
            continue
        fields = dict(re.findall(
            r"(?m)^\s*(OBJECT_ID|NORAD_CAT_ID)\s*=\s*([^\s]+)", document
        ))
        if "OBJECT_ID" in fields:
            identities.append((fields["OBJECT_ID"], fields.get("NORAD_CAT_ID")))
    if not identities:
        raise KogsError("KOGS ephemeris contains no usable object identity")
    cospar_values = {identity[0] for identity in identities}
    catalogs = {identity[1] for identity in identities if identity[1]}
    if len(cospar_values) != 1 or len(catalogs) > 1:
        raise KogsError("KOGS ephemeris identities do not match")
    return identities[0][0], next(iter(catalogs), None)


def get_contact(api_key: str, contact_id: str, *, timeout_seconds: float = KOGS_REQUEST_TIMEOUT_SECONDS) -> Contact:
    return _contact(_request("GET", f"contacts/{contact_id}", api_key, timeout_seconds=timeout_seconds))


def get_spacecraft(api_key: str, spacecraft_id: str, *, timeout_seconds: float = KOGS_REQUEST_TIMEOUT_SECONDS) -> Spacecraft:
    return _spacecraft(_request("GET", f"spacecrafts/{spacecraft_id}", api_key, timeout_seconds=timeout_seconds))


def get_antenna(api_key: str, system_id: str, *, timeout_seconds: float = KOGS_REQUEST_TIMEOUT_SECONDS) -> Antenna:
    return _antenna(_request("GET", f"systems/antennas/{system_id}", api_key, timeout_seconds=timeout_seconds))


def get_ephemeris(api_key: str, ephemeris_id: str, *, timeout_seconds: float = KOGS_REQUEST_TIMEOUT_SECONDS) -> EphemerisMetadata:
    return _ephemeris(_request("GET", f"ephemeris/{ephemeris_id}", api_key, timeout_seconds=timeout_seconds))


def load_contact_metadata(api_key: str, contact_id: str, *, timeout_seconds: float = KOGS_REQUEST_TIMEOUT_SECONDS) -> ContactMetadata:
    """Resolve and cross-check all KOGS metadata for one pass."""

    contact = get_contact(api_key, contact_id, timeout_seconds=timeout_seconds)
    antenna = get_antenna(api_key, contact.system_id, timeout_seconds=timeout_seconds)
    spacecraft = get_spacecraft(api_key, contact.spacecraft_id, timeout_seconds=timeout_seconds)
    ephemeris = get_ephemeris(api_key, contact.ephemeris_id, timeout_seconds=timeout_seconds)
    if contact.id != contact_id:
        raise KogsError("KOGS contact identity does not match the request")
    if antenna.id != contact.system_id or antenna.station_id != contact.station_id:
        raise KogsError("KOGS antenna identity does not match the contact")
    if spacecraft.id != contact.spacecraft_id:
        raise KogsError("KOGS spacecraft identity does not match the contact")
    if ephemeris.spacecraft_id != contact.spacecraft_id:
        raise KogsError("KOGS ephemeris spacecraft does not match the contact")
    cospar, ephemeris_catalog = _ephemeris_identity(ephemeris)
    if ephemeris_catalog and ephemeris_catalog != spacecraft.catalog:
        raise KogsError("KOGS spacecraft and ephemeris catalog IDs do not match")
    vector = sk.itrfcoord(
        latitude_deg=antenna.latitude,
        longitude_deg=antenna.longitude,
        altitude=antenna.altitude,
    ).vector
    return ContactMetadata(
        spacecraft_id=contact.spacecraft_id,
        system_id=contact.system_id,
        station_id=contact.station_id,
        ephemeris_id=contact.ephemeris_id,
        antenna=antenna.name,
        location=antenna.station_name,
        latitude=antenna.latitude,
        longitude=antenna.longitude,
        altitude=antenna.altitude,
        ecef=(float(vector[0]), float(vector[1]), float(vector[2])),
        spacecraft=spacecraft.name,
        cospar=cospar,
        catalog=spacecraft.catalog,
        start=contact.start,
        stop=contact.end,
        contact_id=contact.id,
        ephemeris=ephemeris,
    )


def validate_credentials(api_key: str, *, base_url: str = KOGS_BASE_URL, timeout_seconds: float = KOGS_REQUEST_TIMEOUT_SECONDS) -> None:
    _request("GET", "contacts", api_key, base_url=base_url, timeout_seconds=timeout_seconds, params={"limit": 1})


def list_contacts(
    api_key: str,
    start: dt.datetime,
    stop: dt.datetime,
    *,
    station_ids: Sequence[str] = (),
    system_ids: Sequence[str] = (),
    base_url: str = KOGS_BASE_URL,
    timeout_seconds: float = KOGS_REQUEST_TIMEOUT_SECONDS,
) -> list[Contact]:
    payload = _request(
        "GET", "contacts", api_key, base_url=base_url, timeout_seconds=timeout_seconds,
        params={"start_time": _utc_text(start), "end_time": _utc_text(stop), "station_ids": list(station_ids), "system_ids": list(system_ids)},
    )
    values = payload.get("data")
    if not isinstance(values, list):
        raise KogsError("KOGS contacts response contains no data list")
    return [_contact(value) for value in values]


def book_shadow(
    api_key: str,
    plan: BookingPlan,
    *,
    mutation_contract_confirmed: bool = False,
    base_url: str = KOGS_BASE_URL,
    timeout_seconds: float = KOGS_REQUEST_TIMEOUT_SECONDS,
) -> Contact:
    _require_mutation_contract(mutation_contract_confirmed)
    return _contact(_request(
        "POST", "contacts/shadow", api_key, base_url=base_url, timeout_seconds=timeout_seconds,
        json_body={"source_contact_id": plan.source.id, "system_id": plan.target_antenna_id, "mission_profile_id": plan.mission_profile_id, "external_ref": plan.identity},
    ))


def assign_ephemeris(
    api_key: str,
    contact_id: str,
    ephemeris_id: str,
    *,
    mutation_contract_confirmed: bool = False,
    base_url: str = KOGS_BASE_URL,
    timeout_seconds: float = KOGS_REQUEST_TIMEOUT_SECONDS,
) -> None:
    _require_mutation_contract(mutation_contract_confirmed)
    _request("PUT", f"contacts/{contact_id}/ephemeris", api_key, base_url=base_url, timeout_seconds=timeout_seconds, json_body={"ephemeris_id": ephemeris_id, "mode": "manual"})


def cancel_contact(
    api_key: str,
    contact_id: str,
    *,
    mutation_contract_confirmed: bool = False,
    base_url: str = KOGS_BASE_URL,
    timeout_seconds: float = KOGS_REQUEST_TIMEOUT_SECONDS,
) -> None:
    _require_mutation_contract(mutation_contract_confirmed)
    _request("POST", f"contacts/{contact_id}/cancel", api_key, base_url=base_url, timeout_seconds=timeout_seconds, json_body={})


def _require_mutation_contract(confirmed: bool) -> None:
    if not confirmed:
        raise KogsError("KOGS mutation contract has not been confirmed")


def _utc_text(value: dt.datetime) -> str:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamp must be timezone-aware")
    return value.astimezone(dt.UTC).isoformat().replace("+00:00", "Z")
