"""Shared metadata helpers for compact KSAT delivery writers."""

from __future__ import annotations

import datetime as dt
import os
import re
from dataclasses import dataclass

import satkit as sk

from dart.io import kogs
from dart.io.utils import create_api_auth

BANDS = {"S", "X", "Ka"}
IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
COLUMN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
COSPAR = re.compile(r"^\d{4}-\d{3}[A-Z]{1,3}$")


@dataclass(frozen=True, slots=True)
class ContactMetadata:
    antenna: str
    location: str
    latitude: float
    longitude: float
    altitude: float
    ecef: tuple[float, float, float]
    spacecraft: str
    cospar: str
    catalog: str
    start: dt.datetime
    stop: dt.datetime


def utc(value: object, label: str) -> dt.datetime:
    if isinstance(value, str):
        parsed = dt.datetime.fromisoformat(value)
    elif isinstance(value, dt.datetime):
        parsed = value
    else:
        raise ValueError(f"invalid {label} epoch {value!r}")  # noqa: TRY004
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        parsed = parsed.replace(tzinfo=dt.UTC)
    return parsed.astimezone(dt.UTC)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_contact_metadata(
    contact_id: str,
    band: str,
    timeout_seconds: float,
    product: str,
) -> ContactMetadata:
    """Load and cross-check the KOGS identity for one delivery product."""
    key = os.getenv("KOGS_API_KEY", "")
    require(bool(key), f"KOGS_API_KEY is required for {product} export")
    auth = create_api_auth(key)
    raw = kogs.get_contact(auth, contact_id, timeout_seconds=timeout_seconds)
    contact = kogs.parse_reservation(kogs.unwrap_payload(raw, "contact"))
    required = (
        contact.id,
        contact.spacecraft_id,
        contact.system_id,
        contact.station_id,
        contact.start_time,
        contact.end_time,
        contact.ephemeris_id,
    )
    require(
        not any(value is None for value in required) and contact.id == contact_id,
        "KOGS contact is incomplete or does not match the request",
    )
    antenna = kogs.parse_response(
        kogs.get_antenna(auth, contact.system_id, timeout_seconds=timeout_seconds)
    )
    spacecraft_raw = kogs.get_spacecraft(
        auth, contact.spacecraft_id, timeout_seconds=timeout_seconds
    )
    satellite = kogs.parse_satellite(kogs.unwrap_payload(spacecraft_raw, "spacecraft"))
    ephemeris_raw = kogs.get_ephemeris(
        auth, contact.ephemeris_id, timeout_seconds=timeout_seconds
    )
    ephemeris = kogs.parse_ephemeris(
        kogs.unwrap_payload(ephemeris_raw, "ephemeris")
    )
    require(
        antenna.antenna_id == contact.system_id
        and antenna.station_id == contact.station_id,
        "KOGS antenna identity does not match the contact",
    )
    require(
        satellite.id == contact.spacecraft_id and bool(satellite.name),
        "KOGS spacecraft identity does not match the contact",
    )
    require(
        bool(antenna.antenna_name)
        and None not in (antenna.latitude, antenna.longitude, antenna.altitude),
        "KOGS antenna metadata is incomplete",
    )
    supported_bands = {
        item.strip() for item in (antenna.bands_types or "").split(",") if item.strip()
    }
    require(
        not supported_bands or band in supported_bands,
        f"KOGS antenna does not support {band}-band",
    )
    require(
        not ephemeris.spacecraft_uuid
        or ephemeris.spacecraft_uuid == contact.spacecraft_id,
        "KOGS ephemeris spacecraft does not match the contact",
    )
    catalog_value = satellite.satellite_catalog_number or satellite.norad_id
    require(
        catalog_value is not None and float(catalog_value).is_integer(),
        "KOGS spacecraft has no integral catalog ID",
    )
    catalog = str(int(catalog_value))
    if len(catalog) < 5:
        catalog = catalog.zfill(5)
    require(len(catalog) in {5, 9}, "KOGS catalog ID must contain 5 or 9 digits")
    cospar, ephemeris_catalog = kogs.parse_ephemeris_identity(ephemeris)
    require(
        ephemeris_catalog == catalog,
        "KOGS spacecraft and ephemeris catalog IDs do not match",
    )
    start = utc(contact.start_time, product)
    stop = utc(contact.end_time, product)
    require(stop > start, "KOGS contact stop must be later than its start")
    vector = sk.itrfcoord(
        latitude_deg=antenna.latitude,
        longitude_deg=antenna.longitude,
        altitude=antenna.altitude,
    ).vector
    return ContactMetadata(
        antenna=antenna.antenna_name,
        location=antenna.station_name or "UNKNOWN",
        latitude=antenna.latitude,
        longitude=antenna.longitude,
        altitude=antenna.altitude,
        ecef=tuple(float(value) for value in vector),
        spacecraft=satellite.name,
        cospar=cospar,
        catalog=catalog,
        start=start,
        stop=stop,
    )


def validate_filename_identity(metadata: ContactMetadata) -> None:
    if not IDENTIFIER.fullmatch(metadata.antenna) or not COSPAR.fullmatch(
        metadata.cospar
    ):
        raise ValueError("KOGS identifiers are not valid KSAT filename components")
