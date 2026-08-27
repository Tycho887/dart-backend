"""Authoritative KOGS and geocoder metadata for KSAT TDM headers."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from datetime import date

import requests
import satkit as sk

from dart.io.kogs import (
    KOGS_REQUEST_TIMEOUT_SECONDS,
    get_antenna,
    get_contact,
    get_spacecraft,
    parse_reservation,
    parse_response,
    parse_satellite,
)
from dart.io.ksat_tdm import KsatSite, KsatSpacecraft
from dart.io.utils import create_api_auth


DEFAULT_GEOCODER_URL = "https://nominatim.openstreetmap.org/reverse"
DEFAULT_GEOCODER_USER_AGENT = "DART-KSAT-TDM/1.0"
DEFAULT_GEOCODER_TIMEOUT_SECONDS = 10.0


def _positive_timeout(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite positive number")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be a finite positive number")
    return result


@dataclass(frozen=True, slots=True)
class KogsMetadataConfig:
    """Expected KOGS identities for the one contact being exported."""

    spacecraft_id: str
    system_id: str
    station_id: str
    timeout_seconds: float = KOGS_REQUEST_TIMEOUT_SECONDS

    def __post_init__(self) -> None:
        for name in ("spacecraft_id", "system_id", "station_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
            object.__setattr__(self, name, value.strip())
        object.__setattr__(
            self,
            "timeout_seconds",
            _positive_timeout("kogs.timeout_seconds", self.timeout_seconds),
        )


@dataclass(frozen=True, slots=True)
class GeocoderConfig:
    """Reverse-geocoder endpoint settings; no credentials are stored here."""

    url: str = DEFAULT_GEOCODER_URL
    user_agent: str = DEFAULT_GEOCODER_USER_AGENT
    timeout_seconds: float = DEFAULT_GEOCODER_TIMEOUT_SECONDS

    def __post_init__(self) -> None:
        for name in ("url", "user_agent"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"geocoder.{name} must be a non-empty string")
            object.__setattr__(self, name, value.strip())
        if not self.url.startswith(("https://", "http://")):
            raise ValueError("geocoder.url must be an HTTP(S) URL")
        object.__setattr__(
            self,
            "timeout_seconds",
            _positive_timeout("geocoder.timeout_seconds", self.timeout_seconds),
        )


@dataclass(frozen=True, slots=True)
class KsatSiteOverrides:
    """Reviewed site facts that are not currently exposed by KOGS."""

    name: str | None = None
    pedestal_offset_m: float | None = None
    tlt_calibration_date: date | None = None
    tlt_band: str | None = None

    def __post_init__(self) -> None:
        # Reuse the typed writer's validation without supplying fabricated
        # identity or coordinate values.
        probe = KsatSite(
            identifier="VALIDATION",
            name=self.name,
            pedestal_offset_m=self.pedestal_offset_m,
            tlt_calibration_date=self.tlt_calibration_date,
            tlt_band=self.tlt_band,
        )
        object.__setattr__(self, "name", probe.name)
        object.__setattr__(self, "pedestal_offset_m", probe.pedestal_offset_m)
        object.__setattr__(self, "tlt_calibration_date", probe.tlt_calibration_date)
        object.__setattr__(self, "tlt_band", probe.tlt_band)
        if (self.tlt_band is None) != (self.tlt_calibration_date is None):
            raise ValueError(
                "site.tlt_band and site.tlt_calibration_date must be supplied together"
            )


@dataclass(frozen=True, slots=True)
class KsatMetadataResult:
    site: KsatSite
    spacecraft: KsatSpacecraft
    warnings: tuple[str, ...] = field(default_factory=tuple)


def _ascii_component(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    candidate = " ".join(value.split()).strip(" ,")
    if not candidate or not candidate.isascii():
        return None
    return candidate


def reverse_geocode_location(
    latitude_deg: float,
    longitude_deg: float,
    config: GeocoderConfig,
) -> str:
    """Resolve an ASCII locality, region, and country from WGS-84 coordinates."""

    response = requests.get(
        config.url,
        params={
            "format": "jsonv2",
            "lat": str(latitude_deg),
            "lon": str(longitude_deg),
            "zoom": "18",
            "addressdetails": "1",
            "accept-language": "en",
        },
        headers={
            "User-Agent": config.user_agent,
            "Accept": "application/json",
            "Accept-Language": "en",
        },
        timeout=config.timeout_seconds,
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict) or not isinstance(payload.get("address"), dict):
        raise ValueError("reverse geocoder returned no address")
    address = payload["address"]
    locality = next(
        (
            component
            for key in ("city", "town", "village", "municipality", "hamlet")
            if (component := _ascii_component(address.get(key))) is not None
        ),
        None,
    )
    region = next(
        (
            component
            for key in ("state", "region", "state_district", "county")
            if (component := _ascii_component(address.get(key))) is not None
        ),
        None,
    )
    country = _ascii_component(address.get("country"))
    if locality is None or country is None:
        raise ValueError("reverse geocoder returned no usable ASCII locality/country")
    components: list[str] = []
    seen: set[str] = set()
    for component in (locality, region, country):
        if component is None or component.casefold() in seen:
            continue
        seen.add(component.casefold())
        components.append(component)
    return ", ".join(components)


def _catalog_id(value: float | None) -> str | None:
    if value is None or not math.isfinite(value):
        return None
    return str(int(value)) if value.is_integer() else None


def load_ksat_contact_metadata(
    *,
    contact_id: str,
    kogs: KogsMetadataConfig,
    geocoder: GeocoderConfig,
    site: KsatSiteOverrides,
    spacecraft: KsatSpacecraft,
) -> KsatMetadataResult:
    """Fetch and validate KOGS facts, then construct one runtime TDM header site."""

    api_key = os.getenv("KOGS_API_KEY", "").strip()
    if not api_key:
        raise ValueError("KOGS_API_KEY is required for KSAT TDM export")
    auth = create_api_auth(api_key)
    timeout = kogs.timeout_seconds

    contact_payload = get_contact(auth, contact_id, timeout_seconds=timeout)
    contact_raw = (
        contact_payload.get("contact") if isinstance(contact_payload, dict) else None
    )
    if not isinstance(contact_raw, dict):
        raise ValueError("KOGS contact response contains no contact")
    reservation = parse_reservation(contact_raw)
    expected = {
        "contact": (reservation.id, contact_id),
        "spacecraft": (reservation.spacecraft_id, kogs.spacecraft_id),
        "system": (reservation.system_id, kogs.system_id),
        "station": (reservation.station_id, kogs.station_id),
    }
    for label, (actual, configured) in expected.items():
        if actual != configured:
            raise ValueError(
                f"KOGS {label} identity mismatch: expected {configured!r}, got {actual!r}"
            )

    antenna = parse_response(
        get_antenna(auth, kogs.system_id, timeout_seconds=timeout)
    )
    if antenna.antenna_id != kogs.system_id:
        raise ValueError(
            "KOGS antenna identity mismatch: "
            f"expected {kogs.system_id!r}, got {antenna.antenna_id!r}"
        )
    if antenna.station_id != kogs.station_id:
        raise ValueError(
            "KOGS antenna station mismatch: "
            f"expected {kogs.station_id!r}, got {antenna.station_id!r}"
        )
    if not antenna.antenna_name:
        raise ValueError("KOGS antenna response contains no antenna name")
    if None in (antenna.latitude, antenna.longitude, antenna.altitude):
        raise ValueError("KOGS antenna response contains incomplete WGS-84 coordinates")

    spacecraft_payload = get_spacecraft(
        auth, kogs.spacecraft_id, timeout_seconds=timeout
    )
    spacecraft_raw = (
        spacecraft_payload.get("spacecraft")
        if isinstance(spacecraft_payload, dict) and "spacecraft" in spacecraft_payload
        else spacecraft_payload
    )
    satellite = parse_satellite(spacecraft_raw)
    if satellite.id != kogs.spacecraft_id:
        raise ValueError(
            "KOGS spacecraft identity mismatch: "
            f"expected {kogs.spacecraft_id!r}, got {satellite.id!r}"
        )
    kogs_catalog = _catalog_id(
        satellite.satellite_catalog_number
        if satellite.satellite_catalog_number is not None
        else satellite.norad_id
    )
    if spacecraft.catalog_id and kogs_catalog and spacecraft.catalog_id != kogs_catalog:
        raise ValueError(
            "KOGS spacecraft catalog mismatch: "
            f"configured {spacecraft.catalog_id!r}, got {kogs_catalog!r}"
        )
    runtime_spacecraft = KsatSpacecraft(
        identifier=spacecraft.identifier,
        name=satellite.name or spacecraft.name,
        cospar_id=spacecraft.cospar_id,
        catalog_id=kogs_catalog or spacecraft.catalog_id,
    )

    latitude = float(antenna.latitude)
    longitude = float(antenna.longitude)
    altitude = float(antenna.altitude)
    vector = sk.itrfcoord(
        latitude_deg=latitude,
        longitude_deg=longitude,
        altitude=altitude,
    ).vector
    ecef = tuple(round(float(component), 3) for component in vector)

    warnings: list[str] = []
    try:
        location = reverse_geocode_location(latitude, longitude, geocoder)
    except (OSError, ValueError, requests.RequestException) as exc:
        location = None
        warnings.append(f"ground-station location is UNKNOWN: {exc}")
    if site.pedestal_offset_m is None:
        warnings.append("pedestal offset is UNKNOWN; no reviewed value was configured")
    if site.tlt_calibration_date is None:
        warnings.append("TLT calibration date is UNKNOWN; no reviewed value was configured")

    runtime_site = KsatSite(
        identifier=antenna.antenna_name,
        name=site.name,
        location=location,
        latitude_deg=latitude,
        longitude_deg=longitude,
        altitude_m=altitude,
        ecef_x_m=ecef[0],
        ecef_y_m=ecef[1],
        ecef_z_m=ecef[2],
        pedestal_offset_m=site.pedestal_offset_m,
        tlt_calibration_date=site.tlt_calibration_date,
        tlt_band=site.tlt_band,
    )
    return KsatMetadataResult(runtime_site, runtime_spacecraft, tuple(warnings))


__all__ = [
    "DEFAULT_GEOCODER_TIMEOUT_SECONDS",
    "DEFAULT_GEOCODER_URL",
    "DEFAULT_GEOCODER_USER_AGENT",
    "GeocoderConfig",
    "KogsMetadataConfig",
    "KsatMetadataResult",
    "KsatSiteOverrides",
    "load_ksat_contact_metadata",
    "reverse_geocode_location",
]
