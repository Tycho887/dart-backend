"""Authoritative KOGS and geocoder metadata for KSAT TDM headers."""

from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Protocol

import requests
import satkit as sk

from dart.io.kogs import (
    KOGS_REQUEST_TIMEOUT_SECONDS,
    get_antenna,
    get_contact,
    get_ephemeris,
    get_spacecraft,
    parse_ephemeris,
    parse_reservation,
    parse_response,
    parse_satellite,
)
from dart.io.ksat_sources import KsatAuthority, KsatProvenance
from dart.io.ksat_tdm import KsatSite, KsatSpacecraft, TrackMetadata
from dart.io.utils import create_api_auth

DEFAULT_GEOCODER_URL = "https://nominatim.openstreetmap.org/reverse"
DEFAULT_GEOCODER_USER_AGENT = "DART-KSAT-TDM/1.0"
DEFAULT_GEOCODER_TIMEOUT_SECONDS = 10.0

_COSPAR_ID = re.compile(r"^(?P<year>\d{4})-(?P<number>\d{3})(?P<piece>[A-Z]{1,3})$")
_TLE_DESIGNATOR = re.compile(r"^(?P<year>\d{2})(?P<number>\d{3})(?P<piece>[A-Z]{1,3})$")
_CATALOG_ID = re.compile(r"^(?:\d{5}|\d{9})$")


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
class KsatSpacecraftOverrides:
    """Reviewed fallbacks for values absent from KOGS and its ephemeris.

    ``identifier`` is retained as a deprecated compatibility fallback. New
    configurations should let the exporter select COSPAR, then catalog ID.
    """

    identifier: str | None = None
    name: str | None = None
    cospar_id: str | None = None
    catalog_id: str | None = None

    def __post_init__(self) -> None:
        probe = KsatSpacecraft(
            identifier=(
                self.identifier or self.cospar_id or self.catalog_id or "2000-001A"
            ),
            name=self.name,
            cospar_id=self.cospar_id,
            catalog_id=self.catalog_id,
        )
        object.__setattr__(
            self, "identifier", probe.identifier if self.identifier else None
        )
        object.__setattr__(self, "name", probe.name)
        object.__setattr__(self, "cospar_id", probe.cospar_id)
        object.__setattr__(self, "catalog_id", probe.catalog_id)
        if self.cospar_id is not None and not _COSPAR_ID.fullmatch(self.cospar_id):
            raise ValueError("spacecraft.cospar_id must have form YYYY-NNNP")
        if self.catalog_id is not None and not _CATALOG_ID.fullmatch(self.catalog_id):
            raise ValueError("spacecraft.catalog_id must contain 5 or 9 digits")


@dataclass(frozen=True, slots=True)
class SpacecraftIdentityRequest:
    auth: str
    ephemeris_id: str | None
    catalog_id: str | None
    kogs_name: str | None
    configured: KsatSpacecraftOverrides
    timeout_seconds: float


@dataclass(frozen=True, slots=True)
class SpacecraftIdentityResult:
    spacecraft: KsatSpacecraft
    provenance: tuple[KsatProvenance, ...]
    warnings: tuple[str, ...] = ()


class SpacecraftIdentityProvider(Protocol):
    """Resolve the spacecraft identifiers used in the TDM header and filename."""

    def resolve(
        self, request: SpacecraftIdentityRequest
    ) -> SpacecraftIdentityResult: ...


@dataclass(frozen=True, slots=True)
class CalibrationRequest:
    site_identifier: str
    contact_start: datetime | None
    track: TrackMetadata | None


@dataclass(frozen=True, slots=True)
class CalibrationResult:
    site: KsatSiteOverrides
    track: TrackMetadata | None
    provenance: tuple[KsatProvenance, ...]


class CalibrationProvider(Protocol):
    """Resolve antenna calibration facts without applying physical models."""

    def resolve(self, request: CalibrationRequest) -> CalibrationResult: ...


@dataclass(frozen=True, slots=True)
class ConfigCalibrationProvider:
    """Current reviewed-configuration source, replaceable by a MEOS adapter."""

    site: KsatSiteOverrides

    def resolve(self, request: CalibrationRequest) -> CalibrationResult:
        calibration_date = self.site.tlt_calibration_date
        if (
            calibration_date is not None
            and request.contact_start is not None
            and calibration_date > request.contact_start.date()
        ):
            raise ValueError(
                "TLT calibration date must not be later than the contact start"
            )
        facts: list[KsatProvenance] = []
        for field_name, value in (
            ("site.pedestal_offset_m", self.site.pedestal_offset_m),
            ("site.tlt_calibration_date", self.site.tlt_calibration_date),
            ("site.tlt_band", self.site.tlt_band),
        ):
            if value is not None:
                facts.append(KsatProvenance(field_name, KsatAuthority.CONFIG))
        if request.track is not None:
            for field_name in (
                "transmit_delay_s",
                "receive_delay_s",
                "correction_range_s",
                "correction_doppler_hz",
            ):
                if getattr(request.track, field_name) is not None:
                    facts.append(
                        KsatProvenance(f"track.{field_name}", KsatAuthority.CONFIG)
                    )
        return CalibrationResult(self.site, request.track, tuple(facts))


class MeosCalibrationProvider:
    """Reserved integration point for the antenna-local MEOS service."""

    def resolve(self, request: CalibrationRequest) -> CalibrationResult:
        raise NotImplementedError(
            "MEOS calibration integration is not available; use reviewed configuration"
        )


@dataclass(frozen=True, slots=True)
class KsatMetadataResult:
    site: KsatSite
    spacecraft: KsatSpacecraft
    warnings: tuple[str, ...] = field(default_factory=tuple)
    track: TrackMetadata | None = None
    provenance: tuple[KsatProvenance, ...] = field(default_factory=tuple)
    antenna_bands: tuple[str, ...] = field(default_factory=tuple)


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
    result = str(int(value)) if value.is_integer() else None
    return result if result is not None and _CATALOG_ID.fullmatch(result) else None


def _antenna_bands(value: str | None) -> tuple[str, ...]:
    if not value:
        return ()
    normalized: list[str] = []
    aliases = {"S": "S", "X": "X", "KA": "Ka"}
    for item in value.split(","):
        candidate = item.strip().upper().removesuffix("-BAND").strip()
        band = aliases.get(candidate)
        if band is not None and band not in normalized:
            normalized.append(band)
    return tuple(normalized)


def _tle_identity(inline_tle: str | None) -> tuple[str | None, str | None]:
    if not inline_tle:
        return None, None
    line_1 = next(
        (line.rstrip() for line in inline_tle.splitlines() if line.startswith("1 ")),
        None,
    )
    if line_1 is None or len(line_1) < 17:
        return None, None
    catalog = line_1[2:7].strip()
    designator = line_1[9:17].strip().upper()
    match = _TLE_DESIGNATOR.fullmatch(designator)
    if match is None:
        return None, catalog if _CATALOG_ID.fullmatch(catalog) else None
    short_year = int(match.group("year"))
    year = 1900 + short_year if short_year >= 57 else 2000 + short_year
    cospar = f"{year:04d}-{match.group('number')}{match.group('piece')}"
    return cospar, catalog if _CATALOG_ID.fullmatch(catalog) else None


def _omm_value(inline_omm: str, key: str) -> str | None:
    escaped = re.escape(key)
    patterns = (
        rf"(?mi)^\s*{escaped}\s*=\s*([^\s#]+)",
        rf'(?i)["\']{escaped}["\']\s*:\s*["\']([^"\']+)',
        rf"(?is)<{escaped}>\s*([^<]+)\s*</{escaped}>",
    )
    for pattern in patterns:
        match = re.search(pattern, inline_omm)
        if match is not None:
            return match.group(1).strip()
    return None


def _omm_identity(inline_omm: str | None) -> tuple[str | None, str | None]:
    if not inline_omm:
        return None, None
    cospar = (_omm_value(inline_omm, "OBJECT_ID") or "").upper() or None
    catalog = _omm_value(inline_omm, "NORAD_CAT_ID")
    return (
        cospar if cospar is not None and _COSPAR_ID.fullmatch(cospar) else None,
        catalog if catalog is not None and _CATALOG_ID.fullmatch(catalog) else None,
    )


def extract_ephemeris_identity(
    *, inline_tle: str | None, inline_omm: str | None
) -> tuple[str | None, str | None]:
    """Return ``(COSPAR, catalog)`` from contact-specific TLE/OMM content."""

    tle_cospar, tle_catalog = _tle_identity(inline_tle)
    omm_cospar, omm_catalog = _omm_identity(inline_omm)
    if tle_cospar and omm_cospar and tle_cospar != omm_cospar:
        raise ValueError(
            f"KOGS ephemeris COSPAR mismatch: TLE {tle_cospar!r}, OMM {omm_cospar!r}"
        )
    if tle_catalog and omm_catalog and tle_catalog != omm_catalog:
        raise ValueError(
            f"KOGS ephemeris catalog mismatch: TLE {tle_catalog!r}, OMM {omm_catalog!r}"
        )
    return tle_cospar or omm_cospar, tle_catalog or omm_catalog


class KogsEphemerisIdentityProvider:
    """Resolve participant identity from KOGS spacecraft and contact ephemeris."""

    def resolve(self, request: SpacecraftIdentityRequest) -> SpacecraftIdentityResult:
        configured = request.configured
        cospar: str | None = None
        ephemeris_catalog: str | None = None
        facts: list[KsatProvenance] = []
        warnings: list[str] = []
        if request.ephemeris_id:
            ephemeris_payload = get_ephemeris(
                request.auth,
                request.ephemeris_id,
                timeout_seconds=request.timeout_seconds,
            )
            ephemeris_raw = (
                ephemeris_payload.get("ephemeris")
                if isinstance(ephemeris_payload, dict)
                and isinstance(ephemeris_payload.get("ephemeris"), dict)
                else ephemeris_payload
            )
            ephemeris = parse_ephemeris(ephemeris_raw)
            cospar, ephemeris_catalog = extract_ephemeris_identity(
                inline_tle=ephemeris.inline_tle,
                inline_omm=ephemeris.inline_omm,
            )
            if cospar is not None:
                facts.append(
                    KsatProvenance("spacecraft.cospar_id", KsatAuthority.KOGS_EPHEMERIS)
                )
        catalog = request.catalog_id or ephemeris_catalog or configured.catalog_id
        if (
            request.catalog_id
            and ephemeris_catalog
            and request.catalog_id != ephemeris_catalog
        ):
            raise ValueError(
                "KOGS spacecraft/ephemeris catalog mismatch: "
                f"{request.catalog_id!r} != {ephemeris_catalog!r}"
            )
        if configured.catalog_id and catalog and configured.catalog_id != catalog:
            raise ValueError(
                "KOGS spacecraft catalog mismatch: "
                f"configured {configured.catalog_id!r}, got {catalog!r}"
            )
        if configured.cospar_id and cospar and configured.cospar_id != cospar:
            raise ValueError(
                "KOGS ephemeris COSPAR mismatch: "
                f"configured {configured.cospar_id!r}, got {cospar!r}"
            )
        if cospar is None and configured.cospar_id is not None:
            cospar = configured.cospar_id
            facts.append(KsatProvenance("spacecraft.cospar_id", KsatAuthority.CONFIG))
            warnings.append("COSPAR ID came from reviewed fallback configuration")
        if catalog is not None:
            authority = (
                KsatAuthority.KOGS
                if request.catalog_id is not None
                else KsatAuthority.KOGS_EPHEMERIS
                if ephemeris_catalog is not None
                else KsatAuthority.CONFIG
            )
            facts.append(KsatProvenance("spacecraft.catalog_id", authority))

        participant = cospar or catalog
        if participant is None and configured.identifier is not None:
            participant = configured.identifier
            warnings.append(
                "spacecraft.identifier is a deprecated fallback; configure or derive "
                "COSPAR/catalog identity"
            )
            facts.append(KsatProvenance("spacecraft.identifier", KsatAuthority.CONFIG))
        if participant is None:
            raise ValueError(
                "KOGS supplied no usable COSPAR or catalog participant identity"
            )
        if not (
            _COSPAR_ID.fullmatch(participant) or _CATALOG_ID.fullmatch(participant)
        ):
            raise ValueError(
                "spacecraft participant must be a COSPAR ID or a 5-/9-digit catalog ID"
            )
        name = request.kogs_name or configured.name
        if name is not None:
            facts.append(
                KsatProvenance(
                    "spacecraft.name",
                    KsatAuthority.KOGS if request.kogs_name else KsatAuthority.CONFIG,
                )
            )
        return SpacecraftIdentityResult(
            spacecraft=KsatSpacecraft(
                identifier=participant,
                name=name,
                cospar_id=cospar,
                catalog_id=catalog,
            ),
            provenance=tuple(facts),
            warnings=tuple(warnings),
        )


def load_ksat_contact_metadata(
    *,
    contact_id: str,
    kogs: KogsMetadataConfig,
    geocoder: GeocoderConfig,
    site: KsatSiteOverrides,
    spacecraft: KsatSpacecraft | KsatSpacecraftOverrides,
    contact_start: datetime | None = None,
    track: TrackMetadata | None = None,
    identity_provider: SpacecraftIdentityProvider | None = None,
    calibration_provider: CalibrationProvider | None = None,
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
                f"KOGS {label} identity mismatch: expected {configured!r}, "
                f"got {actual!r}"
            )

    antenna = parse_response(get_antenna(auth, kogs.system_id, timeout_seconds=timeout))
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
    configured_spacecraft = (
        spacecraft
        if isinstance(spacecraft, KsatSpacecraftOverrides)
        else KsatSpacecraftOverrides(
            identifier=spacecraft.identifier,
            name=spacecraft.name,
            cospar_id=spacecraft.cospar_id,
            catalog_id=spacecraft.catalog_id,
        )
    )
    identity = (identity_provider or KogsEphemerisIdentityProvider()).resolve(
        SpacecraftIdentityRequest(
            auth=auth,
            ephemeris_id=reservation.ephemeris_id,
            catalog_id=kogs_catalog,
            kogs_name=satellite.name,
            configured=configured_spacecraft,
            timeout_seconds=timeout,
        )
    )
    calibration = (calibration_provider or ConfigCalibrationProvider(site)).resolve(
        CalibrationRequest(
            site_identifier=antenna.antenna_name,
            contact_start=contact_start,
            track=track,
        )
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

    warnings: list[str] = list(identity.warnings)
    try:
        location = reverse_geocode_location(latitude, longitude, geocoder)
    except (OSError, ValueError, requests.RequestException) as exc:
        location = None
        warnings.append(f"ground-station location is UNKNOWN: {exc}")
    resolved_site = calibration.site
    if resolved_site.pedestal_offset_m is None:
        warnings.append("pedestal offset is UNKNOWN; no reviewed value was configured")
    if resolved_site.tlt_calibration_date is None:
        warnings.append(
            "TLT calibration date is UNKNOWN; no reviewed value was configured"
        )

    runtime_site = KsatSite(
        identifier=antenna.antenna_name,
        name=resolved_site.name,
        location=location,
        latitude_deg=latitude,
        longitude_deg=longitude,
        altitude_m=altitude,
        ecef_x_m=ecef[0],
        ecef_y_m=ecef[1],
        ecef_z_m=ecef[2],
        pedestal_offset_m=resolved_site.pedestal_offset_m,
        tlt_calibration_date=resolved_site.tlt_calibration_date,
        tlt_band=resolved_site.tlt_band,
    )
    provenance = (
        KsatProvenance("site.identifier", KsatAuthority.KOGS),
        KsatProvenance("site.wgs84", KsatAuthority.KOGS),
        KsatProvenance("site.ecef", KsatAuthority.DERIVED, "satkit ITRF"),
        *(
            (KsatProvenance("site.location", KsatAuthority.GEOCODER),)
            if location is not None
            else ()
        ),
        *identity.provenance,
        *calibration.provenance,
    )
    return KsatMetadataResult(
        runtime_site,
        identity.spacecraft,
        tuple(warnings),
        calibration.track,
        provenance,
        _antenna_bands(antenna.bands_types),
    )


__all__ = [
    "DEFAULT_GEOCODER_TIMEOUT_SECONDS",
    "DEFAULT_GEOCODER_URL",
    "DEFAULT_GEOCODER_USER_AGENT",
    "GeocoderConfig",
    "CalibrationProvider",
    "CalibrationRequest",
    "CalibrationResult",
    "ConfigCalibrationProvider",
    "KogsEphemerisIdentityProvider",
    "KogsMetadataConfig",
    "KsatMetadataResult",
    "KsatSiteOverrides",
    "KsatSpacecraftOverrides",
    "MeosCalibrationProvider",
    "SpacecraftIdentityProvider",
    "SpacecraftIdentityRequest",
    "SpacecraftIdentityResult",
    "extract_ephemeris_identity",
    "load_ksat_contact_metadata",
    "reverse_geocode_location",
]
