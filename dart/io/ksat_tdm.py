"""Strict KSAT profile writer for CCSDS 503.0-B-2 TDM files.

This module intentionally lives beside, rather than replacing, :mod:`dart.io.tdm`.
The latter serializes DART solver inputs; this module represents the KSAT TRACK,
ANGLE, and SIGMET delivery products described by the documents in ``TDM-docs``.

Values supplied to these models are already in their wire units: seconds for
round-trip range delay, Hz for carrier frequency, degrees for angles, dBW for
carrier power, and dB-Hz for noise-density ratios.  No physical observable is
derived or reinterpreted by this serializer.
"""

from __future__ import annotations

import datetime as dt
import math
import re
from dataclasses import dataclass, field
from decimal import Decimal
from enum import StrEnum
from pathlib import Path
from typing import TypeAlias


TDM_VERSION = "2.0"
ORIGINATOR = "KSAT"

_BANDS = frozenset({"S", "X", "Ka"})
_ANGLE_TYPES = frozenset({"AZEL", "XEYN", "XSYE"})
_TRACKING_MODES = frozenset({"AUTO", "PROGRAM", "SCAN"})
_COSPAR_ID = re.compile(r"^\d{4}-\d{3}[A-Z]{1,3}$")
_CATALOG_ID = re.compile(r"^(?:\d{5}|\d{9})$")
_TURNAROUND_RATIOS = {
    "S": frozenset({(240, 221)}),
    "X": frozenset({(880, 749)}),
    "Ka": frozenset({(2720, 2407), (2760, 2407), (2816, 2407)}),
}


class KsatProduct(StrEnum):
    """KSAT filename/product discriminators.

    METEO is reserved so callers can report a deferred product consistently;
    this module does not serialize it because its normative definition is not
    present in the supplied KSAT documentation.
    """

    TRACK = "TRACK"
    ANGLE = "ANGLE"
    SIGMET = "SIGMET"
    METEO = "METEO"


def _clean_text(name: str, value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    if "\n" in value or "\r" in value:
        raise ValueError(f"{name} must not contain a newline")
    try:
        value.encode("ascii")
    except UnicodeEncodeError as exc:
        raise ValueError(f"{name} must contain ASCII characters only") from exc
    return value.strip()


def _optional_text(name: str, value: str | None) -> str | None:
    return None if value is None else _clean_text(name, value)


def _identifier(name: str, value: str) -> str:
    cleaned = _clean_text(name, value)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", cleaned):
        raise ValueError(
            f"{name} must start with an ASCII letter or digit and contain only "
            "letters, digits, dash, or underscore"
        )
    return cleaned


def _finite(name: str, value: float) -> None:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number")
    try:
        valid = math.isfinite(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not valid:
        raise ValueError(f"{name} must be a finite number")


def _optional_finite(name: str, value: float | None) -> None:
    if value is not None:
        _finite(name, value)


def _band(name: str, value: str) -> str:
    if value not in _BANDS:
        raise ValueError(f"{name} must be one of S, X, or Ka")
    return value


def _utc(name: str, value: dt.datetime) -> dt.datetime:
    if not isinstance(value, dt.datetime) or value.tzinfo is None:
        raise ValueError(f"{name} must be a timezone-aware datetime")
    offset = value.utcoffset()
    if offset is None:
        raise ValueError(f"{name} must be a timezone-aware datetime")
    return value.astimezone(dt.timezone.utc)


@dataclass(frozen=True, slots=True)
class KsatSite:
    """Ground-site identity and reference-point details used in the header."""

    identifier: str
    name: str | None = None
    location: str | None = None
    latitude_deg: float | None = None
    longitude_deg: float | None = None
    altitude_m: float | None = None
    ecef_x_m: float | None = None
    ecef_y_m: float | None = None
    ecef_z_m: float | None = None
    pedestal_offset_m: float | None = None
    tlt_calibration_date: dt.date | None = None
    tlt_band: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "identifier", _identifier("site identifier", self.identifier))
        object.__setattr__(self, "name", _optional_text("site name", self.name))
        object.__setattr__(self, "location", _optional_text("site location", self.location))
        for name in (
            "latitude_deg",
            "longitude_deg",
            "altitude_m",
            "ecef_x_m",
            "ecef_y_m",
            "ecef_z_m",
            "pedestal_offset_m",
        ):
            _optional_finite(name, getattr(self, name))
        if self.latitude_deg is not None and not -90 <= self.latitude_deg <= 90:
            raise ValueError("latitude_deg must be between -90 and 90")
        if self.longitude_deg is not None and not -180 <= self.longitude_deg <= 180:
            raise ValueError("longitude_deg must be between -180 and 180")
        if self.pedestal_offset_m is not None and self.pedestal_offset_m < 0:
            raise ValueError("pedestal_offset_m must be non-negative")
        wgs84 = (self.latitude_deg, self.longitude_deg, self.altitude_m)
        if any(value is not None for value in wgs84) and any(value is None for value in wgs84):
            raise ValueError("WGS84 latitude, longitude, and altitude must be supplied together")
        ecef = (self.ecef_x_m, self.ecef_y_m, self.ecef_z_m)
        if any(value is not None for value in ecef) and any(value is None for value in ecef):
            raise ValueError("ECEF X, Y, and Z must be supplied together")
        if self.tlt_band is not None:
            _band("tlt_band", self.tlt_band)
        if isinstance(self.tlt_calibration_date, dt.datetime):
            raise ValueError("tlt_calibration_date must be a date, not a datetime")
        if self.tlt_calibration_date is not None and not isinstance(
            self.tlt_calibration_date, dt.date
        ):
            raise ValueError("tlt_calibration_date must be a date")


@dataclass(frozen=True, slots=True)
class KsatSpacecraft:
    """Spacecraft identity used in the header and PARTICIPANT_2."""

    identifier: str
    name: str | None = None
    cospar_id: str | None = None
    catalog_id: str | None = None

    def __post_init__(self) -> None:
        identifier = _identifier("spacecraft identifier", self.identifier)
        object.__setattr__(self, "name", _optional_text("spacecraft name", self.name))
        object.__setattr__(self, "cospar_id", _optional_text("COSPAR ID", self.cospar_id))
        object.__setattr__(self, "catalog_id", _optional_text("catalog ID", self.catalog_id))
        if not (_COSPAR_ID.fullmatch(identifier) or _CATALOG_ID.fullmatch(identifier)):
            raise ValueError(
                "spacecraft identifier must be a COSPAR ID or a 5-/9-digit catalog ID"
            )
        if self.cospar_id is not None and not _COSPAR_ID.fullmatch(self.cospar_id):
            raise ValueError("COSPAR ID must have form YYYY-NNNP")
        if self.catalog_id is not None and not _CATALOG_ID.fullmatch(self.catalog_id):
            raise ValueError("catalog ID must contain 5 or 9 digits")
        known = {value for value in (self.cospar_id, self.catalog_id) if value is not None}
        if known and identifier not in known:
            raise ValueError("spacecraft identifier must equal its COSPAR or catalog ID")
        object.__setattr__(self, "identifier", identifier)


@dataclass(frozen=True, slots=True)
class KsatHeader:
    """Common KSAT TDM header information.

    ``creation_date`` is rendered at the required one-second resolution.  The
    same normalized value is used in the filename.
    """

    creation_date: dt.datetime
    site: KsatSite
    spacecraft: KsatSpacecraft
    summary: str | None = None
    comments: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        object.__setattr__(self, "creation_date", _utc("creation_date", self.creation_date))
        if not isinstance(self.site, KsatSite):
            raise ValueError("site must be a KsatSite")
        if not isinstance(self.spacecraft, KsatSpacecraft):
            raise ValueError("spacecraft must be a KsatSpacecraft")
        object.__setattr__(self, "summary", _optional_text("summary", self.summary))
        comments = tuple(_clean_text("comment", comment) for comment in self.comments)
        object.__setattr__(self, "comments", comments)


@dataclass(frozen=True, slots=True)
class TrackMetadata:
    """Metadata for one KSAT TRACK segment (supported modes 1, 3, and 4)."""

    mode: int
    transmit_band: str
    receive_band: str
    integration_interval_s: float
    turnaround_numerator: int | None = None
    turnaround_denominator: int | None = None
    transmit_delay_s: float | None = None
    receive_delay_s: float | None = None
    correction_range_s: float | None = None
    correction_doppler_hz: float | None = None

    def __post_init__(self) -> None:
        if isinstance(self.mode, bool) or self.mode not in (1, 3, 4):
            raise ValueError("TRACK mode must be 1, 3, or 4; mode 2 is not supported")
        _band("transmit_band", self.transmit_band)
        _band("receive_band", self.receive_band)
        if self.transmit_band != self.receive_band:
            raise ValueError("cross-band TRACK operation is not supported")
        _finite("integration_interval_s", self.integration_interval_s)
        if not 0.01 <= self.integration_interval_s <= 60.0:
            raise ValueError("integration_interval_s must be between 0.01 and 60 seconds")
        for name in (
            "transmit_delay_s",
            "receive_delay_s",
            "correction_range_s",
            "correction_doppler_hz",
        ):
            _optional_finite(name, getattr(self, name))
        for name in ("transmit_delay_s", "receive_delay_s"):
            value = getattr(self, name)
            if value is not None and value < 0:
                raise ValueError(f"{name} must be non-negative")
        self._validate_mode_fields()

    def _validate_mode_fields(self) -> None:
        range_fields = (
            self.transmit_delay_s,
            self.receive_delay_s,
            self.correction_range_s,
        )
        ratio = (self.turnaround_numerator, self.turnaround_denominator)
        if self.mode == 1:
            if any(value is None for value in range_fields):
                raise ValueError("TRACK mode 1 requires range delay and correction metadata")
            if ratio != (None, None) or self.correction_doppler_hz is not None:
                raise ValueError("TRACK mode 1 must not contain Doppler metadata")
            return

        if None in ratio:
            raise ValueError(f"TRACK mode {self.mode} requires a turnaround ratio")
        if not all(isinstance(value, int) and not isinstance(value, bool) for value in ratio):
            raise ValueError("turnaround ratio values must be integers")
        if ratio not in _TURNAROUND_RATIOS[self.transmit_band]:
            raise ValueError(
                f"unsupported {self.transmit_band}-band turnaround ratio "
                f"{ratio[0]}/{ratio[1]}"
            )
        if self.correction_doppler_hz is None:
            raise ValueError(f"TRACK mode {self.mode} requires correction_doppler_hz")

        if self.mode == 3 and any(value is None for value in range_fields):
            raise ValueError("TRACK mode 3 requires range delay and correction metadata")
        if self.mode == 4 and any(value is not None for value in range_fields):
            raise ValueError("TRACK mode 4 must not contain range metadata")


@dataclass(frozen=True, slots=True)
class TrackObservation:
    """Raw TRACK values sharing one microsecond-resolution epoch."""

    epoch: dt.datetime
    range_s: float | None = None
    transmit_frequency_hz: float | None = None
    receive_frequency_hz: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "epoch", _utc("TRACK epoch", self.epoch))
        for name in ("range_s", "transmit_frequency_hz", "receive_frequency_hz"):
            _optional_finite(name, getattr(self, name))
        if self.range_s is not None and self.range_s < 0:
            raise ValueError("range_s must be non-negative")
        if self.transmit_frequency_hz is not None and self.transmit_frequency_hz <= 0:
            raise ValueError("transmit_frequency_hz must be positive")
        if self.receive_frequency_hz is not None and self.receive_frequency_hz <= 0:
            raise ValueError("receive_frequency_hz must be positive")


@dataclass(frozen=True, slots=True)
class TrackSegment:
    metadata: TrackMetadata
    observations: tuple[TrackObservation, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.metadata, TrackMetadata):
            raise ValueError("metadata must be TrackMetadata")
        observations = tuple(self.observations)
        if not observations or not all(
            isinstance(item, TrackObservation) for item in observations
        ):
            raise ValueError("a TRACK segment requires TrackObservation values")
        object.__setattr__(self, "observations", observations)
        _validate_chronology("TRACK", observations)
        mode = self.metadata.mode
        if mode == 1:
            if any(
                item.range_s is None
                or item.transmit_frequency_hz is not None
                or item.receive_frequency_hz is not None
                for item in observations
            ):
                raise ValueError("TRACK mode 1 observations must contain only RANGE")
        elif mode == 3:
            if any(
                item.range_s is None or item.receive_frequency_hz is None
                for item in observations
            ):
                raise ValueError("TRACK mode 3 observations require RANGE and RECEIVE_FREQ_1")
            if not any(item.transmit_frequency_hz is not None for item in observations):
                raise ValueError("TRACK mode 3 requires at least one TRANSMIT_FREQ_1 value")
        else:
            if any(
                item.range_s is not None or item.receive_frequency_hz is None
                for item in observations
            ):
                raise ValueError("TRACK mode 4 observations require RECEIVE_FREQ_1 and no RANGE")
            if not any(item.transmit_frequency_hz is not None for item in observations):
                raise ValueError("TRACK mode 4 requires at least one TRANSMIT_FREQ_1 value")


@dataclass(frozen=True, slots=True)
class AngleObservation:
    epoch: dt.datetime
    angle_1_deg: float
    angle_2_deg: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "epoch", _utc("ANGLE epoch", self.epoch))
        _finite("angle_1_deg", self.angle_1_deg)
        _finite("angle_2_deg", self.angle_2_deg)


@dataclass(frozen=True, slots=True)
class AngleSegment:
    receive_band: str
    angle_type: str
    tracking_mode: str
    observations: tuple[AngleObservation, ...]

    def __post_init__(self) -> None:
        _band("receive_band", self.receive_band)
        if self.angle_type not in _ANGLE_TYPES:
            raise ValueError("angle_type must be AZEL, XEYN, or XSYE; RADEC is not supported")
        if self.tracking_mode not in _TRACKING_MODES:
            raise ValueError("tracking_mode must be AUTO, PROGRAM, or SCAN")
        observations = tuple(self.observations)
        if not observations or not all(
            isinstance(item, AngleObservation) for item in observations
        ):
            raise ValueError("an ANGLE segment requires AngleObservation values")
        object.__setattr__(self, "observations", observations)
        _validate_chronology("ANGLE", observations)


@dataclass(frozen=True, slots=True)
class SignalMetricsObservation:
    epoch: dt.datetime
    carrier_power_dbw: float | None = None
    pc_n0_db_hz: float | None = None
    pr_n0_db_hz: float | None = None

    def __post_init__(self) -> None:
        epoch = _utc("SIGMET epoch", self.epoch)
        if epoch.microsecond:
            raise ValueError("SIGMET epochs must have one-second resolution")
        object.__setattr__(self, "epoch", epoch)
        values = (self.carrier_power_dbw, self.pc_n0_db_hz, self.pr_n0_db_hz)
        if all(value is None for value in values):
            raise ValueError("a SIGMET observation requires at least one metric")
        for name in ("carrier_power_dbw", "pc_n0_db_hz", "pr_n0_db_hz"):
            _optional_finite(name, getattr(self, name))


@dataclass(frozen=True, slots=True)
class SignalMetricsSegment:
    transmit_band: str
    receive_band: str
    observations: tuple[SignalMetricsObservation, ...]

    def __post_init__(self) -> None:
        _band("transmit_band", self.transmit_band)
        _band("receive_band", self.receive_band)
        observations = tuple(self.observations)
        if not observations or not all(
            isinstance(item, SignalMetricsObservation) for item in observations
        ):
            raise ValueError("a SIGMET segment requires SignalMetricsObservation values")
        object.__setattr__(self, "observations", observations)
        _validate_chronology("SIGMET", observations)


KsatSegment: TypeAlias = TrackSegment | AngleSegment | SignalMetricsSegment


@dataclass(frozen=True, slots=True)
class KsatDocument:
    product: KsatProduct
    header: KsatHeader
    segments: tuple[KsatSegment, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.product, KsatProduct):
            raise ValueError("product must be a KsatProduct")
        if not isinstance(self.header, KsatHeader):
            raise ValueError("header must be a KsatHeader")
        segments = tuple(self.segments)
        if not segments:
            raise ValueError("a KSAT TDM document requires at least one segment")
        object.__setattr__(self, "segments", segments)
        expected: type[object] | None = {
            KsatProduct.TRACK: TrackSegment,
            KsatProduct.ANGLE: AngleSegment,
            KsatProduct.SIGMET: SignalMetricsSegment,
            KsatProduct.METEO: None,
        }[self.product]
        if expected is None:
            raise NotImplementedError("METEO serialization is deferred")
        if not all(isinstance(segment, expected) for segment in segments):
            raise ValueError(f"{self.product.value} documents contain the wrong segment type")
        if self.product is KsatProduct.ANGLE:
            angle_segments = tuple(
                segment for segment in segments if isinstance(segment, AngleSegment)
            )
            if any(
                left.tracking_mode == right.tracking_mode
                for left, right in zip(angle_segments, angle_segments[1:])
            ):
                raise ValueError("adjacent ANGLE segments must represent a tracking-mode change")


def _validate_chronology(name: str, observations: tuple[object, ...]) -> None:
    epochs = [getattr(item, "epoch") for item in observations]
    if epochs != sorted(epochs):
        raise ValueError(f"{name} observations must be in chronological order")


def _format_epoch(value: dt.datetime, *, microseconds: bool) -> str:
    value = value.astimezone(dt.timezone.utc)
    timespec = "microseconds" if microseconds else "seconds"
    return value.replace(tzinfo=None).isoformat(timespec=timespec)


def _kv(key: str, value: object) -> str:
    return f"{key} = {value}"


def _number(value: float) -> str:
    """Stable metadata number formatting without changing its precision."""
    return format(Decimal(str(value)), "f")


def _ground_antenna_comment(site: KsatSite) -> str:
    """Format identifier, optional common name, and location without repeats."""

    label = site.identifier
    seen = {site.identifier.casefold()}
    if site.name is not None and site.name.casefold() not in seen:
        label += f" {site.name}"
        seen.add(site.name.casefold())
    location_parts = [
        part.strip()
        for part in (site.location or "UNKNOWN").split(",")
        if part.strip() and part.strip().casefold() not in seen
    ]
    location = ", ".join(dict.fromkeys(location_parts)) or "UNKNOWN"
    return f"Ground antenna: {label}, {location}"


def _header_lines(document: KsatDocument) -> list[str]:
    header = document.header
    site = header.site
    spacecraft = header.spacecraft
    default_summary = {
        KsatProduct.TRACK: "Radiometric Tracking Data",
        KsatProduct.ANGLE: "Antenna Pointing Angles",
        KsatProduct.SIGMET: "Signal Metrics",
        KsatProduct.METEO: "Meteorology",
    }[document.product]

    if None not in (site.latitude_deg, site.longitude_deg, site.altitude_m):
        wgs84 = (
            f"lat={site.latitude_deg}, long={site.longitude_deg}, alt={site.altitude_m}"
        )
    else:
        wgs84 = "UNKNOWN"
    if None not in (site.ecef_x_m, site.ecef_y_m, site.ecef_z_m):
        ecef = (
            f"X={site.ecef_x_m:.3f}, Y={site.ecef_y_m:.3f}, "
            f"Z={site.ecef_z_m:.3f}"
        )
    else:
        ecef = "UNKNOWN"
    pedestal = (
        f"Lg={site.pedestal_offset_m} meters"
        if site.pedestal_offset_m is not None
        else "UNKNOWN"
    )
    calibration = (
        f"{site.tlt_band}-band TLT calibration date: {site.tlt_calibration_date.isoformat()}"
        if site.tlt_band is not None and site.tlt_calibration_date is not None
        else "TLT calibration date: UNKNOWN"
    )
    cospar = spacecraft.cospar_id or "UNKNOWN"
    catalog = spacecraft.catalog_id or "UNKNOWN"

    comments = [
        header.summary or default_summary,
        _ground_antenna_comment(site),
        f"WGS84 coordinates: {wgs84}",
        f"ECEF coordinates: {ecef}",
        f"Pedestal offset: {pedestal}",
        calibration,
        f"Spacecraft: {spacecraft.name or 'UNKNOWN'}",
        f"COSPAR: {cospar} Catalog: {catalog}",
        *header.comments,
    ]
    return [
        _kv("CCSDS_TDM_VERS", TDM_VERSION),
        *(f"COMMENT {comment}" for comment in comments),
        _kv("CREATION_DATE", _format_epoch(header.creation_date, microseconds=False)),
        _kv("ORIGINATOR", ORIGINATOR),
    ]


def _common_metadata(header: KsatHeader) -> list[str]:
    return [
        _kv("TIME_SYSTEM", "UTC"),
        _kv("PARTICIPANT_1", header.site.identifier),
        _kv("PARTICIPANT_2", header.spacecraft.identifier),
    ]


def _track_segment_lines(header: KsatHeader, segment: TrackSegment) -> list[str]:
    metadata = segment.metadata
    descriptions = {
        1: "range only",
        3: "range and Doppler",
        4: "Doppler only",
    }
    lines = [
        "META_START",
        f"COMMENT Range Tracking Data: Mode {metadata.mode}={descriptions[metadata.mode]}",
        *_common_metadata(header),
        _kv("MODE", "SEQUENTIAL"),
        _kv("TRANSMIT_BAND", metadata.transmit_band),
        _kv("RECEIVE_BAND", metadata.receive_band),
    ]
    if metadata.mode in (3, 4):
        lines.extend(
            (
                _kv("TURNAROUND_NUMERATOR", metadata.turnaround_numerator),
                _kv("TURNAROUND_DENOMINATOR", metadata.turnaround_denominator),
            )
        )
    lines.extend(
        (
            _kv("INTEGRATION_INTERVAL", _number(metadata.integration_interval_s)),
            _kv("INTEGRATION_REF", "END"),
        )
    )
    if metadata.mode in (1, 3):
        lines.extend(
            (
                _kv("RANGE_UNITS", "s"),
                _kv("TRANSMIT_DELAY_1", _number(metadata.transmit_delay_s)),
                _kv("RECEIVE_DELAY_1", _number(metadata.receive_delay_s)),
                _kv("CORRECTION_RANGE", _number(metadata.correction_range_s)),
            )
        )
    if metadata.mode in (3, 4):
        lines.append(_kv("CORRECTION_DOPPLER", _number(metadata.correction_doppler_hz)))
    lines.extend(
        (
            _kv("CORRECTIONS_APPLIED", "NO"),
            _kv("PATH", "1,2,1"),
            "META_STOP",
            "",
            "DATA_START",
        )
    )
    for observation in segment.observations:
        epoch = _format_epoch(observation.epoch, microseconds=True)
        if observation.transmit_frequency_hz is not None:
            lines.append(
                _kv(
                    "TRANSMIT_FREQ_1",
                    f"{epoch} {observation.transmit_frequency_hz:.6f}",
                )
            )
        if observation.receive_frequency_hz is not None:
            lines.append(_kv("RECEIVE_FREQ_1", f"{epoch} {observation.receive_frequency_hz:.6f}"))
        if observation.range_s is not None:
            lines.append(_kv("RANGE", f"{epoch} {observation.range_s:.12f}"))
    lines.append("DATA_STOP")
    return lines


def _angle_segment_lines(header: KsatHeader, segment: AngleSegment) -> list[str]:
    lines = [
        "META_START",
        f"COMMENT TRACKING_MODE = {segment.tracking_mode}",
        *_common_metadata(header),
        _kv("RECEIVE_BAND", segment.receive_band),
        _kv("ANGLE_TYPE", segment.angle_type),
        "META_STOP",
        "",
        "DATA_START",
    ]
    for observation in segment.observations:
        epoch = _format_epoch(observation.epoch, microseconds=True)
        lines.extend(
            (
                _kv("ANGLE_1", f"{epoch} {observation.angle_1_deg:.6f}"),
                _kv("ANGLE_2", f"{epoch} {observation.angle_2_deg:.6f}"),
            )
        )
    lines.append("DATA_STOP")
    return lines


def _signal_metrics_segment_lines(
    header: KsatHeader, segment: SignalMetricsSegment
) -> list[str]:
    lines = [
        "META_START",
        *_common_metadata(header),
        _kv("TRANSMIT_BAND", segment.transmit_band),
        _kv("RECEIVE_BAND", segment.receive_band),
        "META_STOP",
        "",
        "DATA_START",
    ]
    for observation in segment.observations:
        epoch = _format_epoch(observation.epoch, microseconds=False)
        if observation.carrier_power_dbw is not None:
            lines.append(_kv("CARRIER_POWER", f"{epoch} {observation.carrier_power_dbw:.2f}"))
        if observation.pc_n0_db_hz is not None:
            lines.append(_kv("PC_N0", f"{epoch} {observation.pc_n0_db_hz:.2f}"))
        if observation.pr_n0_db_hz is not None:
            lines.append(_kv("PR_N0", f"{epoch} {observation.pr_n0_db_hz:.2f}"))
    lines.append("DATA_STOP")
    return lines


def render_ksat_tdm(document: KsatDocument) -> str:
    """Serialize a validated KSAT document to CCSDS KVN text."""
    if not isinstance(document, KsatDocument):
        raise TypeError("document must be a KsatDocument")
    lines = _header_lines(document)
    for segment in document.segments:
        lines.append("")
        if isinstance(segment, TrackSegment):
            lines.extend(_track_segment_lines(document.header, segment))
        elif isinstance(segment, AngleSegment):
            lines.extend(_angle_segment_lines(document.header, segment))
        elif isinstance(segment, SignalMetricsSegment):
            lines.extend(_signal_metrics_segment_lines(document.header, segment))
        else:  # KsatDocument validation makes this unreachable.
            raise TypeError(f"unsupported KSAT segment {type(segment).__name__}")
    return "\n".join(lines) + "\n"


def _filename_component(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", value):
        raise ValueError(
            "KSAT filename identifiers must start with an ASCII letter or digit "
            "and contain only letters, digits, dash, or underscore"
        )
    return value


def ksat_tdm_filename(document: KsatDocument) -> str:
    """Return ``<TYPE>_<GSID>_<SVID>_<DATE>.tdm`` for ``document``."""
    if not isinstance(document, KsatDocument):
        raise TypeError("document must be a KsatDocument")
    creation = _format_epoch(document.header.creation_date, microseconds=False).replace(":", "-")
    return "_".join(
        (
            document.product.value,
            _filename_component(document.header.site.identifier),
            _filename_component(document.header.spacecraft.identifier),
            creation,
        )
    ) + ".tdm"


def write_ksat_tdm(
    document: KsatDocument,
    path: str | Path | None = None,
) -> str:
    """Render a KSAT TDM and optionally write it to ``path``.

    If ``path`` names an existing directory, the standard KSAT filename is
    appended.  Parent directories are never created implicitly.
    """
    text = render_ksat_tdm(document)
    if path is not None:
        destination = Path(path)
        if destination.is_dir():
            destination /= ksat_tdm_filename(document)
        destination.write_text(text, encoding="ascii")
    return text


__all__ = [
    "AngleObservation",
    "AngleSegment",
    "KsatDocument",
    "KsatHeader",
    "KsatProduct",
    "KsatSegment",
    "KsatSite",
    "KsatSpacecraft",
    "ORIGINATOR",
    "SignalMetricsObservation",
    "SignalMetricsSegment",
    "TDM_VERSION",
    "TrackMetadata",
    "TrackObservation",
    "TrackSegment",
    "ksat_tdm_filename",
    "render_ksat_tdm",
    "write_ksat_tdm",
]
