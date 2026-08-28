"""Compact KSAT CCSDS 503.0-B-2 ANGLE exporter."""

from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass, field
from pathlib import Path

import polars as pl

from dart.io import azure, meos
from dart.tdm.common import (
    BANDS,
    COLUMN,
    IDENTIFIER,
    ContactMetadata,
    load_contact_metadata,
    utc,
    validate_filename_identity,
)

_TRACKING_MODES = {"AUTO", "PROGRAM", "SCAN"}
_DEFAULT_ANGLE_1 = "antenna1_position_azimuth"
_DEFAULT_ANGLE_2 = "antenna1_position_elevation"


@dataclass(frozen=True, slots=True)
class AngleColumns:
    timestamp: str = "timestamp"
    contact_id: str = "contact_id"
    station: str = "antenna_name"
    angle_1: str = _DEFAULT_ANGLE_1
    angle_2: str = _DEFAULT_ANGLE_2

    def __post_init__(self) -> None:
        names = (
            self.timestamp,
            self.contact_id,
            self.station,
            self.angle_1,
            self.angle_2,
        )
        if any(not COLUMN.fullmatch(name) for name in names):
            raise ValueError("ANGLE columns must be Kusto identifiers")
        if self.angle_1 == self.angle_2:
            raise ValueError("ANGLE_1 and ANGLE_2 require distinct ADX columns")


@dataclass(frozen=True, slots=True)
class AngleRequest:
    contact_id: str
    band: str
    tracking_mode: str
    angle_type: str = "AZEL"
    columns: AngleColumns = field(default_factory=AngleColumns)

    def __post_init__(self) -> None:
        if not IDENTIFIER.fullmatch(self.contact_id):
            raise ValueError("contact_id contains unsupported characters")
        if self.band not in BANDS:
            raise ValueError("ANGLE receive band must be S, X, or Ka")
        if self.tracking_mode not in _TRACKING_MODES:
            raise ValueError("tracking_mode must be AUTO, PROGRAM, or SCAN")
        if self.angle_type != "AZEL":
            raise NotImplementedError(
                "ANGLE type requires confirmed telemetry mappings; only AZEL is available"
            )


@dataclass(frozen=True, slots=True)
class AngleResult:
    filename: str
    text: str
    path: Path | None = None
    warnings: tuple[str, ...] = ()


def _calibration(
    metadata: ContactMetadata, band: str, columns: AngleColumns
) -> tuple[meos.TrackCalibration | None, tuple[str, ...]]:
    mapping_warning = (
        f"confirm that ADX columns {columns.angle_1!r} and {columns.angle_2!r} "
        "contain controller pointing readback for the target site"
    )
    try:
        calibration = meos.get_track_calibration(
            metadata.antenna, band, metadata.start
        )
    except LookupError:
        calibration_warning = (
            f"no reviewed pedestal offset or TLT calibration for "
            f"{metadata.antenna}/{band}; ANGLE calibration comments use UNKNOWN"
        )
        return None, (mapping_warning, calibration_warning)
    return calibration, (mapping_warning,)


def _value(value: object, column: str) -> float:
    if isinstance(value, bool):
        raise ValueError(  # noqa: TRY004
            f"ADX column {column!r} contains an invalid angle {value!r}"
        )
    try:
        angle = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"ADX column {column!r} contains an invalid angle {value!r}"
        ) from exc
    if not math.isfinite(angle):
        raise ValueError(f"ADX column {column!r} contains a non-finite angle")
    return angle


def _observations(
    frame: pl.DataFrame,
    request: AngleRequest,
    metadata: ContactMetadata,
) -> list[tuple[dt.datetime, float, float]]:
    columns = request.columns
    observations: dict[dt.datetime, tuple[float, float]] = {}
    for row in frame.to_dicts():
        angle_1 = row.get(columns.angle_1)
        angle_2 = row.get(columns.angle_2)
        if angle_1 is None and angle_2 is None:
            continue
        if angle_1 is None or angle_2 is None:
            raise ValueError("ADX ANGLE row contains only one pointing coordinate")
        if row.get(columns.contact_id) != request.contact_id:
            raise ValueError("ADX row contact does not match the request")
        if row.get(columns.station) != metadata.antenna:
            raise ValueError("ADX station does not match the KOGS antenna")
        epoch = utc(row.get(columns.timestamp), "ANGLE")
        values = (
            _value(angle_1, columns.angle_1),
            _value(angle_2, columns.angle_2),
        )
        if epoch in observations and observations[epoch] != values:
            raise ValueError(f"conflicting ANGLE values at {epoch.isoformat()}")
        observations[epoch] = values
    if not observations:
        raise ValueError("ADX returned no complete ANGLE observations")
    return [(epoch, *observations[epoch]) for epoch in sorted(observations)]


def _render(
    request: AngleRequest,
    metadata: ContactMetadata,
    calibration: meos.TrackCalibration | None,
    observations: list[tuple[dt.datetime, float, float]],
    created: dt.datetime,
) -> str:
    validate_filename_identity(metadata)
    if created.tzinfo is None or created.utcoffset() is None:
        raise ValueError("creation_date must be timezone-aware")
    created = created.astimezone(dt.UTC).replace(microsecond=0)
    pedestal = (
        "UNKNOWN"
        if calibration is None
        else f"Lg={calibration.pedestal_offset_m} meters"
    )
    calibration_date = (
        "UNKNOWN" if calibration is None else calibration.tlt_calibration_date.isoformat()
    )
    # fmt: off
    lines = [
        "CCSDS_TDM_VERS = 2.0", "COMMENT Antenna Pointing Angles",
        f"COMMENT Ground antenna: {metadata.antenna} {metadata.location}",
        f"COMMENT WGS84 coordinates: lat={metadata.latitude}, long={metadata.longitude}, alt={metadata.altitude}",
        f"COMMENT ECEF coordinates: X={metadata.ecef[0]:.3f}, Y={metadata.ecef[1]:.3f}, Z={metadata.ecef[2]:.3f}",
        f"COMMENT Pedestal offset: {pedestal}", f"COMMENT TLT calibration date: {calibration_date}",
        f"COMMENT Spacecraft: {metadata.spacecraft}", f"COMMENT COSPAR: {metadata.cospar} Catalog: {metadata.catalog}",
        f"CREATION_DATE = {created.isoformat(timespec='seconds').replace('+00:00', '')}", "ORIGINATOR = KSAT", "",
        "META_START", f"COMMENT TRACKING_MODE = {request.tracking_mode}", "TIME_SYSTEM = UTC",
        f"PARTICIPANT_1 = {metadata.antenna}", f"PARTICIPANT_2 = {metadata.cospar}",
        f"RECEIVE_BAND = {request.band}", f"ANGLE_TYPE = {request.angle_type}", "META_STOP", "", "DATA_START",
    ]
    # fmt: on
    for epoch, angle_1, angle_2 in observations:
        stamp = epoch.strftime("%Y-%m-%dT%H:%M:%S.%f")
        lines.append(f"ANGLE_1 = {stamp} {angle_1:.6f}")
        lines.append(f"ANGLE_2 = {stamp} {angle_2:.6f}")
    text = "\n".join((*lines, "DATA_STOP", ""))
    text.encode("ascii")
    return text


def write_angle_tdm(
    request: AngleRequest,
    output_dir: str | Path | None = None,
    *,
    overwrite: bool = False,
    creation_date: dt.datetime | None = None,
    timeout_seconds: float = azure.ADX_QUERY_TIMEOUT_SECONDS,
) -> AngleResult:
    """Fetch, validate, render, and optionally write one KSAT ANGLE file."""
    metadata = load_contact_metadata(
        request.contact_id, request.band, timeout_seconds, "ANGLE"
    )
    columns = request.columns
    selected = (
        columns.timestamp,
        columns.contact_id,
        columns.station,
        columns.angle_1,
        columns.angle_2,
    )
    frame = azure.fetch_contact_columns(
        request.contact_id,
        metadata.start,
        metadata.stop,
        selected,
        order_by=columns.timestamp,
        contact_column=columns.contact_id,
        timeout_seconds=timeout_seconds,
    )
    calibration, warnings = _calibration(metadata, request.band, columns)
    created = creation_date or dt.datetime.now(dt.UTC)
    text = _render(
        request,
        metadata,
        calibration,
        _observations(frame, request, metadata),
        created,
    )
    stamp = created.astimezone(dt.UTC).strftime("%Y-%m-%dT%H-%M-%S")
    filename = f"ANGLE_{metadata.antenna}_{metadata.cospar}_{stamp}.tdm"
    if output_dir is None:
        return AngleResult(filename, text, warnings=warnings)
    destination = Path(output_dir) / filename
    if destination.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(text, encoding="ascii")
    return AngleResult(filename, text, destination, warnings)
