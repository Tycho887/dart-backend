"""Compact KSAT CCSDS 503.0-B-2 TRACK mode-4 exporter."""

from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from pathlib import Path

import polars as pl

from dart.io import azure, ctrl_config, meos
from dart.io.contact import ContactMetadata, load_contact_metadata
from dart.io.utils import utc
from dart.tdm.common import BANDS, COLUMN, IDENTIFIER, validate_filename_identity

_RATIOS = {
    "S": {(240, 221)},
    "X": {(880, 749)},
    "Ka": {(2720, 2407), (2760, 2407), (2816, 2407)},
}
_UNITS = {"Hz": Decimal(1), "kHz": Decimal(1_000), "MHz": Decimal(1_000_000)}

@dataclass(frozen=True, slots=True)
class FrequencySource:
    link_name: str
    offset_column: str | None = None
    offset_unit: str = "Hz"
    offset_sign: int = 1

    def __post_init__(self) -> None:
        if not self.link_name:
            raise ValueError("frequency source requires a ctrl-config link name")
        if self.offset_column is not None and not COLUMN.fullmatch(self.offset_column):
            raise ValueError("frequency offset column must be a Kusto identifier")
        if self.offset_unit not in _UNITS:
            raise ValueError("frequency offset unit must be Hz, kHz, or MHz")
        if isinstance(self.offset_sign, bool) or self.offset_sign not in {-1, 1}:
            raise ValueError("frequency offset sign must be -1 or 1")

@dataclass(frozen=True, slots=True)
class TrackColumns:
    integration_end: str = "timestamp"
    contact_id: str = "contact_id"
    station: str = "antenna_name"

    def __post_init__(self) -> None:
        names = (self.integration_end, self.contact_id, self.station)
        if any(not COLUMN.fullmatch(name) for name in names):
            raise ValueError("TRACK columns must be Kusto identifiers")

@dataclass(frozen=True, slots=True)
class TrackRequest:
    contact_id: str
    band: str
    integration_interval_s: float
    turnaround_numerator: int
    turnaround_denominator: int
    transmit: FrequencySource
    receive: FrequencySource
    columns: TrackColumns = field(default_factory=TrackColumns)
    expected_station: str | None = None
    calibration: meos.TrackCalibration | None = None
    mode: int = 4

    def __post_init__(self) -> None:
        errors = {
            1: "TRACK mode 1 requires range data, which is not implemented",
            2: "TRACK mode 2 carrier phase is not supported by the KSAT profile",
            3: "TRACK mode 3 requires range data, which is not implemented",
        }
        if self.mode != 4:
            raise NotImplementedError(errors.get(self.mode, f"unsupported TRACK mode {self.mode}"))
        if not IDENTIFIER.fullmatch(self.contact_id):
            raise ValueError("contact_id contains unsupported characters")
        if self.band not in BANDS:
            raise ValueError("TRACK band must be S, X, or Ka")
        if (self.turnaround_numerator, self.turnaround_denominator) not in _RATIOS[self.band]:
            raise ValueError(f"unsupported {self.band}-band turnaround ratio")
        if not math.isfinite(self.integration_interval_s) or not 0.01 <= self.integration_interval_s <= 60:
            raise ValueError("integration interval must be between 0.01 and 60 seconds")
        if self.receive.offset_column is None:
            raise ValueError("mode 4 requires a receive-frequency offset column")
        if self.expected_station is not None and not IDENTIFIER.fullmatch(
            self.expected_station
        ):
            raise ValueError("expected station contains unsupported characters")

@dataclass(frozen=True, slots=True)
class TrackResult:
    filename: str
    text: str
    path: Path | None = None
    metadata: ContactMetadata | None = None

@dataclass(frozen=True, slots=True)
class _Metadata:
    contact: ContactMetadata
    calibration: meos.TrackCalibration
    transmit_center: float
    receive_center: float

def _utc(value: object) -> dt.datetime:
    return utc(value, "TRACK")

def _metadata(
    request: TrackRequest, timeout: float, kogs_api_key: str | None
) -> _Metadata:
    contact = load_contact_metadata(
        request.contact_id, request.band, timeout, "TRACK", kogs_api_key
    )
    if request.expected_station and contact.antenna != request.expected_station:
        raise ValueError("KOGS antenna does not match the TDM profile station")
    return _Metadata(
        contact,
        request.calibration
        or meos.get_track_calibration(contact.antenna, request.band, contact.start),
        ctrl_config.get_link_frequency(contact.spacecraft, request.transmit.link_name, "up"),
        ctrl_config.get_link_frequency(contact.spacecraft, request.receive.link_name, "down"),
    )

def _frequency(center: float, value: object, source: FrequencySource) -> Decimal:
    if value is None and source.offset_column is not None:
        raise ValueError(f"ADX column {source.offset_column!r} contains a null value")
    try:
        offset = Decimal(0) if value is None else Decimal(str(value)) * _UNITS[source.offset_unit]
        result = Decimal(str(center)) + source.offset_sign * offset
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ValueError(f"invalid frequency offset {value!r}") from exc
    if not result.is_finite() or result <= 0:
        raise ValueError("TRACK frequency must be positive and finite")
    return result

def _observations(frame: pl.DataFrame, request: TrackRequest, metadata: _Metadata) -> list[tuple[dt.datetime, Decimal, Decimal]]:
    observations: dict[dt.datetime, tuple[Decimal, Decimal]] = {}
    for row in frame.to_dicts():
        if row.get(request.columns.contact_id) != request.contact_id:
            raise ValueError("ADX row contact does not match the request")
        if row.get(request.columns.station) != metadata.contact.antenna:
            raise ValueError("ADX station does not match the KOGS antenna")
        epoch = _utc(row.get(request.columns.integration_end))
        transmit = _frequency(
            metadata.transmit_center,
            row.get(request.transmit.offset_column),
            request.transmit,
        )
        receive = _frequency(
            metadata.receive_center,
            row.get(request.receive.offset_column),
            request.receive,
        )
        values = (transmit, receive)
        if epoch in observations and observations[epoch] != values:
            raise ValueError(f"conflicting TRACK values at {epoch.isoformat()}")
        observations[epoch] = values
    if not observations:
        raise ValueError("ADX returned no complete TRACK mode-4 observations")
    return [(epoch, *observations[epoch]) for epoch in sorted(observations)]


def _render(
    request: TrackRequest,
    metadata: _Metadata,
    observations: list[tuple[dt.datetime, Decimal, Decimal]],
    created: dt.datetime,
) -> str:
    contact = metadata.contact
    validate_filename_identity(contact)
    if created.tzinfo is None or created.utcoffset() is None:
        raise ValueError("creation_date must be timezone-aware")
    created = created.astimezone(dt.UTC).replace(microsecond=0)
    # fmt: off
    lines = [
        "CCSDS_TDM_VERS = 2.0", "COMMENT Radiometric Tracking Data",
        f"COMMENT Ground antenna: {contact.antenna} {contact.location}",
        f"COMMENT WGS84 coordinates: lat={contact.latitude}, long={contact.longitude}, alt={contact.altitude}",
        f"COMMENT ECEF coordinates: X={contact.ecef[0]:.3f}, Y={contact.ecef[1]:.3f}, Z={contact.ecef[2]:.3f}",
        f"COMMENT Pedestal offset: Lg={metadata.calibration.pedestal_offset_m} meters",
        f"COMMENT {request.band}-band TLT calibration date: {metadata.calibration.tlt_calibration_date.isoformat()}",
        f"COMMENT Spacecraft: {contact.spacecraft}", f"COMMENT COSPAR: {contact.cospar} Catalog: {contact.catalog}",
        f"CREATION_DATE = {created.isoformat(timespec='seconds').replace('+00:00', '')}", "ORIGINATOR = KSAT", "",
        "META_START", "COMMENT Range Tracking Data: Mode 4=Doppler only", "TIME_SYSTEM = UTC",
        f"PARTICIPANT_1 = {contact.antenna}", f"PARTICIPANT_2 = {contact.cospar}", "MODE = SEQUENTIAL",
        f"TRANSMIT_BAND = {request.band}", f"RECEIVE_BAND = {request.band}",
        f"TURNAROUND_NUMERATOR = {request.turnaround_numerator}", f"TURNAROUND_DENOMINATOR = {request.turnaround_denominator}",
        f"INTEGRATION_INTERVAL = {request.integration_interval_s}", "INTEGRATION_REF = END",
        f"CORRECTION_DOPPLER = {metadata.calibration.correction_doppler_hz}", "CORRECTIONS_APPLIED = NO", "PATH = 1,2,1",
        "META_STOP", "", "DATA_START",
    ]
    # fmt: on
    previous_transmit: Decimal | None = None
    for epoch, transmit, receive in observations:
        stamp = epoch.strftime("%Y-%m-%dT%H:%M:%S.%f")
        if transmit != previous_transmit:
            lines.append(f"TRANSMIT_FREQ_1 = {stamp} {transmit:.6f}")
            previous_transmit = transmit
        lines.append(f"RECEIVE_FREQ_1 = {stamp} {receive:.6f}")
    text = "\n".join((*lines, "DATA_STOP", ""))
    text.encode("ascii")
    return text


def write_track_tdm(
    request: TrackRequest,
    output_dir: str | Path | None = None,
    *,
    overwrite: bool = False,
    creation_date: dt.datetime | None = None,
    timeout_seconds: float = azure.ADX_QUERY_TIMEOUT_SECONDS,
    kogs_api_key: str | None = None,
) -> TrackResult:
    """Fetch, validate, render, and optionally write one KSAT TRACK file."""
    metadata = _metadata(request, timeout_seconds, kogs_api_key)
    sources = (request.transmit, request.receive)
    columns = tuple(
        dict.fromkeys(
            (
                request.columns.integration_end,
                request.columns.contact_id,
                request.columns.station,
                *(source.offset_column for source in sources if source.offset_column),
            )
        )
    )
    frame = azure.fetch_contact_columns(
        request.contact_id,
        metadata.contact.start,
        metadata.contact.stop,
        columns,
        order_by=request.columns.integration_end,
        contact_column=request.columns.contact_id,
        timeout_seconds=timeout_seconds,
    )
    created = creation_date or dt.datetime.now(dt.UTC)
    text = _render(request, metadata, _observations(frame, request, metadata), created)
    stamp = created.astimezone(dt.UTC).strftime("%Y-%m-%dT%H-%M-%S")
    filename = f"TRACK_{metadata.contact.antenna}_{metadata.contact.cospar}_{stamp}.tdm"
    if output_dir is None:
        return TrackResult(filename, text, metadata=metadata.contact)
    destination = Path(output_dir) / filename
    if destination.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(text, encoding="ascii")
    return TrackResult(filename, text, destination, metadata.contact)
