"""ADX-backed assembly of KSAT Tracking Data Message products.

This module deliberately separates telemetry extraction from TDM rendering.
ADX installations expose many similarly named measurements whose engineering
meaning is site dependent, so only timestamp and antenna angles have defaults.
Range, frequency, and signal-metric columns must be mapped explicitly together
with their source units before they can be emitted as KSAT data.
"""

from __future__ import annotations

import datetime
import math
import re
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Literal

import polars as pl
from azure.kusto.data.helpers import dataframe_from_result_table

from dart.io.azure import ADX_QUERY_TIMEOUT_SECONDS, _query_properties, get_client
from dart.io.ksat_tdm import (
    AngleObservation,
    AngleSegment,
    KsatDocument,
    KsatHeader,
    KsatProduct,
    SignalMetricsObservation,
    SignalMetricsSegment,
    TrackMetadata,
    TrackObservation,
    TrackSegment,
    ksat_tdm_filename,
    render_ksat_tdm,
)


ProductName = Literal["TRACK", "ANGLE", "SIGMET", "METEO"]
SourceUnit = Literal[
    "s",
    "ms",
    "us",
    "ns",
    "ps",
    "Hz",
    "kHz",
    "MHz",
    "GHz",
    "deg",
    "rad",
    "dBW",
    "dB-Hz",
]

_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_CONTACT_ID = re.compile(r"^[A-Za-z0-9_-]+$")


@dataclass(frozen=True)
class AdxField:
    """One explicitly typed ADX source column."""

    column: str
    unit: SourceUnit

    def __post_init__(self) -> None:
        _validate_identifier(self.column)


@dataclass(frozen=True)
class KsatAdxColumnMap:
    """ADX columns that carry KSAT observables.

    Angle defaults match the existing DART telemetry query. All other
    observable mappings are intentionally absent until a site owner confirms
    that the ADX column has the KSAT meaning represented by the target field.
    """

    timestamp: str = "timestamp"
    track_timestamp: str | None = None
    angle_timestamp: str | None = None
    signal_metrics_timestamp: str | None = None
    contact_id: str = "contact_id"
    station_id: str = "system_id"
    angle_1: AdxField | None = field(
        default_factory=lambda: AdxField("antenna1_position_azimuth", "deg")
    )
    angle_2: AdxField | None = field(
        default_factory=lambda: AdxField("antenna1_position_elevation", "deg")
    )
    tracking_mode: str | None = None
    range_delay: AdxField | None = None
    transmit_frequency: AdxField | None = None
    receive_frequency: AdxField | None = None
    carrier_power: AdxField | None = None
    pc_n0: AdxField | None = None
    pr_n0: AdxField | None = None

    def __post_init__(self) -> None:
        for name in (self.timestamp, self.contact_id, self.station_id):
            _validate_identifier(name)
        for name in (
            self.track_timestamp,
            self.angle_timestamp,
            self.signal_metrics_timestamp,
            self.tracking_mode,
        ):
            if name is not None:
                _validate_identifier(name)

    def projected_columns(self) -> tuple[str, ...]:
        names = [self.timestamp, self.contact_id, self.station_id]
        for timestamp in (
            self.track_timestamp,
            self.angle_timestamp,
            self.signal_metrics_timestamp,
        ):
            if timestamp is not None:
                names.append(timestamp)
        for mapping in (
            self.angle_1,
            self.angle_2,
            self.range_delay,
            self.transmit_frequency,
            self.receive_frequency,
            self.carrier_power,
            self.pc_n0,
            self.pr_n0,
        ):
            if mapping is not None:
                names.append(mapping.column)
        if self.tracking_mode is not None:
            names.append(self.tracking_mode)
        return tuple(dict.fromkeys(names))


@dataclass(frozen=True)
class KsatAdxQuery:
    """A bounded selection from the ADX ``contacts`` table."""

    contact_ids: tuple[str, ...]
    start_time: datetime.datetime
    stop_time: datetime.datetime
    columns: KsatAdxColumnMap = field(default_factory=KsatAdxColumnMap)

    def __post_init__(self) -> None:
        if not self.contact_ids:
            raise ValueError("at least one contact_id is required")
        if len(self.contact_ids) != 1:
            raise ValueError("KSAT TDM export requires exactly one contact_id")
        for contact_id in self.contact_ids:
            if not _CONTACT_ID.fullmatch(contact_id):
                raise ValueError(f"invalid ADX contact_id {contact_id!r}")
        start = _as_utc(self.start_time, "start_time")
        stop = _as_utc(self.stop_time, "stop_time")
        if stop <= start:
            raise ValueError("stop_time must be later than start_time")


@dataclass(frozen=True)
class GeneratedTdm:
    product: ProductName
    filename: str
    text: str
    path: Path | None = None


@dataclass(frozen=True)
class KsatExportResult:
    generated: dict[ProductName, GeneratedTdm] = field(default_factory=dict)
    skipped: dict[ProductName, str] = field(default_factory=dict)
    warnings: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class AngleExportConfig:
    receive_band: str
    angle_type: str = "AZEL"
    tracking_mode: str | None = None


@dataclass(frozen=True)
class SignalMetricsExportConfig:
    transmit_band: str
    receive_band: str


def _validate_identifier(name: str) -> None:
    if not _IDENTIFIER.fullmatch(name):
        raise ValueError(f"invalid ADX column name {name!r}")


def _as_utc(value: datetime.datetime, field_name: str) -> datetime.datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")
    return value.astimezone(datetime.timezone.utc)


def _kusto_datetime(value: datetime.datetime) -> str:
    utc = _as_utc(value, "query datetime")
    return utc.isoformat(timespec="microseconds").replace("+00:00", "Z")


def build_ksat_adx_query(selection: KsatAdxQuery) -> str:
    """Build a validated, time-bounded KQL query for TDM source telemetry."""

    ids = ", ".join(f"'{value}'" for value in selection.contact_ids)
    columns = ", ".join(selection.columns.projected_columns())
    timestamp = selection.columns.timestamp
    contact_id = selection.columns.contact_id
    return (
        "contacts\n"
        f"| where {contact_id} in ({ids})\n"
        f"| where {timestamp} between "
        f"(datetime({_kusto_datetime(selection.start_time)}) .. "
        f"datetime({_kusto_datetime(selection.stop_time)}))\n"
        f"| project {columns}\n"
        f"| order by {timestamp} asc"
    )


def fetch_ksat_tdm_data(
    selection: KsatAdxQuery,
    *,
    timeout_seconds: float = ADX_QUERY_TIMEOUT_SECONDS,
) -> pl.DataFrame:
    """Execute a bounded TDM telemetry query and return a Polars frame."""

    query = build_ksat_adx_query(selection)
    with get_client() as client:
        response = client.execute_query(
            "telemetry", query, _query_properties(timeout_seconds)
        )
        raw = dataframe_from_result_table(response.primary_results[0])
    frame = pl.from_pandas(raw)
    if selection.columns.timestamp in frame.columns:
        frame = frame.sort(selection.columns.timestamp)
    return frame


_LINEAR_FACTORS: dict[tuple[SourceUnit, SourceUnit], float] = {
    ("s", "s"): 1.0,
    ("ms", "s"): 1e-3,
    ("us", "s"): 1e-6,
    ("ns", "s"): 1e-9,
    ("ps", "s"): 1e-12,
    ("Hz", "Hz"): 1.0,
    ("kHz", "Hz"): 1e3,
    ("MHz", "Hz"): 1e6,
    ("GHz", "Hz"): 1e9,
    ("deg", "deg"): 1.0,
    ("rad", "deg"): 180.0 / math.pi,
    ("dBW", "dBW"): 1.0,
    ("dB-Hz", "dB-Hz"): 1.0,
}


def convert_adx_value(value: object, mapping: AdxField, target_unit: SourceUnit) -> float:
    """Convert one finite ADX scalar into a documented KSAT target unit."""

    if value is None:
        raise ValueError(f"ADX column {mapping.column!r} contains a null value")
    if isinstance(value, bool):
        raise ValueError(
            f"ADX column {mapping.column!r} contains non-numeric value {value!r}"
        )
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"ADX column {mapping.column!r} contains non-numeric value {value!r}"
        ) from exc
    if not math.isfinite(number):
        raise ValueError(f"ADX column {mapping.column!r} contains non-finite data")
    try:
        factor = _LINEAR_FACTORS[(mapping.unit, target_unit)]
    except KeyError as exc:
        raise ValueError(
            f"cannot convert ADX column {mapping.column!r} from "
            f"{mapping.unit} to {target_unit}"
        ) from exc
    if mapping.unit == "rad" and target_unit == "deg":
        return number * factor
    # Decimal conversion prevents binary multiplication artifacts from leaking
    # into the writer's microhertz precision (for example 8447.53 MHz).
    try:
        return float(Decimal(str(value)) * Decimal(str(factor)))
    except InvalidOperation as exc:  # defensive for unusual numeric wrappers
        raise ValueError(
            f"ADX column {mapping.column!r} contains non-numeric value {value!r}"
        ) from exc


def _epoch(value: object, field_name: str) -> datetime.datetime:
    if isinstance(value, datetime.datetime):
        parsed = value
    elif isinstance(value, str):
        try:
            parsed = datetime.datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError(f"invalid {field_name} value {value!r}") from exc
    else:
        raise ValueError(f"invalid {field_name} value {value!r}")
    # ADX datetime values are UTC. Pandas may return them without tzinfo.
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=datetime.timezone.utc)
    return parsed.astimezone(datetime.timezone.utc)


def _mapped_value(
    row: dict[str, object], mapping: AdxField, target_unit: SourceUnit
) -> float | None:
    value = row.get(mapping.column)
    if value is None:
        return None
    return convert_adx_value(value, mapping, target_unit)


def _has_single_station(
    frame: pl.DataFrame, column: str, expected_station: str
) -> tuple[bool, str | None]:
    if column not in frame.columns:
        return False, f"ADX result does not contain station column {column!r}"
    stations = {
        str(value)
        for value in frame[column].drop_nulls().unique().to_list()
        if str(value).strip()
    }
    if not stations:
        return False, "ADX result has no station identifier"
    if len(stations) != 1:
        return False, "KSAT TDM files require exactly one ground station"
    source_station = next(iter(stations))
    if source_station != expected_station:
        return (
            False,
            f"ADX station {source_station} does not match header site "
            f"{expected_station}",
        )
    return True, None


def _build_track_document(
    frame: pl.DataFrame,
    header: KsatHeader,
    columns: KsatAdxColumnMap,
    metadata: TrackMetadata,
) -> KsatDocument | str:
    required = {
        "range delay": columns.range_delay if metadata.mode in (1, 3) else True,
        "transmit frequency": columns.transmit_frequency if metadata.mode in (3, 4) else True,
        "receive frequency": columns.receive_frequency if metadata.mode in (3, 4) else True,
    }
    missing = [name for name, mapping in required.items() if mapping is None]
    if missing:
        return "missing explicit ADX mapping for " + ", ".join(missing)

    if columns.track_timestamp is None:
        return (
            "TRACK requires an explicit integration-end timestamp column; "
            "set track_timestamp after confirming its semantics"
        )
    timestamp_column = columns.track_timestamp
    observations: list[TrackObservation] = []
    last_transmit: float | None = None
    for index, row in enumerate(frame.sort(timestamp_column).to_dicts()):
        if row.get(timestamp_column) is None:
            mappings = (
                columns.range_delay,
                columns.transmit_frequency,
                columns.receive_frequency,
            )
            if any(
                row.get(mapping.column) is not None
                for mapping in mappings
                if mapping is not None
            ):
                return f"TRACK source row {index} is missing its timestamp"
            continue
        range_s = (
            _mapped_value(row, columns.range_delay, "s")
            if isinstance(columns.range_delay, AdxField)
            else None
        )
        transmit_hz = (
            _mapped_value(row, columns.transmit_frequency, "Hz")
            if isinstance(columns.transmit_frequency, AdxField)
            else None
        )
        receive_hz = (
            _mapped_value(row, columns.receive_frequency, "Hz")
            if isinstance(columns.receive_frequency, AdxField)
            else None
        )
        target_values = (range_s, transmit_hz, receive_hz)
        if all(value is None for value in target_values):
            continue
        if metadata.mode in (1, 3) and range_s is None:
            return f"TRACK source row {index} is missing range delay"
        if metadata.mode in (3, 4) and transmit_hz is None:
            return f"TRACK source row {index} is missing transmit frequency"
        if metadata.mode in (3, 4) and receive_hz is None:
            return f"TRACK source row {index} is missing receive frequency"
        # KSAT reports transmit frequency once unless pre-steering changes it.
        emit_transmit = transmit_hz
        if transmit_hz is not None and transmit_hz == last_transmit:
            emit_transmit = None
        elif transmit_hz is not None:
            last_transmit = transmit_hz
        observations.append(
            TrackObservation(
                epoch=_epoch(row[timestamp_column], timestamp_column),
                range_s=range_s,
                transmit_frequency_hz=emit_transmit,
                receive_frequency_hz=receive_hz,
            )
        )
    if not observations:
        return "ADX result contains no complete TRACK observations"
    try:
        segment = TrackSegment(metadata=metadata, observations=tuple(observations))
    except ValueError as exc:
        return str(exc)
    return KsatDocument(KsatProduct.TRACK, header, (segment,))


def _build_angle_document(
    frame: pl.DataFrame,
    header: KsatHeader,
    columns: KsatAdxColumnMap,
    config: AngleExportConfig,
) -> KsatDocument | str:
    if columns.angle_1 is None or columns.angle_2 is None:
        return "missing explicit ADX angle mappings"
    if columns.tracking_mode is None and config.tracking_mode is None:
        return "ANGLE requires an ADX tracking-mode column or configured tracking mode"

    timestamp_column = columns.angle_timestamp or columns.timestamp
    groups: list[tuple[str, list[AngleObservation]]] = []
    for row in frame.sort(timestamp_column).to_dicts():
        if row.get(timestamp_column) is None:
            if any(
                row.get(mapping.column) is not None
                for mapping in (columns.angle_1, columns.angle_2)
            ):
                return "ANGLE source row is missing its timestamp"
            continue
        angle_1 = _mapped_value(row, columns.angle_1, "deg")
        angle_2 = _mapped_value(row, columns.angle_2, "deg")
        if angle_1 is None and angle_2 is None:
            continue
        if angle_1 is None or angle_2 is None:
            return "ANGLE source row has only one of ANGLE_1 and ANGLE_2"
        raw_mode = (
            row.get(columns.tracking_mode)
            if columns.tracking_mode is not None
            else config.tracking_mode
        )
        mode = str(raw_mode).strip().upper() if raw_mode is not None else ""
        observation = AngleObservation(
            epoch=_epoch(row[timestamp_column], timestamp_column),
            angle_1_deg=angle_1,
            angle_2_deg=angle_2,
        )
        if not groups or groups[-1][0] != mode:
            groups.append((mode, [observation]))
        else:
            groups[-1][1].append(observation)
    if not groups:
        return "ADX result contains no complete ANGLE observations"
    try:
        segments = tuple(
            AngleSegment(
                receive_band=config.receive_band,
                angle_type=config.angle_type,
                tracking_mode=mode,
                observations=tuple(observations),
            )
            for mode, observations in groups
        )
    except ValueError as exc:
        return str(exc)
    return KsatDocument(KsatProduct.ANGLE, header, segments)


def _build_signal_metrics_document(
    frame: pl.DataFrame,
    header: KsatHeader,
    columns: KsatAdxColumnMap,
    config: SignalMetricsExportConfig,
) -> KsatDocument | str:
    mappings = (columns.carrier_power, columns.pc_n0, columns.pr_n0)
    if all(mapping is None for mapping in mappings):
        return "missing explicit ADX mapping for signal metrics"
    timestamp_column = columns.signal_metrics_timestamp or columns.timestamp
    observations: list[SignalMetricsObservation] = []
    for row in frame.sort(timestamp_column).to_dicts():
        if row.get(timestamp_column) is None:
            if any(
                row.get(mapping.column) is not None
                for mapping in mappings
                if mapping is not None
            ):
                return "SIGMET source row is missing its timestamp"
            continue
        carrier = (
            _mapped_value(row, columns.carrier_power, "dBW")
            if isinstance(columns.carrier_power, AdxField)
            else None
        )
        pc_n0 = (
            _mapped_value(row, columns.pc_n0, "dB-Hz")
            if isinstance(columns.pc_n0, AdxField)
            else None
        )
        pr_n0 = (
            _mapped_value(row, columns.pr_n0, "dB-Hz")
            if isinstance(columns.pr_n0, AdxField)
            else None
        )
        if carrier is None and pc_n0 is None and pr_n0 is None:
            continue
        try:
            observations.append(
                SignalMetricsObservation(
                    epoch=_epoch(row[timestamp_column], timestamp_column),
                    carrier_power_dbw=carrier,
                    pc_n0_db_hz=pc_n0,
                    pr_n0_db_hz=pr_n0,
                )
            )
        except ValueError as exc:
            return str(exc)
    if not observations:
        return "ADX result contains no SIGMET observations"
    try:
        segment = SignalMetricsSegment(
            transmit_band=config.transmit_band,
            receive_band=config.receive_band,
            observations=tuple(observations),
        )
    except ValueError as exc:
        return str(exc)
    return KsatDocument(KsatProduct.SIGMET, header, (segment,))


def build_ksat_tdm_bundle(
    frame: pl.DataFrame,
    header: KsatHeader,
    columns: KsatAdxColumnMap,
    *,
    track: TrackMetadata | None = None,
    angle: AngleExportConfig | None = None,
    signal_metrics: SignalMetricsExportConfig | None = None,
    products: tuple[KsatProduct, ...] = (
        KsatProduct.TRACK,
        KsatProduct.ANGLE,
        KsatProduct.SIGMET,
        KsatProduct.METEO,
    ),
) -> KsatExportResult:
    """Build every requested KSAT product supported by the supplied ADX frame."""

    if not isinstance(frame, pl.DataFrame):
        raise TypeError("frame must be a polars DataFrame")
    station_ok, station_error = _has_single_station(
        frame, columns.station_id, header.site.identifier
    )
    generated: dict[ProductName, GeneratedTdm] = {}
    skipped: dict[ProductName, str] = {}
    builders = {
        KsatProduct.TRACK: (
            track,
            lambda: _build_track_document(frame, header, columns, track),
        ),
        KsatProduct.ANGLE: (
            angle,
            lambda: _build_angle_document(frame, header, columns, angle),
        ),
        KsatProduct.SIGMET: (
            signal_metrics,
            lambda: _build_signal_metrics_document(
                frame, header, columns, signal_metrics
            ),
        ),
    }
    for product in dict.fromkeys(products):
        if product is KsatProduct.METEO:
            skipped[product.value] = (
                "METEO serialization is deferred until the normative KSAT definition "
                "and weather data source are available"
            )
            continue
        if not station_ok:
            skipped[product.value] = station_error or "invalid station selection"
            continue
        config, builder = builders[product]
        if config is None:
            skipped[product.value] = f"{product.value} was not configured"
            continue
        try:
            document_or_error = builder()
        except ValueError as exc:
            skipped[product.value] = str(exc)
            continue
        if isinstance(document_or_error, str):
            skipped[product.value] = document_or_error
            continue
        text = render_ksat_tdm(document_or_error)
        generated[product.value] = GeneratedTdm(
            product=product.value,
            filename=ksat_tdm_filename(document_or_error),
            text=text,
        )
    return KsatExportResult(generated=generated, skipped=skipped)


def export_ksat_tdm_bundle(
    selection: KsatAdxQuery,
    header: KsatHeader,
    *,
    track: TrackMetadata | None = None,
    angle: AngleExportConfig | None = None,
    signal_metrics: SignalMetricsExportConfig | None = None,
    products: tuple[KsatProduct, ...] = (
        KsatProduct.TRACK,
        KsatProduct.ANGLE,
        KsatProduct.SIGMET,
        KsatProduct.METEO,
    ),
    output_dir: str | Path | None = None,
    overwrite: bool = False,
    timeout_seconds: float = ADX_QUERY_TIMEOUT_SECONDS,
) -> KsatExportResult:
    """Fetch ADX telemetry, build available products, and optionally write them."""

    frame = fetch_ksat_tdm_data(selection, timeout_seconds=timeout_seconds)
    result = build_ksat_tdm_bundle(
        frame,
        header,
        selection.columns,
        track=track,
        angle=angle,
        signal_metrics=signal_metrics,
        products=products,
    )
    if output_dir is None or not result.generated:
        return result

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    destinations = {
        product: directory / generated.filename
        for product, generated in result.generated.items()
    }
    existing = [path for path in destinations.values() if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"refusing to overwrite {existing[0]}")
    written: dict[ProductName, GeneratedTdm] = {}
    for product, generated in result.generated.items():
        destination = destinations[product]
        destination.write_text(generated.text, encoding="ascii")
        written[product] = GeneratedTdm(
            product=generated.product,
            filename=generated.filename,
            text=generated.text,
            path=destination,
        )
    return KsatExportResult(
        generated=written,
        skipped=result.skipped,
        warnings=result.warnings,
    )


__all__ = [
    "AdxField",
    "AngleExportConfig",
    "GeneratedTdm",
    "KsatAdxColumnMap",
    "KsatAdxQuery",
    "KsatExportResult",
    "SignalMetricsExportConfig",
    "build_ksat_adx_query",
    "build_ksat_tdm_bundle",
    "convert_adx_value",
    "export_ksat_tdm_bundle",
    "fetch_ksat_tdm_data",
]
