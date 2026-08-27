"""Configuration and orchestration for ADX-backed KSAT TDM exports."""

from __future__ import annotations

import datetime as dt
import math
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

from dart.io.azure import ADX_QUERY_TIMEOUT_SECONDS
from dart.io.ksat_adx import (
    AdxField,
    AngleExportConfig,
    KsatAdxColumnMap,
    KsatAdxQuery,
    KsatExportResult,
    SignalMetricsExportConfig,
    export_ksat_tdm_bundle,
)
from dart.io.ksat_metadata import (
    GeocoderConfig,
    KogsMetadataConfig,
    KsatSiteOverrides,
    load_ksat_contact_metadata,
)
from dart.io.ksat_tdm import (
    KsatHeader,
    KsatProduct,
    KsatSite,
    KsatSpacecraft,
    TrackMetadata,
)


DEFAULT_TIMEOUT_SECONDS = ADX_QUERY_TIMEOUT_SECONDS

_ROOT_KEYS = frozenset(
    {
        "site",
        "spacecraft",
        "kogs",
        "geocoder",
        "header",
        "adx",
        "track",
        "angle",
        "sigmet",
    }
)
_SITE_KEYS = frozenset(
    {
        "name",
        "pedestal_offset_m",
        "tlt_calibration_date",
        "tlt_band",
    }
)
_SPACECRAFT_KEYS = frozenset({"identifier", "name", "cospar_id", "catalog_id"})
_KOGS_KEYS = frozenset(
    {"spacecraft_id", "system_id", "station_id", "timeout_seconds"}
)
_GEOCODER_KEYS = frozenset({"url", "user_agent", "timeout_seconds"})
_HEADER_KEYS = frozenset({"summary", "comments"})
_FIELD_UNITS = {
    "angle_1": frozenset({"deg", "rad"}),
    "angle_2": frozenset({"deg", "rad"}),
    "range_delay": frozenset({"s", "ms", "us", "ns", "ps"}),
    "transmit_frequency": frozenset({"Hz", "kHz", "MHz", "GHz"}),
    "receive_frequency": frozenset({"Hz", "kHz", "MHz", "GHz"}),
    "carrier_power": frozenset({"dBW"}),
    "pc_n0": frozenset({"dB-Hz"}),
    "pr_n0": frozenset({"dB-Hz"}),
}
_ADX_SCALAR_KEYS = frozenset(
    {
        "timestamp",
        "track_timestamp",
        "angle_timestamp",
        "signal_metrics_timestamp",
        "contact_id",
        "station_id",
        "tracking_mode",
    }
)
_ADX_KEYS = _ADX_SCALAR_KEYS | _FIELD_UNITS.keys()
_TRACK_KEYS = frozenset(
    {
        "mode",
        "transmit_band",
        "receive_band",
        "integration_interval_s",
        "turnaround_numerator",
        "turnaround_denominator",
        "transmit_delay_s",
        "receive_delay_s",
        "correction_range_s",
        "correction_doppler_hz",
    }
)
_ANGLE_KEYS = frozenset({"receive_band", "angle_type", "tracking_mode"})
_SIGMET_KEYS = frozenset({"transmit_band", "receive_band"})
_BANDS = frozenset({"S", "X", "Ka"})
_ANGLE_TYPES = frozenset({"AZEL", "XEYN", "XSYE"})
_TRACKING_MODES = frozenset({"AUTO", "PROGRAM", "SCAN"})


@dataclass(frozen=True, slots=True)
class KsatExportConfig:
    """Validated static configuration for one KSAT site and spacecraft."""

    site: KsatSiteOverrides
    spacecraft: KsatSpacecraft
    kogs: KogsMetadataConfig
    geocoder: GeocoderConfig
    columns: KsatAdxColumnMap
    header_summary: str | None = None
    header_comments: tuple[str, ...] = field(default_factory=tuple)
    track: TrackMetadata | None = None
    angle: AngleExportConfig | None = None
    signal_metrics: SignalMetricsExportConfig | None = None

    @property
    def products(self) -> tuple[KsatProduct, ...]:
        """Products configured by the corresponding TOML sections."""

        configured: list[KsatProduct] = []
        if self.track is not None:
            configured.append(KsatProduct.TRACK)
        if self.angle is not None:
            configured.append(KsatProduct.ANGLE)
        if self.signal_metrics is not None:
            configured.append(KsatProduct.SIGMET)
        return tuple(configured)

    @property
    def configured_products(self) -> tuple[KsatProduct, ...]:
        """Alias that makes product discovery explicit to callers."""

        return self.products

    @property
    def sigmet(self) -> SignalMetricsExportConfig | None:
        """Return the configuration loaded from the ``[sigmet]`` section."""

        return self.signal_metrics


def _reject_unknown(
    section: str,
    values: Mapping[str, object],
    allowed: set[str] | frozenset[str],
) -> None:
    unknown = sorted(set(values) - set(allowed))
    if unknown:
        names = ", ".join(repr(name) for name in unknown)
        raise ValueError(f"unknown key(s) in {section}: {names}")


def _table(
    document: Mapping[str, object], name: str, *, required: bool
) -> Mapping[str, object] | None:
    if name not in document:
        if required:
            raise ValueError(f"missing required [{name}] section")
        return None
    value = document[name]
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a TOML table")
    return value


def _require_keys(section: str, values: Mapping[str, object], required: set[str]) -> None:
    missing = sorted(required - set(values))
    if missing:
        names = ", ".join(repr(name) for name in missing)
        raise ValueError(f"missing required key(s) in {section}: {names}")


def _string(section: str, name: str, value: object) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{section}.{name} must be a string")
    return value


def _parse_adx_field(name: str, value: object) -> AdxField:
    section = f"adx.{name}"
    if not isinstance(value, dict):
        raise ValueError(f"{section} must be a TOML table with column and unit")
    _reject_unknown(section, value, frozenset({"column", "unit"}))
    _require_keys(section, value, {"column", "unit"})
    column = _string(section, "column", value["column"])
    unit = _string(section, "unit", value["unit"])
    allowed = _FIELD_UNITS[name]
    if unit not in allowed:
        expected = ", ".join(sorted(allowed))
        raise ValueError(
            f"{section}.unit must be one of {expected}; got {unit!r}"
        )
    return AdxField(column=column, unit=unit)  # type: ignore[arg-type]


def _construct(section: str, factory: type[object], values: dict[str, object]) -> object:
    try:
        return factory(**values)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid [{section}] configuration: {exc}") from exc


def _parse_site(document: Mapping[str, object]) -> KsatSiteOverrides:
    values = _table(document, "site", required=True)
    assert values is not None
    _reject_unknown("site", values, _SITE_KEYS)
    return _construct("site", KsatSiteOverrides, dict(values))  # type: ignore[return-value]


def _parse_spacecraft(document: Mapping[str, object]) -> KsatSpacecraft:
    values = _table(document, "spacecraft", required=True)
    assert values is not None
    _reject_unknown("spacecraft", values, _SPACECRAFT_KEYS)
    _require_keys("spacecraft", values, {"identifier"})
    return _construct("spacecraft", KsatSpacecraft, dict(values))  # type: ignore[return-value]


def _parse_kogs(document: Mapping[str, object]) -> KogsMetadataConfig:
    values = _table(document, "kogs", required=True)
    assert values is not None
    _reject_unknown("kogs", values, _KOGS_KEYS)
    _require_keys("kogs", values, {"spacecraft_id", "system_id", "station_id"})
    return _construct("kogs", KogsMetadataConfig, dict(values))  # type: ignore[return-value]


def _parse_geocoder(document: Mapping[str, object]) -> GeocoderConfig:
    values = _table(document, "geocoder", required=False)
    if values is None:
        return GeocoderConfig()
    _reject_unknown("geocoder", values, _GEOCODER_KEYS)
    return _construct("geocoder", GeocoderConfig, dict(values))  # type: ignore[return-value]


def _parse_header(document: Mapping[str, object]) -> tuple[str | None, tuple[str, ...]]:
    values = _table(document, "header", required=False)
    if values is None:
        return None, ()
    _reject_unknown("header", values, _HEADER_KEYS)
    summary_value = values.get("summary")
    summary = (
        None if summary_value is None else _string("header", "summary", summary_value)
    )
    comments_value = values.get("comments", [])
    if not isinstance(comments_value, list) or not all(
        isinstance(comment, str) for comment in comments_value
    ):
        raise ValueError("header.comments must be an array of strings")
    # Reuse KsatHeader validation for ASCII, blank, and newline checks.
    probe = _construct(
        "header",
        KsatHeader,
        {
            "creation_date": dt.datetime(2000, 1, 1, tzinfo=dt.timezone.utc),
            "site": KsatSite("VALIDATION"),
            "spacecraft": KsatSpacecraft("VALIDATION"),
            "summary": summary,
            "comments": tuple(comments_value),
        },
    )
    assert isinstance(probe, KsatHeader)
    return probe.summary, probe.comments


def _parse_columns(document: Mapping[str, object]) -> KsatAdxColumnMap:
    values = _table(document, "adx", required=True)
    assert values is not None
    _reject_unknown("adx", values, _ADX_KEYS)
    kwargs: dict[str, object] = {}
    for name, value in values.items():
        if name in _FIELD_UNITS:
            kwargs[name] = _parse_adx_field(name, value)
        else:
            kwargs[name] = _string("adx", name, value)
    return _construct("adx", KsatAdxColumnMap, kwargs)  # type: ignore[return-value]


def _parse_product(
    document: Mapping[str, object],
    name: str,
    allowed: frozenset[str],
    required: set[str],
    factory: type[object],
) -> object | None:
    values = _table(document, name, required=False)
    if values is None:
        return None
    _reject_unknown(name, values, allowed)
    _require_keys(name, values, required)
    return _construct(name, factory, dict(values))


def _validate_product_mappings(config: KsatExportConfig) -> None:
    columns = config.columns
    if config.track is not None:
        required: list[tuple[str, object | None]] = [
            ("adx.track_timestamp", columns.track_timestamp),
        ]
        if config.track.mode in (1, 3):
            required.append(("adx.range_delay", columns.range_delay))
        if config.track.mode in (3, 4):
            required.extend(
                (
                    ("adx.transmit_frequency", columns.transmit_frequency),
                    ("adx.receive_frequency", columns.receive_frequency),
                )
            )
        missing = [name for name, value in required if value is None]
        if missing:
            raise ValueError(
                "[track] requires mapping(s): " + ", ".join(missing)
            )
    if config.angle is not None:
        if config.angle.receive_band not in _BANDS:
            raise ValueError("angle.receive_band must be one of S, X, or Ka")
        if config.angle.angle_type not in _ANGLE_TYPES:
            raise ValueError("angle.angle_type must be AZEL, XEYN, or XSYE")
        if (
            config.angle.tracking_mode is not None
            and config.angle.tracking_mode not in _TRACKING_MODES
        ):
            raise ValueError(
                "angle.tracking_mode must be AUTO, PROGRAM, or SCAN"
            )
        missing_angles = [
            name
            for name, value in (
                ("adx.angle_1", columns.angle_1),
                ("adx.angle_2", columns.angle_2),
            )
            if value is None
        ]
        if missing_angles:
            raise ValueError(
                "[angle] requires mapping(s): " + ", ".join(missing_angles)
            )
        if columns.tracking_mode is None and config.angle.tracking_mode is None:
            raise ValueError(
                "[angle] requires adx.tracking_mode or angle.tracking_mode"
            )
    if config.signal_metrics is not None:
        if config.signal_metrics.transmit_band not in _BANDS:
            raise ValueError("sigmet.transmit_band must be one of S, X, or Ka")
        if config.signal_metrics.receive_band not in _BANDS:
            raise ValueError("sigmet.receive_band must be one of S, X, or Ka")
        if all(
            mapping is None
            for mapping in (columns.carrier_power, columns.pc_n0, columns.pr_n0)
        ):
            raise ValueError(
                "[sigmet] requires at least one of adx.carrier_power, "
                "adx.pc_n0, or adx.pr_n0"
            )


def load_ksat_export_config(path: str | Path) -> KsatExportConfig:
    """Load and fully validate a strict KSAT export TOML file."""

    source = Path(path)
    try:
        with source.open("rb") as stream:
            document = tomllib.load(stream)
    except tomllib.TOMLDecodeError as exc:
        raise ValueError(f"invalid TOML in {source}: {exc}") from exc
    if not isinstance(document, dict):  # pragma: no cover - guaranteed by tomllib
        raise ValueError("KSAT export configuration must be a TOML document")
    _reject_unknown("root", document, _ROOT_KEYS)

    site = _parse_site(document)
    spacecraft = _parse_spacecraft(document)
    kogs = _parse_kogs(document)
    geocoder = _parse_geocoder(document)
    summary, comments = _parse_header(document)
    columns = _parse_columns(document)
    track = _parse_product(
        document,
        "track",
        _TRACK_KEYS,
        {"mode", "transmit_band", "receive_band", "integration_interval_s"},
        TrackMetadata,
    )
    angle = _parse_product(
        document,
        "angle",
        _ANGLE_KEYS,
        {"receive_band"},
        AngleExportConfig,
    )
    signal_metrics = _parse_product(
        document,
        "sigmet",
        _SIGMET_KEYS,
        {"transmit_band", "receive_band"},
        SignalMetricsExportConfig,
    )
    config = KsatExportConfig(
        site=site,
        spacecraft=spacecraft,
        kogs=kogs,
        geocoder=geocoder,
        columns=columns,
        header_summary=summary,
        header_comments=comments,
        track=track if isinstance(track, TrackMetadata) else None,
        angle=angle if isinstance(angle, AngleExportConfig) else None,
        signal_metrics=(
            signal_metrics
            if isinstance(signal_metrics, SignalMetricsExportConfig)
            else None
        ),
    )
    _validate_product_mappings(config)
    return config


def parse_utc_datetime(value: str) -> dt.datetime:
    """Parse an offset-aware ISO-8601 timestamp and normalize it to UTC."""

    if not isinstance(value, str) or not value.strip():
        raise ValueError("timestamp must be a non-empty ISO-8601 string")
    candidate = value.strip()
    if candidate.endswith(("Z", "z")):
        candidate = candidate[:-1] + "+00:00"
    try:
        parsed = dt.datetime.fromisoformat(candidate)
    except ValueError as exc:
        raise ValueError(f"invalid ISO-8601 timestamp {value!r}") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"timestamp must include a UTC offset: {value!r}")
    return parsed.astimezone(dt.timezone.utc)


def _normalize_products(
    products: Sequence[KsatProduct | str] | None,
    configured: tuple[KsatProduct, ...],
) -> tuple[KsatProduct, ...]:
    selected = configured if products is None else products
    normalized: list[KsatProduct] = []
    for product in selected:
        try:
            item = (
                product
                if isinstance(product, KsatProduct)
                else KsatProduct(product.upper())
            )
        except (AttributeError, ValueError) as exc:
            raise ValueError(f"unsupported KSAT product {product!r}") from exc
        if item is KsatProduct.METEO:
            raise ValueError("METEO export is not supported")
        if item not in normalized:
            normalized.append(item)
    if not normalized:
        raise ValueError("no KSAT products were requested or configured")
    return tuple(normalized)


def export_ksat_contact(
    config: KsatExportConfig,
    *,
    contact_id: str,
    start_time: dt.datetime,
    stop_time: dt.datetime,
    output_dir: str | Path,
    products: Sequence[KsatProduct | str] | None = None,
    overwrite: bool = False,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
) -> KsatExportResult:
    """Export selected products for exactly one bounded ADX contact."""

    if not isinstance(config, KsatExportConfig):
        raise TypeError("config must be a KsatExportConfig")
    if isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, (int, float)):
        raise ValueError("timeout_seconds must be a finite positive number")
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be a finite positive number")
    selected_products = _normalize_products(products, config.products)
    metadata = load_ksat_contact_metadata(
        contact_id=contact_id,
        kogs=config.kogs,
        geocoder=config.geocoder,
        site=config.site,
        spacecraft=config.spacecraft,
    )
    selection = KsatAdxQuery(
        contact_ids=(contact_id,),
        start_time=start_time,
        stop_time=stop_time,
        columns=config.columns,
    )
    header = KsatHeader(
        creation_date=dt.datetime.now(dt.timezone.utc).replace(microsecond=0),
        site=metadata.site,
        spacecraft=metadata.spacecraft,
        summary=config.header_summary,
        comments=config.header_comments,
    )
    result = export_ksat_tdm_bundle(
        selection,
        header,
        track=config.track,
        angle=config.angle,
        signal_metrics=config.signal_metrics,
        products=selected_products,
        output_dir=output_dir,
        overwrite=overwrite,
        timeout_seconds=float(timeout_seconds),
    )
    return KsatExportResult(
        generated=result.generated,
        skipped=result.skipped,
        warnings=(*metadata.warnings, *result.warnings),
    )


__all__ = [
    "DEFAULT_TIMEOUT_SECONDS",
    "KsatExportConfig",
    "export_ksat_contact",
    "load_ksat_export_config",
    "parse_utc_datetime",
]
