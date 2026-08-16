"""Offline loader: recorded parquet telemetry → ``Sgp4Input``.

The parquet frames (e.g. ``doppler_parquet/forest*.parquet``) carry the ADX
telemetry columns plus the metadata the online path would fetch from KOGS —
TLE, station coordinates, nominal center frequency — so the SGP4 optimizer
can be exercised offline with recorded passes and no ADX/KOGS credentials.

Filtering mirrors the ADX query gates in ``dart.io.azure`` (elevation,
doppler range, optional carrier-lock state); normalization itself is
delegated to ``dart.loaders.leo.build_sgp4_input`` so the wire contract is
identical to the online path.
"""

from __future__ import annotations

import glob as _glob
from pathlib import Path

import polars as pl

from dart.loaders.leo import build_sgp4_input
from dart.schema import Sgp4Input, Station, Tle

#: columns the loader reads directly (a superset of what the observations
#: normalization needs); missing ones are rejected loudly up front.
_REQUIRED_COLUMNS = {
    "timestamp",
    "contact_id",
    "system_id",
    "lr1_receiver1_actualCarrierFrequencyOffset",
    "antenna1_position_azimuth",
    "antenna1_position_elevation",
    "tle_line1",
    "tle_line2",
    "spacecraft_id",
    "expected_frequency",
    "groundStation",
    "antenna_name",
    "station_lat",
    "station_lon",
    "station_alt",
}

_LOCK_COLUMN = "lr1_receiver1_carrierLockState"


def _parquet_files(source: str | Path) -> list[Path]:
    """Resolve a file, glob pattern, or directory into parquet paths."""
    path = Path(source)
    if path.is_dir():
        files = sorted(path.glob("*.parquet"))
    else:
        files = sorted(Path(p) for p in _glob.glob(str(path)))
    if not files:
        raise ValueError(f"no parquet files found at {source!r}")
    return files


def read_parquet(source: str | Path) -> pl.DataFrame:
    """Read one or more parquet files into a single timestamp-sorted frame."""
    frames = [pl.read_parquet(p) for p in _parquet_files(source)]
    return pl.concat(frames).sort("timestamp")


def stations_from_frame(df: pl.DataFrame) -> dict[str, Station]:
    """``system_id`` → Station from the coordinates embedded in the frame.

    ``station_alt`` is in meters (ADX convention), converted to km to match
    ``dart.loaders.leo.station_from_kogs``.
    """
    rows = (
        df.group_by("system_id")
        .first()
        .select(
            "system_id",
            "groundStation",
            "antenna_name",
            "station_lat",
            "station_lon",
            "station_alt",
        )
        .iter_rows(named=True)
    )
    stations = {}
    for row in rows:
        lat, lon, alt = row["station_lat"], row["station_lon"], row["station_alt"]
        if None in (lat, lon, alt):
            raise ValueError(
                f"incomplete station coordinates for system {row['system_id']!r}"
            )
        stations[row["system_id"]] = Station(
            id=row["system_id"],
            name=row["groundStation"] or row["antenna_name"],
            lat_deg=float(lat),
            lon_deg=float(lon),
            alt_km=float(alt) / 1000.0,
        )
    return stations


def _filter_frame(
    df: pl.DataFrame,
    *,
    require_lock: bool,
    min_elevation_deg: float,
    min_doppler_hz: float,
    max_doppler_hz: float,
) -> pl.DataFrame:
    """Apply the ADX-style gates; raise on missing columns.

    The doppler gate is magnitude-based (``0Hz/100kHz`` rule): keep rows with
    ``min_doppler_hz <= |offset| <= max_doppler_hz``, i.e. drop no-contact
    zeros and clipped outliers while retaining both doppler signs.
    """
    missing = [c for c in sorted(_REQUIRED_COLUMNS) if c not in df.columns]
    if require_lock and _LOCK_COLUMN not in df.columns:
        missing.append(_LOCK_COLUMN)
    if missing:
        raise ValueError(f"parquet frame is missing columns: {', '.join(missing)}")

    filtered = df.drop_nulls(
        [
            "timestamp",
            "contact_id",
            "system_id",
            "lr1_receiver1_actualCarrierFrequencyOffset",
            "antenna1_position_azimuth",
            "antenna1_position_elevation",
        ]
    ).filter(
        pl.col("antenna1_position_elevation") >= min_elevation_deg,
        pl.col("lr1_receiver1_actualCarrierFrequencyOffset").abs() >= min_doppler_hz,
        pl.col("lr1_receiver1_actualCarrierFrequencyOffset").abs() <= max_doppler_hz,
    )
    if require_lock:
        filtered = filtered.filter(pl.col(_LOCK_COLUMN) == "Locked")
    return filtered


def _drop_short_passes(df: pl.DataFrame, min_pass_measurements: int) -> pl.DataFrame:
    """Drop passes (contacts) with fewer than ``min_pass_measurements`` rows.

    Passes with too little post-filter data cannot constrain the per-pass bias
    plus the shared elements; gating them keeps the fit well-conditioned.
    """
    if min_pass_measurements <= 0:
        return df
    counts = df.group_by("contact_id").len()
    keep = counts.filter(pl.col("len") >= min_pass_measurements)["contact_id"]
    return df.filter(pl.col("contact_id").is_in(keep))


def build_sgp4_input_from_parquet(
    source: str | Path,
    *,
    require_lock: bool = False,
    min_elevation_deg: float = 1.0,
    min_doppler_hz: float = 1.0,
    max_doppler_hz: float = 1e5,
    min_pass_measurements: int = 0,
    timestamp_offset_s: float = 0.0,
    max_rows: int | None = None,
    tle: Tle | None = None,
    fit_model: str = "mean_anomaly",
) -> Sgp4Input:
    """Recorded parquet telemetry → ``Sgp4Input`` for the SGP4 optimizer.

    Metadata is resolved from the frame itself, mirroring the online loader's
    v1 assumptions: the TLE is the first row's (first-contact semantics),
    stations come from the embedded coordinates, and the nominal center
    frequency is ``expected_frequency``. Each frame must contain a single
    spacecraft — ``Sgp4Input`` is a single-TLE batch.

    Gates (defaults match the production pipeline): ``min_doppler_hz`` /
    ``max_doppler_hz`` implement the magnitude ``0Hz/100kHz`` filter and
    ``min_pass_measurements`` drops passes with too little post-filter data
    (0 disables). ``timestamp_offset_s`` shifts the measurement epochs by a
    constant (backend latency calibration: recorded timestamps arrive late,
    e.g. +0.35 s); 0.0 leaves them untouched.
    """
    df = read_parquet(source)
    if timestamp_offset_s:
        df = df.with_columns(
            (pl.col("timestamp") + pl.duration(seconds=timestamp_offset_s)).alias("timestamp")
        )
    filtered = _filter_frame(
        df,
        require_lock=require_lock,
        min_elevation_deg=min_elevation_deg,
        min_doppler_hz=min_doppler_hz,
        max_doppler_hz=max_doppler_hz,
    )
    if filtered.is_empty():
        raise ValueError("no observations after filtering")
    filtered = _drop_short_passes(filtered, min_pass_measurements)
    if filtered.is_empty():
        raise ValueError(
            "no observations after filtering "
            f"(all passes below min_pass_measurements={min_pass_measurements})"
        )

    resolved_tle = tle
    if resolved_tle is None:
        resolved_tle = Tle(
            line1=filtered["tle_line1"][0], line2=filtered["tle_line2"][0]
        )

    spacecraft_ids = filtered["spacecraft_id"].unique().to_list()
    if len(spacecraft_ids) != 1:
        raise ValueError(
            f"expected a single spacecraft, got {sorted(spacecraft_ids)}"
        )

    centers = filtered["expected_frequency"].drop_nulls().unique().to_list()
    if len(centers) != 1:
        raise ValueError(
            f"expected a single expected_frequency, got {sorted(centers)}"
        )

    stations = stations_from_frame(filtered)
    if max_rows is not None:
        filtered = filtered.head(max_rows)

    return build_sgp4_input(
        filtered,
        tle=resolved_tle,
        stations=stations,
        nominal_center_frequency_hz=float(centers[0]),
        fit_model=fit_model,
        spacecraft_id=spacecraft_ids[0],
    )


def load_offline_sgp4_inputs(
    source: str | Path = "doppler_parquet",
    **kwargs,
) -> list[Sgp4Input]:
    """One ``Sgp4Input`` per parquet file in ``source``.

    Each file is expected to be a single-spacecraft batch (as in
    ``doppler_parquet/``); ``kwargs`` are forwarded to
    ``build_sgp4_input_from_parquet`` (e.g. ``max_rows=500`` for a quick
    subset).
    """
    return [build_sgp4_input_from_parquet(f, **kwargs) for f in _parquet_files(source)]
