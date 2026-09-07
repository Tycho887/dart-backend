"""Canonical column names and types for provider-neutral measurement frames."""

from __future__ import annotations

import polars as pl

MEASUREMENT_COLUMNS = (
    "timestamp",
    "contact_id",
    "spacecraft_id",
    "system_id",
    "antenna_name",
    "tracking_epoch_offset_s",
    "azimuth_deg",
    "elevation_deg",
    "carrier_lock",
    "ebn0",
    "doppler_hz",
)

_STRING_COLUMNS = (
    "contact_id",
    "spacecraft_id",
    "system_id",
    "antenna_name",
    "carrier_lock",
)
_FLOAT_COLUMNS = (
    "tracking_epoch_offset_s",
    "azimuth_deg",
    "elevation_deg",
    "ebn0",
    "doppler_hz",
)


def canonical_measurements(frame: pl.DataFrame) -> pl.DataFrame:
    """Validate, type, and order a frame without filtering measurement rows."""

    missing = set(MEASUREMENT_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(
            f"measurement frame is missing columns: {', '.join(sorted(missing))}"
        )
    timestamp_type = frame.schema["timestamp"]
    if not isinstance(timestamp_type, pl.Datetime):
        raise ValueError("measurement timestamp must be a datetime column")
    timestamp = pl.col("timestamp")
    if timestamp_type.time_zone is None:
        timestamp = timestamp.dt.replace_time_zone("UTC")
    else:
        timestamp = timestamp.dt.convert_time_zone("UTC")
    result = frame.select(MEASUREMENT_COLUMNS).with_columns(
        timestamp.cast(pl.Datetime("us", "UTC")),
        *(pl.col(name).cast(pl.String) for name in _STRING_COLUMNS),
        *(pl.col(name).cast(pl.Float64) for name in _FLOAT_COLUMNS),
    )
    required = ("timestamp", "contact_id", "spacecraft_id", "system_id")
    if any(result[name].null_count() for name in required):
        raise ValueError(
            "measurement identity and timestamp columns may not contain nulls"
        )
    return result.sort("timestamp")
