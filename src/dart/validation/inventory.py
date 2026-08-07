"""Auditable contact-level inventory before estimator eligibility filtering."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from ..io.forest import DOPPLER_COLUMN


def forest_contact_inventory(
    path: Path | str,
    satellite: str,
    *,
    min_samples: int = 301,
) -> list[dict]:
    frame = pl.read_parquet(path).sort("timestamp")
    result = []
    for contact_id in frame["contact_id"].unique(maintain_order=True).to_list():
        raw = frame.filter(pl.col("contact_id") == contact_id)
        nonnull = raw.filter(pl.col(DOPPLER_COLUMN).is_not_null())
        nonzero = nonnull.filter(pl.col(DOPPLER_COLUMN) != 0.0)
        above_floor = nonzero.filter(pl.col(DOPPLER_COLUMN).abs() >= 0.1)
        usable = above_floor.filter(
            (pl.col("antenna1_position_elevation") > 1.0)
            & (pl.col("antenna1_position_elevation") < 89.0)
        )
        metadata_columns = [
            "groundStation",
            "station_lat",
            "station_lon",
            "station_alt",
            "expected_frequency",
            "tle_line1",
            "tle_line2",
        ]
        result.append(
            {
                "satellite": satellite,
                "contact_id": str(contact_id),
                "station": str(raw["groundStation"][0]),
                "raw_samples": raw.height,
                "doppler_nonnull": nonnull.height,
                "doppler_nonzero": nonzero.height,
                "doppler_above_floor": above_floor.height,
                "presented_samples": usable.height,
                "eligible": usable.height >= min_samples,
                "metadata_constant": all(
                    raw[column].n_unique() == 1 for column in metadata_columns
                ),
                "raw_start_utc": raw["timestamp"][0].isoformat(),
                "raw_end_utc": raw["timestamp"][-1].isoformat(),
                "presented_start_utc": (
                    None if usable.is_empty() else usable["timestamp"][0].isoformat()
                ),
                "presented_end_utc": (
                    None if usable.is_empty() else usable["timestamp"][-1].isoformat()
                ),
            }
        )
    return result
