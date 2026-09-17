"""DEPRECATED experiment runner: frozen FOREST historical timing regression.

The Parquet input loader remains usable; new studies use experiment.py v5.

No credentials, KOGS requests, or imports from dart-python are required.
"""

from __future__ import annotations

import hashlib
from datetime import datetime
from pathlib import Path
from typing import cast

import polars as pl
import satkit as sk

from dart.io import ContactMetadata, EphemerisMetadata
from dart.io.measurement import canonical_measurements
from dart.io.parquet import _ADX_NAMES
from experiments._benchmark_io import save_json
from experiments.legacy_metrics import warn_legacy_metrics
from experiments.time_offset import TimeOffsetResult, run_time_offset_loaded


def _contact(frame: pl.DataFrame, satellite: str) -> ContactMetadata:
    names = [
        "contact_id",
        "spacecraft_id",
        "system_id",
        "antenna_name",
        "groundStation",
        "station_lat",
        "station_lon",
        "station_alt",
        "tle_line1",
        "tle_line2",
    ]
    metadata = frame.select(names).unique()
    if metadata.height != 1 or any(metadata.null_count().row(0)):
        raise ValueError("recorded contact metadata must be present and constant")
    row = metadata.row(0, named=True)
    raw_tle = row["tle_line1"] + "\n" + row["tle_line2"]
    tle = sk.TLE.from_lines(raw_tle.splitlines())
    prior = EphemerisMetadata(
        ephemeris_id="parquet:tle:" + hashlib.sha256(raw_tle.encode()).hexdigest(),
        spacecraft_id=row["spacecraft_id"],
        kind="TLE",
        origin="recorded Parquet",
        tenant_id=None,
        epoch=tle.epoch.as_datetime(),
        last_usable_at=None,
        submitted_at=None,
        submitted_by=None,
        tle=raw_tle,
        omm=None,
        oem=None,
        is_cui=None,
        payload=None,
    )
    coordinate = sk.itrfcoord(
        latitude_deg=row["station_lat"],
        longitude_deg=row["station_lon"],
        altitude=row["station_alt"],
    )
    return ContactMetadata(
        spacecraft_id=row["spacecraft_id"],
        system_id=row["system_id"],
        station_id=row["groundStation"],
        ephemeris_id=prior.ephemeris_id,
        antenna=row["antenna_name"],
        location=row["groundStation"],
        latitude=row["station_lat"],
        longitude=row["station_lon"],
        altitude=row["station_alt"],
        ecef=tuple(coordinate.vector),
        spacecraft=satellite,
        # The snapshot has no authoritative COSPAR ID. Do not invent one or
        # publish an OEM from these experiment-only contact records.
        cospar="",
        catalog=row["tle_line1"][2:7].strip(),
        start=cast(datetime, frame["timestamp"].min()),
        stop=cast(datetime, frame["timestamp"].max()),
        contact_id=row["contact_id"],
        ephemeris=prior,
    )


def load_experiment(
    path: Path,
    *,
    spacecraft_id: str,
    satellite: str,
    center_frequency_hz: float,
) -> tuple[list[ContactMetadata], pl.DataFrame, dict[str, EphemerisMetadata]]:
    """Normalize recorded measurements and retain each contact's recorded TLE.

    Contact bounds are recorded telemetry extents, not KOGS reservation bounds.
    An absent tracking-offset column stays null; it never adjusts timestamps.
    """
    raw = pl.read_parquet(path).sort("timestamp")
    if raw.is_empty() or set(raw["spacecraft_id"].unique()) != {spacecraft_id}:
        raise ValueError("Parquet must contain the expected spacecraft")
    if set(raw["expected_frequency"].unique()) != {center_frequency_hz}:
        raise ValueError("Parquet nominal frequency differs from the case")
    contacts = [
        _contact(group, satellite)
        for group in raw.partition_by("contact_id", maintain_order=True)
    ]
    aliases = {
        name: target for name, target in _ADX_NAMES.items() if name in raw.columns
    }
    frame = raw.rename(aliases)
    if "tracking_epoch_offset_s" not in frame.columns:
        frame = frame.with_columns(
            pl.lit(None, dtype=pl.Float64).alias("tracking_epoch_offset_s")
        )
    return (
        contacts,
        canonical_measurements(frame),
        {c.contact_id: c.ephemeris for c in contacts},
    )


def run_comparison(
    path: Path,
    *,
    spacecraft_id: str,
    satellite: str,
    center_frequency_hz: float,
    gps_directory: Path,
    output_dir: Path,
    max_evaluations: int = 1000,
) -> list[TimeOffsetResult]:
    """Run recorded-prior regression through the shared timing fitter and scorer."""
    warn_legacy_metrics("Historical offline FOREST experiment runner")
    contacts, frame, priors = load_experiment(
        path,
        spacecraft_id=spacecraft_id,
        satellite=satellite,
        center_frequency_hz=center_frequency_hz,
    )
    results = run_time_offset_loaded(
        contacts,
        frame,
        priors,
        spacecraft_id=spacecraft_id,
        satellite=satellite,
        center_frequency_hz=center_frequency_hz,
        gps_directory=gps_directory,
        output_dir=output_dir,
        max_evaluations=max_evaluations,
    )
    raw = path.read_bytes()
    (output_dir / "source.parquet").write_bytes(raw)
    save_json(
        output_dir / "acquisition.json",
        {
            "provider": "parquet",
            "path": str(path),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "prior_policy": "recorded TLE per contact",
            "contact_window_policy": "recorded telemetry extents",
        },
    )
    return results


def main() -> None:
    import argparse
    import runpy

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument(
        "--parquet", type=Path, help="override the case Doppler snapshot"
    )
    parser.add_argument(
        "--gps-directory", type=Path, help="override raw BESTXYZ directory"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    case = runpy.run_path(str(args.case))
    results = run_comparison(
        args.parquet or case["DOPPLER_PARQUET"],
        spacecraft_id=case["SPACECRAFT_ID"],
        satellite=case["REFERENCE_OBJECT_ID"],
        center_frequency_hz=case["CENTER_FREQUENCY_HZ"],
        gps_directory=args.gps_directory or case["RAW_GPS_DIRECTORY"],
        output_dir=args.output,
    )
    print(f"Saved {len(results)} single-pass SGP4 time-offset fits to {args.output}")


if __name__ == "__main__":
    main()
