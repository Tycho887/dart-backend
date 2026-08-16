"""Shared normalization helpers for the loaders."""

from __future__ import annotations

import datetime

import polars as pl

from dart.schema import Observation, Station


def tle_epoch_unix(line1: str) -> float:
    """Parse the epoch out of a TLE line 1 (cols 19-32, ``YYDDD.dddddddd``).

    Two-digit year window per convention: 57-99 → 19xx, 00-56 → 20xx.
    Returns unix seconds (UTC), float.
    """
    epoch_str = line1[18:32].strip()
    if len(epoch_str) < 5:
        raise ValueError(f"TLE line 1 has no parseable epoch: {line1!r}")
    year_yy = int(epoch_str[:2])
    day_of_year = float(epoch_str[2:])
    year = 1900 + year_yy if year_yy >= 57 else 2000 + year_yy
    # TLE day-of-year is 1-based: "001.00000000" is Jan 1 00:00:00
    epoch = datetime.datetime(year, 1, 1, tzinfo=datetime.timezone.utc) + datetime.timedelta(
        days=day_of_year - 1.0
    )
    return epoch.timestamp()


def observations_from_frame(
    df: pl.DataFrame,
    stations: dict[str, Station],
) -> list[Observation]:
    """Normalize the ADX telemetry frame into schema Observations.

    Expects the column layout of ``dart.io.azure.fetch_tracking_data`` and a
    ``system_id`` → ``Station`` map. Raises ``ValueError`` when telemetry
    references a station without metadata (fail loudly, never guess).
    """
    unix_s = df["timestamp"].dt.epoch("s").to_list()
    dopplers = df["lr1_receiver1_actualCarrierFrequencyOffset"].to_list()
    azimuths = df["antenna1_position_azimuth"].to_list()
    elevations = df["antenna1_position_elevation"].to_list()
    station_ids = df["system_id"].to_list()

    observations = []
    for t, doppler, az, el, sid in zip(unix_s, dopplers, azimuths, elevations, station_ids):
        if sid not in stations:
            raise ValueError(f"no station metadata for system {sid!r}")
        observations.append(
            Observation(
                epoch_unix=float(t),
                doppler_hz=float(doppler),
                azimuth_deg=float(az),
                elevation_deg=float(el),
                station_id=str(sid),
            )
        )
    return observations
