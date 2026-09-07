"""Read recorded measurements from Parquet into the canonical IO schema."""

from __future__ import annotations

import glob
from pathlib import Path

import polars as pl

from .measurement import canonical_measurements

_ADX_NAMES = {
    "antenna1_tracking_epochOffset": "tracking_epoch_offset_s",
    "antenna1_position_azimuth": "azimuth_deg",
    "antenna1_position_elevation": "elevation_deg",
    "lr1_receiver1_carrierLockState": "carrier_lock",
    "lr1_receiver1_ebN0": "ebn0",
    "lr1_receiver1_actualCarrierFrequencyOffset": "doppler_hz",
}


def files(source: str | Path) -> list[Path]:
    """Resolve a Parquet file, directory, or glob into deterministic paths."""

    path = Path(source)
    paths = (
        sorted(path.glob("*.parquet"))
        if path.is_dir()
        else sorted(Path(value) for value in glob.glob(str(path)))
    )
    if not paths:
        raise ValueError(f"no parquet files found at {source!r}")
    return paths


def read_measurements(source: str | Path) -> pl.DataFrame:
    """Read and normalize one or more recorded measurement files."""

    frame = pl.concat([pl.read_parquet(path) for path in files(source)])
    aliases = {
        name: canonical
        for name, canonical in _ADX_NAMES.items()
        if name in frame.columns
    }
    frame = frame.rename(aliases)
    try:
        return canonical_measurements(frame)
    except ValueError as exc:
        raise ValueError(f"invalid parquet measurements: {exc}") from exc
