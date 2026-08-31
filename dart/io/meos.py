"""Reviewed station constants standing in for the future MEOS integration."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, datetime


@dataclass(frozen=True, slots=True)
class TrackCalibration:
    pedestal_offset_m: float
    tlt_calibration_date: date
    correction_doppler_hz: float

    def __post_init__(self) -> None:
        values = (self.pedestal_offset_m, self.correction_doppler_hz)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("MEOS calibration values must be finite")
        if self.pedestal_offset_m < 0:
            raise ValueError("pedestal offset must be non-negative")


# Populate only with reviewed operational values, keyed by (antenna, band).
TRACK_CALIBRATIONS: dict[tuple[str, str], TrackCalibration] = {}


def get_track_calibration(
    antenna: str, band: str, contact_start: datetime
) -> TrackCalibration:
    """Return reviewed constants for one antenna/band or fail closed."""
    try:
        calibration = TRACK_CALIBRATIONS[(antenna, band)]
    except KeyError as exc:
        raise LookupError(
            f"no reviewed MEOS TRACK calibration for {antenna}/{band}"
        ) from exc
    if calibration.tlt_calibration_date > contact_start.date():
        raise ValueError("TLT calibration date is later than the contact start")
    return calibration
