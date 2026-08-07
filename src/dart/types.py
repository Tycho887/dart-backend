"""Public domain types and unit-bearing configuration objects."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np


class MeasurementMode(str, Enum):
    """RF channels consumed by an estimator."""

    DOPPLER = "doppler"
    DOPPLER_PHASE = "doppler_phase"


@dataclass(frozen=True, slots=True)
class Station:
    """Ground station in geodetic degrees and metres."""

    station_id: str
    latitude_deg: float
    longitude_deg: float
    altitude_m: float


@dataclass(frozen=True, slots=True)
class PhaseBaseline:
    """Antenna-1 minus antenna-2 baseline in local ENU metres."""

    east_m: float
    north_m: float
    up_m: float
    covariance_m2: np.ndarray | None = field(default=None, repr=False)

    @property
    def enu_m(self) -> np.ndarray:
        return np.array([self.east_m, self.north_m, self.up_m], dtype=float)

    def __post_init__(self) -> None:
        if self.covariance_m2 is not None:
            covariance = np.asarray(self.covariance_m2, dtype=float)
            if covariance.shape != (3, 3):
                raise ValueError("baseline covariance must have shape (3, 3)")
            object.__setattr__(self, "covariance_m2", covariance)


@dataclass(frozen=True, slots=True)
class TLEContext:
    """Immutable orbit and RF context used by measurement models."""

    tle: Any
    station: Station
    carrier_hz: float
    baseline: PhaseBaseline | None = None

    def __post_init__(self) -> None:
        if self.carrier_hz <= 0.0:
            raise ValueError("carrier_hz must be positive")


@dataclass(frozen=True, slots=True)
class RFObservation:
    """One passive-RF sample at its UTC measurement epoch.

    ``doppler_hz`` is a carrier-frequency offset.  ``phase_rad`` is optional;
    when present it may be wrapped into [-pi, pi].
    """

    epoch: Any
    station_id: str
    doppler_hz: float | None
    phase_rad: float | None = None
    valid: bool = True
    commanded_az_deg: float | None = None
    commanded_el_deg: float | None = None
    applied_offset_s: float | None = None
    sequence: int | None = None
    quality: dict[str, Any] = field(default_factory=dict)

    @property
    def mode(self) -> MeasurementMode:
        return (
            MeasurementMode.DOPPLER_PHASE
            if self.phase_rad is not None
            else MeasurementMode.DOPPLER
        )

    def measurement(self, mode: MeasurementMode | None = None) -> np.ndarray:
        selected = self.mode if mode is None else mode
        if self.doppler_hz is None:
            raise ValueError("a usable RF observation requires Doppler")
        if selected is MeasurementMode.DOPPLER:
            return np.array([self.doppler_hz], dtype=float)
        if self.phase_rad is None:
            raise ValueError("phase measurement requested but not present")
        return np.array([self.doppler_hz, self.phase_rad], dtype=float)


@dataclass(frozen=True, slots=True)
class Estimate:
    """Estimator output after one accepted or rejected observation."""

    epoch: Any
    mode: MeasurementMode
    offset_s: float
    frequency_bias_hz: float
    covariance: np.ndarray
    phase_bias_rad: float | None = None
    innovation: np.ndarray | None = None
    nis: float | None = None
    accepted: bool = True
    healthy: bool = True
    reason: str | None = None

    @property
    def offset_std_s(self) -> float:
        return float(np.sqrt(max(0.0, self.covariance[0, 0])))
