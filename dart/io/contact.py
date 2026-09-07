"""Resolve KOGS contacts and provide forward-model estimation containers.

This module resolves delivery metadata from KOGS and structures observation
batches and sensor configurations for the Rust forward-models backend.
"""

from __future__ import annotations

import datetime as dt
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Sequence

import satkit as sk

from dart.io import kogs
from dart.io.auth import create_api_auth
from dart.io.utils import require, utc


class MeasurementKind(str, Enum):
    """Supported measurement types matching the Rust forward model."""
    DOPPLER = "Doppler"
    TRUE_RANGE = "TrueRange"
    PSEUDORANGE_PHASE = "PseudorangePhase"


@dataclass(frozen=True, slots=True)
class ContactMetadata:
    """Identity, station, and ephemeris metadata for one KOGS contact."""
    spacecraft_id: str
    system_id: str
    station_id: str
    ephemeris_id: str
    antenna: str
    location: str
    latitude: float
    longitude: float
    altitude: float
    ecef: tuple[float, float, float]
    spacecraft: str
    cospar: str
    catalog: str
    start: dt.datetime
    stop: dt.datetime

    def to_itrfcoord(self) -> sk.itrfcoord:
        """Convert antenna coordinates into a satkit ITRF coordinate object."""
        return sk.itrfcoord(
            latitude_deg=self.latitude,
            longitude_deg=self.longitude,
            altitude=self.altitude,
        )


@dataclass(slots=True)
class ForwardObservation:
    """Generic container matching Rust's ObservationRecord wire format.

    All numerical fields use standard SI units (Hz for Doppler, meters for Range).
    """
    time: sk.instant
    observed: list[float] = field(default_factory=list)
    noise_cov: list[list[float]] = field(default_factory=list)
    receiver_id: int = 0
    pass_index: int = 0
    kind: MeasurementKind = MeasurementKind.DOPPLER
    contact_id: str | None = None

    @classmethod
    def from_scalar(
        cls,
        time: sk.instant | dt.datetime | float,
        value: float,
        variance: float,
        receiver_id: int = 0,
        pass_index: int = 0,
        kind: MeasurementKind = MeasurementKind.DOPPLER,
        contact_id: str | None = None,
    ) -> ForwardObservation:
        """Construct an observation for scalar observables (e.g., Doppler shift)."""
        instant = cls._coerce_instant(time)
        return cls(
            time=instant,
            observed=[float(value)],
            noise_cov=[[float(variance)]],
            receiver_id=receiver_id,
            pass_index=pass_index,
            kind=kind,
            contact_id=contact_id,
        )

    @staticmethod
    def _coerce_instant(time: sk.instant | dt.datetime | float) -> sk.instant:
        """Convert datetime or Unix epoch float into satkit Instant."""
        if isinstance(time, sk.instant):
            return time
        if isinstance(time, dt.datetime):
            return sk.instant.from_datetime(time)
        if isinstance(time, (int, float)):
            return sk.instant.from_unixtime(float(time))
        raise TypeError(f"Unsupported time type: {type(time)}")


@dataclass(slots=True)
class ForwardModelContext:
    """Batch estimation context matching Rust's EstimationEngine configuration.

    Consolidates receivers, pass-to-index mapping, and observations to avoid
    discrepancies between Python ingestion and the Rust calculation engine.
    """
    center_frequency_hz: float
    receivers: list[sk.itrfcoord] = field(default_factory=list)
    contacts: dict[str, ContactMetadata] = field(default_factory=dict)
    contact_to_pass_idx: dict[str, int] = field(default_factory=dict)
    system_to_receiver_idx: dict[str, int] = field(default_factory=dict)
    observations: list[ForwardObservation] = field(default_factory=list)

    @property
    def num_passes(self) -> int:
        """Return the number of registered passes."""
        return len(self.contact_to_pass_idx)

    def register_contact(self, contact: ContactMetadata) -> tuple[int, int]:
        """Register a contact, tracking receiver indices and pass allocations.

        Returns a tuple of (receiver_id, pass_index).
        """
        if contact.system_id not in self.system_to_receiver_idx:
            receiver_idx = len(self.receivers)
            self.receivers.append(contact.to_itrfcoord())
            self.system_to_receiver_idx[contact.system_id] = receiver_idx
        else:
            receiver_idx = self.system_to_receiver_idx[contact.system_id]

        contact_key = f"{contact.spacecraft_id}_{contact.start.isoformat()}"
        if contact_key not in self.contact_to_pass_idx:
            pass_idx = len(self.contact_to_pass_idx)
            self.contact_to_pass_idx[contact_key] = pass_idx
            self.contacts[contact_key] = contact
        else:
            pass_idx = self.contact_to_pass_idx[contact_key]

        return receiver_idx, pass_idx

    def add_observation(
        self,
        time: sk.instant | dt.datetime | float,
        value: float,
        variance: float,
        system_id: str,
        contact_key: str,
        kind: MeasurementKind = MeasurementKind.DOPPLER,
    ) -> None:
        """Append an observation with automatic receiver and pass resolution."""
        require(system_id in self.system_to_receiver_idx, f"Unknown system ID: {system_id}")
        require(contact_key in self.contact_to_pass_idx, f"Unknown contact key: {contact_key}")

        receiver_id = self.system_to_receiver_idx[system_id]
        pass_index = self.contact_to_pass_idx[contact_key]

        obs = ForwardObservation.from_scalar(
            time=time,
            value=value,
            variance=variance,
            receiver_id=receiver_id,
            pass_index=pass_index,
            kind=kind,
            contact_id=contact_key,
        )
        self.observations.append(obs)

    def validate(self) -> None:
        """Validate context consistency against the Rust EstimationEngine constraints."""
        require(
            self.center_frequency_hz > 0.0,
            "Center frequency must be finite and positive",
        )
        require(len(self.receivers) > 0, "At least one receiver must be registered")
        for i, obs in enumerate(self.observations):
            require(
                obs.receiver_id < len(self.receivers),
                f"Observation {i} receiver_id {obs.receiver_id} out of bounds",
            )
            if self.num_passes > 0:
                require(
                    obs.pass_index < self.num_passes,
                    f"Observation {i} pass_index {obs.pass_index} out of bounds",
                )


@dataclass(frozen=True, slots=True)
class BatchEvaluationResult:
    """Evaluated whitened residuals and Jacobian matrix from Rust."""
    residuals: list[float]
    residual_jacobian: list[list[float]]


def load_contact_metadata(
    contact_id: str,
    band: str,
    timeout_seconds: float,
    product: str,
    kogs_api_key: str | None = None,
) -> ContactMetadata:
    """Load and cross-check the KOGS identity for one delivery product."""
    key = kogs_api_key or os.getenv("KOGS_API_KEY", "")
    require(bool(key), f"KOGS_API_KEY is required for {product} export")
    auth = create_api_auth(key)
    raw = kogs.get_contact(auth, contact_id, timeout_seconds=timeout_seconds)
    contact = kogs.parse_reservation(kogs.unwrap_payload(raw, "contact"))
    required = (
        contact.id,
        contact.spacecraft_id,
        contact.system_id,
        contact.station_id,
        contact.start_time,
        contact.end_time,
        contact.ephemeris_id,
    )
    require(
        not any(value is None for value in required) and contact.id == contact_id,
        "KOGS contact is incomplete or does not match the request",
    )
    antenna = kogs.parse_response(
        kogs.get_antenna(auth, contact.system_id, timeout_seconds=timeout_seconds)
    )
    spacecraft_raw = kogs.get_spacecraft(
        auth, contact.spacecraft_id, timeout_seconds=timeout_seconds
    )
    satellite = kogs.parse_satellite(kogs.unwrap_payload(spacecraft_raw, "spacecraft"))
    ephemeris_raw = kogs.get_ephemeris(
        auth, contact.ephemeris_id, timeout_seconds=timeout_seconds
    )
    ephemeris = kogs.parse_ephemeris(
        kogs.unwrap_payload(ephemeris_raw, "ephemeris")
    )
    require(
        antenna.antenna_id == contact.system_id
        and antenna.station_id == contact.station_id,
        "KOGS antenna identity does not match the contact",
    )
    require(
        satellite.id == contact.spacecraft_id and bool(satellite.name),
        "KOGS spacecraft identity does not match the contact",
    )
    require(
        bool(antenna.antenna_name)
        and None not in (antenna.latitude, antenna.longitude, antenna.altitude),
        "KOGS antenna metadata is incomplete",
    )
    supported_bands = {
        item.strip() for item in (antenna.bands_types or "").split(",") if item.strip()
    }
    require(
        not supported_bands or band in supported_bands,
        f"KOGS antenna does not support {band}-band",
    )
    require(
        not ephemeris.spacecraft_uuid
        or ephemeris.spacecraft_uuid == contact.spacecraft_id,
        "KOGS ephemeris spacecraft does not match the contact",
    )
    catalog_value = satellite.satellite_catalog_number or satellite.norad_id
    require(
        catalog_value is not None and float(catalog_value).is_integer(),
        "KOGS spacecraft has no integral catalog ID",
    )
    catalog = str(int(catalog_value))
    if len(catalog) < 5:
        catalog = catalog.zfill(5)
    require(len(catalog) in {5, 9}, "KOGS catalog ID must contain 5 or 9 digits")
    cospar, ephemeris_catalog = kogs.parse_ephemeris_identity(ephemeris)
    require(
        ephemeris_catalog == catalog,
        "KOGS spacecraft and ephemeris catalog IDs do not match",
    )
    start = utc(contact.start_time, product)
    stop = utc(contact.end_time, product)
    require(stop > start, "KOGS contact stop must be later than its start")
    vector = sk.itrfcoord(
        latitude_deg=antenna.latitude,
        longitude_deg=antenna.longitude,
        altitude=antenna.altitude,
    ).vector
    return ContactMetadata(
        spacecraft_id=contact.spacecraft_id,
        system_id=contact.system_id,
        station_id=contact.station_id,
        ephemeris_id=contact.ephemeris_id,
        antenna=antenna.antenna_name,
        location=antenna.station_name or "UNKNOWN",
        latitude=antenna.latitude,
        longitude=antenna.longitude,
        altitude=antenna.altitude,
        ecef=tuple(float(value) for value in vector),
        spacecraft=satellite.name,
        cospar=cospar,
        catalog=catalog,
        start=start,
        stop=stop,
    )