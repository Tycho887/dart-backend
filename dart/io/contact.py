"""Provider-independent data structures returned by :mod:`dart.io`."""

from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass, field
from enum import Enum

import satkit as sk


class MeasurementKind(str, Enum):
    """Measurement types understood by the shared forward-model interface."""

    DOPPLER = "Doppler"
    TRUE_RANGE = "TrueRange"
    PSEUDORANGE_PHASE = "PseudorangePhase"


@dataclass(frozen=True, slots=True)
class EphemerisMetadata:
    """KOGS ephemeris content preserved without selecting a propagation model."""

    ephemeris_id: str
    spacecraft_id: str
    kind: str
    origin: str | None
    tenant_id: str | None
    epoch: dt.datetime | None
    last_usable_at: dt.datetime | None
    submitted_at: dt.datetime | None
    submitted_by: str | None
    tle: str | None
    omm: str | None
    oem: str | None
    is_cui: bool | None
    payload: str | None


@dataclass(frozen=True, slots=True)
class ContactMetadata:
    """Identity, station, and source-orbit metadata for one KOGS contact.

    Existing field names are retained for downstream migration. Coordinates
    are WGS-84 degrees/metres; ``ecef`` is metres in ITRF.
    """

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
    contact_id: str
    ephemeris: EphemerisMetadata

    def to_itrfcoord(self) -> sk.itrfcoord:
        """Convert the station coordinates into satkit's ITRF type."""

        return sk.itrfcoord(
            latitude_deg=self.latitude,
            longitude_deg=self.longitude,
            altitude=self.altitude,
        )


@dataclass(slots=True)
class ForwardObservation:
    """One model-independent observation in native SI units."""

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
        """Construct an observation for a scalar measurement."""

        if not math.isfinite(value):
            raise ValueError("observation value must be finite")
        if not math.isfinite(variance) or variance <= 0:
            raise ValueError("observation variance must be finite and positive")
        return cls(
            time=cls._coerce_instant(time),
            observed=[float(value)],
            noise_cov=[[float(variance)]],
            receiver_id=receiver_id,
            pass_index=pass_index,
            kind=kind,
            contact_id=contact_id,
        )

    @staticmethod
    def _coerce_instant(time: sk.instant | dt.datetime | float) -> sk.instant:
        if isinstance(time, sk.instant):
            return time
        if isinstance(time, dt.datetime):
            if time.tzinfo is None or time.utcoffset() is None:
                raise ValueError("observation datetime must be timezone-aware")
            return sk.instant.from_datetime(time)
        if isinstance(time, (int, float)):
            return sk.instant.from_unixtime(float(time))
        raise TypeError(f"unsupported time type: {type(time)}")


@dataclass(slots=True)
class ForwardModelContext:
    """Receivers, passes, and observations for a shared forward model."""

    center_frequency_hz: float
    receivers: list[sk.itrfcoord] = field(default_factory=list)
    contacts: dict[str, ContactMetadata] = field(default_factory=dict)
    contact_to_pass_idx: dict[str, int] = field(default_factory=dict)
    system_to_receiver_idx: dict[str, int] = field(default_factory=dict)
    observations: list[ForwardObservation] = field(default_factory=list)

    @property
    def num_passes(self) -> int:
        return len(self.contact_to_pass_idx)

    def register_contact(self, contact: ContactMetadata) -> tuple[int, int]:
        """Register a contact and return its receiver and pass indices."""

        if contact.system_id not in self.system_to_receiver_idx:
            receiver_idx = len(self.receivers)
            self.receivers.append(contact.to_itrfcoord())
            self.system_to_receiver_idx[contact.system_id] = receiver_idx
        else:
            receiver_idx = self.system_to_receiver_idx[contact.system_id]

        if contact.contact_id not in self.contact_to_pass_idx:
            pass_idx = len(self.contact_to_pass_idx)
            self.contact_to_pass_idx[contact.contact_id] = pass_idx
            self.contacts[contact.contact_id] = contact
        else:
            pass_idx = self.contact_to_pass_idx[contact.contact_id]
        return receiver_idx, pass_idx

    def add_observation(
        self,
        time: sk.instant | dt.datetime | float,
        value: float,
        variance: float,
        system_id: str,
        contact_id: str,
        kind: MeasurementKind = MeasurementKind.DOPPLER,
    ) -> None:
        """Append an observation using registered provider identities."""

        if system_id not in self.system_to_receiver_idx:
            raise ValueError(f"unknown system ID: {system_id}")
        if contact_id not in self.contact_to_pass_idx:
            raise ValueError(f"unknown contact ID: {contact_id}")
        self.observations.append(
            ForwardObservation.from_scalar(
                time=time,
                value=value,
                variance=variance,
                receiver_id=self.system_to_receiver_idx[system_id],
                pass_index=self.contact_to_pass_idx[contact_id],
                kind=kind,
                contact_id=contact_id,
            )
        )

    def validate(self) -> None:
        """Validate indices and scalar values before calling a model."""

        if not math.isfinite(self.center_frequency_hz) or self.center_frequency_hz <= 0:
            raise ValueError("center frequency must be finite and positive")
        if not self.receivers:
            raise ValueError("at least one receiver must be registered")
        if not self.observations:
            raise ValueError("at least one observation must be registered")
        for index, observation in enumerate(self.observations):
            if observation.receiver_id >= len(self.receivers):
                raise ValueError(f"observation {index} receiver_id is out of bounds")
            if observation.pass_index >= self.num_passes:
                raise ValueError(f"observation {index} pass_index is out of bounds")
