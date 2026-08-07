"""Interfaces separating estimation from antenna hardware transports."""

from __future__ import annotations

from enum import Enum
from typing import Protocol, runtime_checkable

from ..types import RFObservation


class OffsetConvention(str, Enum):
    """Sign used by an antenna backend.

    Dart's public convention is ``PROPAGATION_ADVANCE``.  Legacy Autofinder
    and some antenna systems expose a positive lag and therefore need the
    sign-converting adapter.
    """

    PROPAGATION_ADVANCE = "propagation_advance"
    POSITIVE_LAG = "positive_lag"


@runtime_checkable
class AntennaBackend(Protocol):
    convention: OffsetConvention

    def apply_offset(self, offset_s: float) -> None:
        """Apply an offset in the backend's declared sign convention."""

    def read(self) -> RFObservation | None:
        """Return the next delivered measurement, or ``None`` after the pass."""


class SignConvertingBackend:
    """Expose any backend using Dart's propagation-advance convention."""

    convention = OffsetConvention.PROPAGATION_ADVANCE

    def __init__(self, backend: AntennaBackend):
        self.backend = backend
        self._factor = (
            -1.0 if backend.convention is OffsetConvention.POSITIVE_LAG else 1.0
        )

    def apply_offset(self, offset_s: float) -> None:
        self.backend.apply_offset(self._factor * float(offset_s))

    def read(self) -> RFObservation | None:
        observation = self.backend.read()
        if observation is None or observation.applied_offset_s is None:
            return observation
        # RFObservation is frozen; reconstruct only the convention-dependent
        # field and retain all receiver metadata.
        return RFObservation(
            epoch=observation.epoch,
            station_id=observation.station_id,
            doppler_hz=observation.doppler_hz,
            phase_rad=observation.phase_rad,
            valid=observation.valid,
            commanded_az_deg=observation.commanded_az_deg,
            commanded_el_deg=observation.commanded_el_deg,
            applied_offset_s=self._factor * observation.applied_offset_s,
            sequence=observation.sequence,
            quality=dict(observation.quality),
        )

