from __future__ import annotations

import satkit as sk

from dart.control import DitherConfig, OffsetConvention, SignConvertingBackend, acquire
from dart.types import RFObservation


class VisibilityBackend:
    convention = OffsetConvention.PROPAGATION_ADVANCE

    def __init__(self, true_offset=30.0):
        self.true_offset = true_offset
        self.offset = 0.0
        self.sequence = 0
        self.start = sk.time(2024, 1, 1)

    def apply_offset(self, offset_s):
        self.offset = float(offset_s)

    def read(self):
        self.sequence += 1
        return RFObservation(
            epoch=self.start + sk.duration(seconds=float(self.sequence)),
            station_id="test",
            doppler_hz=1.0,
            valid=abs(self.offset - self.true_offset) <= 15.0,
            commanded_el_deg=30.0,
            applied_offset_s=self.offset,
            sequence=self.sequence,
        )


class LagBackend(VisibilityBackend):
    convention = OffsetConvention.POSITIVE_LAG


class UntaggedBackend(VisibilityBackend):
    def read(self):
        observation = super().read()
        return RFObservation(
            epoch=observation.epoch,
            station_id=observation.station_id,
            doppler_hz=observation.doppler_hz,
            valid=observation.valid,
            commanded_el_deg=observation.commanded_el_deg,
            applied_offset_s=None,
            sequence=observation.sequence,
        )


def test_dither_locks_at_valid_run_center():
    result = acquire(VisibilityBackend(), DitherConfig(max_sweeps=1))
    assert result.locked
    assert abs(result.offset_s - 30.0) <= 1.0


def test_sign_adapter_translates_legacy_positive_lag():
    backend = LagBackend(true_offset=-30.0)
    result = acquire(SignConvertingBackend(backend), DitherConfig(max_sweeps=1))
    assert result.locked
    assert abs(result.offset_s - 30.0) <= 1.0


def test_acquisition_rejects_untagged_delayed_measurements():
    result = acquire(
        UntaggedBackend(),
        DitherConfig(max_sweeps=1, max_delivery_reads=1),
    )
    assert not result.locked
    assert all(probe.status == "delivery_timeout" for probe in result.probes)
